"""
Stage 2 PPO Training Script (Phase 3B: Randomized Goal Navigation with Fixed Obstacles).
Loads the Stage 1 trained brain (models/stage1/best_model/best_model.zip),
attaches the Stage 2 environment with central obstacles enabled,
and trains for 250,000 additional Stage 2 timesteps headlessly.
Supports one-command training and visualization via --watch.
"""

import os
import sys
import argparse
import random
import subprocess
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    OBSERVATION_SIZE,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_SEED,
    PPO_STAGE2_TOTAL_TIMESTEPS,
    PPO_STAGE2_CHECKPOINT_FREQ,
    PPO_STAGE2_EVAL_EPISODES,
    PPO_STAGE2_EVAL_SEED,
)
from rl_environment import StealthGymEnv
from training_callbacks import Stage2EvalCallback


def set_seed(seed: int):
    """Seed python, numpy, and pytorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on Curriculum Stage 2 with Fixed Obstacles (Phase 3B)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=PPO_STAGE2_TOTAL_TIMESTEPS,
        help=f"Total Stage 2 training timesteps (default: {PPO_STAGE2_TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PPO_SEED,
        help=f"Random seed (default: {PPO_SEED})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default="models/stage1/best_model/best_model.zip",
        help="Path to Stage 1 best model to continue learning from (default: 'models/stage1/best_model/best_model.zip')"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing Stage 2 checkpoint to resume training from"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=PPO_STAGE2_CHECKPOINT_FREQ,
        help=f"Checkpoints and evaluation frequency (default: {PPO_STAGE2_CHECKPOINT_FREQ})"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=PPO_STAGE2_EVAL_EPISODES,
        help=f"Number of deterministic episodes per evaluation (default: {PPO_STAGE2_EVAL_EPISODES})"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/stage2",
        help="Directory to save Stage 2 models and checkpoints (default: 'models/stage2')"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/stage2",
        help="Directory for Stage 2 logs and summaries (default: 'logs/stage2')"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Automatically launch live Pygame spectator in a separate window"
    )
    parser.add_argument(
        "--watch-stochastic",
        action="store_true",
        help="When --watch is active, display stochastic sampled actions instead of default deterministic"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 65)
    print("PHASE 3B: CURRICULUM STAGE 2 PPO TRAINING")
    print("RANDOMIZED GOAL NAVIGATION WITH FIXED OBSTACLES")
    print("=" * 65)
    print(f"Target Timesteps:    {args.timesteps:,} (Stage 2)")
    print(f"Random Seed:         {args.seed}")
    print(f"Compute Device:      {args.device}")
    print(f"Evaluation Freq:     Every {args.checkpoint_freq:,} steps ({args.eval_episodes} deterministic eps)")
    print(f"Evaluation Seed:     {PPO_STAGE2_EVAL_SEED}")
    print(f"Hyperparameters:     lr={PPO_LEARNING_RATE}, n_steps={PPO_N_STEPS}, batch_size={PPO_BATCH_SIZE}")
    print(f"                     n_epochs={PPO_N_EPOCHS}, gamma={PPO_GAMMA}, gae_lambda={PPO_GAE_LAMBDA}, clip={PPO_CLIP_RANGE}")
    print(f"Live Spectator:      {'ACTIVE (--watch)' if args.watch else 'OFF (headless training)'}")
    print("=" * 65)

    # Directories setup (Stage 2 isolation)
    checkpoints_dir = os.path.join(args.model_dir, "checkpoints")
    best_model_dir = os.path.join(args.model_dir, "best_model")
    final_model_dir = os.path.join(args.model_dir, "final")
    monitor_dir = os.path.join(args.log_dir, "monitor")
    tensorboard_dir = os.path.join(args.log_dir, "tensorboard")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Check Stage 1 model existence and compatibility if not resuming
    if not args.resume:
        if not os.path.exists(args.stage1_model):
            print(f"[ERROR] Stage 1 best model not found at: {args.stage1_model}")
            print("Curriculum Stage 2 requires an existing Stage 1 model as starting brain.")
            sys.exit(1)
        try:
            temp_model = PPO.load(args.stage1_model, device="cpu")
            saved_obs_dim = (
                temp_model.observation_space.shape[0]
                if (temp_model.observation_space is not None and len(temp_model.observation_space.shape) > 0)
                else None
            )
            del temp_model
            if saved_obs_dim != OBSERVATION_SIZE:
                print("[ERROR] Stage 1 model is incompatible with the new 43-dimensional observation space. Retrain Stage 1 first.")
                sys.exit(1)
        except Exception:
            print("[ERROR] Stage 1 model is incompatible with the new 43-dimensional observation space. Retrain Stage 1 first.")
            sys.exit(1)
    else:
        if not os.path.exists(args.resume):
            print(f"[ERROR] Resume checkpoint not found at: {args.resume}")
            sys.exit(1)
        try:
            temp_model = PPO.load(args.resume, device="cpu")
            saved_obs_dim = (
                temp_model.observation_space.shape[0]
                if (temp_model.observation_space is not None and len(temp_model.observation_space.shape) > 0)
                else None
            )
            del temp_model
            if saved_obs_dim != OBSERVATION_SIZE:
                print(f"[ERROR] Resumed checkpoint is incompatible with the new 43-dimensional observation space (found {saved_obs_dim}-dim).")
                sys.exit(1)
        except Exception:
            print("[ERROR] Resumed checkpoint is incompatible with the new 43-dimensional observation space.")
            sys.exit(1)

    # 1. Exactly ONE training environment, headless, Stage 2
    def make_train_env():
        raw_env = StealthGymEnv(stage=2, render_mode=None)
        monitored_env = Monitor(raw_env, filename=os.path.join(monitor_dir, "train_monitor.csv"))
        return monitored_env

    train_vec_env = DummyVecEnv([make_train_env])

    # 2. Separate evaluation environment, headless, Stage 2
    eval_env = StealthGymEnv(stage=2, render_mode=None)

    # 3. Callback configuration (Deterministic Best Model Ranking)
    callback = Stage2EvalCallback(
        eval_env=eval_env,
        eval_freq=args.checkpoint_freq,
        n_eval_episodes=args.eval_episodes,
        eval_seed=PPO_STAGE2_EVAL_SEED,
        checkpoints_dir=checkpoints_dir,
        best_model_dir=best_model_dir,
        log_dir=args.log_dir,
        verbose=1
    )

    # 4. Optional live spectator launch (--watch)
    spectator_process: Optional[subprocess.Popen] = None
    if args.watch:
        print("[INFO] Launching live visual spectator process...")
        spectator_cmd = [
            sys.executable,
            "live_watch_stage2.py",
            "--checkpoint-dir", checkpoints_dir,
            "--stage1-baseline", args.stage1_model,
            "--summary-csv", os.path.join(args.log_dir, "stage2_eval_summary.csv"),
            "--device", args.device,
        ]
        if args.watch_stochastic:
            spectator_cmd.append("--stochastic")

        try:
            spectator_process = subprocess.Popen(spectator_cmd)
            print(f"[INFO] Spectator PID {spectator_process.pid} running in separate window.")
        except Exception as e:
            print(f"[WARNING] Failed to spawn live spectator: {e}")
            print("Continuing headless training...")

    # 5. Model Loading (Stage 1 transfer or Stage 2 resume)
    try:
        import tensorboard
        tb_log_dir = tensorboard_dir
    except ImportError:
        tb_log_dir = None
        print("[INFO] TensorBoard not installed; continuing with CSV logs only.")

    if args.resume:
        print(f"Resuming Stage 2 training from checkpoint: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_vec_env,
            device=args.device,
            tensorboard_log=tb_log_dir
        )
        reset_num_timesteps = False
    else:
        print(f"Loading Stage 1 brain from: {args.stage1_model}")
        model = PPO.load(
            args.stage1_model,
            env=train_vec_env,
            device=args.device,
            tensorboard_log=tb_log_dir
        )
        # Ensure hyperparameters match Stage 2 config
        from stable_baselines3.common.utils import get_schedule_fn
        model.lr_schedule = get_schedule_fn(PPO_LEARNING_RATE)
        model.n_steps = PPO_N_STEPS
        model.batch_size = PPO_BATCH_SIZE
        model.n_epochs = PPO_N_EPOCHS
        model.gamma = PPO_GAMMA
        model.gae_lambda = PPO_GAE_LAMBDA
        model.clip_range = get_schedule_fn(PPO_CLIP_RANGE)
        # Reset timestep counter so Stage 2 progress is tracked 0 -> 250k
        reset_num_timesteps = True
        print("[INFO] Successfully transferred Stage 1 policy and value network weights.")
        print("[INFO] Stage 2 learning starting on top of Stage 1 brain.")

    # 6. Execute training
    print(f"\nBeginning headless Stage 2 training ({args.timesteps:,} steps)...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    # 7. Save final Stage 2 model
    final_model_path = os.path.join(final_model_dir, "stage2_final.zip")
    model.save(final_model_path)
    print("=" * 65)
    print("STAGE 2 TRAINING COMPLETE")
    print(f"Final Model Saved:   {final_model_path}")
    print(f"Best Model Saved:    {os.path.join(best_model_dir, 'best_model.zip')}")
    print(f"Evaluation Summary:  {callback.summary_csv_path}")
    print("=" * 65)

    # Clean up training environments
    train_vec_env.close()
    eval_env.close()

    if spectator_process is not None:
        print("[INFO] Spectator process remains active for visual review.")
        print("Close the spectator window or press ESC to exit spectator.")


if __name__ == "__main__":
    main()
