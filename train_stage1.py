"""
Stage 1 PPO Training Script (Phase 3A).
Trains Stable-Baselines3 PPO on curriculum Stage 1 headlessly with 64x64 MLP architecture.
"""

import os
import sys
import subprocess
import argparse
import random
from typing import Optional
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_SEED,
    PPO_NET_ARCH,
    PPO_TOTAL_TIMESTEPS,
    PPO_CHECKPOINT_FREQ,
    PPO_EVAL_EPISODES,
)
from rl_environment import StealthGymEnv
from training_callbacks import Stage1EvalCallback


def set_seed(seed: int):
    """Seed python, numpy, and pytorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Curriculum Stage 1 (Phase 3A).")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=PPO_TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {PPO_TOTAL_TIMESTEPS})"
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
        "--resume",
        type=str,
        default=None,
        help="Path to existing model checkpoint to resume training from"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=PPO_CHECKPOINT_FREQ,
        help=f"Checkpoints and evaluation frequency (default: {PPO_CHECKPOINT_FREQ})"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=PPO_EVAL_EPISODES,
        help=f"Number of episodes per evaluation (default: {PPO_EVAL_EPISODES})"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/stage1",
        help="Directory to save models and checkpoints (default: 'models/stage1')"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/stage1",
        help="Directory for logs and summaries (default: 'logs/stage1')"
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

    print("=" * 60)
    print("PHASE 3A: STAGE 1 PPO TRAINING INFRASTRUCTURE")
    print("=" * 60)
    print(f"Target Timesteps:    {args.timesteps:,}")
    print(f"Random Seed:         {args.seed}")
    print(f"Compute Device:      {args.device}")
    print(f"Evaluation Freq:     Every {args.checkpoint_freq:,} steps ({args.eval_episodes} stochastic eps)")
    print(f"Network Arch:        MLP Policy [64, 64], Value [64, 64]")
    print(f"Hyperparameters:     lr={PPO_LEARNING_RATE}, n_steps={PPO_N_STEPS}, batch_size={PPO_BATCH_SIZE}")
    print(f"                     n_epochs={PPO_N_EPOCHS}, gamma={PPO_GAMMA}, gae_lambda={PPO_GAE_LAMBDA}, clip={PPO_CLIP_RANGE}")
    print("=" * 60)

    # Directories setup
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

    # 1. Exactly ONE training environment, headless, Stage 1
    def make_train_env():
        raw_env = StealthGymEnv(stage=1, render_mode=None)
        monitored_env = Monitor(raw_env, filename=os.path.join(monitor_dir, "train_monitor.csv"))
        return monitored_env

    train_vec_env = DummyVecEnv([make_train_env])

    # 2. Separate evaluation environment, headless, Stage 1
    eval_env = StealthGymEnv(stage=1, render_mode=None)

    # 3. Callback configuration
    # 3. Callback configuration
    callback = Stage1EvalCallback(
        eval_env=eval_env,
        eval_freq=args.checkpoint_freq,
        n_eval_episodes=args.eval_episodes,
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
            "live_watch_training.py",
            "--checkpoint-dir", checkpoints_dir,
            "--device", args.device,
        ]
        if not args.watch_stochastic:
            spectator_cmd.append("--deterministic")

        try:
            spectator_process = subprocess.Popen(spectator_cmd)
            print(f"[INFO] Spectator PID {spectator_process.pid} running in separate window.")
        except Exception as e:
            print(f"[WARNING] Failed to spawn live spectator: {e}")
            print("Continuing headless training...")

    # 5. Model initialization or resuming
    policy_kwargs = dict(
        net_arch=PPO_NET_ARCH
    )

    # Check if tensorboard is available
    try:
        import tensorboard
        tb_log_dir = tensorboard_dir
    except ImportError:
        tb_log_dir = None
        print("[INFO] TensorBoard not installed; continuing with CSV logs only.")

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_vec_env,
            device=args.device
        )
        reset_num_timesteps = False
    else:
        print("Initializing fresh PPO model with 2x64 MLP architecture...")
        model = PPO(
            policy="MlpPolicy",
            env=train_vec_env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=args.seed,
            device=args.device,
            tensorboard_log=tb_log_dir
        )
        reset_num_timesteps = True

    # 6. Execute training
    print(f"Beginning headless training run ({args.timesteps:,} steps)...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    # 7. Save final model
    final_model_path = os.path.join(final_model_dir, "stage1_final.zip")
    model.save(final_model_path)
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final Model Saved:   {final_model_path}")
    print(f"Best Model Saved:    {os.path.join(best_model_dir, 'best_model.zip')}")
    print(f"Evaluation Summary:  {callback.summary_csv_path}")
    print("=" * 60)

    # Clean up environments
    train_vec_env.close()
    eval_env.close()

    if spectator_process is not None:
        print("[INFO] Spectator process remains active for visual review.")
        print("Close the spectator window or press ESC to exit spectator.")


if __name__ == "__main__":
    main()
