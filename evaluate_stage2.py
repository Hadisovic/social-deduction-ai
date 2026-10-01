"""
Evaluation script for Stage 2 PPO models (Phase 3B: Fixed Obstacle Navigation).
Runs evaluation episodes over randomized Stage 2 layouts using a fixed evaluation seed.
Evaluates both deterministic (argmax) and stochastic (sampled) policy modes.
Reports full metrics including straight-line efficiency and wall contact diagnostics.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO

from config import (
    PPO_STAGE2_FINAL_EVAL_EPISODES,
    PPO_STAGE2_EVAL_SEED,
)
from rl_environment import StealthGymEnv
from training_callbacks import run_evaluation_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Stage 2 PPO model.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/stage2/best_model/best_model.zip",
        help="Path to the model zip checkpoint to evaluate (default: 'models/stage2/best_model/best_model.zip')"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=PPO_STAGE2_FINAL_EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {PPO_STAGE2_FINAL_EVAL_EPISODES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PPO_STAGE2_EVAL_SEED,
        help=f"Evaluation layout seed for reproducible layout sequence (default: {PPO_STAGE2_EVAL_SEED})"
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Run only deterministic evaluation mode"
    )
    parser.add_argument(
        "--stochastic-only",
        action="store_true",
        help="Run only stochastic evaluation mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    return parser.parse_args()


def print_mode_report(mode_name: str, metrics: dict):
    print(f"\n--- {mode_name} Evaluation ({metrics['n_episodes']} episodes) ---")
    print(f"Successes:                    {metrics['successes']} / {metrics['n_episodes']}")
    print(f"Failures:                     {metrics['failures']}")
    print(f"Success Rate:                 {metrics['success_rate'] * 100:.2f}%")
    print(f"Mean Episode Reward:          {metrics['mean_reward']:.2f}")
    print(f"Median Episode Reward:        {metrics['median_reward']:.2f}")
    print(f"Std Dev Reward:               {metrics['std_reward']:.2f}")
    print(f"Average Initial Distance:     {metrics['avg_initial_distance']:.2f} px")

    if metrics["successes"] > 0:
        print(f"Average Successful Time:      {metrics['avg_success_time']:.2f} s")
        print(f"Fastest Successful Time:      {metrics['fastest_success_time']:.2f} s")
        print(f"Slowest Successful Time:      {metrics['slowest_success_time']:.2f} s")
        print(f"Straight-Line Efficiency:     {metrics['avg_path_efficiency']:.2%} (diagnostic)")
    else:
        print("Average Successful Time:      N/A (0 successes)")

    if metrics["failures"] > 0:
        print(f"Avg Final Dist (Failures):    {metrics['avg_fail_distance']:.2f} px")
    else:
        print("Avg Final Dist (Failures):    N/A (0 failures)")

    avg_wall = metrics.get("avg_wall_contact_fraction", 0.0)
    print(f"Avg Wall Contact Fraction:    {avg_wall:.2%} of episode steps (diagnostic)")


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return

    print("=" * 65)
    print("STAGE 2 MODEL EVALUATION (FIXED OBSTACLE NAVIGATION)")
    print("=" * 65)
    print(f"Model Path:          {args.model}")
    print(f"Episodes:            {args.episodes}")
    print(f"Layout Base Seed:    {args.seed}")
    print(f"Compute Device:      {args.device}")
    print("=" * 65)

    eval_env = StealthGymEnv(stage=2, render_mode=None)
    model = PPO.load(args.model, env=eval_env, device=args.device)

    run_det = not args.stochastic_only
    run_stoch = not args.deterministic_only

    det_metrics = None
    stoch_metrics = None

    if run_det:
        print(f"\nRunning {args.episodes} DETERMINISTIC evaluation episodes...")
        det_metrics = run_evaluation_episodes(
            model=model,
            env=eval_env,
            n_episodes=args.episodes,
            deterministic=True,
            eval_seed=args.seed
        )
        print_mode_report("DETERMINISTIC MODE", det_metrics)

    if run_stoch:
        print(f"\nRunning {args.episodes} STOCHASTIC evaluation episodes...")
        stoch_metrics = run_evaluation_episodes(
            model=model,
            env=eval_env,
            n_episodes=args.episodes,
            deterministic=False,
            eval_seed=args.seed
        )
        print_mode_report("STOCHASTIC MODE", stoch_metrics)

    eval_env.close()

    # Pass / Fail criteria check
    print("\n" + "=" * 65)
    print("FINAL EVALUATION VERDICT")
    print("=" * 65)
    if stoch_metrics is not None:
        sr = stoch_metrics["success_rate"]
        print(f"Stochastic Success Rate:      {sr * 100:.2f}% (Threshold: >= 95.00%)")
        if det_metrics is not None:
            print(f"Deterministic Success Rate:   {det_metrics['success_rate'] * 100:.2f}%")
        if sr >= 0.95:
            print("RESULT: >>> STAGE 2 PASSED <<<")
        else:
            print(f"RESULT: >>> STAGE 2 NOT YET PASSED <<< (requires >= 95.0% stochastic success rate)")
    elif det_metrics is not None:
        sr = det_metrics["success_rate"]
        print(f"Deterministic Success Rate:   {sr * 100:.2f}%")
        if sr >= 0.95:
            print("RESULT: >>> STAGE 2 DETERMINISTIC PASSED <<<")
        else:
            print(f"RESULT: >>> STAGE 2 NOT YET PASSED <<<")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
