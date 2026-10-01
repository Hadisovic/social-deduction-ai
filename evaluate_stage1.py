"""
Evaluation script for Stage 1 PPO models (Phase 3A).
Runs headless stochastic (or deterministic) evaluation episodes and reports PASS/FAIL.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO

from config import PPO_FINAL_EVAL_EPISODES, PPO_EVAL_SEED
from rl_environment import StealthGymEnv
from training_callbacks import run_evaluation_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Stage 1 PPO model.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/stage1/final/stage1_final.zip",
        help="Path to the model zip checkpoint to evaluate"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=PPO_FINAL_EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {PPO_FINAL_EVAL_EPISODES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PPO_EVAL_SEED,
        help=f"Evaluation layout seed for reproducible layout sequence (default: {PPO_EVAL_SEED})"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions (argmax) instead of stochastic sampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return

    print("=" * 60)
    print("STAGE 1 MODEL EVALUATION")
    print("=" * 60)
    print(f"Model Path:          {args.model}")
    print(f"Episodes:            {args.episodes}")
    print(f"Evaluation Mode:     {'Deterministic (no noise)' if args.deterministic else 'Stochastic (sampled actions)'}")
    print(f"Layout Base Seed:    {args.seed}")
    print(f"Device:              {args.device}")
    print("=" * 60)

    # 1. Load model and evaluation environment
    eval_env = StealthGymEnv(stage=1, render_mode=None)
    model = PPO.load(args.model, env=eval_env, device=args.device)

    # 2. Run evaluation
    print(f"Evaluating {args.episodes} episodes headlessly...")
    metrics = run_evaluation_episodes(
        model=model,
        env=eval_env,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        eval_seed=args.seed
    )
    eval_env.close()

    # 3. Print clean report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Episodes:               {metrics['n_episodes']}")
    print(f"Successes:                    {metrics['successes']}")
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
        print(f"Average Path Efficiency:      {metrics['avg_path_efficiency']:.2%}")
    else:
        print("Average Successful Time:      N/A (0 successes)")

    if metrics["failures"] > 0:
        print(f"Avg Final Dist (Failures):    {metrics['avg_fail_distance']:.2f} px")
    else:
        print("Avg Final Dist (Failures):    N/A (0 failures)")

    print("=" * 60)

    # 4. Official Stage 1 Criterion: Success Rate >= 95%
    if metrics["success_rate"] >= 0.95:
        print("RESULT: >>> STAGE 1 PASSED <<<")
    else:
        print("RESULT: >>> STAGE 1 NOT YET PASSED <<< (requires >= 95.0% success rate)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
