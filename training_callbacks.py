"""
Training and Evaluation callbacks and utilities for Stage 1 PPO (Phase 3A).
Handles headless periodic evaluation, best-model selection, checkpointing, and logging.
"""

import os
import csv
import json
import math
from typing import Dict, Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    RL_DT,
    PPO_EVAL_SEED,
    PPO_STAGE2_EVAL_SEED,
    PPO_STAGE2_CHECKPOINT_FREQ,
    PPO_STAGE2_EVAL_EPISODES,
)


def run_evaluation_episodes(
    model,
    env,
    n_episodes: int = 20,
    deterministic: bool = False,
    eval_seed: Optional[int] = PPO_EVAL_SEED
) -> Dict[str, Any]:
    """
    Run evaluation episodes headlessly on an independent evaluation environment.
    
    Args:
        model: Stable-Baselines3 model (or policy).
        env: Gymnasium environment (StealthGymEnv).
        n_episodes: Number of episodes to evaluate.
        deterministic: Whether to sample stochastically (False) or take argmax (True).
        eval_seed: Base seed for reproducible evaluation layout sequence.
        
    Returns:
        Structured dictionary of evaluation statistics.
    """
    episode_records = []
    successes = 0
    total_rewards = []
    success_times = []
    fail_distances = []
    initial_distances = []
    success_path_efficiencies = []

    for ep_idx in range(n_episodes):
        # Episode 0 re-seeds the eval environment so every evaluation point sees
        # the identical sequence of randomized layouts across checkpoints.
        if ep_idx == 0 and eval_seed is not None:
            obs, info = env.reset(seed=eval_seed)
        else:
            obs, info = env.reset()

        done = False
        ep_reward = 0.0
        ep_steps = 0
        is_success = False
        last_distance = info.get("distance_to_goal", 0.0)
        initial_dist = info.get("initial_distance_to_goal", last_distance)
        initial_distances.append(initial_dist)

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            last_distance = info.get("distance_to_goal", last_distance)
            done = terminated or truncated

        is_success = bool(info.get("is_success", False))
        elapsed_time = float(info.get("elapsed_time", ep_steps * RL_DT))
        dist_travelled = float(info.get("distance_travelled", 0.0))
        path_eff = float(info.get("path_efficiency", 1.0))

        wall_steps = int(info.get("wall_contact_steps", 0))
        wall_frac = float(info.get("wall_contact_fraction", 0.0))

        if is_success:
            successes += 1
            success_times.append(elapsed_time)
            success_path_efficiencies.append(path_eff)
        else:
            fail_distances.append(last_distance)

        total_rewards.append(ep_reward)
        episode_records.append({
            "episode": ep_idx + 1,
            "success": is_success,
            "reward": round(ep_reward, 3),
            "steps": ep_steps,
            "elapsed_time": round(elapsed_time, 3),
            "initial_distance": round(initial_dist, 2),
            "final_distance": round(last_distance, 2),
            "distance_travelled": round(dist_travelled, 2),
            "path_efficiency": round(path_eff, 3),
            "wall_contact_steps": wall_steps,
            "wall_contact_fraction": round(wall_frac, 4),
        })

    success_rate = float(successes / n_episodes) if n_episodes > 0 else 0.0
    mean_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    median_reward = float(np.median(total_rewards)) if total_rewards else 0.0
    std_reward = float(np.std(total_rewards)) if total_rewards else 0.0

    avg_success_time = float(np.mean(success_times)) if len(success_times) > 0 else float("inf")
    fastest_success_time = float(np.min(success_times)) if len(success_times) > 0 else None
    slowest_success_time = float(np.max(success_times)) if len(success_times) > 0 else None
    avg_fail_distance = float(np.mean(fail_distances)) if len(fail_distances) > 0 else 0.0
    avg_initial_distance = float(np.mean(initial_distances)) if initial_distances else 0.0
    avg_path_efficiency = float(np.mean(success_path_efficiencies)) if success_path_efficiencies else 0.0
    avg_wall_contact_fraction = (
        float(np.mean([e["wall_contact_fraction"] for e in episode_records]))
        if episode_records
        else 0.0
    )

    return {
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "eval_seed": eval_seed,
        "successes": successes,
        "failures": n_episodes - successes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "median_reward": median_reward,
        "std_reward": std_reward,
        "avg_success_time": avg_success_time,
        "fastest_success_time": fastest_success_time,
        "slowest_success_time": slowest_success_time,
        "avg_fail_distance": avg_fail_distance,
        "avg_initial_distance": avg_initial_distance,
        "avg_path_efficiency": avg_path_efficiency,
        "avg_wall_contact_fraction": avg_wall_contact_fraction,
        "episodes": episode_records
    }


class Stage1EvalCallback(BaseCallback):
    """
    Periodic evaluation and checkpointing callback for Stage 1 PPO.
    
    Features:
    - Independent headless evaluation on separate eval environment every eval_freq steps.
    - Uses stochastic PPO action selection (deterministic=False).
    - Best model selection:
        1. Higher success rate.
        2. If tied and success_rate > 0: lower average successful completion time.
        3. If tied at 0: higher mean evaluation reward.
    - Saves periodic checkpoints to models/stage1/checkpoints/stage1_<step>_steps.zip.
    - Saves best model to models/stage1/best_model/best_model.zip.
    - Writes evaluation summaries to CSV and JSON logs.
    - Prints concise, clean progress blocks to terminal.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        eval_seed: int = PPO_EVAL_SEED,
        checkpoints_dir: str = "models/stage1/checkpoints",
        best_model_dir: str = "models/stage1/best_model",
        log_dir: str = "logs/stage1",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.checkpoints_dir = checkpoints_dir
        self.best_model_dir = best_model_dir
        self.log_dir = log_dir

        self.last_eval_step = 0
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.best_timestep: int = 0

        # Create output directories
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "evaluations"), exist_ok=True)
        self.summary_csv_path = os.path.join(self.log_dir, "stage1_eval_summary.csv")

        # Initialize CSV log with header if not exists
        if not os.path.exists(self.summary_csv_path):
            with open(self.summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep",
                    "success_rate",
                    "mean_reward",
                    "avg_success_time",
                    "avg_initial_distance",
                    "avg_path_efficiency",
                    "best_so_far"
                ])

    def _is_better_model(self, candidate: Dict[str, Any]) -> bool:
        """
        Rank Stage 1 models strictly using deterministic metrics:
        1. PRIMARY: highest deterministic success rate.
        2. SECONDARY (tied & success_rate > 0): lowest average successful completion time.
        3. TERTIARY (still tied): higher mean evaluation reward.
        4. If still tied: preserve earlier existing best model (return False).
        """
        if self.best_metrics is None:
            return True

        cand_sr = candidate["success_rate"]
        best_sr = self.best_metrics["success_rate"]

        # 1. Primary: Highest deterministic success rate
        if cand_sr > best_sr:
            return True
        elif cand_sr < best_sr:
            return False

        # 2. Secondary: If success rate > 0 and tied, lowest avg completion time wins
        if cand_sr > 0.0:
            if candidate["avg_success_time"] < (self.best_metrics["avg_success_time"] - 1e-4):
                return True
            elif candidate["avg_success_time"] > (self.best_metrics["avg_success_time"] + 1e-4):
                return False

        # 3. Tertiary: Higher mean evaluation reward
        if candidate["mean_reward"] > (self.best_metrics["mean_reward"] + 1e-4):
            return True
        elif candidate["mean_reward"] < (self.best_metrics["mean_reward"] - 1e-4):
            return False

        # 4. If identical, preserve earlier model
        return False

    def _on_step(self) -> bool:
        # Check if periodic evaluation interval reached
        if (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._run_eval_and_checkpoint()

        return True

    def _run_eval_and_checkpoint(self):
        curr_step = self.num_timesteps
        metrics = run_evaluation_episodes(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            deterministic=True,  # Mandatory deterministic evaluation for ranking
            eval_seed=self.eval_seed
        )

        is_best = self._is_better_model(metrics)
        if is_best:
            self.best_metrics = metrics
            self.best_timestep = curr_step
            best_model_path = os.path.join(self.best_model_dir, "best_model.zip")
            self.model.save(best_model_path)

        # Save periodic checkpoint
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"stage1_{curr_step:06d}_steps.zip"
        )
        self.model.save(checkpoint_path)

        # Log to summary CSV
        avg_time_str = (
            f"{metrics['avg_success_time']:.2f}"
            if metrics["avg_success_time"] != float("inf")
            else "N/A"
        )
        avg_eff_str = (
            f"{metrics['avg_path_efficiency']:.4f}"
            if metrics["avg_path_efficiency"] > 0
            else "N/A"
        )
        with open(self.summary_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                curr_step,
                f"{metrics['success_rate']:.4f}",
                f"{metrics['mean_reward']:.2f}",
                avg_time_str,
                f"{metrics['avg_initial_distance']:.2f}",
                avg_eff_str,
                "YES" if is_best else "NO"
            ])

        # Write detailed JSON log
        detail_path = os.path.join(
            self.log_dir,
            "evaluations",
            f"eval_step_{curr_step:06d}.json"
        )
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestep": curr_step,
                "is_best": is_best,
                "metrics": metrics
            }, f, indent=2)

        # Concise terminal output
        if self.verbose > 0:
            print("\n" + "=" * 50)
            print(f"[Stage 1 Evaluation (Deterministic)]")
            print(f"Timesteps:            {curr_step:,}")
            print(f"Eval success:         {metrics['success_rate'] * 100:.1f}% ({metrics['successes']}/{self.n_eval_episodes})")
            print(f"Mean reward:          {metrics['mean_reward']:.2f}")
            print(f"Avg initial distance: {metrics['avg_initial_distance']:.1f} px")
            if metrics["avg_success_time"] != float("inf"):
                print(f"Avg success time:     {metrics['avg_success_time']:.2f} s")
                print(f"Avg path efficiency:  {metrics['avg_path_efficiency']:.2%}")
            else:
                print(f"Avg success time:     N/A (0 successes)")
            if is_best:
                print(f"Best model:           UPDATED (step {curr_step:,})")
            else:
                print(f"Best model:           Unchanged (from step {self.best_timestep:,})")
            print("=" * 50 + "\n")


class Stage2EvalCallback(BaseCallback):
    """
    Periodic evaluation and checkpointing callback for Stage 2 PPO (Phase 3B).
    
    Features:
    - Independent headless evaluation on separate eval environment every eval_freq steps.
    - Uses deterministic PPO action selection (model.predict(..., deterministic=True)).
    - Fixed evaluation seed (PPO_STAGE2_EVAL_SEED) so all checkpoints see identical layout sequences.
    - Best model selection:
        1. PRIMARY: Higher deterministic success rate.
        2. SECONDARY (tied & success_rate > 0): Lower average successful completion time.
        3. TERTIARY (still tied): Higher mean evaluation reward.
        4. If still tied: Keep earlier existing best model.
        (Straight-line path efficiency is NOT used for ranking, but logged as diagnostic).
    - Saves periodic checkpoints to models/stage2/checkpoints/stage2_<step>_steps.zip.
    - Saves best model to models/stage2/best_model/best_model.zip.
    - Writes evaluation summaries to CSV and JSON logs.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = PPO_STAGE2_CHECKPOINT_FREQ,
        n_eval_episodes: int = PPO_STAGE2_EVAL_EPISODES,
        eval_seed: int = PPO_STAGE2_EVAL_SEED,
        checkpoints_dir: str = "models/stage2/checkpoints",
        best_model_dir: str = "models/stage2/best_model",
        log_dir: str = "logs/stage2",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.checkpoints_dir = checkpoints_dir
        self.best_model_dir = best_model_dir
        self.log_dir = log_dir

        self.last_eval_step = 0
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.best_timestep: int = 0

        # Create output directories
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "evaluations"), exist_ok=True)
        self.summary_csv_path = os.path.join(self.log_dir, "stage2_eval_summary.csv")

        # Initialize CSV log with header if not exists
        if not os.path.exists(self.summary_csv_path):
            with open(self.summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep",
                    "success_rate",
                    "mean_reward",
                    "avg_success_time",
                    "avg_initial_distance",
                    "avg_straight_line_efficiency",
                    "avg_wall_contact_fraction",
                    "best_so_far"
                ])

    def _is_better_model(self, candidate: Dict[str, Any]) -> bool:
        """
        Rank Stage 2 models strictly using deterministic metrics:
        1. PRIMARY: highest deterministic success rate.
        2. SECONDARY (tied & success_rate > 0): lowest average successful completion time.
        3. TERTIARY (still tied): higher mean evaluation reward.
        4. If still tied: keep earlier existing best model (return False).
        Straight-line efficiency is deliberately NOT used for ranking because necessary
        detours around obstacles naturally lower straight-line efficiency.
        """
        if self.best_metrics is None:
            return True

        cand_sr = candidate["success_rate"]
        best_sr = self.best_metrics["success_rate"]

        # 1. Primary: Highest deterministic success rate
        if cand_sr > best_sr:
            return True
        elif cand_sr < best_sr:
            return False

        # 2. Secondary: If success rate > 0 and tied, lowest avg completion time wins
        if cand_sr > 0.0:
            if candidate["avg_success_time"] < (self.best_metrics["avg_success_time"] - 1e-4):
                return True
            elif candidate["avg_success_time"] > (self.best_metrics["avg_success_time"] + 1e-4):
                return False

        # 3. Tertiary: Higher mean evaluation reward
        if candidate["mean_reward"] > (self.best_metrics["mean_reward"] + 1e-4):
            return True
        elif candidate["mean_reward"] < (self.best_metrics["mean_reward"] - 1e-4):
            return False

        # If still tied: keep earlier model rather than replacing for noise
        return False

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._run_eval_and_checkpoint()

        return True

    def _run_eval_and_checkpoint(self):
        curr_step = self.num_timesteps
        metrics = run_evaluation_episodes(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            deterministic=True,  # Mandatory deterministic evaluation for ranking
            eval_seed=self.eval_seed
        )

        is_best = self._is_better_model(metrics)
        if is_best:
            self.best_metrics = metrics
            self.best_timestep = curr_step
            best_model_path = os.path.join(self.best_model_dir, "best_model.zip")
            self.model.save(best_model_path)

        # Save periodic checkpoint
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"stage2_{curr_step:06d}_steps.zip"
        )
        self.model.save(checkpoint_path)

        # Log to summary CSV
        avg_time_str = (
            f"{metrics['avg_success_time']:.2f}"
            if metrics["avg_success_time"] != float("inf")
            else "N/A"
        )
        avg_eff_str = (
            f"{metrics['avg_path_efficiency']:.4f}"
            if metrics["avg_path_efficiency"] > 0
            else "N/A"
        )
        avg_wall_str = (
            f"{metrics.get('avg_wall_contact_fraction', 0.0):.4f}"
        )
        with open(self.summary_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                curr_step,
                f"{metrics['success_rate']:.4f}",
                f"{metrics['mean_reward']:.2f}",
                avg_time_str,
                f"{metrics['avg_initial_distance']:.2f}",
                avg_eff_str,
                avg_wall_str,
                "YES" if is_best else "NO"
            ])

        # Write detailed JSON log
        detail_path = os.path.join(
            self.log_dir,
            "evaluations",
            f"eval_step_{curr_step:06d}.json"
        )
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestep": curr_step,
                "is_best": is_best,
                "metrics": metrics
            }, f, indent=2)

        # Concise terminal output
        if self.verbose > 0:
            print("\n" + "=" * 50)
            print(f"[Stage 2 Evaluation (Deterministic)]")
            print(f"Timesteps:            {curr_step:,}")
            print(f"Eval success:         {metrics['success_rate'] * 100:.1f}% ({metrics['successes']}/{self.n_eval_episodes})")
            print(f"Mean reward:          {metrics['mean_reward']:.2f}")
            print(f"Avg initial distance: {metrics['avg_initial_distance']:.1f} px")
            if metrics["avg_success_time"] != float("inf"):
                print(f"Avg success time:     {metrics['avg_success_time']:.2f} s")
                print(f"Straight-line eff:    {metrics['avg_path_efficiency']:.2%} (diagnostic)")
                print(f"Avg wall contact:     {metrics.get('avg_wall_contact_fraction', 0.0):.2%}")
            else:
                print(f"Avg success time:     N/A (0 successes)")
            if is_best:
                print(f"Best model:           UPDATED (step {curr_step:,})")
            else:
                print(f"Best model:           Unchanged (from step {self.best_timestep:,})")
            print("=" * 50 + "\n")

