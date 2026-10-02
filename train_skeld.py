"""
train_skeld.py
PPO training script for Stage 2.5: The Skeld navigation environment.

=============================================================================
STATUS: EXPERIMENTAL / NOT YET APPROVED FOR FULL TRAINING
DO NOT LAUNCH FULL TRAINING RUNS.
Training architecture decisions (PPO observation size, CNN perception,
curriculum transfer) will be finalized only after manual visual map inspection.
=============================================================================

This script provides an experimental skeleton for training PPO on The Skeld.
It is intended for short smoke tests and integration verification only.

Usage:
    python train_skeld.py [--timesteps 2000] [--n-envs 1]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from skeld_environment import SkeldNavEnv


# ==========================================
# Defaults
# ==========================================
DEFAULT_TIMESTEPS       = 250_000
DEFAULT_CHECKPOINT_FREQ = 10_000
DEFAULT_EVAL_EPISODES   = 20
DEFAULT_N_ENVS          = 4
MODEL_DIR = Path("models/skeld")
LOG_DIR   = Path("logs/skeld")


class SkeldProgressCallback(BaseCallback):
    """Lightweight callback that prints periodic training summaries."""

    def __init__(self, print_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_print = 0
        self._ep_rewards = []
        self._ep_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])

        if self.n_calls - self._last_print >= self.print_freq:
            self._last_print = self.n_calls
            if self._ep_rewards:
                mean_r = np.mean(self._ep_rewards[-50:])
                mean_l = np.mean(self._ep_lengths[-50:])
                print(f"  [Skeld] step={self.n_calls:>7d} | "
                      f"ep_rew={mean_r:+.1f} | ep_len={mean_l:.0f}")
            else:
                print(f"  [Skeld] step={self.n_calls:>7d} | (no episodes yet)")
        return True


def make_env(rank: int = 0, seed: int = 42):
    def _init():
        env = SkeldNavEnv(render_mode=None, task_goal_mode=True)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ,
    eval_episodes:   int = DEFAULT_EVAL_EPISODES,
    n_envs:          int = DEFAULT_N_ENVS,
    resume:          bool = False,
):
    print("=" * 70)
    print("STATUS: EXPERIMENTAL / NOT YET APPROVED FOR FULL TRAINING")
    print("Do not run full production training runs until architecture is decided.")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Environments
    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i, 42) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0, 42)])

    eval_env = Monitor(SkeldNavEnv(render_mode=None, task_goal_mode=True))
    eval_env.reset(seed=99999)

    # Model definition
    latest = MODEL_DIR / "latest_model.zip"
    if resume and latest.exists():
        print(f"Resuming from {latest}")
        model = PPO.load(str(latest), env=vec_env)
    else:
        print("Initializing experimental Skeld PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            seed=42,
            tensorboard_log=str(LOG_DIR),
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="skeld_ppo",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best_model"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=max(checkpoint_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )
    progress_cb = SkeldProgressCallback(print_freq=5000)

    t0 = time.time()
    print(f"Executing experimental training for {total_timesteps:,} steps across {n_envs} envs...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb, progress_cb],
        reset_num_timesteps=not resume,
    )
    elapsed = time.time() - t0
    print(f"Experimental run complete in {elapsed:.0f}s")

    model.save(str(MODEL_DIR / "final_model"))
    model.save(str(latest))
    print(f"Saved checkpoint to {MODEL_DIR}")

    vec_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental PPO script for The Skeld")
    parser.add_argument("--timesteps",       type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--eval-episodes",   type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--n-envs",          type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--resume",          action="store_true")
    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        resume=args.resume,
    )
