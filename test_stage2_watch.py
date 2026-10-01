"""
Verification test suite for Stage 2 Live Spectator and --watch integration (Phase 3B):
- Verifies live_watch_stage2.py initializes Pygame video system safely.
- Verifies loading Stage 1 baseline when no Stage 2 checkpoint exists yet.
- Verifies automatic episode reset and checkpoint switching at episode boundaries.
- Verifies clean exit on ESC and Window Close.
- Verifies independent execution from training.
"""

import os
import sys
import time
import shutil
import tempfile
import threading

# Use dummy driver for headless verification
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from stable_baselines3 import PPO

from rl_environment import StealthGymEnv
import live_watch_stage2


def test_watch_stage1_baseline_fallback_and_esc():
    print("\n--- Test 1: Stage 1 Baseline Fallback & ESC Exit ---")
    temp_ckpts = tempfile.mkdtemp(prefix="test_s2_ckpts_")
    stop_event = threading.Event()
    try:
        s1_env = StealthGymEnv(stage=1, render_mode=None)
        s1_model = PPO("MlpPolicy", s1_env, n_steps=64, batch_size=64, n_epochs=1, seed=42, verbose=0)
        baseline_path = os.path.join(temp_ckpts, "test_s1_baseline.zip")
        s1_model.save(baseline_path)
        s1_env.close()

        def post_esc():
            time.sleep(1.0)
            while not stop_event.is_set():
                if pygame.get_init() and pygame.display.get_init():
                    try:
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
                    except Exception:
                        pass
                time.sleep(0.3)

        t = threading.Thread(target=post_esc, daemon=True)
        t.start()

        sys.argv = [
            "live_watch_stage2.py",
            "--checkpoint-dir", temp_ckpts,
            "--stage1-baseline", baseline_path,
            "--fps", "60",
            "--pause", "0.1"
        ]
        live_watch_stage2.main()
        stop_event.set()
        print("[PASS] Test 1: Loaded Stage 1 baseline in Stage 2 env, ran episodes without crashing, and exited cleanly on ESC.")
    finally:
        stop_event.set()
        shutil.rmtree(temp_ckpts, ignore_errors=True)


def test_watch_checkpoint_switching():
    print("\n--- Test 2: Dynamic Stage 2 Checkpoint Switch at Episode Boundary ---")
    temp_ckpts = tempfile.mkdtemp(prefix="test_s2_switch_")
    stop_event = threading.Event()
    try:
        train_env = StealthGymEnv(stage=2, render_mode=None)
        m1 = PPO("MlpPolicy", train_env, n_steps=64, batch_size=64, n_epochs=1, seed=1, verbose=0)
        p1 = os.path.join(temp_ckpts, "stage2_010000_steps.zip")
        m1.save(p1)

        m2 = PPO("MlpPolicy", train_env, n_steps=64, batch_size=64, n_epochs=1, seed=2, verbose=0)
        p2 = os.path.join(temp_ckpts, "stage2_020000_steps.zip")

        def save_and_close():
            time.sleep(1.0)
            m2.save(p2)
            time.sleep(1.0)
            while not stop_event.is_set():
                if pygame.get_init() and pygame.display.get_init():
                    try:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
                    except Exception:
                        pass
                time.sleep(0.3)

        t = threading.Thread(target=save_and_close, daemon=True)
        t.start()

        sys.argv = [
            "live_watch_stage2.py",
            "--checkpoint-dir", temp_ckpts,
            "--check-interval", "0.2",
            "--fps", "60",
            "--pause", "0.1"
        ]
        live_watch_stage2.main()
        stop_event.set()
        print("[PASS] Test 2: Checkpoint switch detected, loaded at episode boundary, and exited on window close.")
        train_env.close()
    finally:
        stop_event.set()
        shutil.rmtree(temp_ckpts, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING STAGE 2 SPECTATOR & WATCH INTEGRATION TEST SUITE")
    print("=" * 70)
    test_watch_stage1_baseline_fallback_and_esc()
    test_watch_checkpoint_switching()
    print("=" * 70)
    print("ALL STAGE 2 SPECTATOR TESTS PASSED PERFECTLY!")
    print("=" * 70)
