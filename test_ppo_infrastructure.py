"""
Automated verification suite for Phase 3A: Stage 1 PPO Training Infrastructure.
Covers all 14 infrastructure requirements specified in the project specification.
"""

import os
import sys
import shutil
import tempfile
import math
import numpy as np

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    STAGE1_STEP_PENALTY,
    OBSERVATION_SIZE,
    PPO_NET_ARCH,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_SEED,
)
from rl_environment import StealthGymEnv
from training_callbacks import run_evaluation_episodes, Stage1EvalCallback
from test_stealth_env import run_acceptance_tests
from test_rl_env import run_phase_2_tests


def run_infrastructure_tests():
    print("=" * 70)
    print("RUNNING PHASE 3A PPO INFRASTRUCTURE VERIFICATION SUITE")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp(prefix="test_stage1_ppo_")

    try:
        # -----------------------------------------------------------------
        # Test 1: Stage 1 step penalty exists
        # -----------------------------------------------------------------
        env1 = StealthGymEnv(stage=1, render_mode=None)
        env1.reset()
        _, r1, _, _, _ = env1.step(np.array([0.0, 0.0], dtype=np.float32))
        assert math.isclose(r1, STAGE1_STEP_PENALTY, abs_tol=1e-6), (
            f"Stage 1 reward {r1} != STAGE1_STEP_PENALTY {STAGE1_STEP_PENALTY}"
        )
        assert STAGE1_STEP_PENALTY == -0.01, f"STAGE1_STEP_PENALTY is {STAGE1_STEP_PENALTY}, expected -0.01"
        print(f"[PASS] Test 1: Stage 1 step penalty exists and equals {STAGE1_STEP_PENALTY}.")

        # -----------------------------------------------------------------
        # Test 2: Stage 2 receives STAGE2_STEP_PENALTY (-0.01), Stage 3 does not
        # -----------------------------------------------------------------
        env2 = StealthGymEnv(stage=2, render_mode=None)
        env2.reset()
        _, r2, _, _, _ = env2.step(np.array([0.0, 0.0], dtype=np.float32))
        assert math.isclose(r2, -0.01, abs_tol=1e-6), (
            f"Stage 2 reward is unexpected: {r2}, expected -0.01"
        )
        env3 = StealthGymEnv(stage=3, render_mode=None)
        env3.reset()
        _, r3, _, _, _ = env3.step(np.array([0.0, 0.0], dtype=np.float32))
        assert math.isclose(r3, 0.0, abs_tol=1e-6), (
            f"Stage 3 reward is unexpected: {r3}, expected 0.0"
        )
        print("[PASS] Test 2: Stage 2 receives step penalty (-0.01) and Stage 3 does NOT receive penalty (0.0).")

        # -----------------------------------------------------------------
        # Test 3: PPO model can initialize with the environment
        # -----------------------------------------------------------------
        train_env = DummyVecEnv([lambda: StealthGymEnv(stage=1, render_mode=None)])
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=256,      # Smaller n_steps just for fast unit testing
            batch_size=64,
            n_epochs=2,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            policy_kwargs=dict(net_arch=PPO_NET_ARCH),
            seed=PPO_SEED,
            device="cpu",
            verbose=0
        )
        assert model is not None, "Failed to instantiate PPO model"
        print("[PASS] Test 3: PPO model initializes cleanly with Stage 1 environment.")

        # -----------------------------------------------------------------
        # Test 4: PPO policy accepts the 43-value observation
        # -----------------------------------------------------------------
        sample_obs = env1.observation_space.sample()
        assert sample_obs.shape == (43,), f"Expected (43,), got {sample_obs.shape}"
        assert sample_obs.dtype == np.float32
        obs_tensor, _ = model.policy.obs_to_tensor(sample_obs)
        assert obs_tensor.shape == (1, 43), f"Policy tensor shape mismatch: {obs_tensor.shape}"
        print("[PASS] Test 4: PPO policy strictly accepts the 43-value observation vector.")

        # -----------------------------------------------------------------
        # Test 5: PPO action output is compatible with 2D Box action space
        # -----------------------------------------------------------------
        pred_action, _ = model.predict(sample_obs, deterministic=False)
        assert pred_action.shape == (2,), f"Predicted action shape {pred_action.shape} != (2,)"
        assert env1.action_space.contains(pred_action), f"Predicted action {pred_action} out of bounds"
        print("[PASS] Test 5: PPO action output is compatible with 2D Box([-1, 1], shape=(2,)).")

        # -----------------------------------------------------------------
        # Test 6: A tiny training smoke test runs without crashing
        # -----------------------------------------------------------------
        model.learn(total_timesteps=512)
        assert model.num_timesteps >= 512, f"Timesteps {model.num_timesteps} < 512"
        print("[PASS] Test 6: Training smoke test completed without crashing.")

        # -----------------------------------------------------------------
        # Test 7: Model can save
        # -----------------------------------------------------------------
        save_path = os.path.join(temp_dir, "test_model.zip")
        model.save(save_path)
        assert os.path.exists(save_path), f"Saved model file not found at {save_path}"
        print("[PASS] Test 7: Model saves cleanly to disk.")

        # -----------------------------------------------------------------
        # Test 8: Saved model can reload
        # -----------------------------------------------------------------
        loaded_model = PPO.load(save_path, env=train_env, device="cpu")
        assert loaded_model is not None, "Failed to reload model from disk"
        print("[PASS] Test 8: Saved model reloads successfully.")

        # -----------------------------------------------------------------
        # Test 9: Reloaded model can predict an action
        # -----------------------------------------------------------------
        act, _ = loaded_model.predict(sample_obs, deterministic=True)
        assert act.shape == (2,), f"Reloaded model action shape {act.shape} != (2,)"
        assert env1.action_space.contains(act)
        print("[PASS] Test 9: Reloaded model predicts valid actions.")

        # -----------------------------------------------------------------
        # Test 10: Evaluation helper can complete an episode
        # -----------------------------------------------------------------
        eval_metrics = run_evaluation_episodes(
            model=loaded_model,
            env=env1,
            n_episodes=3,
            deterministic=False
        )
        assert eval_metrics["n_episodes"] == 3
        assert "success_rate" in eval_metrics
        assert "mean_reward" in eval_metrics
        assert len(eval_metrics["episodes"]) == 3
        print(f"[PASS] Test 10: Evaluation helper runs episodes and produces structured metrics.")

        # -----------------------------------------------------------------
        # Test 11: Checkpoint callback can save
        # -----------------------------------------------------------------
        cb_checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        cb_best_dir = os.path.join(temp_dir, "best_model")
        cb_log_dir = os.path.join(temp_dir, "logs")

        cb = Stage1EvalCallback(
            eval_env=env1,
            eval_freq=256,
            n_eval_episodes=2,
            checkpoints_dir=cb_checkpoint_dir,
            best_model_dir=cb_best_dir,
            log_dir=cb_log_dir,
            verbose=0
        )
        model.learn(total_timesteps=512, callback=cb)

        checkpoints = os.listdir(cb_checkpoint_dir)
        assert len(checkpoints) > 0, "No checkpoint files saved by callback"
        assert os.path.exists(os.path.join(cb_best_dir, "best_model.zip")), "best_model.zip not saved"
        assert os.path.exists(os.path.join(cb_log_dir, "stage1_eval_summary.csv")), "Summary CSV not saved"
        print("[PASS] Test 11: Callback saves periodic checkpoints, best model, and CSV logs.")

        # -----------------------------------------------------------------
        # Test 12: Existing Phase 1 tests still pass
        # -----------------------------------------------------------------
        print("\nExecuting Phase 1 acceptance suite...")
        run_acceptance_tests()
        print("[PASS] Test 12: Phase 1 acceptance tests passed.")

        # -----------------------------------------------------------------
        # Test 13: Existing Phase 2 tests still pass
        # -----------------------------------------------------------------
        print("\nExecuting Phase 2 RL verification suite...")
        run_phase_2_tests()
        print("[PASS] Test 13: Phase 2 verification tests passed.")

        # -----------------------------------------------------------------
        # Test 14: Gymnasium environment checker still passes
        # -----------------------------------------------------------------
        print("\nExecuting Gymnasium environment check...")
        check_env(StealthGymEnv(stage=1, render_mode=None))
        print("[PASS] Test 14: Gymnasium environment checker passes cleanly.")

        print("=" * 70)
        print("ALL 14 INFRASTRUCTURE VERIFICATION TESTS PASSED PERFECTLY!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_infrastructure_tests()
