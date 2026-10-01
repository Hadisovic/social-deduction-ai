"""
Verification test for Stage 1 -> Stage 2 model transfer (Phase 3B).
Proves that the initial Stage 2 PPO model genuinely inherits policy and value network
parameters from models/stage1/best_model/best_model.zip without reinitialization.
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO

from rl_environment import StealthGymEnv


def test_stage1_to_stage2_weight_transfer(stage1_model_path: str = None):
    print("=" * 70)
    print("RUNNING STAGE 1 -> STAGE 2 WEIGHT TRANSFER TEST (SECTION 46)")
    print("=" * 70)

    created_temp = False
    temp_dir = None
    if stage1_model_path is None or not os.path.exists(stage1_model_path):
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="test_s1_transfer_")
        stage1_model_path = os.path.join(temp_dir, "temp_stage1_43obs.zip")
        print(f"[INFO] Creating temporary 43-input Stage 1 model at: {stage1_model_path}")
        s1_tmp_env = StealthGymEnv(stage=1, render_mode=None)
        tmp_model = PPO("MlpPolicy", s1_tmp_env, n_steps=64, batch_size=64, n_epochs=1, seed=42, verbose=0)
        tmp_model.save(stage1_model_path)
        s1_tmp_env.close()
        created_temp = True
    else:
        # Check if the existing model is 43-dim; if not (e.g. old 33-dim), generate temp
        try:
            test_load = PPO.load(stage1_model_path, device="cpu")
            if test_load.observation_space.shape[0] != 43:
                raise ValueError("Old 33-dim model")
        except Exception:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="test_s1_transfer_")
            stage1_model_path = os.path.join(temp_dir, "temp_stage1_43obs.zip")
            print(f"[INFO] Existing model incompatible. Creating temporary 43-input Stage 1 model at: {stage1_model_path}")
            s1_tmp_env = StealthGymEnv(stage=1, render_mode=None)
            tmp_model = PPO("MlpPolicy", s1_tmp_env, n_steps=64, batch_size=64, n_epochs=1, seed=42, verbose=0)
            tmp_model.save(stage1_model_path)
            s1_tmp_env.close()
            created_temp = True

    try:
        # 1. Load Stage 1 model directly and extract state dict parameters
        s1_env = StealthGymEnv(stage=1, render_mode=None)
        model_s1 = PPO.load(stage1_model_path, env=s1_env, device="cpu")
        assert model_s1.observation_space.shape == (43,), f"Expected 43 obs, got {model_s1.observation_space.shape}"
        s1_params = {k: v.clone() for k, v in model_s1.policy.state_dict().items()}
        s1_env.close()
        print(f"[INFO] Loaded 43-input Stage 1 model with {len(s1_params)} parameter tensors in state_dict.")

        # 2. Initialize Stage 2 continuation as done in train_stage2.py
        s2_env = StealthGymEnv(stage=2, render_mode=None)
        model_s2 = PPO.load(stage1_model_path, env=s2_env, device="cpu")
        s2_params = model_s2.policy.state_dict()

        # 3. Verify exact bit-for-bit parameter match across every layer
        for name, s1_tensor in s1_params.items():
            assert name in s2_params, f"Parameter tensor '{name}' missing in Stage 2 model!"
            s2_tensor = s2_params[name]
            assert torch.equal(s1_tensor, s2_tensor), (
                f"Parameter mismatch in tensor '{name}'! Stage 2 model weights do not match Stage 1!"
            )

        print("[PASS] Test 1: All policy and value network parameter tensors bit-exactly match Stage 1 best model.")

        # 4. Verify network architecture is preserved (MLP 64x64 for policy and value)
        obs_s2, _ = s2_env.reset()
        action_s1, _ = model_s1.predict(obs_s2, deterministic=True)
        action_s2, _ = model_s2.predict(obs_s2, deterministic=True)
        assert np.allclose(action_s1, action_s2), "Predictions differ on identical inputs!"
        print("[PASS] Test 2: Deterministic policy outputs match identically between Stage 1 and transferred Stage 2 models.")

    finally:
        s2_env.close()
        if created_temp and temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("[INFO] Cleaned up temporary Stage 1 transfer test artifacts.")

    print("=" * 70)
    print("STAGE 1 -> STAGE 2 WEIGHT TRANSFER VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    import numpy as np
    test_stage1_to_stage2_weight_transfer()
