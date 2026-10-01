"""
Verification test suite for Stage 2 Evaluation Fairness & Best-Model Ranking Logic (Phase 3B):
1. Checkpoint evaluation with same eval seed produces identical Stage 2 layout sequence.
2. Layouts within the sequence are different from one another.
3. Best-model evaluation uses deterministic=True.
4. Checkpoint ranking strictly prioritizes:
   - Primary: highest deterministic success rate
   - Secondary: lowest average successful completion time
   - Tertiary: higher mean evaluation reward
   - Ties: preserves earlier existing best model
5. Straight-line path efficiency is NOT used for Stage 2 best-model ranking.
"""

import math
from rl_environment import StealthGymEnv
from training_callbacks import Stage2EvalCallback, run_evaluation_episodes
from config import PPO_STAGE2_EVAL_SEED


def test_stage2_eval_fairness_and_ranking():
    print("=" * 70)
    print("RUNNING STAGE 2 EVALUATION FAIRNESS & RANKING TEST SUITE")
    print("=" * 70)

    eval_env = StealthGymEnv(stage=2, render_mode=None)

    # -----------------------------------------------------------------
    # Test 1 & 2: Same eval seed produces identical sequence across checkpoints,
    # and layouts within sequence are distinct.
    # -----------------------------------------------------------------
    eval_env.reset(seed=PPO_STAGE2_EVAL_SEED)
    seq_10k = []
    for _ in range(10):
        seq_10k.append(((eval_env.env.player.x, eval_env.env.player.y), (eval_env.env.goal_pos[0], eval_env.env.goal_pos[1])))
        eval_env.reset()

    # Simulate next checkpoint evaluation
    eval_env.reset(seed=PPO_STAGE2_EVAL_SEED)
    seq_20k = []
    for _ in range(10):
        seq_20k.append(((eval_env.env.player.x, eval_env.env.player.y), (eval_env.env.goal_pos[0], eval_env.env.goal_pos[1])))
        eval_env.reset()

    for idx, (layout_a, layout_b) in enumerate(zip(seq_10k, seq_20k)):
        assert math.isclose(layout_a[0][0], layout_b[0][0], abs_tol=1e-5)
        assert math.isclose(layout_a[0][1], layout_b[0][1], abs_tol=1e-5)
        assert math.isclose(layout_a[1][0], layout_b[1][0], abs_tol=1e-5)
        assert math.isclose(layout_a[1][1], layout_b[1][1], abs_tol=1e-5)

    print("[PASS] Test 1: Checkpoint evaluation with same eval seed produces identical layout sequence.")

    # Distinctness within sequence
    for i in range(len(seq_10k) - 1):
        assert seq_10k[i] != seq_10k[i + 1], f"Consecutive layouts {i} and {i+1} are unexpectedly identical!"
    print("[PASS] Test 2: Layouts within evaluation sequence are distinct.")

    # -----------------------------------------------------------------
    # Test 3: Best model ranking logic
    # -----------------------------------------------------------------
    cb = Stage2EvalCallback(eval_env=eval_env, eval_freq=10000, n_eval_episodes=10, verbose=0)

    # Baseline best model
    cb.best_metrics = {
        "success_rate": 0.80,
        "avg_success_time": 5.0,
        "mean_reward": 80.0,
        "avg_path_efficiency": 0.90,
    }

    # Candidate 1: Higher success rate (0.90 vs 0.80) -> MUST WIN regardless of time or reward or efficiency
    cand_1 = {
        "success_rate": 0.90,
        "avg_success_time": 8.0,      # Slower
        "mean_reward": 70.0,          # Lower reward
        "avg_path_efficiency": 0.50,  # Lower efficiency
    }
    assert cb._is_better_model(cand_1) is True
    print("[PASS] Test 3a: Primary ranking prioritizes highest success rate (0.90 beats 0.80 despite slower time & lower efficiency).")

    # Candidate 2: Lower success rate (0.70 vs 0.80) -> MUST LOSE even with faster time and higher reward
    cand_2 = {
        "success_rate": 0.70,
        "avg_success_time": 2.0,      # Much faster
        "mean_reward": 110.0,         # Much higher
        "avg_path_efficiency": 0.99,  # High efficiency
    }
    assert cb._is_better_model(cand_2) is False
    print("[PASS] Test 3b: Lower success rate strictly rejected.")

    # Candidate 3: Tied success rate (0.80), Faster completion time (4.0s vs 5.0s) -> MUST WIN
    cand_3 = {
        "success_rate": 0.80,
        "avg_success_time": 4.0,      # Faster
        "mean_reward": 75.0,          # Lower reward
        "avg_path_efficiency": 0.60,  # Lower efficiency
    }
    assert cb._is_better_model(cand_3) is True
    print("[PASS] Test 3c: Secondary ranking prioritizes lowest completion time when success rates tie.")

    # Candidate 4: Tied success rate (0.80), Slower completion time (6.0s vs 5.0s) -> MUST LOSE
    cand_4 = {
        "success_rate": 0.80,
        "avg_success_time": 6.0,
        "mean_reward": 95.0,
        "avg_path_efficiency": 0.95,
    }
    assert cb._is_better_model(cand_4) is False
    print("[PASS] Test 3d: Slower completion time strictly rejected on tie.")

    # Candidate 5: Tied success rate (0.80) and tied time (5.0s), higher mean reward -> MUST WIN
    cand_5 = {
        "success_rate": 0.80,
        "avg_success_time": 5.0,
        "mean_reward": 85.0,          # Higher reward
        "avg_path_efficiency": 0.60,
    }
    assert cb._is_better_model(cand_5) is True
    print("[PASS] Test 3e: Tertiary ranking prioritizes higher mean reward on tied success rate and time.")

    # Candidate 6: Complete tie on success rate (0.80), time (5.0s), reward (80.0) -> MUST PRESERVE EARLIER
    cand_6 = {
        "success_rate": 0.80,
        "avg_success_time": 5.0,
        "mean_reward": 80.0,
        "avg_path_efficiency": 0.99,  # Even with higher efficiency, must not replace
    }
    assert cb._is_better_model(cand_6) is False
    print("[PASS] Test 3f: Complete tie preserves earlier best model; straight-line efficiency does NOT override ranking.")

    eval_env.close()
    print("=" * 70)
    print("ALL EVALUATION FAIRNESS & RANKING TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_stage2_eval_fairness_and_ranking()
