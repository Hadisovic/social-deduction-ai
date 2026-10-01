"""
Visual Watch Mode for trained Stage 1 PPO checkpoints (Phase 3A).
Renders the agent's behavior live in the Pygame window without modifying weights.
"""

import os
import argparse
import pygame
from stable_baselines3 import PPO

from rl_environment import StealthGymEnv
from config import RL_DT


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained Stage 1 PPO agent in Pygame.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/stage1/final/stage1_final.zip",
        help="Path to the trained model zip file to visualize"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to watch (default: 1)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of deterministic argmax"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate limit (default: 30 fps)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return

    deterministic = not args.stochastic

    print("=" * 60)
    print("STAGE 1 VISUAL WATCH MODE")
    print("=" * 60)
    print(f"Model Path:          {args.model}")
    print(f"Episodes:            {args.episodes}")
    print(f"Action Mode:         {'Deterministic (argmax)' if deterministic else 'Stochastic (sampling)'}")
    print(f"Playback FPS:        {args.fps}")
    print("Close the Pygame window or press ESC to exit early.")
    print("=" * 60)

    # 1. Initialize Stage 1 environment with human rendering mode
    env = StealthGymEnv(stage=1, render_mode="human")
    env.metadata["render_fps"] = args.fps

    # 2. Load trained PPO model
    model = PPO.load(args.model, env=env, device=args.device)

    running = True

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        print(f"\n--- Episode {ep + 1}/{args.episodes} Started ---")

        while not done and running:
            # Process Pygame events for responsive UI and clean exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break

            if not running:
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            done = terminated or truncated

        if not running:
            print("\n[INFO] Visual playback interrupted by user.")
            break

        is_success = info.get("is_success", False)
        elapsed_time = info.get("elapsed_time", ep_steps * RL_DT)
        status = "SUCCESS (ESCAPED)" if is_success else ("TIMEOUT" if info.get("timeout") else "DETECTED")

        print(f"Episode {ep + 1} Finished:")
        print(f"  Outcome:      {status}")
        print(f"  Total Reward: {ep_reward:.2f}")
        print(f"  Steps Taken:  {ep_steps}")
        print(f"  Sim Time:     {elapsed_time:.2f} s")
        print(f"  Goal Dist:    {info.get('distance_to_goal', 0.0):.1f} px")

    env.close()
    pygame.quit()
    print("\nVisual playback closed cleanly.")


if __name__ == "__main__":
    main()
