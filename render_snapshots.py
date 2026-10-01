"""
Render snapshots of the game in various states for visual inspection.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from environment import StealthEnvironment
from config import WINDOW_WIDTH, WINDOW_HEIGHT

def generate_snapshots():
    pygame.init()
    surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    env = StealthEnvironment()

    # 1. Normal State Snapshot
    env.render(surface, debug=False, fps=60.0)
    pygame.image.save(surface, "snapshot_normal.png")
    print("Saved snapshot_normal.png")

    # 2. Debug State Snapshot (F1 mode)
    # Step simulation slightly so observers have moved and computed LOS
    for _ in range(30):
        env.step((0.0, 0.0), 0.016)
    env.render(surface, debug=True, fps=60.0)
    pygame.image.save(surface, "snapshot_debug.png")
    print("Saved snapshot_debug.png")

    # 3. Detected Modal Snapshot
    env.reset()
    obs = env.observers[0]
    # Place player in direct vision of observer 1
    front_dist = 80.0
    import math
    env.player.x = obs.x + math.cos(obs.facing_angle) * front_dist
    env.player.y = obs.y + math.sin(obs.facing_angle) * front_dist
    env.step((0.0, 0.0), 0.016)
    env.render(surface, debug=False, fps=60.0)
    pygame.image.save(surface, "snapshot_detected.png")
    print("Saved snapshot_detected.png")

    # 4. Escaped Modal Snapshot
    env.reset()
    env.player.x = env.goal_pos[0]
    env.player.y = env.goal_pos[1]
    env.step((0.0, 0.0), 0.016)
    env.render(surface, debug=False, fps=60.0)
    pygame.image.save(surface, "snapshot_escaped.png")
    print("Saved snapshot_escaped.png")

if __name__ == "__main__":
    generate_snapshots()
