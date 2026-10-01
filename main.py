"""
Main entry point for Phase 1 of the 2D Stealth Environment.
Handles Pygame event loop, user input translation, time delta, and debug toggling.
"""

import sys
import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    FPS,
)
from environment import StealthEnvironment


def get_keyboard_action() -> tuple[float, float]:
    """
    Read continuous keyboard state and return directional intent vector (dx, dy).
    Decoupled from movement logic to facilitate future RL agent action swapping.
    """
    keys = pygame.key.get_pressed()

    dx = 0.0
    dy = 0.0

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        dy -= 1.0
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        dy += 1.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        dx -= 1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        dx += 1.0

    return dx, dy


def main():
    """Run the Pygame main game loop."""
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    env = StealthEnvironment()
    debug_mode = False
    running = True

    while running:
        # Compute delta time in seconds, clamped to avoid physics explosion on window drag
        raw_dt = clock.tick(FPS) / 1000.0
        dt = min(raw_dt, 0.05)

        # -------------------------------------------------------------
        # EVENT HANDLING
        # -------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_F1:
                    debug_mode = not debug_mode
                elif event.key == pygame.K_r:
                    env.reset()

        if not running:
            break

        # -------------------------------------------------------------
        # INPUT & SIMULATION STEP
        # -------------------------------------------------------------
        action_vector = get_keyboard_action()
        env.step(action_vector, dt)

        # -------------------------------------------------------------
        # RENDER
        # -------------------------------------------------------------
        env.render(screen, debug=debug_mode, fps=clock.get_fps())
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
