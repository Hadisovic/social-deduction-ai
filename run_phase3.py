"""Watch a real five-player scripted match: python run_phase3.py.

Space pauses, Right steps, +/- changes speed, R repeats the seed, N advances it.
The optional flags support reproducible visual inspection; none is required.
"""
import argparse
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--speed", type=float, default=1.)
    result.add_argument("--max-frames", type=int, help="Close after this many displayed frames (smoke testing)")
    result.add_argument("--capture", type=Path, help="Save representative real simulator frames to this directory")
    result.add_argument("--exit-on-finish", action="store_true", help="Close after displaying the terminal result")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.speed <= 32:
        parser().error("--speed must be greater than zero and at most 32")
    if args.max_frames is not None and args.max_frames <= 0:
        parser().error("--max-frames must be positive")
    import pygame
    from phase3_renderer import Phase3Renderer, ViewOptions
    from phase3_runner import ScriptedMatch
    from social_deduction.actor import Phase

    pygame.init()
    screen = pygame.display.set_mode((1440, 900))
    pygame.display.set_caption("Social Deduction AI | Phase 3 | Loading the Skeld")
    screen.fill((12, 20, 33))
    font = pygame.font.SysFont("segoeui", 24)
    screen.blit(font.render("Preparing the Skeld and five scripted players...", True, (226, 237, 249)), (40, 60))
    pygame.display.flip()
    pygame.event.pump()
    match = ScriptedMatch(seed=args.seed)
    renderer = Phase3Renderer(match.game.map)
    options = ViewOptions()
    clock = pygame.time.Clock()
    speed, paused, accumulator, frames = args.speed, False, 0., 0
    captured = set()
    captures = args.capture
    if captures:
        captures.mkdir(parents=True, exist_ok=True)
    toggle_keys = {pygame.K_TAB: "hud", pygame.K_1: "roles", pygame.K_g: "routes",
                   pygame.K_v: "visibility", pygame.K_t: "tasks", pygame.K_l: "labels",
                   pygame.K_c: "collision", pygame.K_e: "events", pygame.K_b: "bodies"}
    running = True
    try:
        while running:
            wall_dt = min(clock.tick(60) / 1000, .25)
            one_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        accumulator = 0.
                    elif event.key == pygame.K_RIGHT and paused:
                        one_step = True
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed = min(32., speed * 2)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = max(.25, speed / 2)
                    elif event.key in (pygame.K_r, pygame.K_n):
                        match.reset(match.game.seed + (event.key == pygame.K_n))
                        paused, accumulator = False, 0.
                        captured.clear()
                    elif event.key in toggle_keys:
                        key = toggle_keys[event.key]
                        setattr(options, key, not getattr(options, key))
            if not running:
                break
            game = match.game
            # Every step uses config.dt. Display time only schedules how many
            # identical ticks are performed; it is never passed into the world.
            if one_step and game.result is None:
                match.step()
            elif not paused and game.result is None and frames > 0:
                accumulator += wall_dt * speed
                steps = 0
                while accumulator >= game.config.dt and steps < 32 and game.result is None:
                    match.step()
                    accumulator -= game.config.dt
                    steps += 1
            surface = renderer.draw(game, options, paused, speed, clock.get_fps())
            screen.blit(surface, (0, 0))
            pygame.display.set_caption(f"Social Deduction AI | seed {game.seed} | {game.phase.value} | {game.time:.1f}s | {speed:g}x")
            pygame.display.flip()
            if captures:
                candidates = []
                if frames == 0:
                    candidates.append("spawn")
                if game.phase == Phase.ROAMING and game.time >= 5:
                    candidates.append("exploration")
                if any(player.interaction_task for player in game.players.values()):
                    candidates.append("task")
                if any(not body.reported for body in game.bodies.values()):
                    candidates.append("body")
                if game.phase == Phase.DISCUSSION:
                    candidates.append("discussion")
                    if any(hasattr(event.payload, "speaker_id") for event in game.publications):
                        candidates.append("claims")
                if game.phase == Phase.VOTING:
                    candidates.append("voting")
                    if game.votes:
                        candidates.append("votes")
                if game.result is not None:
                    candidates.append("finished")
                for name in candidates:
                    if name not in captured:
                        path = captures / f"seed-{game.seed}-{name}.png"
                        pygame.image.save(surface, str(path))
                        print(f"Captured {path} at {game.time:.1f}s", flush=True)
                        captured.add(name)
            frames += 1
            if args.max_frames is not None and frames >= args.max_frames:
                running = False
            if args.exit_on_finish and game.result is not None:
                running = False
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    from phase3_cli import use_project_environment
    use_project_environment()
    raise SystemExit(main())
