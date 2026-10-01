"""
Live Training Spectator for Stage 1 PPO (Phase 3A).
Provides a persistent, live Pygame window that monitors checkpoints in
models/stage1/checkpoints/ and automatically switches to newer checkpoints
at episode boundaries without interrupting training.
"""

import os
import re
import time
import argparse
from typing import Optional, Tuple

import pygame
from stable_baselines3 import PPO

from config import WINDOW_WIDTH, WINDOW_HEIGHT, RL_DT
from rl_environment import StealthGymEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live Pygame spectator for Stage 1 PPO training."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/stage1/checkpoints",
        help="Directory to monitor for PPO checkpoints (default: 'models/stage1/checkpoints')"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic action selection instead of stochastic default"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Visual playback FPS cap (default: 30)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="Seconds between scanning directory for newer checkpoints (default: 1.0)"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.8,
        help="Seconds to pause on final frame after SUCCESS/TIMEOUT before reset (default: 0.8)"
    )
    return parser.parse_args()


def get_latest_checkpoint(checkpoints_dir: str) -> Optional[Tuple[int, str]]:
    """
    Scan checkpoints directory and return (timestep, filepath) of the checkpoint
    with the largest timestep number. Returns None if no valid checkpoints exist.
    """
    if not os.path.exists(checkpoints_dir):
        return None

    pattern = re.compile(r"^stage1_(\d+)_steps\.zip$")
    candidates = []

    try:
        filenames = os.listdir(checkpoints_dir)
    except OSError:
        return None

    for fname in filenames:
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            fpath = os.path.join(checkpoints_dir, fname)
            candidates.append((step, fpath))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def safely_load_model(checkpoint_path: str, env: StealthGymEnv, device: str = "cpu") -> Optional[PPO]:
    """
    Attempt to load a PPO model checkpoint.
    Safely catches exceptions in case the checkpoint is still being written to disk.
    """
    try:
        if not os.path.exists(checkpoint_path):
            return None
        # Verify file has non-trivial size (not empty file created by open)
        if os.path.getsize(checkpoint_path) < 1000:
            return None

        model = PPO.load(checkpoint_path, env=env, device=device)
        return model
    except Exception:
        # File may be mid-write or temporarily locked
        return None


def draw_spectator_overlay(
    surface: pygame.Surface,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
    checkpoint_step: Optional[int],
    episode_num: int,
    is_deterministic: bool,
    current_time: float,
    prev_outcome: Optional[str],
    prev_success_time: Optional[float],
    waiting_mode: bool = False
):
    """
    Draw an unobtrusive, stylish HUD card in the upper-left of the arena.
    Does not obstruct player start (lower-left), goal (upper-right), or center.
    """
    card_w = 285
    card_h = 175
    card_x = 48
    card_y = 48

    # Translucent rounded backdrop
    card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card_surf, (15, 23, 42, 220), (0, 0, card_w, card_h), border_radius=8)
    pygame.draw.rect(card_surf, (51, 65, 85, 240), (0, 0, card_w, card_h), width=1, border_radius=8)

    # Title
    title_text = "Stage 1 Training Spectator"
    title_render = font_title.render(title_text, True, (241, 245, 249))
    card_surf.blit(title_render, (14, 12))

    if waiting_mode:
        msg = font_body.render("Waiting for first checkpoint...", True, (251, 191, 36))
        card_surf.blit(msg, (14, 45))
        hint = font_body.render("Scanning checkpoints dir...", True, (148, 163, 184))
        card_surf.blit(hint, (14, 70))
        surface.blit(card_surf, (card_x, card_y))
        return

    # Checkpoint line
    if checkpoint_step is not None:
        ckpt_str = f"{checkpoint_step:,} steps"
    else:
        ckpt_str = "None"
    t_ckpt = font_body.render(f"Checkpoint: {ckpt_str}", True, (56, 189, 248))
    card_surf.blit(t_ckpt, (14, 38))

    # Episode number
    t_ep = font_body.render(f"Episode: {episode_num}", True, (203, 213, 225))
    card_surf.blit(t_ep, (14, 60))

    # Mode: Stochastic or Deterministic
    mode_str = "Deterministic" if is_deterministic else "Stochastic"
    t_mode = font_body.render(f"Mode: {mode_str}", True, (203, 213, 225))
    card_surf.blit(t_mode, (14, 82))

    # Current episode time
    t_time = font_body.render(f"Episode Time: {current_time:.2f} s", True, (203, 213, 225))
    card_surf.blit(t_time, (14, 104))

    # Previous outcome
    if prev_outcome:
        outcome_color = (52, 211, 153) if prev_outcome == "SUCCESS" else (248, 113, 113)
        t_prev = font_body.render(f"Previous Result: {prev_outcome}", True, outcome_color)
    else:
        t_prev = font_body.render("Previous Result: N/A", True, (148, 163, 184))
    card_surf.blit(t_prev, (14, 126))

    # Previous completion time if successful
    if prev_success_time is not None:
        t_succ = font_body.render(f"Previous Time: {prev_success_time:.2f} s", True, (52, 211, 153))
    else:
        t_succ = font_body.render("Previous Time: N/A", True, (148, 163, 184))
    card_surf.blit(t_succ, (14, 148))

    surface.blit(card_surf, (card_x, card_y))


def main():
    args = parse_args()
    action_mode_str = "deterministic" if args.deterministic else "stochastic"

    # Startup banner as required
    print("=" * 60)
    print("STAGE 1 LIVE TRAINING SPECTATOR")
    print("=" * 60)
    print(f"Watching directory: {args.checkpoint_dir}")
    print(f"Action mode:        {action_mode_str}")
    print(f"Playback FPS:       {args.fps}")
    print("Waiting/checking for newer checkpoints automatically.")
    print("Press ESC or close window to exit.")
    print("=" * 60)

    # 1. Properly initialize Pygame video system BEFORE any display or event calls
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()

    # Initialize Stage 1 environment in human render mode
    env = StealthGymEnv(stage=1, render_mode="human")
    env.metadata["render_fps"] = args.fps

    # Ensure persistent window surface and clock exist on the environment immediately
    if env.screen is None:
        env.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Stealth Sandbox - Stage 1 Training Spectator")
        env.clock = pygame.time.Clock()

    # Initialize Pygame fonts for HUD overlay
    if not pygame.font.get_init():
        pygame.font.init()
    try:
        font_title = pygame.font.SysFont("Segoe UI", 15, bold=True)
        font_body = pygame.font.SysFont("Segoe UI", 13)
    except Exception:
        font_title = pygame.font.Font(None, 20)
        font_body = pygame.font.Font(None, 18)

    # State tracking variables
    current_model: Optional[PPO] = None
    current_checkpoint_step: Optional[int] = None
    current_checkpoint_path: Optional[str] = None
    pending_checkpoint: Optional[Tuple[int, str]] = None

    episode_num = 0
    current_episode_time = 0.0
    prev_outcome: Optional[str] = None
    prev_success_time: Optional[float] = None
    is_waiting = True

    # Register overlay callback
    def render_overlay(surface: pygame.Surface):
        draw_spectator_overlay(
            surface=surface,
            font_title=font_title,
            font_body=font_body,
            checkpoint_step=current_checkpoint_step,
            episode_num=episode_num,
            is_deterministic=args.deterministic,
            current_time=current_episode_time,
            prev_outcome=prev_outcome,
            prev_success_time=prev_success_time,
            waiting_mode=is_waiting
        )

    env.overlay_callback = render_overlay

    running = True
    last_check_time = 0.0

    # Render initial frame so window appears and shows waiting message immediately
    env.render()

    # -------------------------------------------------------------
    # 1. WAIT LOOP: If no checkpoints exist yet, wait gracefully
    # -------------------------------------------------------------
    while running and current_model is None:
        if pygame.get_init() and pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    break

        if not running or env.screen is None:
            break

        now = time.time()
        if now - last_check_time >= args.check_interval:
            last_check_time = now
            latest = get_latest_checkpoint(args.checkpoint_dir)
            if latest is not None:
                step, path = latest
                loaded = safely_load_model(path, env, device=args.device)
                if loaded is not None:
                    current_model = loaded
                    current_checkpoint_step = step
                    current_checkpoint_path = path
                    is_waiting = False
                    print(f"\n[Live Watch] Initial checkpoint loaded: {step:,} steps")
                    break

        # Render waiting screen smoothly
        env.render()
        if env.screen is None:
            running = False
            break
        pygame.time.Clock().tick(15)

    if not running:
        env.close()
        if pygame.get_init():
            pygame.quit()
        return

    # -------------------------------------------------------------
    # 2. MAIN SPECTATOR LOOP: Continuous episodes with model switching
    # -------------------------------------------------------------
    while running:
        # Check if a pending newer checkpoint is ready to be loaded at episode boundary
        if pending_checkpoint is not None:
            cand_step, cand_path = pending_checkpoint
            new_model = safely_load_model(cand_path, env, device=args.device)
            if new_model is not None:
                current_model = new_model
                current_checkpoint_step = cand_step
                current_checkpoint_path = cand_path
                pending_checkpoint = None
                print(f"[Live Watch] Loaded checkpoint: {cand_step:,} steps")

        episode_num += 1
        current_episode_time = 0.0
        obs, info = env.reset()
        done = False
        ep_steps = 0

        # Episode step loop
        while not done and running:
            # Handle user input / window close
            if pygame.get_init() and pygame.display.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                        break

            if not running or env.screen is None:
                break

            # Check directory periodically for newer checkpoint
            now = time.time()
            if now - last_check_time >= args.check_interval:
                last_check_time = now
                latest = get_latest_checkpoint(args.checkpoint_dir)
                if latest is not None and current_checkpoint_step is not None:
                    lat_step, lat_path = latest
                    if lat_step > current_checkpoint_step:
                        if pending_checkpoint is None or lat_step > pending_checkpoint[0]:
                            pending_checkpoint = latest
                            print(f"\n[Live Watch] New checkpoint detected:")
                            print(f"{current_checkpoint_step:,} -> {lat_step:,} steps")
                            print("Will switch after current episode.\n")

            # Step agent policy (stochastic by default: deterministic=False)
            action, _ = current_model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            current_episode_time = float(info.get("elapsed_time", ep_steps * RL_DT))
            done = terminated or truncated

        if not running or env.screen is None:
            break

        # Record outcome of completed attempt
        is_success = bool(info.get("is_success", False))
        if is_success:
            prev_outcome = "SUCCESS"
            prev_success_time = current_episode_time
        elif info.get("timeout", False):
            prev_outcome = "TIMEOUT"
        else:
            prev_outcome = "TERMINATED"

        # Pause briefly on final frame so user can register outcome
        if args.pause > 0.0:
            pause_start = time.time()
            while (time.time() - pause_start) < args.pause and running and env.screen is not None:
                if pygame.get_init() and pygame.display.get_init():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            running = False
                            break
                env.render()
                pygame.time.Clock().tick(args.fps)

    env.close()
    if pygame.get_init():
        pygame.quit()
    print("\n[Live Watch] Spectator window closed cleanly.")


if __name__ == "__main__":
    main()
