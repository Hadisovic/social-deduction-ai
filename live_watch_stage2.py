"""
Live Training Spectator for Stage 2 PPO (Phase 3B).
Provides a persistent, live Pygame window that monitors checkpoints in
models/stage2/checkpoints/ and automatically switches to newer checkpoints
at episode boundaries without interrupting training.
Before the first Stage 2 checkpoint exists, it visualizes the Stage 1 best model
operating in the Stage 2 environment.
"""

import os
import re
import csv
import time
import argparse
from typing import Optional, Tuple

import pygame
from stable_baselines3 import PPO

from config import WINDOW_WIDTH, WINDOW_HEIGHT, RL_DT
from rl_environment import StealthGymEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live Pygame spectator for Stage 2 PPO training."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/stage2/checkpoints",
        help="Directory to monitor for Stage 2 PPO checkpoints (default: 'models/stage2/checkpoints')"
    )
    parser.add_argument(
        "--stage1-baseline",
        type=str,
        default="models/stage1/best_model/best_model.zip",
        help="Path to Stage 1 best model used before first Stage 2 checkpoint (default: 'models/stage1/best_model/best_model.zip')"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling instead of default deterministic argmax"
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
        default=0.5,
        help="Seconds to pause on final frame after SUCCESS/TIMEOUT before reset (default: 0.5)"
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="logs/stage2/stage2_eval_summary.csv",
        help="Path to evaluation summary CSV to read latest eval success rate"
    )
    return parser.parse_args()


def get_latest_checkpoint(checkpoints_dir: str) -> Optional[Tuple[int, str]]:
    """
    Scan checkpoints directory and return (timestep, filepath) of the checkpoint
    with the largest timestep number. Returns None if no valid checkpoints exist.
    """
    if not os.path.exists(checkpoints_dir):
        return None

    pattern = re.compile(r"^stage2_(\d+)_steps\.zip$")
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


def get_latest_eval_success(csv_path: str) -> Optional[float]:
    """Read the latest evaluation success rate from summary CSV if available."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if reader:
                last_row = reader[-1]
                return float(last_row.get("success_rate", 0.0))
    except Exception:
        return None
    return None


def draw_spectator_overlay(
    surface: pygame.Surface,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
    checkpoint_step: Optional[int],
    is_stage1_baseline: bool,
    episode_num: int,
    is_deterministic: bool,
    current_time: float,
    prev_outcome: Optional[str],
    prev_success_time: Optional[float],
    latest_eval_success: Optional[float] = None,
    waiting_mode: bool = False,
    is_blocked: bool = False,
    blocked_streak: int = 0,
    is_stuck: bool = False,
    stagnation_progress: float = 0.0,
    current_cell: Optional[Tuple[int, int]] = None,
    cell_visits: int = 1,
    event_banner_text: Optional[str] = None,
    event_banner_color: Tuple[int, int, int] = (248, 113, 113),
):
    """
    Draw an unobtrusive HUD card in the upper-left of the arena.
    Does not obstruct player start (lower-left), goal (upper-right), or obstacles.
    """
    card_w = 300
    card_h = 285
    card_x = 48
    card_y = 48

    # Translucent rounded backdrop
    card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card_surf, (15, 23, 42, 225), (0, 0, card_w, card_h), border_radius=8)
    pygame.draw.rect(card_surf, (51, 65, 85, 240), (0, 0, card_w, card_h), width=1, border_radius=8)

    # Title
    title_text = "STAGE 2 TRAINING SPECTATOR"
    title_render = font_title.render(title_text, True, (241, 245, 249))
    card_surf.blit(title_render, (14, 10))

    if waiting_mode:
        msg = font_body.render("Waiting for model...", True, (251, 191, 36))
        card_surf.blit(msg, (14, 40))
        hint = font_body.render("Scanning for Stage 1 baseline or Stage 2 ckpt...", True, (148, 163, 184))
        card_surf.blit(hint, (14, 65))
        surface.blit(card_surf, (card_x, card_y))
        return

    # Checkpoint line
    if is_stage1_baseline:
        t_brain = font_body.render("Checkpoint: Stage 1 Baseline (0 steps)", True, (245, 158, 11))
    elif checkpoint_step is not None:
        t_brain = font_body.render(f"Checkpoint: {checkpoint_step:,} steps", True, (56, 189, 248))
    else:
        t_brain = font_body.render("Checkpoint: Unknown", True, (148, 163, 184))
    card_surf.blit(t_brain, (14, 32))

    # Episode number
    t_ep = font_body.render(f"Episode: {episode_num}", True, (203, 213, 225))
    card_surf.blit(t_ep, (14, 52))

    # Mode: Deterministic or Stochastic
    mode_str = "Deterministic" if is_deterministic else "Stochastic"
    t_mode = font_body.render(f"Mode: {mode_str}", True, (203, 213, 225))
    card_surf.blit(t_mode, (14, 72))

    # Current episode time
    t_time = font_body.render(f"Episode Time: {current_time:.2f} s", True, (203, 213, 225))
    card_surf.blit(t_time, (14, 92))

    # Stuck YES/NO
    if is_stuck:
        t_stk = font_body.render("Stuck: YES", True, (239, 68, 68))
    else:
        t_stk = font_body.render("Stuck: NO", True, (148, 163, 184))
    card_surf.blit(t_stk, (14, 112))

    # Blocked Streak
    blk_color = (248, 113, 113) if is_blocked else (203, 213, 225)
    t_blk = font_body.render(f"Blocked Streak: {blocked_streak}", True, blk_color)
    card_surf.blit(t_blk, (14, 132))

    # Stagnation: XX%
    stag_pct = int(round(stagnation_progress * 100))
    stag_color = (239, 68, 68) if stag_pct >= 100 else ((245, 158, 11) if stag_pct > 50 else (203, 213, 225))
    t_stag = font_body.render(f"Stagnation: {stag_pct}%", True, stag_color)
    card_surf.blit(t_stag, (14, 152))

    # Previous outcome
    if prev_outcome:
        outcome_color = (52, 211, 153) if prev_outcome == "SUCCESS" else (248, 113, 113)
        t_prev = font_body.render(f"Previous Result: {prev_outcome}", True, outcome_color)
    else:
        t_prev = font_body.render("Previous Result: N/A", True, (148, 163, 184))
    card_surf.blit(t_prev, (14, 172))

    # Previous completion time if successful
    if prev_success_time is not None:
        t_succ = font_body.render(f"Previous Time: {prev_success_time:.2f} s", True, (52, 211, 153))
    else:
        t_succ = font_body.render("Previous Time: N/A", True, (148, 163, 184))
    card_surf.blit(t_succ, (14, 192))

    # Current Cell: (x, y) & Cell Visits: N
    if current_cell is not None:
        t_cell = font_body.render(f"Current Cell: {current_cell}", True, (203, 213, 225))
        vst_color = (239, 68, 68) if cell_visits >= 3 else ((245, 158, 11) if cell_visits == 2 else (203, 213, 225))
        t_vst = font_body.render(f"Cell Visits: {cell_visits}", True, vst_color)
    else:
        t_cell = font_body.render("Current Cell: N/A", True, (148, 163, 184))
        t_vst = font_body.render("Cell Visits: 1", True, (148, 163, 184))
    card_surf.blit(t_cell, (14, 212))
    card_surf.blit(t_vst, (14, 232))

    # Optional brief event banner (STAGNATION -25, RECOVERED +X.XX, REVISIT -2)
    if event_banner_text:
        t_event = font_title.render(event_banner_text, True, event_banner_color)
        card_surf.blit(t_event, (14, 254))

    surface.blit(card_surf, (card_x, card_y))


def main():
    args = parse_args()
    is_deterministic = not args.stochastic
    action_mode_str = "deterministic" if is_deterministic else "stochastic"

    # Startup banner
    print("=" * 60)
    print("STAGE 2 LIVE TRAINING SPECTATOR")
    print("=" * 60)
    print(f"Watching directory: {args.checkpoint_dir}")
    print(f"Stage 1 baseline:   {args.stage1_baseline}")
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

    # Initialize Stage 2 environment in human render mode
    env = StealthGymEnv(stage=2, render_mode="human")
    env.metadata["render_fps"] = args.fps

    # Ensure persistent window surface and clock exist on the environment immediately
    if env.screen is None:
        env.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Stealth Sandbox - Stage 2 Training Spectator")
        env.clock = pygame.time.Clock()

    # Initialize Pygame fonts for HUD overlay
    if not pygame.font.get_init():
        pygame.font.init()
    try:
        font_title = pygame.font.SysFont("Segoe UI", 14, bold=True)
        font_body = pygame.font.SysFont("Segoe UI", 12)
    except Exception:
        font_title = pygame.font.Font(None, 20)
        font_body = pygame.font.Font(None, 18)

    # State tracking variables
    current_model: Optional[PPO] = None
    current_checkpoint_step: Optional[int] = None
    current_checkpoint_path: Optional[str] = None
    is_stage1_baseline = False
    pending_checkpoint: Optional[Tuple[int, str]] = None

    episode_num = 0
    current_episode_time = 0.0
    prev_outcome: Optional[str] = None
    prev_success_time: Optional[float] = None
    latest_eval_success: Optional[float] = None
    is_waiting = True
    is_current_blocked = False
    current_blocked_streak = 0
    is_current_stuck = False
    current_stagnation_progress = 0.0
    current_cell: Optional[Tuple[int, int]] = None
    current_cell_visits: int = 1
    event_banner_text: Optional[str] = None
    event_banner_color = (248, 113, 113)
    event_banner_expiry = 0.0

    # Register overlay callback
    def render_overlay(surface: pygame.Surface):
        now = time.time()
        active_banner = event_banner_text if now < event_banner_expiry else None
        draw_spectator_overlay(
            surface=surface,
            font_title=font_title,
            font_body=font_body,
            checkpoint_step=current_checkpoint_step,
            is_stage1_baseline=is_stage1_baseline,
            episode_num=episode_num,
            is_deterministic=is_deterministic,
            current_time=current_episode_time,
            prev_outcome=prev_outcome,
            prev_success_time=prev_success_time,
            latest_eval_success=latest_eval_success,
            waiting_mode=is_waiting,
            is_blocked=is_current_blocked,
            blocked_streak=current_blocked_streak,
            is_stuck=is_current_stuck,
            stagnation_progress=current_stagnation_progress,
            current_cell=current_cell,
            cell_visits=current_cell_visits,
            event_banner_text=active_banner,
            event_banner_color=event_banner_color,
        )

    env.overlay_callback = render_overlay

    running = True
    last_check_time = 0.0

    # Render initial frame so window appears and shows waiting message immediately
    env.render()

    # -------------------------------------------------------------
    # 1. INITIAL LOAD: Check for Stage 2 checkpoint first; fallback to Stage 1 baseline
    # -------------------------------------------------------------
    latest = get_latest_checkpoint(args.checkpoint_dir)
    if latest is not None:
        step, path = latest
        loaded = safely_load_model(path, env, device=args.device)
        if loaded is not None:
            current_model = loaded
            current_checkpoint_step = step
            current_checkpoint_path = path
            is_stage1_baseline = False
            is_waiting = False
            print(f"[Live Watch Stage 2] Initial Stage 2 checkpoint loaded: {step:,} steps")
    elif os.path.exists(args.stage1_baseline):
        loaded = safely_load_model(args.stage1_baseline, env, device=args.device)
        if loaded is not None:
            current_model = loaded
            current_checkpoint_step = 0
            current_checkpoint_path = args.stage1_baseline
            is_stage1_baseline = True
            is_waiting = False
            print("[Live Watch Stage 2] Loaded Stage 1 Best Model baseline in Stage 2 environment.")
            print("[Live Watch Stage 2] Observing how Stage 1 brain responds to central obstacles.")

    # If neither exists yet, wait gracefully
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
                    is_stage1_baseline = False
                    is_waiting = False
                    print(f"\n[Live Watch Stage 2] Initial checkpoint loaded: {step:,} steps")
                    break
            elif os.path.exists(args.stage1_baseline):
                loaded = safely_load_model(args.stage1_baseline, env, device=args.device)
                if loaded is not None:
                    current_model = loaded
                    current_checkpoint_step = 0
                    current_checkpoint_path = args.stage1_baseline
                    is_stage1_baseline = True
                    is_waiting = False
                    print("\n[Live Watch Stage 2] Loaded Stage 1 baseline model.")
                    break

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
                is_stage1_baseline = False
                pending_checkpoint = None
                print(f"[Live Watch Stage 2] Loaded Stage 2 checkpoint: {cand_step:,} steps")

        # Read latest eval summary success rate if available
        latest_eval_success = get_latest_eval_success(args.summary_csv)

        episode_num += 1
        current_episode_time = 0.0
        is_current_blocked = False
        current_blocked_streak = 0
        is_current_stuck = False
        current_stagnation_progress = 0.0
        obs, info = env.reset()
        current_cell = info.get("current_confirmed_cell")
        current_cell_visits = int(info.get("cell_visits", 1))
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
                if latest is not None:
                    lat_step, lat_path = latest
                    # Switch if transitioning from Stage 1 baseline (step 0) or newer step
                    if is_stage1_baseline or (current_checkpoint_step is not None and lat_step > current_checkpoint_step):
                        if pending_checkpoint is None or lat_step > pending_checkpoint[0]:
                            pending_checkpoint = latest
                            print(f"\n[Live Watch Stage 2] New checkpoint detected:")
                            from_str = "Stage 1 Baseline" if is_stage1_baseline else f"{current_checkpoint_step:,} steps"
                            print(f"{from_str} -> {lat_step:,} steps")
                            print("Will switch after current episode.\n")

            # Step agent policy (deterministic by default)
            action, _ = current_model.predict(obs, deterministic=is_deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            current_episode_time = float(info.get("elapsed_time", ep_steps * RL_DT))
            is_current_blocked = bool(info.get("is_blocked", False))
            current_blocked_streak = int(info.get("consecutive_blocked_steps", 0))
            is_current_stuck = bool(info.get("is_stuck", False))
            current_stagnation_progress = float(info.get("stagnation_progress", 0.0))
            current_cell = info.get("current_confirmed_cell")
            current_cell_visits = int(info.get("cell_visits", 1))

            # Check for stagnation penalty event
            stag_pen = float(info.get("stagnation_penalty", 0.0))
            if stag_pen < -1.0:
                event_banner_text = "STAGNATION -25"
                event_banner_color = (239, 68, 68)
                event_banner_expiry = time.time() + 1.5

            # Check for recovery reward event
            rec_reward = float(info.get("recovery_reward", 0.0))
            if rec_reward > 0.0:
                event_banner_text = f"RECOVERED +{rec_reward:.2f}"
                event_banner_color = (52, 211, 153)
                event_banner_expiry = time.time() + 1.5

            # Check for revisit penalty event
            revis_pen = float(info.get("revisit_penalty_this_step", 0.0))
            if revis_pen < -1.0:
                event_banner_text = "REVISIT -2"
                event_banner_color = (239, 68, 68)
                event_banner_expiry = time.time() + 1.5

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
    print("\n[Live Watch Stage 2] Spectator window closed cleanly.")


if __name__ == "__main__":
    main()
