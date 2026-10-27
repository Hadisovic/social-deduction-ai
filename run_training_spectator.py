"""run_training_spectator.py — Social Deduction AI Spectator & Replay Studio

Launch:
    python run_training_spectator.py
    python run_training_spectator.py --watch "C:\\AI\\crewmate-runs"
    python run_training_spectator.py --replay path/to/match.jsonl
    python run_training_spectator.py --demo --seed 42

Keyboard shortcuts
------------------
SPACE           Play / Pause
RIGHT           Step forward one simulation tick
LEFT            Step backward one simulation tick
.  (period)     Jump to next event marker
,  (comma)      Jump to previous event marker
N               Next episode / replay
P               Previous episode / replay
R               Replay the current episode from start
+  / =          Double playback speed
-               Halve playback speed
F               Browse for replay folder
O               Open a specific replay file
D               Generate and play a demo match
TAB             Cycle selected player
1               Toggle role reveal
G               Toggle route overlay
V               Toggle sight-range circle
T               Toggle task markers
L               Toggle player labels
C               Toggle collision geometry
M               Cycle playback mode (manual → auto-next → follow latest)
★ (S)           Toggle favorite on current replay
ESC / Q         Quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path

# ── Make sure the repository root is on sys.path ─────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Social Deduction AI — Spectator & Replay Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--watch", metavar="DIR",
                   help="Directory to monitor for new replay files")
    p.add_argument("--replay", metavar="FILE",
                   help="Open a specific replay file immediately")
    p.add_argument("--demo", action="store_true",
                   help="Generate and play a scripted demo match on startup")
    p.add_argument("--seed", type=int, default=7,
                   help="Seed for the demo match (default: 7)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Initial playback speed multiplier (default: 1.0)")
    p.add_argument("--mode", choices=["manual", "auto_next", "follow_latest"],
                   default="manual",
                   help="Initial playback mode")
    p.add_argument("--fullscreen", action="store_true",
                   help="Start in fullscreen")
    p.add_argument("--width", type=int, default=0,
                   help="Window width (0 = auto-fit to monitor)")
    p.add_argument("--height", type=int, default=0,
                   help="Window height (0 = auto-fit to monitor)")
    p.add_argument("--max-frames", type=int,
                   help="Exit after this many frames (smoke test mode)")
    p.add_argument("--capture", type=Path,
                   help="Save one screenshot and exit (CI/documentation)")
    p.add_argument('--status-file',type=Path,help='Optional atomic training progress JSON')
    p.add_argument('--state-file',type=Path,help='Optional local GUI verification state')
    return p.parse_args(argv)


# ── Replay directory (default next to the script) ────────────────────────────
_DEFAULT_REPLAYS = _ROOT / "replays"

SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
MODES = ["manual", "auto_next", "follow_latest"]


def _nearest_speed(current: float, direction: int) -> float:
    """direction +1 = faster, -1 = slower."""
    for i, s in enumerate(SPEED_STEPS):
        if abs(s - current) < 0.01:
            idx = i + direction
            return SPEED_STEPS[max(0, min(len(SPEED_STEPS) - 1, idx))]
    # Snap to nearest
    nearest = min(SPEED_STEPS, key=lambda s: abs(s - current))
    return nearest


def _open_folder_dialog() -> str | None:
    """Show a native Windows folder picker (falls back gracefully)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select Replay Directory")
        root.destroy()
        return folder or None
    except Exception:
        return None


def _open_file_dialog() -> str | None:
    """Show a native file picker for .jsonl files."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        f = filedialog.askopenfilename(
            title="Open Replay File",
            filetypes=[("JSONL replay", "*.jsonl"), ("All files", "*.*")],
        )
        root.destroy()
        return f or None
    except Exception:
        return None


def _load_replay_player(path: Path):
    """Parse a replay file and return a ReplayPlayer, or None on failure."""
    from spectator.replay_format import parse_replay
    from spectator.playback import ReplayPlayer
    try:
        data = parse_replay(path)
        if not data["ticks"]:
            return None
        return ReplayPlayer(data)
    except Exception as exc:
        print(f"[spectator] Could not load {path}: {exc}", flush=True)
        return None


def main(argv=None) -> int:
    args = _parse_args(argv)

    # ── Pygame init ──────────────────────────────────────────────────────────
    import pygame
    from among_us_map import get_map
    from spectator_display import fitted_window_size

    pygame.init()
    desktops = []
    for i in range(pygame.display.get_num_displays()):
        mode = pygame.display.get_desktop_sizes()[i] if hasattr(pygame.display, "get_desktop_sizes") else (1920, 1080)
        desktops.append(mode)
    if not desktops:
        desktops = [(1920, 1080)]

    if args.width and args.height:
        win_size = (args.width, args.height)
    else:
        try:
            win_size = fitted_window_size(desktops, canvas=(1680, 960))
        except Exception:
            win_size = (1440, 900)

    flags = pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
        win_size = desktops[0]

    screen = pygame.display.set_mode(win_size, flags)
    pygame.display.set_caption("Social Deduction AI  |  Spectator & Replay Studio")
    icon_color = (100, 230, 200)

    # Loading splash
    screen.fill((6, 10, 18))
    font_load = pygame.font.SysFont("segoeui", 22)
    screen.blit(font_load.render("Loading Skeld map…", True, (226, 237, 249)), (30, 30))
    pygame.display.flip()
    pygame.event.pump()

    # ── Build world and renderer ─────────────────────────────────────────────
    world = get_map()
    from spectator.ui_renderer import SpectatorRenderer, Fonts, draw_welcome_screen
    from spectator.app_state import SpectatorAppState
    from spectator.replay_format import ReplayLibrary
    from spectator.watcher import WatchService
    from spectator.playback import ReplayPlayer

    renderer = SpectatorRenderer(world, win_size)
    fonts = Fonts.get()
    app = SpectatorAppState()
    app.playback_speed = args.speed
    app.playback_mode = args.mode

    # ── Library & watch service ──────────────────────────────────────────────
    library = ReplayLibrary()
    watcher = WatchService(library)

    watch_dir = args.watch or str(_DEFAULT_REPLAYS)
    if args.watch:
        library.set_directory(watch_dir)
        app.watch_dir = watch_dir

    _player: ReplayPlayer | None = None

    def _set_replay(path: str | Path) -> None:
        nonlocal _player
        p = Path(path)
        rp = _load_replay_player(p)
        if rp is None:
            return
        _player = rp
        app.current_replay_path = str(p)
        app.player_duration = rp.duration
        app.event_markers = rp.markers
        app.episode_info = rp.episode_info()
        app.selected_player=app.episode_info.get('focal_player') or next(iter(rp.current_state().players),None)
        app.playing = True
        pygame.display.set_caption('Social Deduction AI | Spectator & Replay Studio | '+app.episode_info['condition'])
        print(f"[spectator] Loaded {p.name}: {app.episode_info['condition']}",flush=True)

    # ── Initial replay ───────────────────────────────────────────────────────
    if args.replay:
        _set_replay(args.replay)
    elif args.demo or (not args.watch and not args.replay):
        # Generate a demo replay if nothing else is specified
        _DEFAULT_REPLAYS.mkdir(parents=True, exist_ok=True)
        splash_msg = "Generating demo match (scripted)…"
        screen.fill((6, 10, 18))
        screen.blit(font_load.render(splash_msg, True, (226, 237, 249)), (30, 30))
        screen.blit(font_load.render("This only happens once.", True, (80, 110, 140)), (30, 60))
        pygame.display.flip()
        pygame.event.pump()
        from spectator.demo import ensure_demo_replay
        demo_path = ensure_demo_replay(_DEFAULT_REPLAYS, seed=args.seed)
        library.set_directory(str(_DEFAULT_REPLAYS))
        app.watch_dir = str(_DEFAULT_REPLAYS)
        library.refresh()
        _set_replay(demo_path)

    # ── Main loop ────────────────────────────────────────────────────────────
    clock = pygame.time.Clock()
    frames = 0
    anim_time = 0.0
    lib_refresh_countdown = 0
    next_status=0.
    running = True

    toggle_keys = {
        pygame.K_1: "roles",
        pygame.K_g: "routes",
        pygame.K_v: "visibility",
        pygame.K_t: "tasks",
        pygame.K_l: "labels",
        pygame.K_c: "collision",
        pygame.K_b: "bodies",
    }

    while running:
        wall_dt = min(clock.tick(60) / 1000.0, 0.1)
        anim_time += wall_dt

        # ── Events ─────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                win_size = (event.w, event.h)
                screen = pygame.display.set_mode(win_size, pygame.RESIZABLE)
                renderer.resize(win_size)

            elif event.type == pygame.KEYDOWN:
                k = event.key

                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif k == pygame.K_SPACE:
                    if _player:
                        app.playing = not app.playing

                elif k == pygame.K_RIGHT and _player:
                    _player.step_forward(1)
                    app.playing = False

                elif k == pygame.K_LEFT and _player:
                    _player.step_backward(1)
                    app.playing = False

                elif k == pygame.K_PERIOD and _player:   # .  → next event
                    t = _player.next_event_time()
                    if t is not None:
                        _player.seek_to_time(t)

                elif k == pygame.K_COMMA and _player:    # ,  → prev event
                    t = _player.prev_event_time()
                    if t is not None:
                        _player.seek_to_time(t)

                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    app.playback_speed = _nearest_speed(app.playback_speed, +1)

                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    app.playback_speed = _nearest_speed(app.playback_speed, -1)

                elif k == pygame.K_r and _player:        # Replay from start
                    _player.seek_to_time(0.0)
                    app.playing = True

                elif k == pygame.K_n:                    # Next episode
                    nxt = watcher.next_after(app.current_replay_path)
                    if nxt:
                        _set_replay(nxt)
                    elif _player:
                        _player.seek_to_time(0.0)
                        app.playing = True

                elif k == pygame.K_p and _player:        # Previous episode — seek start for now
                    _player.seek_to_time(0.0)

                elif k == pygame.K_f:                    # Browse folder
                    pygame.display.iconify()
                    folder = _open_folder_dialog()
                    if folder:
                        library.set_directory(folder)
                        app.watch_dir = folder
                        library.refresh()
                        app.library_entries = library.entries
                        latest = watcher.latest_complete_path()
                        if latest:
                            _set_replay(latest)

                elif k == pygame.K_o:                    # Open file
                    pygame.display.iconify()
                    f = _open_file_dialog()
                    if f:
                        _set_replay(f)

                elif k == pygame.K_d:                    # Demo
                    screen.fill((6, 10, 18))
                    screen.blit(font_load.render("Generating demo…", True, (226, 237, 249)), (30, 30))
                    pygame.display.flip()
                    pygame.event.pump()
                    from spectator.demo import ensure_demo_replay
                    demo_p = ensure_demo_replay(_DEFAULT_REPLAYS, seed=args.seed)
                    _set_replay(demo_p)

                elif k == pygame.K_m:                    # Cycle mode
                    idx = MODES.index(app.playback_mode)
                    app.playback_mode = MODES[(idx + 1) % len(MODES)]

                elif k == pygame.K_s:                    # Star / favorite
                    if app.current_replay_path:
                        library.toggle_favorite(app.current_replay_path)
                        library.refresh()
                        app.library_entries = library.entries

                elif k == pygame.K_TAB:                  # Cycle selected player
                    if _player and app.frame_state:
                        pids = sorted(app.frame_state.players.keys())
                        if not pids:
                            pass
                        elif app.selected_player not in pids:
                            app.selected_player = pids[0]
                        else:
                            idx = pids.index(app.selected_player)
                            app.selected_player = pids[(idx + 1) % len(pids)]

                elif k in toggle_keys:
                    key = toggle_keys[k]
                    app.view_options[key] = not app.view_options.get(key, False)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    selected=_handle_mouse_click(event.pos, app, _player)
                    if selected:
                        _set_replay(selected)
                        app.playback_mode='manual'

        # ── Watch service poll ──────────────────────────────────────────────
        if watcher.poll():
            app.library_entries = library.entries
            if app.playback_mode == "follow_latest":
                latest = watcher.latest_complete_path()
                if latest and str(latest) != app.current_replay_path:
                    _set_replay(latest)
            elif app.playback_mode=='auto_next' and (_player is None or (_player.at_end and not app.playing)):
                nxt=watcher.next_after(app.current_replay_path)
                if nxt:
                    _set_replay(nxt)

        # ── Advance playback ────────────────────────────────────────────────
        if _player and app.playing:
            sim_delta = wall_dt * app.playback_speed
            finished = _player.advance_sim_time(sim_delta)
            if finished:
                app.playing = False
                if app.playback_mode == "auto_next":
                    nxt = watcher.next_after(app.current_replay_path)
                    if nxt:
                        _set_replay(nxt)
                    # Stay at end until the watcher discovers the next complete game.

        # ── Build frame state ──────────────────────────────────────────────
        if _player:
            dt_within_tick = (wall_dt * app.playback_speed) / max(0.01, 0.2)  # config.dt ~0.2
            alpha = min(0.95, dt_within_tick) if app.playing else 0.0
            app.frame_state = _player.current_state(interpolation_alpha=alpha)
        else:
            app.frame_state = None

        # ── Render ─────────────────────────────────────────────────────────
        renderer.set_anim_time(anim_time)
        app.fps = clock.get_fps()

        if _player:
            frame_surf = renderer.draw(app)
        else:
            frame_surf = pygame.Surface(win_size)
            draw_welcome_screen(frame_surf, fonts, win_size, app.watch_dir)

        screen.blit(frame_surf, (0, 0))
        pygame.display.flip()
        if time.monotonic()>=next_status:
            next_status=time.monotonic()+2
            if args.status_file:
                try:
                    app.training_metrics=json.loads(args.status_file.read_text())
                except (OSError,ValueError):
                    pass
            if args.state_file:
                args.state_file.parent.mkdir(parents=True,exist_ok=True)
                temporary=args.state_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(pid=os.getpid(),replay=app.current_replay_path,
                    condition=(app.episode_info or {}).get('condition'),selected_player=app.selected_player,
                    sim_time=app.frame_state.sim_time if app.frame_state else None,
                    library_count=len(app.library_entries),mode=app.playback_mode,playing=app.playing,
                    updated_unix=time.time())),encoding='utf-8')
                temporary.replace(args.state_file)

        # ── Capture mode (CI/documentation) ───────────────────────────────
        if args.capture and frames == 30:  # After 30 frames — settled state
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(args.capture))
            print(f"[spectator] Captured screenshot: {args.capture}", flush=True)
            running = False

        if args.max_frames and frames >= args.max_frames:
            running = False

        frames += 1

    pygame.quit()
    return 0


def _handle_mouse_click(pos: tuple[int, int], app: SpectatorAppState,
                        player: "ReplayPlayer | None") -> None:
    """Handle left-click on the timeline scrub bar or map."""
    from spectator.app_state import SpectatorAppState
    for pid,rect in app._player_hitboxes.items():
        if rect.collidepoint(pos):
            app.selected_player=pid
            return
    for rect,path in app._library_hitboxes:
        if rect.collidepoint(pos):
            return path

    # Scrub bar seek
    if app._scrub_rect and app._scrub_rect.collidepoint(pos):
        if player and app.player_duration > 0:
            t = (pos[0] - app._scrub_rect.x) / app._scrub_rect.width
            player.seek_to_time(t * app.player_duration)
        return


if __name__ == "__main__":
    # Support running from a worktree by resolving the correct sys.path
    from phase3_cli import use_project_environment
    use_project_environment()
    raise SystemExit(main())
