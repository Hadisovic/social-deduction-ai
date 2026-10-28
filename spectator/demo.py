"""Demo match generator — runs a real scripted CrewmateV2 game and saves a replay.

Used when no replay files exist so the user sees a genuine game immediately.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make sure the repo root is on the path when called from a worktree
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def generate_demo_replay(output_path: Path | str, seed: int = 7) -> Path:
    """Run a full scripted match and write a replay file.

    Returns the path to the written replay.
    """
    output_path = Path(output_path)

    from phase3_runner import ScriptedMatch
    from spectator.recorder import MatchRecorder

    match = ScriptedMatch(seed=seed)
    recorder = MatchRecorder(
        match, output_path,
        episode=0,
        training_session="demo",
        model_architecture="scripted",
    )

    # Run the full match step by step so the recorder can observe each tick
    max_steps = 20000  # Safety limit
    step_count = 0
    while not match.game.is_terminal and step_count < max_steps:
        match.step()
        recorder.after_step()
        step_count += 1

    recorder.finish()
    return output_path


def ensure_demo_replay(replays_dir: Path | str, seed: int = 7) -> Path:
    """Return a demo replay path, generating it if it doesn't already exist."""
    replays_dir = Path(replays_dir)
    replays_dir.mkdir(parents=True, exist_ok=True)
    demo_path = replays_dir / f"demo_seed{seed:04d}.jsonl"
    if not demo_path.exists():
        print(f"[spectator] Generating demo replay for seed {seed}…", flush=True)
        generate_demo_replay(demo_path, seed)
        print(f"[spectator] Demo replay written: {demo_path}", flush=True)
    return demo_path
