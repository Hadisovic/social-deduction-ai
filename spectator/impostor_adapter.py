"""Impostor replay adapter — reads impostor-format replays from the
codex/impostor-environment-v2 branch and maps them into the common
spectator FrameState / event format.

This is an isolated adapter contract.  It does NOT import any training code
from the impostor branch and does NOT modify that branch's files.

Current status
--------------
The adapter reads the standard JSONL spectator format produced by replay
files recorded with spectator.recorder.  If the impostor branch produces
its own format, add a converter function here.

Integration steps required by the impostor developer
-----------------------------------------------------
1. In your training loop, after each episode step, call::

       from spectator.recorder import MatchRecorder
       recorder.after_step()

2. At the end of each episode, call::

       recorder.finish()

3. Point run_training_spectator.py at your log directory::

       python run_training_spectator.py --watch "C:\\AI\\impostor-runs"

The spectator will automatically detect, parse and display your replays.
No other changes to training code are required.
"""
from __future__ import annotations

from pathlib import Path
from spectator.replay_format import parse_replay
from spectator.playback import ReplayPlayer


# ---------------------------------------------------------------------------
# Known impostor-specific action families (from phase3_api.py / bots.py)
# ---------------------------------------------------------------------------

IMPOSTOR_ACTIONS = {
    "wait":       "Wait",
    "go_to_task": "Move",
    "go_to_room": "Move to Room",
    "go_to_location": "Move to Location",
    "go_to_body": "Approach Body",
    "fake_task":  "Fake Task",
    "kill":       "Kill",
    "report":     "Report Body",
    "call_meeting": "Call Meeting",
    "claim":      "Make Claim",
    "vote":       "Vote",
    "skip":       "Skip Vote",
    "cancel_navigation": "Cancel Move",
}


def load_impostor_replay(path: Path | str) -> ReplayPlayer | None:
    """Load an impostor replay in the common JSONL spectator format."""
    path = Path(path)
    try:
        data = parse_replay(path)
    except Exception as exc:
        print(f"[impostor adapter] Could not parse {path}: {exc}")
        return None
    if not data["ticks"]:
        return None
    # Verify at least one impostor is present
    header = data.get("header", {})
    roles = header.get("agent_roles", {})
    if not any(r == "impostor" for r in roles.values()):
        # Still load it — may be a mixed replay
        pass
    return ReplayPlayer(data)


def impostor_action_label(action_kind: str) -> str:
    """Human-readable label for an impostor action kind."""
    return IMPOSTOR_ACTIONS.get(str(action_kind).lower(), action_kind.replace("_", " ").title())
