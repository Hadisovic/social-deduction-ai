"""Recorder adapter for ScriptedMatch / CrewmateV2Game.

This module hooks into a running simulation and writes a replay file.
It is an observer — it does not change tick timing, RNG or actions.

Usage
-----
    match = ScriptedMatch(seed=42)
    recorder = MatchRecorder(match, "replays/ep-0042.jsonl", episode=42)
    while not match.game.is_terminal:
        match.step()
        recorder.after_step()
    recorder.finish()
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from spectator.replay_format import (
    ReplayHeader, ReplayTick, ReplayResult, ReplayRecorder,
    SCHEMA_VERSION,
)

try:
    from phase3_runner import ScriptedMatch
except ImportError:
    ScriptedMatch = None  # type: ignore[assignment,misc]


def _val(v):
    return v.value if isinstance(v, Enum) else v


class MatchRecorder:
    """Wraps a ScriptedMatch (or CrewmateV2Game) and writes a replay file."""

    def __init__(self, match, path: Path | str, *,
                 episode: int = 0,
                 training_session: str = "",
                 checkpoint_path: str = "",
                 model_architecture: str = "", controller_types: dict | None = None,
                 metadata: dict | None = None, source_version: str | None = None,
                 game_schema: str | None = None):
        self.match = match
        self.recorder = ReplayRecorder(path)
        self._episode = episode
        self._cumulative: dict[str, float] = {}
        self._wrote_header = False

        game = getattr(match, "game", match)
        players = game.players

        agent_roles = {pid: _val(p.role) for pid, p in players.items()}
        controller_types = controller_types or {pid: "scripted" for pid in players}

        header = ReplayHeader(
            schema=SCHEMA_VERSION,
            game_schema=game_schema or ("crewmate-v2" if hasattr(game, "observe_v2") else "phase3-v1"),
            source_version=source_version if source_version is not None else self._git_sha(),
            training_session=training_session,
            episode=episode,
            scenario_seed=getattr(game, "seed", 0),
            agent_roles=agent_roles,
            controller_types=controller_types,
            checkpoint_path=checkpoint_path,
            model_architecture=model_architecture,
            simulation_timestamp=time.time(),
            config={**self._safe_config(game), **(metadata or {})},
        )
        self.recorder.write_header(header)
        self._wrote_header = True

        # Write events that happened before recording started (shouldn't be any)
        self._last_event_count = 0

    @staticmethod
    def _git_sha() -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2,
                cwd=str(Path(__file__).parent.parent),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    @staticmethod
    def _safe_config(game) -> dict:
        from dataclasses import fields as dc_fields, is_dataclass
        cfg = getattr(game, "config", None)
        if cfg is None:
            return {}
        if is_dataclass(cfg):
            try:
                from dataclasses import asdict
                return asdict(cfg)
            except Exception:
                pass
        return {}

    def after_step(self) -> None:
        """Call immediately after match.step() on every tick."""
        game = getattr(self.match, "game", self.match)

        # Write any new events since last tick
        event_log = getattr(game, "event_log", [])
        new_events = event_log[self._last_event_count:]
        for event in new_events:
            self.recorder.write_event(dict(event))
        self._last_event_count = len(event_log)

        # Write tick snapshot
        players_snap = {}
        for pid, p in game.players.items():
            owned=[t for t in getattr(p,'tasks',[]) if hasattr(t,'progress')]
            task=next((t for t in owned if t.task_id==getattr(p,'interaction_task',None)),None)
            players_snap[pid] = {
                "position": list(p.position),
                "velocity": list(getattr(p, "velocity", (0, 0))),
                "status": _val(p.status),
                "role": _val(p.role),
                "color": p.color_name,
                "intention": str(getattr(p, "intention", "")),
                "interaction_task": getattr(p, "interaction_task", None),
                "cooldown": getattr(p, "cooldown", 0.0),
                "meetings_remaining": getattr(p, "meetings_remaining", 0),
                "task_progress": task.progress if task is not None else None,
                "tasks_completed": sum(t.progress>=1 for t in owned) if owned else None,
                "tasks_total": len(owned) if owned else None,
                "ghost_position": list(game.ghosts[pid].position) if pid in getattr(game,'ghosts',{}) else None,
            }

        bodies_snap = {}
        for body_id, body in getattr(game, "bodies", {}).items():
            bodies_snap[body_id] = {
                "position": list(body.position),
                "victim_id": body.victim_id,
                "killer_id": body.killer_id,
                "reported": body.reported,
            }

        tick_rec = ReplayTick(
            tick=game.tick,
            sim_time=game.time,
            phase=_val(game.phase),
            players=players_snap,
            bodies=bodies_snap,
        )
        self.recorder.write_tick(tick_rec)

    def finish(self) -> None:
        """Call after the match is terminal."""
        game = getattr(self.match, "game", self.match)
        game_result = getattr(game, "result", None)

        result_rec = ReplayResult(
            tick=game.tick,
            sim_time=game.time,
            winner=_val(game_result.winner) if game_result and game_result.winner is not None else None,
            reason=getattr(game_result, "reason", "") if game_result else "",
            metrics=dict(getattr(game, "metrics", {})),
            actor_trace_hash=getattr(self.match, "actor_trace_hash", ""),
            truth_trace_hash=getattr(self.match, "truth_trace_hash", ""),
        )
        self.recorder.write_result(result_rec)

    def record_neural(self, player_id: str, action_kind: str, action_target=None,
                      probabilities: dict | None = None, value_estimate: float | None = None,
                      entropy: float | None = None, legal_actions: list | None = None,
                      masked_actions: list | None = None, training_step: int | None = None,
                      policy_update: int | None = None, selected_action_index: int | None = None,
                      suspicion_probabilities: dict | None = None) -> None:
        """Optional: record a neural agent's decision for this player."""
        from spectator.replay_format import ReplayNeural
        game = getattr(self.match, "game", self.match)
        neural = ReplayNeural(
            tick=game.tick,
            sim_time=game.time,
            player_id=player_id,
            action_kind=str(action_kind),
            action_target=str(action_target) if action_target is not None else None,
            action_probabilities=dict(probabilities or {}),
            value_estimate=float(value_estimate) if value_estimate is not None else None,
            policy_entropy=float(entropy) if entropy is not None else None,
            legal_actions=list(legal_actions or []),
            masked_actions=list(masked_actions or []),
            training_step=training_step, policy_update=policy_update, selected_action_index=selected_action_index,
            suspicion_probabilities=dict(suspicion_probabilities or {}),
        )
        self.recorder.write_neural(neural)
