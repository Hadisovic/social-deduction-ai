"""Replay playback engine — manages state for seeking, scrubbing and stepping.

This is purely a data layer.  It takes a parsed replay dict and provides:
  - random-access to any tick
  - current_state() returning everything the renderer needs
  - event markers for the timeline
  - smooth interpolated position between ticks
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerState:
    player_id: str
    color: str
    role: str
    status: str
    position: tuple[float, float]
    velocity: tuple[float, float]
    intention: str
    interaction_task: str | None
    cooldown: float
    meetings_remaining: int


@dataclass
class BodyState:
    body_id: str
    victim_id: str
    killer_id: str
    position: tuple[float, float]
    reported: bool


@dataclass
class FrameState:
    """Everything the spectator renderer needs for a single display frame."""
    tick: int
    sim_time: float
    phase: str
    players: dict[str, PlayerState]
    bodies: dict[str, BodyState]
    events_up_to_now: list[dict]    # all events with sim_time <= current
    neural_up_to_now: list[dict]    # all neural records up to current tick
    latest_neural: dict[str, dict]  # player_id -> most recent neural record
    latest_reward: dict[str, dict]  # player_id -> most recent reward record
    result: dict | None


@dataclass
class EventMarker:
    """A timeline event marker (shown on the scrub bar)."""
    sim_time: float
    tick: int
    kind: str
    label: str
    color: tuple[int, int, int]     # RGB
    player_id: str | None = None


# Colours for timeline markers
_MARKER_COLORS = {
    "kill": (255, 80, 80),
    "elimination": (255, 80, 80),
    "report": (80, 200, 120),
    "meeting_started": (80, 150, 255),
    "ejection": (255, 200, 60),
    "task_completed": (100, 220, 200),
    "fake_task_end": (200, 140, 200),
    "exploration_resumed": (140, 200, 255),
    "voting_started": (255, 170, 60),
    "meeting_resolution": (255, 200, 100),
}
_DEFAULT_MARKER_COLOR = (180, 180, 180)

_INTERESTING_EVENTS = {
    "kill", "elimination", "report", "meeting_started", "ejection",
    "task_completed", "voting_started", "meeting_resolution",
}


class ReplayPlayer:
    """Stateful playback engine for a parsed replay."""

    def __init__(self, replay: dict):
        self.replay = replay
        self._ticks: list[dict] = replay.get("ticks", [])
        self._events: list[dict] = replay.get("events", [])
        self._neural: list[dict] = replay.get("neural", [])
        self._rewards: list[dict] = replay.get("rewards", [])
        self._result: dict | None = replay.get("result")
        self._header: dict = replay.get("header", {})

        # Total duration
        if self._ticks:
            self._max_sim_time = self._ticks[-1].get("sim_time", 0.0)
            self._max_tick = self._ticks[-1].get("tick", len(self._ticks) - 1)
        else:
            self._max_sim_time = 0.0
            self._max_tick = 0

        # Current playback position
        self._current_tick_index: int = 0  # index into self._ticks
        self._sim_time: float = self._ticks[0].get("sim_time", 0.0) if self._ticks else 0.0

        # Precompute timeline markers
        self.markers: list[EventMarker] = self._build_markers()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def seek_to_time(self, sim_time: float) -> None:
        sim_time = max(0.0, min(sim_time, self._max_sim_time))
        self._sim_time = sim_time
        # Binary search for closest tick index
        lo, hi = 0, len(self._ticks) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._ticks[mid].get("sim_time", 0) <= sim_time:
                lo = mid
            else:
                hi = mid - 1
        self._current_tick_index = lo

    def seek_to_tick(self, tick: int) -> None:
        lo, hi = 0, len(self._ticks) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._ticks[mid].get("tick", 0) <= tick:
                lo = mid
            else:
                hi = mid - 1
        self._current_tick_index = lo
        self._sim_time = self._ticks[lo].get("sim_time", 0.0) if self._ticks else 0.0

    def step_forward(self, n: int = 1) -> None:
        self._current_tick_index = min(len(self._ticks) - 1, self._current_tick_index + n)
        if self._ticks:
            self._sim_time = self._ticks[self._current_tick_index].get("sim_time", self._sim_time)

    def step_backward(self, n: int = 1) -> None:
        self._current_tick_index = max(0, self._current_tick_index - n)
        if self._ticks:
            self._sim_time = self._ticks[self._current_tick_index].get("sim_time", self._sim_time)

    def next_event_time(self) -> float | None:
        """Return the sim_time of the next interesting event after the current time."""
        for m in self.markers:
            if m.sim_time > self._sim_time + 0.05:
                return m.sim_time
        return None

    def prev_event_time(self) -> float | None:
        """Return the sim_time of the previous interesting event."""
        for m in reversed(self.markers):
            if m.sim_time < self._sim_time - 0.05:
                return m.sim_time
        return None

    def advance_sim_time(self, delta: float) -> bool:
        """Advance by delta simulated seconds.  Returns True when at end."""
        new_time = self._sim_time + delta
        if new_time >= self._max_sim_time:
            self.seek_to_time(self._max_sim_time)
            return True
        self.seek_to_time(new_time)
        return False

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def current_sim_time(self) -> float:
        return self._sim_time

    @property
    def duration(self) -> float:
        return self._max_sim_time

    @property
    def progress(self) -> float:
        return self._sim_time / self._max_sim_time if self._max_sim_time > 0 else 0.0

    @property
    def at_end(self) -> bool:
        return self._current_tick_index >= len(self._ticks) - 1

    def current_state(self, interpolation_alpha: float = 0.0) -> FrameState:
        """Return the current FrameState, optionally interpolating toward the next tick."""
        if not self._ticks:
            return FrameState(0, 0.0, "roaming", {}, {}, [], [], {}, {}, None)

        tick_data = self._ticks[self._current_tick_index]
        tick = tick_data.get("tick", 0)
        sim_time = tick_data.get("sim_time", 0.0)
        phase = tick_data.get("phase", "roaming")

        # Build player states, optionally interpolated
        players: dict[str, PlayerState] = {}
        next_tick_data = None
        if interpolation_alpha > 0 and self._current_tick_index + 1 < len(self._ticks):
            next_tick_data = self._ticks[self._current_tick_index + 1]

        for pid, pdata in tick_data.get("players", {}).items():
            pos = tuple(pdata.get("position", [0, 0]))
            vel = tuple(pdata.get("velocity", [0, 0]))

            if next_tick_data and interpolation_alpha > 0:
                next_pdata = next_tick_data.get("players", {}).get(pid, {})
                next_pos = tuple(next_pdata.get("position", pos))
                next_status = next_pdata.get("status", pdata.get("status", "alive"))
                # Don't interpolate across status changes
                if next_status == pdata.get("status", "alive"):
                    px = pos[0] + (next_pos[0] - pos[0]) * interpolation_alpha
                    py = pos[1] + (next_pos[1] - pos[1]) * interpolation_alpha
                    pos = (px, py)

            players[pid] = PlayerState(
                player_id=pid,
                color=pdata.get("color", "red"),
                role=pdata.get("role", "crewmate"),
                status=pdata.get("status", "alive"),
                position=pos,
                velocity=vel,
                intention=pdata.get("intention", "wait"),
                interaction_task=pdata.get("interaction_task"),
                cooldown=pdata.get("cooldown", 0.0),
                meetings_remaining=pdata.get("meetings_remaining", 1),
            )

        bodies: dict[str, BodyState] = {}
        for bid, bdata in tick_data.get("bodies", {}).items():
            bodies[bid] = BodyState(
                body_id=bid,
                victim_id=bdata.get("victim_id", ""),
                killer_id=bdata.get("killer_id", ""),
                position=tuple(bdata.get("position", [0, 0])),
                reported=bdata.get("reported", False),
            )

        events_up_to_now = [e for e in self._events
                            if e.get("time", e.get("sim_time", 0)) <= sim_time + 0.01]

        neural_up_to_now = [n for n in self._neural
                            if n.get("sim_time", 0) <= sim_time + 0.01]

        # Latest neural per player
        latest_neural: dict[str, dict] = {}
        for n in neural_up_to_now:
            pid = n.get("player_id", "")
            if pid:
                latest_neural[pid] = n

        latest_reward: dict[str, dict] = {}
        for r in self._rewards:
            if r.get("sim_time", 0) <= sim_time + 0.01:
                pid = r.get("player_id", "")
                if pid:
                    latest_reward[pid] = r

        result = self._result if self.at_end else None

        return FrameState(tick, sim_time, phase, players, bodies,
                          events_up_to_now, neural_up_to_now,
                          latest_neural, latest_reward, result)

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------

    def _build_markers(self) -> list[EventMarker]:
        markers = []
        for event in self._events:
            kind = event.get("kind", event.get("type", ""))
            if kind not in _INTERESTING_EVENTS:
                continue
            t = event.get("time", event.get("sim_time", 0.0))
            tick = event.get("tick", 0)
            player_id = event.get("player_id", event.get("killer_id",
                                  event.get("reporter_id", None)))
            label = kind.replace("_", " ").title()
            color = _MARKER_COLORS.get(kind, _DEFAULT_MARKER_COLOR)
            markers.append(EventMarker(t, tick, kind, label, color, player_id))
        if self._result:
            t = self._result.get("sim_time", self._max_sim_time)
            winner = self._result.get("winner", "")
            label = f"{winner.upper()} WIN" if winner else "DRAW"
            color = (100, 255, 140) if winner == "crewmate" else (255, 100, 100) if winner == "impostor" else (200, 200, 200)
            markers.append(EventMarker(t, self._max_tick, "result", label, color))
        return sorted(markers, key=lambda m: m.sim_time)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def episode_info(self) -> dict:
        h = self._header
        return {
            "episode": h.get("episode", 0),
            "seed": h.get("scenario_seed", 0),
            "game_schema": h.get("game_schema", "unknown"),
            "training_session": h.get("training_session", ""),
            "checkpoint": h.get("checkpoint_path", ""),
            "complete": self.replay.get("complete", False),
            "tags": self.replay.get("tags", []),
            "duration": self._max_sim_time,
        }
