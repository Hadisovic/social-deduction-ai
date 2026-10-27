"""Tests for the spectator package.

Run with:
    pytest tests/test_spectator.py -v
or:
    python -m pytest tests/test_spectator.py
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from spectator.replay_format import (
    ReplayHeader, ReplayTick, ReplayResult, ReplayRecorder,
    ReplayLibrary, parse_replay, SCHEMA_VERSION,
)
from spectator.playback import ReplayPlayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_replay(path: Path, n_ticks: int = 10,
                          complete: bool = True) -> Path:
    """Write a synthetic replay file with n_ticks position snapshots."""
    rec = ReplayRecorder(path)
    header = ReplayHeader(
        game_schema="test-v0",
        scenario_seed=42,
        episode=1,
        agent_roles={"red": "crewmate", "blue": "impostor"},
        controller_types={"red": "scripted", "blue": "scripted"},
    )
    rec.write_header(header)

    for t in range(n_ticks):
        tick = ReplayTick(
            tick=t,
            sim_time=float(t) * 0.2,
            phase="roaming",
            players={
                "red":  {"position": [float(t) * 0.1, 0.0], "velocity": [0.1, 0.0],
                         "status": "alive", "role": "crewmate", "color": "red",
                         "intention": "task:console1", "interaction_task": None,
                         "cooldown": 0.0, "meetings_remaining": 1},
                "blue": {"position": [10.0, 5.0], "velocity": [0.0, 0.0],
                         "status": "alive", "role": "impostor", "color": "blue",
                         "intention": "wait", "interaction_task": None,
                         "cooldown": 5.0, "meetings_remaining": 1},
            },
            bodies={},
        )
        rec.write_tick(tick)
        if t == 5:
            rec.write_event({"kind": "task_completed", "player_id": "red",
                             "time": float(t) * 0.2, "task_id": "t1", "console_id": "c1"})

    if complete:
        result = ReplayResult(
            tick=n_ticks,
            sim_time=float(n_ticks) * 0.2,
            winner="crewmate",
            reason="TASKS_COMPLETED",
            metrics={"task_completions": 2},
        )
        rec.write_result(result)
    else:
        rec.close()

    return path


# ---------------------------------------------------------------------------
# Schema / serialisation
# ---------------------------------------------------------------------------

class TestReplayFormat:
    def test_schema_version_string(self):
        assert SCHEMA_VERSION.startswith("spectator-v")

    def test_header_written_and_parsed(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _make_minimal_replay(p)
        with p.open("r") as f:
            first = json.loads(f.readline())
        assert first["record"] == "header"
        assert first["schema"] == SCHEMA_VERSION
        assert first["game_schema"] == "test-v0"
        assert first["scenario_seed"] == 42

    def test_ticks_present(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _make_minimal_replay(p, n_ticks=5)
        parsed = parse_replay(p)
        assert len(parsed["ticks"]) == 5

    def test_events_present(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _make_minimal_replay(p)
        parsed = parse_replay(p)
        assert any(e["kind"] == "task_completed" for e in parsed["events"])

    def test_result_present_when_complete(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _make_minimal_replay(p, complete=True)
        parsed = parse_replay(p)
        assert parsed["complete"] is True
        assert parsed["result"]["winner"] == "crewmate"

    def test_incomplete_file_handled(self, tmp_path):
        p = tmp_path / "partial.jsonl"
        _make_minimal_replay(p, complete=False)
        parsed = parse_replay(p)
        assert parsed["complete"] is False
        assert parsed["result"] is None
        assert len(parsed["ticks"]) > 0

    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        parsed = parse_replay(p)
        assert parsed["ticks"] == []
        assert parsed["complete"] is False

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(
            '{"record":"header","schema":"spectator-v1.0"}\n'
            'NOT_JSON\n'
            '{"record":"tick","tick":0,"sim_time":0.0,"phase":"roaming","players":{},"bodies":{}}\n',
            encoding="utf-8",
        )
        parsed = parse_replay(p)
        assert len(parsed["ticks"]) == 1


# ---------------------------------------------------------------------------
# Interesting-game tagging
# ---------------------------------------------------------------------------

class TestAutoTags:
    def test_crew_win_tasks_tag(self, tmp_path):
        p = tmp_path / "crew_tasks.jsonl"
        _make_minimal_replay(p, complete=True)
        parsed = parse_replay(p)
        # winner = crewmate + reason contains TASKS
        assert "first_crew_win_tasks" in parsed["tags"]

    def test_no_false_positive_impostor_tag(self, tmp_path):
        p = tmp_path / "crew_tasks.jsonl"
        _make_minimal_replay(p, complete=True)
        parsed = parse_replay(p)
        assert "first_impostor_win" not in parsed["tags"]


# ---------------------------------------------------------------------------
# Library scanner
# ---------------------------------------------------------------------------

class TestReplayLibrary:
    def test_detects_complete_replay(self, tmp_path):
        p = tmp_path / "ep001.jsonl"
        _make_minimal_replay(p, complete=True)
        lib = ReplayLibrary(tmp_path)
        changed = lib.refresh()
        assert changed is True
        assert len(lib.entries) == 1
        assert lib.entries[0]["complete"] is True

    def test_skips_writing_files(self, tmp_path):
        writing = tmp_path / "ep001.jsonl.writing"
        writing.write_text("{}\n", encoding="utf-8")
        lib = ReplayLibrary(tmp_path)
        lib.refresh()
        assert len(lib.entries) == 0

    def test_latest_returns_newest(self, tmp_path):
        for i in range(3):
            p = tmp_path / f"ep{i:03d}.jsonl"
            _make_minimal_replay(p, complete=True)
        lib = ReplayLibrary(tmp_path)
        lib.refresh()
        latest = lib.latest
        assert latest is not None
        assert "ep002" in latest["path"]

    def test_favorite_toggle(self, tmp_path):
        p = tmp_path / "ep001.jsonl"
        _make_minimal_replay(p, complete=True)
        lib = ReplayLibrary(tmp_path)
        lib.refresh()
        path_str = str(p)
        lib.toggle_favorite(path_str)
        assert path_str in lib.favorites
        lib.toggle_favorite(path_str)
        assert path_str not in lib.favorites


# ---------------------------------------------------------------------------
# Playback engine
# ---------------------------------------------------------------------------

class TestReplayPlayer:
    def _make_player(self, tmp_path: Path, n: int = 20) -> ReplayPlayer:
        p = tmp_path / "ep.jsonl"
        _make_minimal_replay(p, n_ticks=n, complete=True)
        from spectator.replay_format import parse_replay
        data = parse_replay(p)
        return ReplayPlayer(data)

    def test_seek_to_time_clamps(self, tmp_path):
        player = self._make_player(tmp_path)
        player.seek_to_time(-100)
        assert player.current_sim_time == pytest.approx(0.0, abs=0.01)
        player.seek_to_time(10000)
        assert player.current_sim_time == pytest.approx(player.duration, abs=0.01)

    def test_step_forward_and_backward(self, tmp_path):
        player = self._make_player(tmp_path, n=20)
        player.seek_to_time(0.0)
        t0 = player.current_sim_time
        player.step_forward(5)
        assert player.current_sim_time > t0
        player.step_backward(3)
        assert player.current_sim_time > t0  # still ahead of start

    def test_at_end_after_full_advance(self, tmp_path):
        player = self._make_player(tmp_path)
        player.seek_to_time(player.duration)
        assert player.at_end is True

    def test_progress_in_range(self, tmp_path):
        player = self._make_player(tmp_path)
        player.seek_to_time(player.duration * 0.5)
        assert 0.0 <= player.progress <= 1.0

    def test_current_state_has_players(self, tmp_path):
        player = self._make_player(tmp_path)
        state = player.current_state()
        assert "red" in state.players
        assert "blue" in state.players

    def test_event_markers_built(self, tmp_path):
        player = self._make_player(tmp_path)
        # We injected a task_completed event at tick 5
        assert any(m.kind == "task_completed" for m in player.markers)

    def test_next_event_time(self, tmp_path):
        player = self._make_player(tmp_path)
        player.seek_to_time(0.0)
        t = player.next_event_time()
        # Should return the task_completed marker time (1.0s = tick 5 × 0.2)
        assert t is not None
        assert t > 0.0

    def test_advance_returns_true_at_end(self, tmp_path):
        player = self._make_player(tmp_path, n=5)
        player.seek_to_time(0.0)
        # Advance more than total duration in one step
        finished = player.advance_sim_time(player.duration + 10.0)
        assert finished is True


# ---------------------------------------------------------------------------
# Recorder integration with Phase3 engine
# ---------------------------------------------------------------------------

class TestRecorderIntegration:
    def test_records_real_match(self, tmp_path):
        """Record an actual scripted Phase 3 match and verify the JSONL."""
        from phase3_runner import ScriptedMatch
        from spectator.recorder import MatchRecorder

        match = ScriptedMatch(seed=999)
        out = tmp_path / "ep999.jsonl"
        recorder = MatchRecorder(match, out, episode=999, training_session="test")

        max_steps = 5000
        steps = 0
        while not match.game.is_terminal and steps < max_steps:
            match.step()
            recorder.after_step()
            steps += 1
        recorder.finish()

        assert out.exists()
        parsed = parse_replay(out)
        assert parsed["complete"] is True
        assert len(parsed["ticks"]) >= 1
        assert parsed["result"]["winner"] in ("crewmate", "impostor", None)

    def test_recorder_does_not_change_outcome(self, tmp_path):
        """Two identical seeds must produce the same outcome with/without recording."""
        from phase3_runner import ScriptedMatch
        from spectator.recorder import MatchRecorder

        # Without recorder
        match_a = ScriptedMatch(seed=1234)
        while not match_a.game.is_terminal:
            match_a.step()
        winner_a = match_a.game.result.winner

        # With recorder
        match_b = ScriptedMatch(seed=1234)
        rec = MatchRecorder(match_b, tmp_path / "ep.jsonl")
        while not match_b.game.is_terminal:
            match_b.step()
            rec.after_step()
        rec.finish()
        winner_b = match_b.game.result.winner

        assert winner_a == winner_b, "Recording must not alter simulation outcome"

    def test_replay_hash_preserved(self, tmp_path):
        """Trajectory hash must be identical with or without the recorder."""
        from phase3_runner import ScriptedMatch
        from spectator.recorder import MatchRecorder

        match_a = ScriptedMatch(seed=55)
        while not match_a.game.is_terminal:
            match_a.step()

        match_b = ScriptedMatch(seed=55)
        rec = MatchRecorder(match_b, tmp_path / "ep.jsonl")
        while not match_b.game.is_terminal:
            match_b.step()
            rec.after_step()
        rec.finish()

        assert match_a.trajectory_hash == match_b.trajectory_hash
