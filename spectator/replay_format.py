"""Versioned replay / telemetry schema (schema v1.0).

Files are written as JSONL (one JSON object per line) so they can be tailed
and partially read without parsing the whole file.

Line structure
--------------
{"record": "header",  ...}          # always first — static metadata
{"record": "tick",    ...}          # one per simulation tick — positions etc.
{"record": "event",   ...}          # game events (kills, reports, meetings …)
{"record": "neural",  ...}          # optional — one per agent decision
{"record": "reward",  ...}          # optional — per-tick reward components
{"record": "result",  ...}          # always last — match outcome

Recording is optional and must NOT alter simulation timing, RNG, actions or
rewards.  The recorder is a pure observer.
"""
from __future__ import annotations

import json
import os
import hashlib
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "spectator-v1.0"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serial(value: Any) -> Any:
    """Convert a value to a JSON-serialisable form."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _serial(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    return value


def _dump(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------
# Data classes written to replay files
# ---------------------------------------------------------------------------

@dataclass
class ReplayHeader:
    """Written once at the start of every replay file."""
    schema: str = SCHEMA_VERSION
    game_schema: str = ""          # e.g. "phase3-v1", "crewmate-v2"
    source_version: str = ""       # git SHA or __version__
    training_session: str = ""     # run ID if produced during training
    episode: int = 0
    scenario_seed: int = 0
    agent_roles: dict = field(default_factory=dict)  # player_id -> role string
    controller_types: dict = field(default_factory=dict)  # player_id -> "scripted"|"ppo"|"frozen"
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    model_architecture: str = ""
    simulation_timestamp: float = field(default_factory=time.time)
    config: dict = field(default_factory=dict)


@dataclass
class ReplayTick:
    """One entry per simulation tick — position snapshot."""
    record: str = "tick"
    tick: int = 0
    sim_time: float = 0.0
    phase: str = "roaming"
    players: dict = field(default_factory=dict)  # player_id -> {pos, vel, status, room, intention, task}
    bodies: dict = field(default_factory=dict)   # body_id -> {pos, victim_id, reported}


@dataclass
class ReplayNeural:
    """One entry per agent decision step (optional)."""
    record: str = "neural"
    tick: int = 0
    sim_time: float = 0.0
    player_id: str = ""
    action_kind: str = ""
    action_target: str | None = None
    action_probabilities: dict = field(default_factory=dict)  # kind -> probability
    value_estimate: float | None = None
    policy_entropy: float | None = None
    legal_actions: list = field(default_factory=list)
    masked_actions: list = field(default_factory=list)
    observation_features: dict = field(default_factory=dict)
    training_step: int | None = None
    policy_update: int | None = None
    selected_action_index: int | None = None
    suspicion_probabilities: dict = field(default_factory=dict)


@dataclass
class ReplayReward:
    """Per-tick reward information (optional)."""
    record: str = "reward"
    tick: int = 0
    sim_time: float = 0.0
    player_id: str = ""
    reward: float = 0.0
    cumulative_reward: float = 0.0
    components: dict = field(default_factory=dict)  # component_name -> value


@dataclass
class ReplayResult:
    """Written at the end of the match."""
    record: str = "result"
    tick: int = 0
    sim_time: float = 0.0
    winner: str | None = None   # "crewmate", "impostor", or None
    reason: str = ""
    metrics: dict = field(default_factory=dict)
    actor_trace_hash: str = ""
    truth_trace_hash: str = ""


# ---------------------------------------------------------------------------
# Tags used by automatic interesting-game detection
# ---------------------------------------------------------------------------

INTERESTING_TAGS = {
    "first_impostor_win",
    "first_crew_win_ejection",
    "first_crew_win_tasks",
    "multiple_kills",
    "witnessed_murder",
    "correct_witness_accusation",
    "unexpected_innocent_ejection",
    "rare_vote_pattern",
    "long_pursuit",
    "repeated_fake_task_loop",
    "navigation_loop",
    "extreme_high_reward",
    "extreme_low_reward",
    "very_long_game",
    "simulation_anomaly",
    "manual_favorite",
}


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class ReplayRecorder:
    """Appends JSONL records to a file during simulation.

    The recorder is entirely optional.  It never receives actions or rewards
    unless the calling code explicitly passes them.  It must never block.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(".jsonl.writing")
        if self.path.exists():
            raise FileExistsError(self.path)
        self._stream = self._tmp.open("x", encoding="utf-8", newline="\n", buffering=65536)
        self._closed = False
        self.error = None
        self._tick_count = 0
        self._event_count = 0

    def write_header(self, header: ReplayHeader) -> None:
        self._write({"record": "header", **_serial(asdict(header))})

    def write_tick(self, tick: ReplayTick) -> None:
        self._write(_serial(asdict(tick)))
        self._tick_count += 1

    def write_event(self, event: dict) -> None:
        self._write({"record": "event", **event})
        self._event_count += 1

    def write_neural(self, neural: ReplayNeural) -> None:
        self._write(_serial(asdict(neural)))

    def write_reward(self, reward: ReplayReward) -> None:
        self._write(_serial(asdict(reward)))

    def write_result(self, result: ReplayResult) -> None:
        self._write(_serial(asdict(result)))
        self.close()

    def _write(self, obj: dict) -> None:
        if self._closed:
            return
        try:
            self._stream.write(_dump(obj) + "\n")
        except Exception as error:
            self.error = repr(error)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
            if self.error:
                return
            # Atomic rename so readers never see a partial file
            if self.path.exists():
                raise FileExistsError(self.path)
            self._tmp.rename(self.path)  # Windows atomic rename refuses an existing target.
        except Exception as error:
            self.error = repr(error)

    def __del__(self):
        if not getattr(self, "_closed", True):
            try:
                self._stream.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Reader / parser
# ---------------------------------------------------------------------------

class ReplayParseError(Exception):
    pass


def parse_replay(path: Path | str) -> dict:
    """Parse a spectator JSONL replay file into a structured dict.

    Returns
    -------
    {
        "header": dict,
        "ticks":  list[dict],
        "events": list[dict],
        "neural": list[dict],
        "rewards": list[dict],
        "result": dict | None,
        "tags":   list[str],   # auto-detected interesting-game tags
        "path":   Path,
        "complete": bool,
    }
    """
    path = Path(path)
    result = {
        "header": {},
        "ticks": [],
        "events": [],
        "neural": [],
        "rewards": [],
        "result": None,
        "tags": [],
        "path": path,
        "complete": False,
    }
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # Tolerate partial last line (file still being written)
                if not isinstance(obj, dict):
                    continue
                record = obj.get("record", "")
                if record == "header":
                    result["header"] = obj
                elif record == "tick":
                    import math
                    if (not isinstance(obj.get('players',{}),dict) or not isinstance(obj.get('bodies',{}),dict)
                            or not isinstance(obj.get('sim_time'),(int,float)) or not math.isfinite(obj['sim_time'])
                            or not isinstance(obj.get('tick'),int) or not isinstance(obj.get('phase','roaming'),str)):
                        continue
                    if any(not isinstance(p,dict) or not isinstance(p.get('position'),list)
                           or len(p['position'])!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p['position'])
                           for p in obj.get('players',{}).values()):
                        continue
                    if any(any(not isinstance(p.get(k,''),str) for k in ('status','color','role','intention'))
                           or not isinstance(p.get('velocity',[0,0]),list) or len(p.get('velocity',[0,0]))!=2
                           or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p.get('velocity',[0,0]))
                           for p in obj.get('players',{}).values()):
                        continue
                    if any(not isinstance(b,dict) or not isinstance(b.get('position'),list) or len(b['position'])!=2
                           or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in b['position'])
                           for b in obj.get('bodies',{}).values()):
                        continue
                    result["ticks"].append(obj)
                elif record == "event":
                    result["events"].append(obj)
                elif record == "neural":
                    result["neural"].append(obj)
                elif record == "reward":
                    result["rewards"].append(obj)
                elif record == "result":
                    result["result"] = obj
                    result["complete"] = True
    except (OSError, PermissionError):
        pass  # File locked or missing — return what we have

    result['complete'] = bool(result['result'] and result['ticks'] and result['header'].get('schema')==SCHEMA_VERSION)
    result["tags"] = _detect_tags(result)
    return result


def _detect_tags(replay: dict) -> list[str]:
    """Automatically detect interesting-game tags from parsed replay data."""
    tags = []
    result = replay.get("result") or {}
    events = replay.get("events") or []
    ticks = replay.get("ticks") or []
    rewards = replay.get("rewards") or []

    winner = result.get("winner")
    reason = result.get("reason", "")

    if winner == "impostor":
        tags.append("first_impostor_win")
    if winner == "crewmate" and "EJECT" in reason.upper():
        tags.append("first_crew_win_ejection")
    if winner == "crewmate" and "TASK" in reason.upper():
        tags.append("first_crew_win_tasks")

    kill_events = [e for e in events if e.get("kind") in ("elimination", "kill")]
    if len(kill_events) >= 2:
        tags.append("multiple_kills")

    for e in events:
        kind = e.get("kind", "")
        if kind == "witnessed_elimination":
            tags.append("witnessed_murder")
            break

    ejection_events = [e for e in events if e.get("kind") == "ejection" and e.get("player_id") is not None]
    for ej in ejection_events:
        role_at_ejection = None
        for e in events:
            if e.get("kind") == "confirmed_ejection" and e.get("player_id") == ej.get("player_id"):
                role_at_ejection = e.get("role")
                break
        if role_at_ejection == "crewmate":
            tags.append("unexpected_innocent_ejection")
            break

    sim_time = (result or {}).get("sim_time", 0)
    if sim_time and sim_time > 180:
        tags.append("very_long_game")

    if rewards:
        total = sum(r.get("reward", 0) for r in rewards)
        if total > 50:
            tags.append("extreme_high_reward")
        if total < -20:
            tags.append("extreme_low_reward")

    return tags


# ---------------------------------------------------------------------------
# Replay library scanner
# ---------------------------------------------------------------------------

class ReplayLibrary:
    """Monitors a directory for replay files and maintains an ordered index."""

    EXTENSIONS = (".jsonl",)

    def __init__(self, watch_dir: Path | str | None = None):
        self.watch_dir = Path(watch_dir).resolve() if watch_dir else None
        self._index: list[dict] = []   # list of lightweight summary dicts
        self._mtime: float = 0.0
        self.favorites: set[str] = set()
        self._files = ()
        self._load_favorites()

    def set_directory(self, directory: Path | str) -> None:
        self.watch_dir = Path(directory).resolve()
        self._index.clear()
        self._mtime = 0.0
        self._files = ()
        self._load_favorites()

    def _load_favorites(self):
        self.favorites = set()
        if not self.watch_dir:
            return
        try:
            for name in json.loads((self.watch_dir/'.spectator-favorites.json').read_text()):
                path=(self.watch_dir/name).resolve()
                if path.is_relative_to(self.watch_dir.resolve()):
                    self.favorites.add(str(path))
        except (OSError,ValueError,TypeError):
            pass

    def refresh(self) -> bool:
        """Scan the directory and update the index.  Returns True if changed."""
        if not self.watch_dir or not self.watch_dir.is_dir():
            return False
        try:
            fingerprint=tuple(sorted((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in self.watch_dir.rglob('*.jsonl')))
            current_mtime=max((mtime/1e9 for _,mtime,_ in fingerprint),default=0.)
        except OSError:
            return False
        if fingerprint == self._files:
            return False
        self._files = fingerprint
        self._mtime = current_mtime
        new_index = []
        for path in sorted(self.watch_dir.rglob("*.jsonl")):
            if path.name.endswith(".writing"):
                continue
            summary = self._summarise(path)
            if summary:
                new_index.append(summary)
        new_index.sort(key=lambda s: s.get("mtime", 0))
        self._index = new_index
        return True

    def _summarise(self, path: Path) -> dict | None:
        """Read only the header + result line; skip heavy tick data."""
        try:
            stat = path.stat()
        except OSError:
            return None
        summary = {"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size,
                   "complete": False, "header": {}, "result": None,
                   "favorite": str(path) in self.favorites, "tags": []}
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    if len(lines) >= 2:
                        break  # Got header; stop for speed
            # Also check last few lines for result
            with path.open('rb') as f:
                f.seek(max(0,stat.st_size-65536))
                tail_lines=f.read().decode('utf-8',errors='replace').splitlines()[-5:]
            for line in tail_lines:
                try:
                    obj = json.loads(line.strip())
                    if obj.get("record") == "result":
                        summary["result"] = obj
                        summary["complete"] = True
                    if obj.get("record") == "header":
                        summary["header"] = obj
                except Exception:
                    pass
            for line in lines:
                try:
                    obj = json.loads(line)
                    if obj.get("record") == "header":
                        summary["header"] = obj
                except Exception:
                    pass
        except OSError:
            pass
        if summary['header'].get('schema') != SCHEMA_VERSION:
            return None
        if summary["result"]:
            summary["tags"] = _detect_tags({"result": summary["result"], "events": []})
        return summary

    @property
    def entries(self) -> list[dict]:
        return list(self._index)

    @property
    def latest(self) -> dict | None:
        return self._index[-1] if self._index else None

    def toggle_favorite(self, path: str) -> None:
        if path in self.favorites:
            self.favorites.discard(path)
        else:
            self.favorites.add(path)
        for entry in self._index:
            entry['favorite']=entry['path'] in self.favorites
        if self.watch_dir:
            names=[str(Path(p).resolve().relative_to(self.watch_dir.resolve())) for p in self.favorites
                   if Path(p).resolve().is_relative_to(self.watch_dir.resolve())]
            temporary=self.watch_dir/'.spectator-favorites.tmp'
            temporary.write_text(json.dumps(names),encoding='utf-8')
            temporary.replace(self.watch_dir/'.spectator-favorites.json')
