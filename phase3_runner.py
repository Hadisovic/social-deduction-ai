"""Trusted fixed-timestep orchestration shared by visual and headless modes."""
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from time import perf_counter

from social_deduction.phase3_api import IntentKind
from phase3_bots import ScriptedController
from phase3_engine import GameConfig, Phase3Game


def canonical(value):
    """JSON-compatible values with stable ordering; never used as policy input."""
    if isinstance(value, Enum):
        return value.value
    if type(value) is GameConfig:
        return canonical(value.to_dict())
    if is_dataclass(value):
        return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    return value


def canonical_bytes(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def behavior_seed(seed: int, player_id: str) -> int:
    """Independent per-policy stream; the match seed is not handed to policies."""
    return int.from_bytes(sha256(f"phase3:behavior:{seed}:{player_id}".encode()).digest()[:16], "big")


class ScriptedMatch:
    def __init__(self, seed=0, config=None, planner=None):
        self.game = Phase3Game(seed=seed, config=config or GameConfig(), planner=planner)
        self._initialize(seed)

    def _initialize(self, seed):
        self.seed = seed
        self.controllers = {player: ScriptedController(behavior_seed(seed, player))
                            for player in sorted(self.game.players)}
        self.decision_ticks = max(1, math.ceil(self.game.config.decision_interval / self.game.config.dt - 1e-9))
        self._actor_hash = sha256()
        self._static_observation_hashes = {}
        self.decision_calls = 0
        self.observation_hash_seconds = 0.0
        self.policy_seconds = 0.0

    def reset(self, seed=None):
        selected = self.seed if seed is None else seed
        self.game.reset(selected)
        self._initialize(selected)
        return self

    def _record_observation(self, player, obs):
        # Hash every field in the complete decision input. Large immutable public
        # map/roster metadata is hashed once, then its digest enters each packet.
        # It never contains truth and is never supplied to the policy as a seed.
        static_names = ("map", "destinations", "roster", "settings")
        static = {name: getattr(obs, name) for name in static_names}
        prior = self._static_observation_hashes.get(player)
        if prior is None or prior[0] != static:
            prior = static, sha256(canonical_bytes(static)).hexdigest()
            self._static_observation_hashes[player] = prior
        dynamic = {field.name: getattr(obs, field.name) for field in fields(obs)
                   if field.name not in static_names}
        self._actor_hash.update(canonical_bytes((player, prior[1], dynamic)))
        self._actor_hash.update(b"\n")

    def step(self):
        if self.game.is_terminal:
            return self.game.result
        actions = {}
        if self.game.tick % self.decision_ticks == 0:
            # All snapshots are made before any policy or action is executed.
            # A policy cannot observe another same-tick decision's consequences.
            observations = {player: self.game.observe(player) for player in sorted(self.controllers)}
            self.game.record_observations(observations)
            for player, obs in observations.items():
                started = perf_counter()
                self._record_observation(player, obs)
                self.observation_hash_seconds += perf_counter() - started
                if not obs.own.active or not obs.actions:
                    continue
                started = perf_counter()
                action = self.controllers[player].decide(obs)
                self.policy_seconds += perf_counter() - started
                self.decision_calls += 1
                self._actor_hash.update(canonical_bytes((player, action)))
                self._actor_hash.update(b"\n")
                # WAIT also expresses "no submission" in scheduled discussion
                # and voting. Do not fabricate an illegal action for an idle bot.
                if action.kind is IntentKind.WAIT and not any(
                        candidate.kind is IntentKind.WAIT for candidate in obs.actions):
                    continue
                actions[player] = action
        self.game.step(actions)
        return self.game.result

    def run(self):
        while not self.game.is_terminal:
            self.step()
        return self.game.result

    @property
    def actor_trace_hash(self):
        return self._actor_hash.hexdigest()

    @property
    def truth_trace_hash(self):
        return sha256(canonical_bytes(self.game.event_log)).hexdigest()

    @property
    def trajectory_hash(self):
        return sha256(canonical_bytes((self.truth_trace_hash, self.actor_trace_hash,
                                       self.game.result))).hexdigest()

    def replay_document(self):
        return {
            "schema": "phase3-replay-v1" if self.game.config.rules_version == 1 else "phase6-replay-v1", "seed": self.seed,
            "config": canonical(self.game.config), "result": canonical(self.game.result),
            "actor_trace_hash": self.actor_trace_hash, "truth_trace_hash": self.truth_trace_hash,
            "trajectory_hash": self.trajectory_hash,
            "notice": "Privileged spectator truth log; never passed to a controller.",
            "events": canonical(self.game.event_log),
        }

    def write_replay(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = self.replay_document()
        events = document.pop("events")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"record": "metadata", **document}, sort_keys=True) + "\n")
            for event in events:
                stream.write(json.dumps({"record": "event", **event}, sort_keys=True) + "\n")
