"""Fail closed: valid shared settings are not evidence of a runnable Jane match."""

from dataclasses import dataclass
from contextlib import redirect_stdout
import io
import re

from jane import CONTRACT_SCHEMA
from .settings import GameSettings, bundle_receipt


class CompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class CompatibilityReport:
    engine_id: str
    compatible: bool
    blockers: tuple[str, ...]
    settings_sha256: str
    scope: str = "configuration-only"
    model_execution_verified: bool = False
    notes: tuple[str, ...] = ()


def legacy_lobby(settings: GameSettings):
    """Translate only supported settings to the unchanged V2.2 lobby constructor.

    This constructs configuration only. It neither constructs an engine nor loads
    a checkpoint. Historical weights remain restricted to their registered study.
    """
    blockers = []
    if settings.ruleset_id != "historical-v2.2":
        blockers.append("historical engine does not implement the new classic ruleset")
    if (settings.lobby.players, settings.lobby.impostors) != (5, 1):
        blockers.append("historical engine supports exactly four crew plus one impostor")
    if settings.mechanics.enabled():
        blockers.append("historical engine has no enabled extension: "
                        + ", ".join(sorted(settings.mechanics.enabled())))
    if any(role.enabled for role in settings.optional_roles):
        blockers.append("historical engine supports only base Crewmate and Impostor roles")
    if settings.lobby.kill_distance != "medium":
        blockers.append("short/long native kill distances have not been calibrated by this adapter")
    if settings.simulation.cap_policy != "historical-impostor-loss":
        blockers.append("historical engine uses a terminal match timeout, not truncation")
    if settings.simulation.stall_cap_seconds != 0:
        blockers.append("historical engine has no separate technical stall cut")
    if blockers:
        raise CompatibilityError("; ".join(blockers))
    # The old import chain prints pygame's support banner in a clean shell.
    # Keep the configuration CLI's stdout machine-readable without changing
    # global environment variables or touching the historical source.
    with redirect_stdout(io.StringIO()):
        from crewmate_v2.config import LobbySettings
    try:
        return LobbySettings(
            player_speed=settings.lobby.player_speed,
            crewmate_vision=settings.lobby.crewmate_vision,
            impostor_vision=settings.lobby.impostor_vision,
            kill_cooldown=settings.lobby.kill_cooldown_seconds,
            kill_distance_character_widths=2.,
            confirm_ejects=settings.lobby.confirm_ejects,
            emergency_meetings=settings.meetings.emergency_meetings,
            emergency_cooldown=settings.meetings.emergency_cooldown_seconds,
            discussion_seconds=settings.meetings.discussion_seconds,
            voting_seconds=settings.meetings.voting_seconds,
            anonymous_votes=settings.meetings.anonymous_votes,
            common_tasks=settings.tasks.common,
            long_tasks=settings.tasks.long,
            short_tasks=settings.tasks.short,
            task_bar_updates=settings.tasks.task_bar_updates,
            visual_tasks=settings.tasks.visual_tasks,
            dt=settings.simulation.physics_step_seconds,
            decision_interval=settings.simulation.observation_interval_seconds,
            timeout=settings.simulation.match_cap_seconds,
        )
    except ValueError as error:
        raise CompatibilityError(str(error)) from error


def check_engine(settings: GameSettings, engine_id: str):
    if engine_id == "historical-v2.2":
        try:
            legacy_lobby(settings)
        except CompatibilityError as error:
            blockers = (str(error),)
        else:
            blockers = ()
    elif engine_id == "patrick-jane":
        from .patrick_jane import a2_configuration
        try:
            a2_configuration(settings)
        except ValueError as error:
            blockers = (str(error),)
        else:
            blockers = ()
    else:
        raise ValueError(f"Unknown engine: {engine_id}")
    notes = (("Pinned upstream A2 configuration mapping only; no model load or learned-impostor adapter.",
              "Published weights are task-pursuit-v1, not A2 trained; runtime is Torch 2.7.1+cu128.")
             if engine_id == "patrick-jane" else ())
    return CompatibilityReport(engine_id, not blockers, blockers, settings.fingerprint(), notes=notes)


def check_pair(settings: GameSettings, crewmate: dict, impostor: dict):
    """Validate an integration declaration, never infer checkpoint compatibility.

    The future trusted loader must verify source/weight hashes and perform real
    boundary tests. This declaration check cannot authenticate a caller's claims.
    """
    blockers = []
    for expected_role, manifest in (("crewmate", crewmate), ("impostor", impostor)):
        prefix = expected_role + ": "
        if manifest.get("role") != expected_role:
            blockers.append(prefix + "wrong role")
        if manifest.get("integration_status") != "verified":
            blockers.append(prefix + "integration is not verified")
        if manifest.get("actor_contract") != CONTRACT_SCHEMA:
            blockers.append(prefix + "actor contract mismatch")
        if manifest.get("verified_settings_sha256") != settings.fingerprint():
            blockers.append(prefix + "settings fingerprint mismatch")
        if manifest.get("verified_bundle_sha256") != bundle_receipt(settings)["bundle_sha256"]:
            blockers.append(prefix + "shared bundle fingerprint mismatch")
        for key in ("source_commit", "checkpoint_sha256", "observation_schema", "action_schema", "tensor_schema"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                blockers.append(prefix + "missing " + key)
        for key, length in (("source_commit", 40), ("checkpoint_sha256", 64)):
            if not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", str(manifest.get(key, ""))):
                blockers.append(prefix + "invalid " + key)
    if crewmate.get("observation_schema") != impostor.get("observation_schema"):
        blockers.append("Role adapters do not declare the same semantic observation schema")
    if crewmate.get("action_schema") != impostor.get("action_schema"):
        blockers.append("Role adapters do not declare the same semantic action schema")
    return tuple(blockers)
