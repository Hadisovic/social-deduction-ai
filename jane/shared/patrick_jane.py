"""Configuration-only adapter to the reviewed, pinned Patrick Jane A2 API.

No upstream neural code is vendored or imported. Model loading and learned
impostor execution remain separate integration gates.
"""

import json
from pathlib import Path


DATA = Path(__file__).resolve().parent / "upstream"


def reference_review():
    return json.loads((DATA / "patrick_jane_review.json").read_text(encoding="utf-8"))


def a2_configuration(settings):
    """Return native A2Experiment kwargs, or reject unsupported settings.

    Nested population/lobby/sabotage records must be constructed with their
    native types in the pinned upstream checkout, never mixed into this engine.
    JSON's allowed_sabotages array must be restored to a native tuple; the locked
    upstream registration intentionally rejects a list even with the same items.
    PPO/reward/opponent/commitment settings are preserved from the reviewed A2
    defaults. Acceptance here is not model or gameplay compatibility.
    """
    errors = []
    if settings.ruleset_id != "patrick-jane-a2-v1":
        errors.append("select the explicit patrick-jane-a2-v1 ruleset")
    if settings.crewmates > 12:
        errors.append("native Population supports at most 12 crew: 15 players currently need 3 impostors")
    if settings.crewmates <= settings.lobby.impostors:
        errors.append("native Population requires more crew than impostors")
    required = frozenset(("lights", "doors", "vents"))
    if settings.mechanics.enabled() != required:
        errors.append("registered A2 requires lights, doors and vents only; other extensions are deferred")
    if any(role.enabled for role in settings.optional_roles):
        errors.append("optional roles are not implemented in A2")
    if settings.tasks.pool_id != "patrick-jane-skeld-v1":
        errors.append("use the reviewed Patrick Jane task pool without recategorizing old tasks")
    if (settings.tasks.common, settings.tasks.long, settings.tasks.short) != (1, 1, 2):
        errors.append("native lobby fixes task counts at 1 common + 1 long + 2 short")
    if settings.tasks.task_bar_updates != "always" or settings.meetings.anonymous_votes:
        errors.append("native lobby requires always-visible task bar and non-anonymous votes")
    if settings.meetings.emergency_meetings != 1:
        errors.append("native lobby supports one emergency per living player")
    if settings.meetings.discussion_seconds <= 0 or settings.meetings.voting_seconds <= 0:
        errors.append("native lobby requires positive discussion and voting durations")
    if settings.lobby.kill_distance != "medium":
        errors.append("only the reviewed native distance of 2 character widths is mapped")
    if settings.simulation.cap_policy != "truncate" or settings.simulation.stall_cap_seconds <= 0:
        errors.append("A2 uses separate positive technical/stall cuts, never time-based wins")
    if errors:
        raise ValueError("; ".join(errors))

    native = json.loads((DATA / "patrick_jane_a2_default.json").read_text(encoding="utf-8"))
    native["population"] = {"crew": settings.crewmates, "impostors": settings.lobby.impostors}
    native["lobby"].update(
        player_speed=settings.lobby.player_speed,
        confirm_ejects=settings.lobby.confirm_ejects,
        crewmate_vision=settings.lobby.crewmate_vision,
        impostor_vision=settings.lobby.impostor_vision,
        kill_cooldown=settings.lobby.kill_cooldown_seconds,
        emergency_meetings=settings.meetings.emergency_meetings,
        emergency_cooldown=float(settings.meetings.emergency_cooldown_seconds),
        discussion_seconds=float(settings.meetings.discussion_seconds),
        voting_seconds=float(settings.meetings.voting_seconds),
        visual_tasks=settings.tasks.visual_tasks,
        dt=settings.simulation.physics_step_seconds,
        decision_interval=settings.simulation.observation_interval_seconds,
    )
    # lobby.timeout=900 is the preserved legacy field. normal_match_timeout=False
    # disables that gameplay ending; these are the actual A2 technical cuts.
    native["technical_max_seconds"] = settings.simulation.match_cap_seconds
    native["technical_stall_seconds"] = settings.simulation.stall_cap_seconds
    return native
