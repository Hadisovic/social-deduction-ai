"""Configuration/compatibility regressions only: no match, weights or PPO."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from jane import CONTRACT_SCHEMA, MAX_PLAYERS
from jane.shared.compatibility import CompatibilityError, check_engine, check_pair, legacy_lobby
from jane.shared.settings import GameSettings, OPTIONAL_ROLES, bundle_receipt, load_settings


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "jane" / "configs"
SHARED = ROOT / "jane" / "shared"


def profile(name="jane_12p_base"):
    return load_settings(CONFIGS / (name + ".json"))


@pytest.mark.parametrize("name,players,impostors", [
    ("historical_5p", 5, 1), ("jane_12p_base", 12, 2),
    ("jane_15p_base", 15, 2), ("future_15p_full_classic", 15, 3),
])
def test_profiles_round_trip_and_population(name, players, impostors):
    settings = profile(name)
    assert settings.lobby.players == players
    assert settings.lobby.impostors == impostors
    assert settings.crewmates == players - impostors
    assert GameSettings.from_dict(settings.to_dict()) == settings
    assert MAX_PLAYERS == 15


@pytest.mark.parametrize("players,impostors", [
    (p, i) for i, minimum in ((1, 4), (2, 7), (3, 9)) for p in range(minimum, 16)
])
def test_legal_custom_lobbies_are_representable(players, impostors):
    settings = profile()
    custom = replace(settings, lobby=replace(settings.lobby, players=players, impostors=impostors))
    assert custom.crewmates + impostors == players


@pytest.mark.parametrize("players,impostors", [(3, 1), (16, 1), (6, 2), (8, 3), (12, 0), (12, 4), (True, 1)])
def test_invalid_populations_rejected(players, impostors):
    with pytest.raises(ValueError):
        replace(profile().lobby, players=players, impostors=impostors)


@pytest.mark.parametrize("section,key,value", [
    ("lobby", "player_speed", float("nan")), ("lobby", "crewmate_vision", float("inf")),
    ("lobby", "player_speed", .6), ("lobby", "kill_cooldown_seconds", 11),
    ("lobby", "kill_cooldown_seconds", True), ("lobby", "kill_distance", "far"),
    ("lobby", "confirm_ejects", 1), ("meetings", "anonymous_votes", "false"),
    ("meetings", "emergency_cooldown_seconds", 1), ("meetings", "voting_seconds", 121),
    ("meetings", "discussion_seconds", -15), ("tasks", "common", 3),
    ("tasks", "short", 6), ("tasks", "long", True),
    ("tasks", "task_bar_updates", "sometimes"), ("mechanics", "vents", 1),
    ("simulation", "physics_step_seconds", 0),
    ("simulation", "observation_interval_seconds", .55),
    ("simulation", "cap_policy", "pretend-loss"),
])
def test_invalid_settings_fail_closed(section, key, value):
    data = profile().to_dict()
    data[section][key] = value
    with pytest.raises(ValueError):
        GameSettings.from_dict(data)


@pytest.mark.parametrize("section", [None, "lobby", "meetings", "tasks", "mechanics", "simulation"])
def test_unknown_keys_do_not_get_silently_ignored(section):
    data = profile().to_dict()
    (data if section is None else data[section])["typo_or_unimplemented_option"] = True
    with pytest.raises(ValueError, match="unknown"):
        GameSettings.from_dict(data)


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_settings(path)


def test_loaded_settings_are_immutable_and_exports_are_copies():
    settings = profile()
    with pytest.raises(FrozenInstanceError):
        settings.lobby.players = 15
    exported = settings.public_settings()
    exported["lobby"]["players"] = 4
    assert settings.lobby.players == 12
    assert "simulation" not in exported
    assert "profile_id" not in exported


def test_settings_fingerprint_tracks_changes_not_json_key_order():
    settings = profile()
    reverse = dict(reversed(list(settings.to_dict().items())))
    assert GameSettings.from_dict(reverse).fingerprint() == settings.fingerprint()
    changed = replace(settings, lobby=replace(settings.lobby, kill_cooldown_seconds=42.5))
    assert changed.fingerprint() != settings.fingerprint()


def test_bundle_binds_definition_contents_even_with_same_version(tmp_path):
    for name in bundle_receipt(profile())["shared_files"]:
        shutil.copyfile(SHARED / name, tmp_path / name)
    before = bundle_receipt(profile(), tmp_path)
    with (tmp_path / "actor_contract.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    after = bundle_receipt(profile(), tmp_path)
    assert before["settings_sha256"] == after["settings_sha256"]
    assert before["bundle_sha256"] != after["bundle_sha256"]


def test_default_extensions_and_optional_roles_stay_disabled():
    for name in ("historical_5p", "jane_12p_base", "jane_15p_base"):
        settings = profile(name)
        assert not settings.mechanics.enabled()
        assert {r.role for r in settings.optional_roles} == set(OPTIONAL_ROLES)
        assert all(r.max_count == r.chance_percent == 0 for r in settings.optional_roles)
    assert len(profile("future_15p_full_classic").mechanics.enabled()) == 10


def test_dependencies_and_duplicate_roles_rejected():
    with pytest.raises(ValueError, match="sabotage"):
        replace(profile().mechanics, impostor_ghosts=True)
    with pytest.raises(ValueError, match="occupancy"):
        replace(profile().mechanics, clean_vent=True)
    data = profile().to_dict()
    data["optional_roles"].append(data["optional_roles"][0])
    with pytest.raises(ValueError, match="exactly once"):
        GameSettings.from_dict(data)


def test_historical_translation_matches_existing_defaults_exactly():
    from crewmate_v2.config import LobbySettings
    result = legacy_lobby(profile("historical_5p"))
    assert result.to_dict() == LobbySettings().to_dict()
    assert check_engine(profile("historical_5p"), "historical-v2.2").compatible


def test_new_population_and_features_cannot_fall_back_to_old_engine():
    for name in ("jane_12p_base", "jane_15p_base", "future_15p_full_classic"):
        report = check_engine(profile(name), "historical-v2.2")
        assert not report.compatible
        assert report.blockers
    original = profile("historical_5p")
    with pytest.raises(CompatibilityError, match="extension"):
        legacy_lobby(replace(original, mechanics=replace(original.mechanics, lights=True)))


@pytest.mark.parametrize("section,changes", [
    ("meetings", {"anonymous_votes": True}),
    ("meetings", {"emergency_cooldown_seconds": 5}),
    ("tasks", {"short": 3}), ("tasks", {"task_bar_updates": "meetings"}),
    ("lobby", {"kill_distance": "short"}),
])
def test_historical_adapter_rejects_unsupported_host_options(section, changes):
    settings = profile("historical_5p")
    changed = replace(settings, **{section: replace(getattr(settings, section), **changes)})
    with pytest.raises(CompatibilityError):
        legacy_lobby(changed)


def test_absent_jane_source_is_never_marked_compatible():
    report = check_engine(profile(), "patrick-jane")
    assert not report.compatible
    assert "not supplied" in report.blockers[0]
    with pytest.raises(ValueError, match="Unknown engine"):
        check_engine(profile(), "pretend-engine")


def declaration(role, settings):
    # Synthetic declarations only; never a real checkpoint or trusted registration.
    return dict(role=role, integration_status="verified", source_commit="a" * 40,
                checkpoint_sha256="b" * 64, actor_contract=CONTRACT_SCHEMA,
                observation_schema="synthetic-semantic-obs-v1", action_schema="synthetic-actions-v1",
                tensor_schema=f"synthetic-{role}-tensors-v1", verified_settings_sha256=settings.fingerprint(),
                verified_bundle_sha256=bundle_receipt(settings)["bundle_sha256"])


def test_declaration_check_requires_both_role_versions_and_exact_settings():
    settings = profile()
    crew, imp = declaration("crewmate", settings), declaration("impostor", settings)
    assert not check_pair(settings, crew, imp)
    assert check_pair(settings, crew, dict(imp, action_schema="different"))
    assert check_pair(settings, crew, dict(imp, checkpoint_sha256="made-up"))
    assert check_pair(settings, crew, dict(imp, verified_bundle_sha256="0" * 64))
    assert check_pair(replace(settings, profile_id="new-study"), crew, imp)


def test_bundled_manifests_remain_explicitly_unverified():
    manifests = [json.loads((ROOT / "jane" / role / "manifest.json").read_text(encoding="utf-8"))
                 for role in ("crewmate", "impostor")]
    assert check_pair(profile(), *manifests)
    assert all(m["source_commit"] is None and m["checkpoint_sha256"] is None for m in manifests)


def test_inventory_preserves_all_asset_task_ids_and_source_hash():
    catalog = json.loads((SHARED / "catalog.json").read_text(encoding="utf-8"))
    source = ROOT / catalog["map_source"]["path"]
    original = json.loads(source.read_text(encoding="utf-8"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == catalog["map_source"]["sha256"]
    expected = {entry["taskType"]: entry["length"] for entry in original["tasks"].values()}
    assert {t["id"]: t["historical_category"] for t in catalog["tasks"]} == expected
    assert len(expected) == 19
    assert len({t["canonical_family"] for t in catalog["tasks"]}) == 18
    assert sorted(v for component in catalog["vent_components_from_asset"] for v in component) == list(range(14))
    assert sum(catalog["door_grouping"].values()) == 13


def test_configuration_import_does_not_import_training_or_models():
    code = ("import sys; import jane.shared.settings; import jane.shared.compatibility; "
            "assert 'torch' not in sys.modules; "
            "assert not any('training' in m or 'trainer' in m for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("arguments,expected", [
    (["jane/configs/jane_15p_base.json"], 0),
    (["jane/configs/historical_5p.json", "--engine", "historical-v2.2"], 0),
    (["jane/configs/jane_12p_base.json", "--engine", "patrick-jane", "--check-pair"], 2),
    (["jane/configs/future_15p_full_classic.json", "--engine", "historical-v2.2"], 2),
])
def test_cli_reports_configuration_and_expected_blockers(arguments, expected):
    result = subprocess.run([sys.executable, "-m", "jane", *arguments], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == expected, result.stderr + result.stdout
    data = json.loads(result.stdout)
    assert data["settings_valid"] is True
    assert "no game or training" in data["mode"]


def test_historical_cli_is_json_in_a_clean_shell():
    environment = dict(os.environ)
    environment.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)
    result = subprocess.run([sys.executable, "-m", "jane", "jane/configs/historical_5p.json",
                             "--engine", "historical-v2.2"], cwd=ROOT, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["engine_configuration"]["compatible"]
