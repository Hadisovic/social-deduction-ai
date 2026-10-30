"""Configuration inspection only; never launches a match or PPO."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from jane import MAX_PLAYERS
from jane.shared.compatibility import check_engine, check_pair
from jane.shared.settings import bundle_receipt, load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--engine", choices=("historical-v2.2", "patrick-jane"))
    parser.add_argument("--check-pair", action="store_true",
                        help="Check the registered role declarations; not a model load")
    parser.add_argument("--native-config", action="store_true",
                        help="Print native Patrick Jane A2 kwargs; never launch them")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.profile)
        result = {
            "settings_valid": True, "profile": settings.profile_id,
            "capacity": MAX_PLAYERS, "players": settings.lobby.players,
            "crewmates": settings.crewmates, "impostors": settings.lobby.impostors,
            "settings_sha256": settings.fingerprint(),
            "shared_bundle": bundle_receipt(settings),
            "enabled_extensions": sorted(settings.mechanics.enabled()),
            "mode": "configuration inspection; no game or training launched",
        }
        blocked = False
        if args.engine:
            report = check_engine(settings, args.engine)
            result["engine_configuration"] = asdict(report)
            blocked = not report.compatible
        if args.check_pair:
            root = Path(__file__).resolve().parent
            crew = json.loads((root / "crewmate" / "manifest.json").read_text(encoding="utf-8"))
            imp = json.loads((root / "impostor" / "manifest.json").read_text(encoding="utf-8"))
            blockers = check_pair(settings, crew, imp)
            result["role_declaration_blockers"] = list(blockers)
            blocked = blocked or bool(blockers)
        if args.native_config:
            from jane.shared.patrick_jane import a2_configuration
            result["patrick_jane_a2_kwargs"] = a2_configuration(settings)
        print(json.dumps(result, indent=2))
        return 2 if blocked else 0
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"settings_valid": False, "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
