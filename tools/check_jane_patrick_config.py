"""Verify pinned source/artifacts and construct A2 settings only. Never run a model/game."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jane.shared.patrick_jane import a2_configuration, reference_review
from jane.shared.settings import load_settings


PROBE = r'''
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from patrick_jane.config import Population, ClassicLobby, SabotageSettings
from patrick_jane.a2_config import A2Experiment
import patrick_jane.config
assert Path(patrick_jane.config.__file__).resolve().is_relative_to(root)
value = json.loads(sys.argv[2])
value['population'] = Population(**value['population'])
value['lobby'] = ClassicLobby(**value['lobby'])
value['sabotage'] = SabotageSettings(**value['sabotage'])
value['allowed_sabotages'] = tuple(value['allowed_sabotages'])
experiment = A2Experiment(**value)
assert not any(m.startswith('patrick_jane.network') or m.startswith('patrick_jane.training') for m in sys.modules)
print(json.dumps(experiment.to_dict()))
'''


def check(profile, source_root):
    settings = load_settings(profile)
    native = a2_configuration(settings)
    receipt = reference_review()
    source_root = Path(source_root).resolve()
    sources = {**receipt["source_hashes"], **receipt["config_import_source_hashes"]}
    for name, expected in sources.items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError("Invalid source reference path")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Pinned source mismatch: " + name)
    for name, expected in receipt["published_files_verified"].items():
        path = source_root / "artifacts/patrick-jane/task-pursuit-v1-update102" / name
        data = path.read_bytes()
        if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Published artifact mismatch: " + name)
    environment = dict(os.environ, PYTHONPATH="", PYGAME_HIDE_SUPPORT_PROMPT="1")
    result = subprocess.run([sys.executable, "-I", "-c", PROBE, str(source_root), json.dumps(native)],
                            cwd=source_root, env=environment, capture_output=True, text=True,
                            timeout=60, check=True)
    if json.loads(result.stdout) != native:
        raise ValueError("Native constructor changed translated settings")
    return dict(configuration_verified=True, settings_sha256=settings.fingerprint(),
                implementation_commit=receipt["implementation_commit"], source_files_checked=len(sources),
                artifact_files_checked=len(receipt["published_files_verified"]),
                players=settings.lobby.players, crewmates=settings.crewmates,
                impostors=settings.lobby.impostors, model_loaded=False, matches_started=0,
                training_updates=0, model_runtime_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Separate unmodified checkout/snapshot of the pinned Patrick Jane source")
    args = parser.parse_args(argv)
    try:
        result = check(args.profile, args.source_root)
    except subprocess.CalledProcessError as error:
        print(json.dumps(dict(configuration_verified=False,
                              error="Upstream constructor rejected settings: " + error.stderr.strip())))
        return 2
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(configuration_verified=False, error=str(error))))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
