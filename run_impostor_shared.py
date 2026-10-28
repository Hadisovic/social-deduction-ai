"""Watch the recorded shared-environment learned-impostor engineering match."""
from pathlib import Path
import json
import sys

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    replay=root/'local-runs/shared-integration/learned-b-2760002-v2/replay.jsonl'
    has_replay=any(a=='--replay' or a.startswith('--replay=') for a in sys.argv[1:])
    if not has_replay and not replay.exists():
        raise SystemExit('No complete learned replay yet. Run .venv\\Scripts\\python.exe -m impostor_shared.run --output local-runs/shared-integration/learned-b-2760002-v2 first.')
    from spectator.v2.application import main
    options=[] if has_replay else ['--replay',str(replay)]
    if not has_replay and not any(a=='--selected' or a.startswith('--selected=') for a in sys.argv[1:]):
        with replay.open(encoding='utf-8') as stream:header=json.loads(next(stream))
        owner=next((pid for pid,role in header['agent_roles'].items() if role=='impostor'),None)
        if owner is not None:options+=['--selected',owner]
    if not any(a=='--watch' or a.startswith('--watch=') for a in sys.argv[1:]):
        options+=['--watch',str(root/'local-runs/shared-integration')]
    raise SystemExit(main([*options,*sys.argv[1:]]))
