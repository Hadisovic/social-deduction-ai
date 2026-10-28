import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json
from time import perf_counter

def main():
    import torch
    from patrick_jane.network import PatrickJane
    from patrick_jane.collector import Pool,collect
    from patrick_jane.config import Experiment
    from patrick_jane.progress import atomic_json
    torch.set_num_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    policy=PatrickJane().cuda();root=Path('local-runs/patrick-jane/repro-probe-v1');root.mkdir(exist_ok=False)
    with Pool(1) as pool:r=collect(policy,pool,Experiment(),[2920001],device='cuda',wall_limit=180,replay_dir=root/'replays',record_every=1)
    r.pop('trajectories');atomic_json(root/'metrics.json',r)
    from spectator.replay_format import parse_replay
    paired=json.loads(Path('local-runs/patrick-jane/fullgame-qualified-v4/paired.json').read_text())
    print(json.dumps(dict(fresh=r['matches'][0]['trace'],previous_reference=paired['reference'][1]['trace'],parallel=paired['parallel'][1]['trace'])))
    a=parse_replay(next((root/'replays').glob('*.jsonl')));b=parse_replay(Path('local-runs/patrick-jane/fullgame-qualified-v4/replays/env-01-seed-2920001-update-0.jsonl'))
    for kind in ('patrick_jane_decision','navigate','structured_claim'):
        aa=[x for x in a['events'] if x['kind']==kind];bb=[x for x in b['events'] if x['kind']==kind]
        rows=[]
        for x,y in zip(aa,bb):
            if x!=y:rows.append(dict(left=x,right=y))
            if len(rows)>=2:break
        print(json.dumps(dict(kind=kind,counts=[len(aa),len(bb)],differences=rows)))
if __name__=='__main__':main()
