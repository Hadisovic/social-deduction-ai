"""Bounded spawn/CUDA/PPO engineering check; two 3-second technical segments."""
import os,json
from pathlib import Path


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch,numpy as np
    from dataclasses import replace
    from patrick_jane.a2_config import A2Experiment
    from patrick_jane.network import PatrickJane
    from patrick_jane.collector import Pool,collect
    from patrick_jane.ppo import update
    from patrick_jane.learning_run import audit_lights
    from patrick_jane.a2_report import metrics
    from patrick_jane.checkpoint import save,load
    from patrick_jane.progress import atomic_json
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(17)
    folder=Path('local-runs/a2-engineering-spawn-v1');folder.mkdir(exist_ok=False,parents=True)
    policy=PatrickJane().to('cuda');exp=replace(A2Experiment(opponent_regime='upgraded-a2-v1'),technical_max_seconds=3.)
    with Pool(2) as pool:result=collect(policy,pool,exp,[2920003,2920004],device='cuda',wall_limit=120.,replay_dir=folder/'replays',record_every=1)
    audit_lights(result);optimizer=torch.optim.Adam(policy.parameters(),lr=exp.learning_rate);rng=np.random.default_rng(17)
    learning=update(policy,optimizer,result.pop('trajectories'),exp,rng=rng,device='cuda')
    counters=dict(update=1,individual_decisions=result['individual_decisions'],complete_matches=0)
    sha=save(folder/'engineering-only.pt',policy,optimizer,exp,counters,rng=rng,metadata=dict(engineering_only=True,no_parent_learned_weights=True))
    restored=PatrickJane();load(folder/'engineering-only.pt',restored,restore_rng=False)
    assert all(torch.equal(v.cpu(),restored.state_dict()[k]) for k,v in policy.state_dict().items())
    summary=dict(kind='ENGINEERING ONLY; fresh initialization; no learned update102 weights changed',segments=2,normal_complete_games=0,
        metrics=metrics(result['matches']),ppo=learning,checkpoint_sha256=sha,strict_resume=True)
    atomic_json(folder/'verification.json',summary);print(json.dumps(summary,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
