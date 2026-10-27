"""Engineering only: short matched collection, one PPO update and checkpoint test."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from crewmate_v2.config import LobbySettings,MatchMixture
from crewmate_v2.collection.ppo import update
from crewmate_v2.learning.train import state_hash
from .collector import collect_matches
from .policy import FourCrewPolicy,source_lock
from .config import Rewards


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,choices=(1,2,4),default=1);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    started=perf_counter();torch.set_num_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    policy=FourCrewPolicy();initial=state_hash(policy)
    rollout=collect_matches(policy,[2760000,2760001],workers=a.workers,lobby=LobbySettings(timeout=3.),
                            mixture=MatchMixture(1.),rewards=Rewards(),deadline=started+120)
    optimizer=torch.optim.Adam(policy.parameters(),lr=.0003);rng=np.random.default_rng(17)
    diagnostics=update(policy,optimizer,rollout,rng)
    metadata=dict(engineering_only=True,initialization=17,seeds=[2760000,2760001],
                  lobby=LobbySettings(timeout=3.).to_dict(),rewards=Rewards().to_dict(),deduction_frequency=1.,
                  completed_matches=rollout['episodes'],individual_actor_decisions=len(rollout['actions']),
                  policy_version=1,optimizer=optimizer.state_dict(),update_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
    path=a.output/'engineering-checkpoint.pt';policy.save(path,metadata)
    loaded,_=FourCrewPolicy.load(path);assert state_hash(loaded)==state_hash(policy)
    report=dict(status='engineering_only_no_full_pilot',seconds=perf_counter()-started,workers=a.workers,
                decisions=len(rollout['actions']),trajectory_keys=rollout['trajectory_keys'],initial_sha256=initial,
                final_sha256=state_hash(policy),diagnostics=diagnostics,episodes=rollout['episodes'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__=='__main__':main()
