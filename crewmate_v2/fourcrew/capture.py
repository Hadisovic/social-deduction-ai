"""Bounded real four-actor collection/replay proof; zero learning updates."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from crewmate_v2.config import LobbySettings,MatchMixture
from crewmate_v2.live.protocol import spectator_import
from crewmate_v2.learning.train import state_hash
from .match import FourCrewMatch
from .policy import FourCrewPolicy,source_lock
from .config import Rewards


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--spectator-source',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);module=spectator_import(a.spectator_source)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(17)
    policy=FourCrewPolicy();before=state_hash(policy);results=[];times=[]
    for recording in (False,True):
        start=perf_counter();m=FourCrewMatch(2760003,lobby=LobbySettings(timeout=60.),mixture=MatchMixture(1.),deadline=start+180)
        rec=None
        if recording:rec=module.MatchRecorder(m.match,a.output/'fourcrew-engineering.jsonl',
            controller_types={pid:'ppo' if pid in m.crew else 'scripted' for pid in m.game.players},
            metadata=dict(training_condition='Four private PPO actors — zero-update engineering',reward_profile=Rewards().to_dict(),
                          private_debug='Selected actor only; shared frozen Phase 4 predictor'),source_version='fourcrew-engineering')
        try:r,s=m.collect(policy,recorder=rec)
        finally:
            if rec and not rec.recorder._closed:
                rec.recorder._stream.flush();rec.recorder._stream.close();rec.recorder._closed=True
        times.append(perf_counter()-start);results.append((r,s))
    x,y=results
    for k in ('actions','logp','values','next_values','rewards','discounts','dones','durations','segments'):
        assert x[0][k]==y[0][k],k
    for pa,pb in zip(x[0]['packets'],y[0]['packets']):
        import numpy as np
        assert all(np.array_equal(pa[k],pb[k]) for k in pa)
    assert x[1]==y[1] and state_hash(policy)==before
    report=dict(status='engineering_only_zero_updates',seconds_off=times[0],seconds_on=times[1],ratio=times[1]/times[0],
                exact_recording_noninterference=True,runtime=source_lock(),summary=y[1])
    (a.output/'proof.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runtime','summary')}))


if __name__=='__main__':main()
