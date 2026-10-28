"""One bounded representative engineering benchmark; never uses registered study seeds."""
import argparse
import copy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from crewmate_v2.learning.train import state_hash
from crewmate_v2.learning.provenance import digest
from crewmate_v2.collection.ppo import update
from .policy import FourCrewPolicy
from .config import Rewards
from .registration import plan,runtime,EXPERIMENT
from .execution import jobs,run_jobs
from .experiment import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--spectator-source',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=perf_counter();deadline=start+600
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(17)
    zero=FourCrewPolicy().eval().requires_grad_(False);initial=state_hash(zero)
    locked=runtime(a.spectator_source)
    write(a.output/'registration.json',dict(status='engineering_benchmark_only',wall_limit_seconds=600,
        game_lobby_seconds=900,initial_sha256=initial,runtime=locked,study_seeds_opened=False))
    zero.save(a.output/'benchmark-zero.pt',dict(engineering_only=True,updates=0,initial_sha256=initial))
    payloads=[]
    for i,context in enumerate(('ordinary','body','meeting_crew','meeting_impostor')):
        ctx=dict(seed=2760020+i,seed_partition='engineering',case_index=i,rotation=i%3,context=context,deduction_frequency=float(i!=0))
        payloads+=jobs(zero,[ctx],rewards=Rewards(),label='benchmark-'+context,deadline=deadline,session='registration-readiness',
            greedy=i!=1,replay_dir=a.output/'replays',spectator_source=a.spectator_source)
    samples=[]
    def sample():
        samples.append(dict(elapsed_seconds=perf_counter()-start))
    try:
        rows=run_jobs(payloads,deadline,on_sample=sample,
            on_result=lambda i,r:write(a.output/f'match-{i}.json',r[1]))
        inference_wall=perf_counter()-start
        # Disposable clone only; the reference is never optimized.
        disposable=copy.deepcopy(zero).requires_grad_(True);optimizer=torch.optim.Adam(disposable.parameters(),lr=.0003)
        t=perf_counter();diagnostics=update(disposable,optimizer,rows[1][0],np.random.default_rng(17));update_seconds=perf_counter()-t
        assert state_hash(zero)==initial and runtime(a.spectator_source)==locked
        record=dict(status='completed_representative_engineering_benchmark',seconds=perf_counter()-start,
            simultaneous_jobs=4,batch_inference_wall_seconds=inference_wall,ppo_update_seconds=update_seconds,
            disposable_update_diagnostics=diagnostics,matches=[r[1] for r in rows],zero_unchanged=True,
            resource_measurement='Runtime only; CPU/RAM not instrumented in this benchmark',progress_samples=samples,
            study_seed_partitions_opened=False,full_training_started=False)
        write(a.output/'benchmark.json',record)
        print({k:v for k,v in record.items() if k not in ('matches','progress_samples','disposable_update_diagnostics')})
    except BaseException as error:
        write(a.output/'failure.json',dict(status='benchmark_incomplete',error=repr(error),seconds=perf_counter()-start,
            progress_samples=samples,study_seed_partitions_opened=False));raise


if __name__=='__main__':main()
