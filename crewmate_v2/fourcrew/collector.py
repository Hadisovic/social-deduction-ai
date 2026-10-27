"""Freeze one centrally updated policy per complete-match collection batch."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
from crewmate_v2.learning.train import state_hash
from .policy import FourCrewPolicy
from .match import FourCrewMatch,FIELDS


def worker(payload):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy=FourCrewPolicy();policy.load_state_dict(payload['weights']);policy.eval()
    match=FourCrewMatch(payload['seed'],**payload['options'])
    return match.collect(policy,initialization=payload['initialization'],policy_version=payload['version'])


def collect_matches(policy,seeds,*,workers=1,initialization=17,version=0,**options):
    if workers not in (1,2,4):raise ValueError('Use one, two or four headless match workers')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Unique development matches required')
    before=state_hash(policy)
    payloads=[dict(weights={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()},seed=s,
                   options=options,initialization=initialization,version=version) for s in seeds]
    if workers==1:results=[worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as executor:
            results=list(executor.map(worker,payloads))
    if state_hash(policy)!=before:raise RuntimeError('Policy changed during collection')
    flat={k:[] for k in FIELDS};flat.update(segments=[],trajectory_keys=[],episodes=[])
    for rollout,summary in results:
        offset=len(flat['actions'])
        for key in FIELDS:flat[key].extend(rollout[key])
        flat['segments'].extend((lo+offset,hi+offset) for lo,hi in rollout['segments'])
        flat['trajectory_keys'].extend(rollout['trajectory_keys']);flat['episodes'].append(summary)
    return flat
