"""Bounded engineering proof of the actual PPO/replay connection, never the pilot."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.train import write,state_hash
from crewmate_v2.collection.ppo import update,save
from .collector import LiveCollector
from .protocol import lock,spectator_import


def case(output,source,recording,deadline):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    began=perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=dict(spectator_source=str(Path(source).resolve()),replay_dir=str(output/'replays'),
                streams=list(range(8)),session='engineering-live-connection',workers=4,source_version='engineering-smoke') if recording else None
    with LiveCollector(workers=4,recording=config,deadline=deadline) as collector:
        torch.manual_seed(17);policy=V2Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        rng=np.random.default_rng(17);initial=state_hash(policy)
        rollout=collector.collect(policy,64)
        diagnostics=update(policy,optimizer,rollout,rng);collector.policy_version=1
        continuation=collector.collect(policy,8)  # Actual post-update policy decisions.
        save(output/'checkpoint-72.pt',policy,optimizer,rng,collector,72)
        loaded,_=V2Policy.load(output/'checkpoint-72.pt');assert state_hash(loaded)==state_hash(policy)
        snapshot=collector.snapshot()
        traces=[t.hexdigest() for t in collector.traces]
        telemetry=collector._command('recording_summary') if recording else {}
    result=dict(status='engineering_only_not_learning_evidence',recording=recording,seconds=perf_counter()-began,
                decisions=72,initial_sha256=initial,final_sha256=state_hash(policy),actor_action_hashes=traces,
                final_packets={s:r['packet_sha256'] for s,r in snapshot['environments'].items()},
                rewards=rollout['rewards']+continuation['rewards'],dones=rollout['dones']+continuation['dones'],
                simulated_seconds=sum(rollout['durations'])+sum(continuation['durations']),
                episodes=rollout['episodes']+continuation['episodes'],diagnostics=diagnostics,telemetry=telemetry)
    write(output/'result.json',result);return result


def smoke(output,source,ceiling=600):
    if not 0<ceiling<=600:raise ValueError('Bounded engineering ceiling is ten minutes')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);began=perf_counter();deadline=began+ceiling
    runtime=lock(source);write(output/'registration.json',dict(runtime=runtime,source='fixed c18ff5f collection kernels + live observer',ceiling=ceiling))
    off=case(output/'off',source,False,deadline)
    on=case(output/'on',source,True,deadline)
    keys=('initial_sha256','final_sha256','actor_action_hashes','final_packets','rewards','dones','simulated_seconds','episodes')
    for key in keys:
        if off[key]!=on[key]:raise ValueError('Recording noninterference failed: '+key)
    spectator_import(source)
    from spectator.replay_format import parse_replay
    files=sorted((output/'on'/'replays').rglob('*.jsonl'))
    assert files,'No completed actual PPO episode was recorded'
    episodes={(r['slot'],r['seed']):r for r in on['episodes']}
    descriptions=[]
    for path in files:
        replay=parse_replay(path);header=replay['header'];cfg=header['config']
        assert replay['complete'] and len(replay['ticks'][0]['players'])==5 and replay['neural']
        assert header['controller_types'][cfg['focal_player']]=='ppo'
        actual=episodes[cfg['stream'],header['scenario_seed']]
        assert replay['result']['reason']==actual['reason']
        assert abs(replay['result']['sim_time']-actual['time'])<1e-8
        for neural in replay['neural']:
            assert abs(sum(neural['action_probabilities'].values())-1)<1e-5
            assert neural['action_kind'] in neural['legal_actions']
        descriptions.append(dict(path=str(path),seed=header['scenario_seed'],stream=cfg['stream'],
                                 ticks=len(replay['ticks']),events=len(replay['events']),neural=len(replay['neural']),reason=replay['result']['reason']))
    if lock(source)!=runtime:raise ValueError('Smoke source drift')
    ratio=on['seconds']/off['seconds']
    record=dict(status='passed_engineering_only',runtime=runtime,noninterference=True,real_ppo_replay=True,
                checkpoint_roundtrip=True,seconds=perf_counter()-began,off_seconds=off['seconds'],on_seconds=on['seconds'],
                recording_wall_ratio=ratio,overhead_acceptable=ratio<=1.25,recorded_all_streams_for_conservative_overhead=True,
                pilot_records_streams=[0,1,2,3],completed_replays=descriptions,
                decisions_per_case=72,pilot_started=False,final_test_opened=False)
    write(output/'proof.json',record);print(json.dumps({k:record[k] for k in ('status','seconds','recording_wall_ratio','overhead_acceptable')}),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spectator-source',type=Path,required=True);p.add_argument('--wall-seconds',type=float,default=600)
    args=p.parse_args();smoke(args.output,args.spectator_source,args.wall_seconds)


if __name__=='__main__':main()
