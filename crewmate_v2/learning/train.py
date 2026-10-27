"""Small V2 PPO pilot infrastructure. Only bounded smoke is currently authorized."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from crewmate_v2.config import LobbySettings,MatchMixture
from phase5_options_training import ppo_update
from phase3_runner import canonical_bytes
from .config import RewardConfig,match_seed,pilot_plan
from .env import LearningEnv
from .policy import V2Policy,tensors,check_packet
from .provenance import runtime_lock,digest,plan_hash,validate_approval


def write(path,record):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')


def state_hash(policy):
    h=hashlib.sha256()
    for key,value in sorted(policy.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_one(output,arm,initialization,steps,deadline,*,smoke=False,expected_lock=None,approval=None):
    if arm not in ('original','shaped') or steps!=(32 if smoke else 1024):raise ValueError('Unregistered arm/decision budget')
    if not smoke and expected_lock is None:raise PermissionError('Pilot entry requires an approved runtime lock')
    if not smoke:validate_approval(approval or {},expected_lock)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lock=runtime_lock(LobbySettings());expected_lock=expected_lock or lock
    if lock!=expected_lock:raise ValueError('Source/config drift')
    began=perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    env=LearningEnv(rewards=None if arm=='original' else RewardConfig(),opponents='varied',coalesce=True,
        mixture=MatchMixture(1.),deadline=deadline)
    torch.manual_seed(initialization);rng=np.random.default_rng(initialization)
    policy=V2Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
    frozen={k:v.clone() for k,v in env.belief_model.network.state_dict().items()}
    initial=state_hash(policy);episode=completed=0;simulated=0.;forced=0
    def reset():
        env.rotation=episode%3
        return env.reset(seed=match_seed('smoke' if smoke else 'train',episode//3))[0]
    packet=reset();metadata=dict(runtime=lock,arm=arm,initialization_seed=initialization,plan_sha256=plan_hash(),
        smoke_only=smoke,reward=env.match.ledger.config.to_dict(),execution=dict(coalesce=True,deduction_frequency=1.),
        initialization_sha256=initial)
    write(output/'config.json',metadata)
    updates=[];episodes=[];actions=Counter();trace=hashlib.sha256();used=[]
    while completed<steps:
        if perf_counter()>deadline:raise TimeoutError('Shared compute ceiling reached')
        if runtime_lock(LobbySettings())!=expected_lock:raise ValueError('Source/config drift')
        rollout={k:[] for k in ('packets','actions','logp','values','next_values','rewards','discounts','dones','durations')}
        for _ in range(min(32 if smoke else 128,steps-completed)):
            check_packet(packet)
            action,logp,value=policy.act(packet)
            if not packet['mask'][action] or int(packet['mask'].sum())<2:raise ValueError('Non-actionable or masked learner selection')
            actions[env.options[action].kind]+=1
            trace.update(canonical_bytes((packet,action)))
            new,reward,done,_,info=env.step(action)
            if env.match.game.metrics['illegal_actions'] or env.match.game.metrics['navigation_failures'] or info['masked_selection']:
                raise ValueError('Illegal/navigation/masked transition')
            if not np.isfinite([reward,info['discount'],info['elapsed_seconds']]).all():raise FloatingPointError('Nonfinite transition')
            with torch.no_grad():next_value=0. if done else float(policy(tensors(new))[1].item())
            for key,item in zip(rollout,(packet,action,logp,value,next_value,reward,info['discount'],done,info['elapsed_seconds'])):
                rollout[key].append(item)
            completed+=1;simulated+=info['elapsed_seconds'];forced+=info['coalesced_forced_windows']
            if done:
                episodes.append(dict(seed=match_seed('smoke' if smoke else 'train',episode//3),rotation=episode%3,**info['episode']))
                used.append(dict(seed=match_seed('smoke' if smoke else 'train',episode//3),rotation=episode%3))
                episode+=1;packet=reset()
            else:packet=new
        diagnostics=ppo_update(policy,optimizer,rollout,.02,rng)
        if perf_counter()>deadline:raise TimeoutError('Compute ceiling reached during optimizer update')
        for v in diagnostics.values():
            if v is not None and not np.isfinite(v):raise FloatingPointError('Nonfinite optimizer diagnostic')
        if diagnostics['approximate_kl']>.10 or diagnostics['clip_fraction']>.80 or diagnostics['gradient_norm']>100:
            raise RuntimeError('Registered optimizer stability stop')
        if any(not torch.isfinite(p).all() for p in policy.parameters()):raise FloatingPointError('Nonfinite policy weights')
        if not all(torch.equal(v,frozen[k]) for k,v in env.belief_model.network.state_dict().items()):raise ValueError('Frozen belief changed')
        updates.append(dict(decisions=completed,simulated_seconds=simulated,seconds=perf_counter()-began,**diagnostics))
        print(f'{arm}-{initialization}: '+json.dumps(updates[-1]),flush=True)
        if smoke or completed%256==0:
            training_state=dict(torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
                optimizer=optimizer.state_dict(),decisions=completed,episode_index=episode)
            policy.save(output/f'checkpoint-{completed}.pt',{**metadata,'training_state':training_state})
    final=state_hash(policy)
    if final==initial:raise ValueError('Optimizer smoke/pilot did not update new policy weights')
    loaded,loaded_meta=V2Policy.load(output/f'checkpoint-{steps}.pt')
    if state_hash(loaded)!=final:raise ValueError('Checkpoint serialization mismatch')
    result=dict(status='optimizer_smoke_only' if smoke else 'trained_pending_review',arm=arm,
        initialization_seed=initialization,actionable_decisions=completed,simulated_seconds=simulated,
        coalesced_forced_windows=forced,seconds=perf_counter()-began,episodes_completed=len(episodes),
        episodes=episodes,partial_episode_counts=dict(env.match.ledger.counts),
        partial_episode_components=dict(env.match.ledger.components),actions=dict(actions),
        initial_state_sha256=initial,final_state_sha256=final,actor_action_trace_sha256=trace.hexdigest(),
        frozen_belief_unchanged=True,checkpoint_roundtrip=True,
        checkpoint_sha256=hashlib.sha256((output/f'checkpoint-{steps}.pt').read_bytes()).hexdigest(),
        used_completed_contexts=used,current_context=dict(seed=match_seed('smoke' if smoke else 'train',episode//3),rotation=episode%3),updates=updates)
    write(output/'result.json',result)
    return result


def smoke(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=perf_counter();deadline=started+600.
    lock=runtime_lock(LobbySettings());rows=[]
    for arm in ('original','shaped'):
        try:rows.append(run_one(output/arm,arm,17,32,deadline,smoke=True,expected_lock=lock))
        except BaseException as error:
            write(output/'failure.json',dict(status='smoke_failed_preserve_outputs',reason=repr(error)));raise
    assert rows[0]['initial_state_sha256']==rows[1]['initial_state_sha256']
    assert rows[0]['actor_action_trace_sha256']==rows[1]['actor_action_trace_sha256'],'Rewards changed pre-update actor inputs/actions'
    record=dict(status='bounded_optimizer_smoke_only_no_pilot',seconds=perf_counter()-started,runs=rows,
        plan_sha256=plan_hash(),runtime_sha256=digest(lock),pilot_started=False,final_test_opened=False)
    write(output/'smoke.json',record);return record


def pilot(output,approval,smoke_record):
    lock=runtime_lock(LobbySettings());validate_approval(approval,lock)
    if smoke_record.get('runtime_sha256')!=digest(lock):raise ValueError('Smoke runtime mismatch')
    plan=pilot_plan()
    # Conservative measured forecast: training scaling + worst-duration validation.
    rates=[r['seconds']/r['simulated_seconds'] for r in smoke_record['runs']]
    projected=2.*(sum(r['seconds'] for r in smoke_record['runs'])*2*1024/32+max(rates)*900*128)
    if projected>plan['total_wall_ceiling_seconds']:raise TimeoutError('Projected pilot exceeds approved three-hour ceiling; request revised scope/budget')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=perf_counter();deadline=started+10800
    write(output/'registration.json',dict(plan=plan,plan_sha256=plan_hash(),runtime=lock,projected_seconds=projected))
    from .evaluate import evaluate_game,validation_contexts
    runs=[]
    try:
        for initialization in (17,29):
            for arm in ('original','shaped'):
                folder=output/f'{arm}-{initialization}'
                result=run_one(folder,arm,initialization,1024,deadline,expected_lock=lock,approval=approval)
                policy,_=V2Policy.load(folder/'checkpoint-1024.pt')
                validation=[evaluate_game(policy,deadline=deadline,**context) for context in validation_contexts()]
                write(folder/'validation.json',dict(status='development_only',games=validation))
                runs.append(dict(arm=arm,initialization_seed=initialization,result=result,games=validation))
    except BaseException as error:
        write(output/'failure.json',dict(status='stopped_preserve_checkpoints',reason=repr(error),completed_runs=len(runs)))
        raise
    record=dict(status='pilot_complete_acceptance_pending',seconds=perf_counter()-started,runs=runs,final_test_opened=False)
    from .comparison import compare
    write(output/'comparison.json',compare(runs))
    write(output/'pilot.json',record);return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('smoke','pilot'))
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--approval',type=Path);parser.add_argument('--smoke-record',type=Path)
    args=parser.parse_args()
    if args.mode=='smoke':smoke(args.output)
    else:
        if args.approval is None or args.smoke_record is None:raise PermissionError('Owner approval and matching smoke record required before pilot')
        pilot(args.output,json.loads(args.approval.read_text()),json.loads(args.smoke_record.read_text()))


if __name__=='__main__':main()
