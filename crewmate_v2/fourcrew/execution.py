"""Bounded headless match jobs. Only the approval-gated runner uses study partitions."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import torch
from crewmate_v2.config import LobbySettings,MatchMixture
from crewmate_v2.learning.provenance import digest
from crewmate_v2.live.protocol import spectator_import
from .policy import FourCrewPolicy,source_lock
from .scenarios import ScenarioMatch
from .evaluation import DecisionAudit


def execute(payload):
    started=perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy=FourCrewPolicy();policy.load_state_dict(payload['weights']);policy.eval()
    context=dict(payload['context']);frequency=context.pop('deduction_frequency')
    match=ScenarioMatch(**context,lobby=LobbySettings(),mixture=MatchMixture(frequency),
                        rewards=payload['rewards'],deadline=payload['deadline'])
    recorder=None
    if payload.get('replay'):
        module=spectator_import(payload['spectator_source'])
        recorder=module.MatchRecorder(match.match,payload['replay'],
            controller_types={pid:'ppo' if pid in match.crew else 'scripted' for pid in match.game.players},
            model_architecture=f'FourCrewPolicy {sum(p.numel() for p in policy.parameters())} parameters',
            training_session=payload['session'],source_version='runtime:'+digest(source_lock()),
            metadata=dict(training_condition=payload['label'],context=match.context,policy_version=payload['version'],
                initialization=17,environment_schema=match.game.lobby.schema,
                prefix_control='scripted legal setup only; no policy decisions/rewards until handoff' if match.context!='ordinary' else 'none',
                prefix_handoff_seconds=75.2 if match.context.startswith('meeting') else 75. if match.context=='body' else 0.,
                reward_profile=payload['rewards'].to_dict()))
    audit=DecisionAudit()
    try:
        match.prepare_prefix(recorder)
        controlled_meeting=match.game.meeting_id if match.context.startswith('meeting') else None
        rollout,summary=match.collect(policy,initialization=17,policy_version=payload['version'],recorder=recorder,
            deterministic=payload['greedy'],decision_observer=audit)
    finally:
        if recorder and not recorder.recorder._closed:
            recorder.recorder._stream.flush();recorder.recorder._stream.close();recorder.recorder._closed=True
    summary.update(audit=audit.score(match.game,controlled_meeting),wall_seconds=perf_counter()-started)
    return rollout,summary


def jobs(policy,contexts,*,rewards,label,deadline,session,greedy=False,version=0,replay_dir=None,spectator_source=None):
    return [dict(weights={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()},context=context,
        rewards=rewards,label=label,deadline=deadline,session=session,greedy=greedy,version=version,
        spectator_source=str(spectator_source) if spectator_source else None,
        replay=str(Path(replay_dir)/label/(context['context']+'-'+str(context['seed'])+'.jsonl')) if replay_dir else None)
        for context in contexts]


def run_jobs(payloads,deadline,workers=4,on_result=None,on_sample=None):
    if workers not in (1,2,4):raise ValueError('Unsupported registered worker count')
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
    pending={};results=[None]*len(payloads)
    try:
        for i,payload in enumerate(payloads):pending[executor.submit(execute,payload)]=i
        while pending:
            left=deadline-perf_counter()
            if left<=0:raise TimeoutError('Registered total wall ceiling; preserve partial outputs')
            done,_=wait(pending,timeout=min(left,1.),return_when=FIRST_COMPLETED)
            if on_sample:on_sample()
            for future in done:
                i=pending.pop(future);results[i]=future.result()
                if on_result:on_result(i,results[i])
        executor.shutdown(wait=True)
    except BaseException:
        # Terminate only processes owned by this executor, never unrelated apps/worktrees.
        processes=list((executor._processes or {}).values())
        executor.shutdown(wait=False,cancel_futures=True)
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:process.join(timeout=2)
        raise
    return results
