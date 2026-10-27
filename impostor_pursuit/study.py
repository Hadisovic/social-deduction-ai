"""Conditional pursuit-family continuation inside the ORIGINAL three-hour deadline."""
from collections import Counter
import json
from pathlib import Path
from time import time,perf_counter
import torch
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.learning import write_json,exclusive_worker
from impostor_research.strategic_v2 import StrategicEnv
from impostor_research.pilot import social_metrics
from impostor_research.protocol import ROOT
from .learning import Session,record_for,verify,load,digest

SELECTION=tuple(range(3200060,3200080))
CONFIRM=tuple(range(3200080,3200100))


def progress(output,**state):
    write_json(output/'progress.json',dict(updated_epoch=time(),**state))
    (output/'LIVE_PROGRESS.md').write_text('# Pursuit-family correction — live progress\n\n```json\n'+json.dumps(state,indent=2)+
        '\n```\n\nSame original three-hour deadline; no changed rules, ghosts, frozen crew or final tests.\n',encoding='utf-8')
    print(json.dumps(state,allow_nan=False),flush=True)


def evaluate(output,study,models,method,partition,path):
    directory=output/partition;directory.mkdir(exist_ok=True)
    policy,_=load(path,models,study['deadline_epoch']);env=StrategicEnv(models=models)
    scenarios=SELECTION if partition=='selection' else CONFIRM
    rows=[]
    for scenario in scenarios:
        file=directory/f'{method}-{scenario}.json'
        if file.exists():rows.append(json.loads(file.read_text()));continue
        verify(study['record'],models,study['deadline_epoch']);started=perf_counter();packet,_=env.reset(seed=scenario)
        actions=Counter();decisions=0
        while not env.finished:
            if time()>study['deadline_epoch']:raise TimeoutError('Original overall deadline reached')
            action,_,_=policy.act(packet,deterministic=True);actions[env.options[action].kind]+=1
            packet,_,_,_,_=env.step(action);decisions+=1
            if env.match.game.metrics['illegal_actions'] or env.match.game.metrics['navigation_failures']:raise ValueError('Illegal action/navigation failure')
        row=dict(**env.summary(),**social_metrics(env),method=method,partition=partition,seconds=perf_counter()-started,
            actions=dict(actions),external_decisions=decisions,checkpoint_sha256=digest(path))
        write_json(file,row,exclusive=True);rows.append(row)
        progress(output,stage=partition,method=method,completed=len(list(directory.glob('*.json'))),
            total=60 if partition=='selection' else 40,wins=sum(r['impostor_win'] for r in rows),kills=sum(r['kills'] for r in rows))
    return rows


def run(output,parent_study,parent_checkpoint,resume=False):
    output=Path(output).resolve();parent_study=Path(parent_study).resolve();parent_checkpoint=Path(parent_checkpoint).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Keep the study inside this repository')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parent_result=json.loads((parent_study/'results.json').read_text())
    parent_plan=json.loads((parent_study/'study.json').read_text())
    if parent_result['status']!='complete' or len(parent_result['confirmation'])!=80:raise ValueError('Finish V2 confirmation first')
    wins=sum(r['impostor_win'] for r in parent_result['confirmation'] if r['method']=='selected_family')
    if wins>=5:raise ValueError('Conditional stage not triggered: V2 won at least 5/20 fresh matches')
    if digest(parent_checkpoint)!=parent_result['selected']['sha256']:raise ValueError('Parent is not the locked V2 selection')
    if not resume:
        if parent_plan['deadline_epoch']-time()<3600:raise TimeoutError('Less than one hour remains; no budget extension allowed')
        output.mkdir(parents=True,exist_ok=False)
    if (output/'results.json').exists():return json.loads((output/'results.json').read_text())
    with exclusive_worker(output):
        models=FrozenModels()
        if resume:study=json.loads((output/'study.json').read_text())
        else:
            record=record_for(models,parent_checkpoint)
            if record['core']['dirty']:raise ValueError('Commit the continuation runtime before starting')
            study=dict(record=record,started_epoch=time(),deadline_epoch=parent_plan['deadline_epoch'],
                parent_study=parent_study.relative_to(ROOT).as_posix(),parent_results_sha256=digest(parent_study/'results.json'),
                trigger_wins=wins,additional_decisions=8192,initialization_count=1,continuation_not_independent_replication=True,
                selection=list(SELECTION),confirmation=list(CONFIRM),final_tests_opened=False)
            write_json(output/'study.json',study,exclusive=True)
        verify(study['record'],models,study['deadline_epoch'])
        if digest(parent_study/'results.json')!=study['parent_results_sha256']:raise ValueError('Parent results changed')
        try:
            folder=output/'training';folder.mkdir(exist_ok=True)
            if not (folder/'training.json').exists():
                if not (folder/'start.json').exists():write_json(folder/'start.json',dict(epoch=time()),exclusive=True)
                snapshots=sorted(folder.glob('resume-*.pt'))
                session=Session.restore(snapshots[-1],models,study['deadline_epoch']) if snapshots else Session(study['record']['parent']['seed'],models=models,record=study['record'])
                if not snapshots:
                    session.save(folder/'checkpoint-00000.pt',False);session.save(folder/'resume-00000.pt')
                while session.decisions<8192:
                    row=session.update(min(256,8192-session.decisions),study['deadline_epoch'])
                    session.save(folder/f'resume-{session.decisions:05d}.pt')
                    if session.decisions in (2048,4096,8192):session.save(folder/f'checkpoint-{session.decisions:05d}.pt',False)
                    write_json(folder/'updates.json',session.updates);write_json(folder/'episodes.json',session.episodes)
                    progress(output,stage='training',decisions=session.decisions,episodes=len(session.episodes),
                        wins=sum(r['impostor_win'] for r in session.episodes),kills=sum(r['kills'] for r in session.episodes),
                        kill_probability=row['kill_probability_when_available'],seconds=session.elapsed_seconds)
                if weight_hash(session.policy)==session.initial_hash:raise ValueError('Continuation did not update')
                write_json(folder/'training.json',dict(seed=session.seed,decisions=8192,episodes=len(session.episodes),
                    simulated_seconds=session.simulated_seconds,wall_seconds=time()-json.loads((folder/'start.json').read_text())['epoch'],
                    timed_seconds=session.elapsed_seconds,actions=session.action_counts,policy_changed=True,frozen_weights_unchanged=True),exclusive=True)
            candidates=[]
            for method,path,step,parent in (('parent_v2',parent_checkpoint,0,True),
                ('pursuit4096',folder/'checkpoint-04096.pt',4096,False),('pursuit8192',folder/'checkpoint-08192.pt',8192,False)):
                rows=evaluate(output,study,models,method,'selection',path)
                candidates.append(dict(method=method,path=path.relative_to(ROOT).as_posix(),sha256=digest(path),step=step,parent=parent,
                    wins=sum(r['impostor_win'] for r in rows),kills=sum(r['kills'] for r in rows),timeouts=sum(r['reason']=='TIMEOUT' for r in rows)))
            best=max(candidates,key=lambda c:(c['wins'],c['kills'],-c['timeouts'],c['parent'],-c['step']))
            lock=output/'selection-lock.json'
            if lock.exists():
                if json.loads(lock.read_text())['selected']!=best:raise ValueError('Continuation lock changed')
            else:write_json(lock,dict(selected=best,candidates=candidates,locked_epoch=time(),confirmation_unopened=True),exclusive=True)
            rows=[]
            for method,path in (('selected',ROOT/best['path']),('parent_v2',parent_checkpoint)):
                rows+=evaluate(output,study,models,method,'confirmation',path)
            verify(study['record'],models,study['deadline_epoch'])
            result=dict(status='complete',selected=best,selection=candidates,confirmation=rows,
                seconds=time()-study['started_epoch'],original_deadline_epoch=study['deadline_epoch'],
                frozen_weights_unchanged=True,final_tests_opened=False)
            write_json(output/'results.json',result,exclusive=True);progress(output,stage='complete',selected=best,seconds=result['seconds'])
            return result
        except BaseException as e:
            progress(output,stage='stopped',exception=type(e).__name__,reason=str(e),automatic_retry=False)
            raise
