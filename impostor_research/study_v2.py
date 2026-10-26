"""Bounded v2 learning, validation selection and locked confirmation comparisons."""
from collections import Counter
import hashlib,json,shutil
from pathlib import Path
from time import time,perf_counter
import torch
from .strategic_v2 import StrategicEnv,FamilyPolicy
from .learning_v2 import Session,record_for,verify,load
from .learning import write_json,exclusive_worker,load_inference
from .env import ImpostorEnv
from .frozen import FrozenModels,weight_hash
from .protocol import ROOT
from .pilot import social_metrics

SEEDS=(71,83)
SCREEN=tuple(range(3200020,3200040))
CONFIRM=tuple(range(3200040,3200060))
OLD='artifacts/impostor-pilot-20261027/seed-83/checkpoint-4096.pt'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(output,**state):
    write_json(output/'progress.json',dict(updated_epoch=time(),**state))
    title=state['stage']
    if title=='training':detail=f"Initialization {state['seed']}: {state['decisions']}/8192 decisions; {state['episodes']} games; {state['wins']} wins; {state['kills']} kills."
    elif title in ('selection','confirmation'):detail=f"{title}: {state['completed']}/{state['total']} matches. Method: {state['method']}."
    else:detail=json.dumps(state,indent=2)
    (output/'LIVE_PROGRESS.md').write_text('# Improved impostor — live progress\n\n**'+title+'**\n\n'+detail+
        '\n\nFrozen full-29 + three scripts; canonical ordinary V2; no merge/deploy/final tests.\n',encoding='utf-8')
    print(json.dumps(state,allow_nan=False),flush=True)


def evaluate(output,record,models,deadline,method,partition,checkpoint=None):
    from phase3_bots import ScriptedController,BotStyle
    folder=output/partition;folder.mkdir(exist_ok=True)
    if method=='pilot83':policy,_=load_inference(ROOT/OLD,models=models,deadline=deadline);env=ImpostorEnv(models=models)
    elif method=='scripted_hunter':policy=None;env=ImpostorEnv(models=models)
    elif method=='untrained_family':policy=FamilyPolicy(71).eval();env=StrategicEnv(models=models)
    else:policy,_=load(checkpoint,models,deadline);env=StrategicEnv(models=models)
    scenarios=SCREEN if partition=='selection' else CONFIRM
    rows=[]
    for scenario in scenarios:
        target=folder/f'{method}-{scenario}.json'
        if target.exists():rows.append(json.loads(target.read_text()));continue
        verify(record,models,deadline);start=perf_counter();packet,_=env.reset(seed=scenario)
        script=ScriptedController(91,BotStyle(impostor='hunter'));decisions=0;counts=Counter()
        while not env.finished:
            if time()>deadline:raise TimeoutError('V2 evaluation ceiling reached')
            if policy is None:
                action=script.decide(env.envelope.actor);counts[action.kind.value]+=1;packet,_,_,_,_=env.step_scripted(action)
            else:
                action,_,_=policy.act(packet,deterministic=True);counts[env.options[action].kind]+=1;packet,_,_,_,_=env.step(action)
            decisions+=1
            if env.match.game.metrics['illegal_actions'] or env.match.game.metrics['navigation_failures']:raise ValueError('Illegal/navigation failure')
        row=dict(**env.summary(),**social_metrics(env),method=method,partition=partition,seconds=perf_counter()-start,
            external_decisions=decisions,actions=dict(counts),checkpoint_sha256=digest(checkpoint) if checkpoint else digest(ROOT/OLD) if method=='pilot83' else None)
        write_json(target,row,exclusive=True);rows.append(row)
        progress(output,stage=partition,method=method,completed=len(list(folder.glob('*.json'))),total=80,
                 wins=sum(r['impostor_win'] for r in rows),kills=sum(r['kills'] for r in rows))
    return rows


def run(output,resume=False):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Store this registered study inside the project')
    if not resume:output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with exclusive_worker(output):
        models=FrozenModels()
        if resume:study=json.loads((output/'study.json').read_text())
        else:
            record=record_for(StrategicEnv(models=models))
            if record['dirty']:raise ValueError('Commit the v2 runtime before training')
            study=dict(record=record,started_epoch=time(),deadline_epoch=time()+10800,budget_seconds=10800,
                initializations=list(SEEDS),steps=8192,screen=list(SCREEN),confirmation=list(CONFIRM),
                old_checkpoint=OLD,old_checkpoint_sha256=digest(ROOT/OLD),
                selection_rule='wins, kills, fewer timeouts, earlier step, lower initialization',
                final_tests_opened=False)
            write_json(output/'study.json',study,exclusive=True)
        record=study['record'];deadline=study['deadline_epoch']
        verify(record,models,deadline)
        if digest(ROOT/OLD)!=study['old_checkpoint_sha256']:raise ValueError('Old pilot baseline changed')
        try:
            for seed in SEEDS:
                folder=output/f'seed-{seed}'
                if (folder/'training.json').exists():continue
                folder.mkdir(exist_ok=resume)
                if not (folder/'start.json').exists():write_json(folder/'start.json',{'epoch':time()},exclusive=True)
                snapshots=sorted(folder.glob('resume-*.pt'))
                if snapshots:session=Session.restore(snapshots[-1],models,deadline)
                else:
                    session=Session(seed,models=models,record=record)
                    if not (folder/'checkpoint-00000.pt').exists():session.save(folder/'checkpoint-00000.pt',False)
                    session.save(folder/'resume-00000.pt')
                progress(output,stage='starting_run',seed=seed,decisions=session.decisions)
                while session.decisions<8192:
                    row=session.update(min(256,8192-session.decisions),deadline)
                    session.save(folder/f'resume-{session.decisions:05d}.pt')
                    if session.decisions in (2048,4096,8192):session.save(folder/f'checkpoint-{session.decisions:05d}.pt',False)
                    write_json(folder/'updates.json',session.updates);write_json(folder/'episodes.json',session.episodes)
                    progress(output,stage='training',seed=seed,decisions=session.decisions,episodes=len(session.episodes),
                        wins=sum(r['impostor_win'] for r in session.episodes),kills=sum(r['kills'] for r in session.episodes),
                        seconds=session.elapsed_seconds,entropy=row['entropy'],kill_probability=row['kill_probability_when_available'])
                if not (folder/'checkpoint-08192.pt').exists():session.save(folder/'checkpoint-08192.pt',False)
                if weight_hash(session.policy)==session.initial_hash:raise ValueError('Learner did not update')
                write_json(folder/'updates.json',session.updates);write_json(folder/'episodes.json',session.episodes)
                write_json(folder/'training.json',dict(seed=seed,decisions=8192,episodes=len(session.episodes),
                    simulated_seconds=session.simulated_seconds,wall_seconds=time()-json.loads((folder/'start.json').read_text())['epoch'],
                    timed_seconds=session.elapsed_seconds,policy_changed=True,frozen_weights_unchanged=True,
                    initial_hash=session.initial_hash,final_hash=weight_hash(session.policy),actions=session.action_counts),exclusive=True)
            # Never tune or select while learning is still in progress.
            candidates=[]
            for seed in SEEDS:
                for step in (4096,8192):
                    checkpoint=output/f'seed-{seed}'/f'checkpoint-{step:05d}.pt';method=f'seed{seed}-step{step}'
                    rows=evaluate(output,record,models,deadline,method,'selection',checkpoint)
                    candidates.append(dict(seed=seed,step=step,path=checkpoint.relative_to(ROOT).as_posix(),method=method,sha256=digest(checkpoint),
                        wins=sum(r['impostor_win'] for r in rows),kills=sum(r['kills'] for r in rows),timeouts=sum(r['reason']=='TIMEOUT' for r in rows)))
            best=max(candidates,key=lambda c:(c['wins'],c['kills'],-c['timeouts'],-c['step'],-c['seed']))
            lock=output/'selection-lock.json'
            if lock.exists():
                selected=json.loads(lock.read_text())
                if selected['selected']!=best:raise ValueError('Locked selection changed')
            else:write_json(lock,dict(selected=best,candidates=candidates,locked_epoch=time(),confirmation_unopened=True),exclusive=True)
            # This identity remains fixed regardless of confirmation outcomes.
            rows=[]
            for method,path in (('selected_family',ROOT/best['path']),('pilot83',None),('scripted_hunter',None),('untrained_family',None)):
                rows+=evaluate(output,record,models,deadline,method,'confirmation',path)
            verify(record,models,deadline)
            result=dict(status='complete',selected=best,selection=candidates,confirmation=rows,
                        seconds=time()-study['started_epoch'],frozen_weights_unchanged=True,final_tests_opened=False)
            write_json(output/'results.json',result,exclusive=True)
            progress(output,stage='complete',selected=best,seconds=result['seconds'])
            return result
        except BaseException as e:
            progress(output,stage='stopped',exception=type(e).__name__,reason=str(e),automatic_retry=False)
            raise
