"""Owner-approved two-seed pilot and exactly 40 greedy validation matches."""
import json
from pathlib import Path
from time import perf_counter, time
import hashlib
import torch
from .env import ImpostorEnv
from .frozen import FrozenModels, weight_hash
from .learning import (LearnerSession, exclusive_worker, load_inference, pilot_manifest,
                       verify_record, write_json, PilotConfig)

INITIALIZATIONS=(71,83)
VALIDATION=tuple(range(3200000,3200020))
CAP_SECONDS=10800


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def social_metrics(env):
    """Offline scoring only; never supplied to policy observations."""
    from collections import Counter
    from social_deduction.actor import Role
    game=env.match.game;own=env.match.focal
    claims=[e for e in game.event_log if e['kind']=='claim' and e['speaker_id']==own]
    return dict(impostor_claims=dict(Counter(str(e['claim_kind']) for e in claims)),
        innocent_team_ejections=sum(e['kind']=='ejection' and e.get('player_id') is not None
            and game.players[e['player_id']].role is Role.CREWMATE for e in game.event_log))


def progress(output,**values):
    write_json(output/'progress.json',dict(updated_epoch=time(),**values))
    stage=values['stage']
    if stage=='training':
        details=f"Seed {values['initialization_seed']}: {values['decisions']}/4096 decisions; {values['completed_episodes']} completed episodes."
    elif stage=='validation':
        details=f"{values['completed']}/40 matches complete; initialization {values['initialization_seed']}, scenario {values['scenario']}."
    else:details=json.dumps(values,indent=2)
    (output/'LIVE_PROGRESS.md').write_text('# Impostor pilot — live progress\n\n**Stage: '+stage+'**\n\n'+details+
        '\n\nApproved: two 4096-decision runs and 40 validation matches.\nFrozen crew/belief models; no merges, deployment or reserved final tests.\n',encoding='utf-8')
    print(json.dumps(values,allow_nan=False),flush=True)


def run(output,*,resume=False):
    output=Path(output)
    if resume:
        if not (output/'study.json').exists():raise ValueError('No existing pilot to resume')
    else:output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with exclusive_worker(output):
        models=FrozenModels()
        if resume:study=json.loads((output/'study.json').read_text())
        else:
            record=pilot_manifest(ImpostorEnv(models=models))
            if record['dirty']:raise ValueError('Commit source/config before the real pilot')
            study=dict(schema='impostor-approved-pilot-v1',record=record,initializations=list(INITIALIZATIONS),
                validation=list(VALIDATION),started_epoch=time(),deadline_epoch=time()+CAP_SECONDS,
                approval='Owner: go on, 27 October 2026; two 4096-decision runs and 40 validation matches',
                condition='ordinary_v2_both_safeguards_full29',evaluation_selection='final_4096_preselected',
                final_tests_opened=False,budget_seconds=CAP_SECONDS)
            write_json(output/'study.json',study,exclusive=True)
        if study['initializations']!=list(INITIALIZATIONS) or study['validation']!=list(VALIDATION):
            raise ValueError('Pilot scenario/init manifest changed')
        deadline=study['deadline_epoch'];record=study['record']
        verify_record(record,models=models,deadline=None if (output/'results.json').exists() else deadline)
        if (output/'results.json').exists():
            completed=json.loads((output/'results.json').read_text())
            if completed.get('status')!='completed_pilot_not_final_acceptance':raise ValueError('Unexpected existing result')
            return completed
        try:
            for seed in INITIALIZATIONS:
                directory=output/f'seed-{seed}'
                if (directory/'training.json').exists():continue
                directory.mkdir(exist_ok=resume)
                if not (directory/'run-start.json').exists():
                    write_json(directory/'run-start.json',dict(started_epoch=time()),exclusive=True)
                snapshots=sorted(directory.glob('resume-*.pt'))
                if snapshots:
                    if not resume:raise ValueError('Existing run requires explicit resume')
                    session=LearnerSession.restore(snapshots[-1],models=models,deadline=deadline)
                else:
                    session=LearnerSession(seed,models=models,record=record)
                    if not (directory/'checkpoint-0000.pt').exists():session.save(directory/'checkpoint-0000.pt',resume=False)
                    session.save(directory/'resume-0000.pt')
                progress(output,stage='starting_run',initialization_seed=seed,decisions=session.decisions,
                         target_decisions=4096,deadline_epoch=deadline)
                while session.decisions<4096:
                    row=session.update(min(256,4096-session.decisions),deadline)
                    checkpoint=directory/f'resume-{session.decisions:04d}.pt'
                    session.save(checkpoint)
                    if session.decisions in (1024,2048,4096):
                        session.save(directory/f'checkpoint-{session.decisions:04d}.pt',resume=False)
                    write_json(directory/'updates.json',session.updates)
                    write_json(directory/'episodes.json',session.episodes)
                    progress(output,stage='training',initialization_seed=seed,**row)
                if weight_hash(session.policy)==session.initial_hash:raise RuntimeError('Impostor did not update')
                models.assert_unchanged()
                final=directory/'checkpoint-4096.pt'
                if not final.exists():session.save(final,resume=False)
                write_json(directory/'updates.json',session.updates)
                write_json(directory/'episodes.json',session.episodes)
                write_json(directory/'training.json',dict(seed=seed,decisions=session.decisions,
                    completed_episodes=len(session.episodes),simulated_seconds=session.simulated_seconds,
                    collection_and_update_seconds=session.elapsed_seconds,policy_changed=True,
                    training_wall_seconds=time()-json.loads((directory/'run-start.json').read_text())['started_epoch'],
                    initial_policy_hash=session.initial_hash,final_policy_hash=weight_hash(session.policy),
                    checkpoint_sha256=digest(final),frozen_weights_unchanged=True,
                    partial_episode_seed=session.scenario_seed,partial_episode_actions=len(session.episode_actions)),exclusive=True)
                del session
            # Both learning runs finish before any validation outcomes are opened.
            for seed in INITIALIZATIONS:
                directory=output/f'seed-{seed}'
                final=directory/'checkpoint-4096.pt'
                policy,payload=load_inference(final,models=models,deadline=deadline)
                if payload['decisions']!=4096:raise ValueError('Pilot evaluates only the preselected final checkpoint')
                env=ImpostorEnv(models=models)
                for scenario in VALIDATION:
                    target=directory/f'validation-{scenario}.json'
                    if target.exists():
                        row=json.loads(target.read_text())
                        if row['checkpoint_sha256']!=digest(final) or row['seed']!=scenario:
                            raise ValueError('Existing validation record mismatch')
                        continue
                    verify_record(record,models=models,deadline=deadline)
                    began=perf_counter();packet,_=env.reset(seed=scenario)
                    decisions=0
                    while not env.finished:
                        if time()>deadline:raise TimeoutError('Pilot deadline exceeded in validation')
                        action,_,_=policy.act(packet,deterministic=True)
                        packet,_,_,_,_=env.step(action);decisions+=1
                        if env.match.game.metrics['illegal_actions'] or env.match.game.metrics['navigation_failures']:
                            raise ValueError('Illegal action/navigation failure during validation')
                    row=dict(**env.summary(),**social_metrics(env),initialization_seed=seed,seconds=perf_counter()-began,
                             external_decisions=decisions,checkpoint_sha256=digest(final),partition='validation',
                             action_selection='greedy',checkpoint_selection='preselected_final_4096')
                    write_json(target,row,exclusive=True)
                    progress(output,stage='validation',initialization_seed=seed,scenario=scenario,
                        completed=sum(1 for _ in output.glob('seed-*/validation-*.json')),total=40,
                        impostor_win=row['impostor_win'],reason=row['reason'],kills=row['kills'],seconds=row['seconds'])
            verify_record(record,models=models,deadline=deadline)
            rows=[json.loads(p.read_text()) for p in sorted(output.glob('seed-*/validation-*.json'))]
            if len(rows)!=40:raise ValueError('Pilot validation count mismatch')
            result=dict(status='completed_pilot_not_final_acceptance',initializations=list(INITIALIZATIONS),
                training_decisions=8192,validation_matches=40,seconds=time()-study['started_epoch'],
                frozen_weights_unchanged=True,final_tests_opened=False,checkpoint_search=False,
                checkpoints={str(s):digest(output/f'seed-{s}'/'checkpoint-4096.pt') for s in INITIALIZATIONS},
                validation=rows)
            write_json(output/'results.json',result,exclusive=True)
            progress(output,stage='complete',training_decisions=8192,validation_matches=40,seconds=result['seconds'])
            return result
        except BaseException as error:
            progress(output,stage='stopped',exception=type(error).__name__,reason=str(error),
                     elapsed_seconds=time()-study['started_epoch'],automatic_retry=False)
            raise
