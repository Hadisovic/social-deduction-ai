import hashlib
import json
from pathlib import Path
from time import time
import numpy as np
import pytest
from phase6.config import PARTITIONS, source_hashes
from phase6.study_support import guard, packet_guard, transition_guard, write_json
from phase6.study_training import train_run, CONDITIONS, CHECKPOINTS
from phase6.study_validation import validation_manifest, select_checkpoint, evaluate_job


def test_validation_manifest_is_fresh_bounded_and_final_disjoint():
    manifest=validation_manifest(200)
    assert manifest==tuple(range(1220000,1220200))
    assert validation_manifest(3,20)==(1220020,1220021,1220022)
    for key in ('train','final_id','final_patient','development','smoke_train'):
        lo,hi=PARTITIONS[key]
        assert all(not lo<=seed<=hi for seed in manifest)


@pytest.mark.parametrize('count,offset',[(201,0),(20,181),(0,0),(1,-1),(True,0),(1,True),('200',0)])
def test_validation_rejects_bad_counts_and_offsets(count,offset):
    with pytest.raises(ValueError):validation_manifest(count,offset)


def screening_rows():
    return [dict(method=f'full@{step}',family='ID',seed=seed,crew_win=seed%2==0,
                 own_tasks=1,illegal_actions=0,navigation_failures=0)
            for step in CHECKPOINTS for seed in validation_manifest(20)]


def test_selection_uses_frozen_ranking_not_final_values():
    checkpoints=[dict(decisions=step,name=f'checkpoint-{step}.pt',sha256=str(step)) for step in CHECKPOINTS]
    rows=screening_rows()
    assert select_checkpoint('full',rows,checkpoints)['decisions']==2048
    for row in rows:
        if row['method']=='full@4096':row['own_tasks']=2
    assert select_checkpoint('full',rows,checkpoints)['decisions']==4096
    for row in rows:
        if row['method']=='full@6144':row['crew_win']=True
    assert select_checkpoint('full',rows,checkpoints)['decisions']==6144
    for corrupted in (rows[:-1], [dict(r,family='patient') for r in rows],
                      [dict(r,seed=1230000) for r in rows], [dict(r,navigation_failures=1) for r in rows]):
        with pytest.raises(ValueError):select_checkpoint('full',corrupted,checkpoints)


def test_guards_stop_on_source_deadline_and_bad_transitions():
    sources=source_hashes();guard(sources,time()+60)
    assert {'belief/features.py','belief/model.py','social_deduction/truth.py'}<=sources.keys()
    with pytest.raises(ValueError,match='Source'):guard({},time()+60)
    with pytest.raises(TimeoutError):guard(sources,time()-1)
    with pytest.raises(FloatingPointError):packet_guard({'mask':np.array([1]),'x':np.array([np.nan])})
    with pytest.raises(FloatingPointError):packet_guard({'mask':np.array([0])})
    from types import SimpleNamespace
    env=SimpleNamespace(match=SimpleNamespace(game=SimpleNamespace(metrics={'illegal_actions':0,'navigation_failures':1})))
    with pytest.raises(ValueError,match='navigation'):transition_guard(env,dict(discount=.9,elapsed_seconds=1),0.)
    with pytest.raises(FloatingPointError):transition_guard(env,dict(discount=.9,elapsed_seconds=1),float('inf'))


def test_final_partition_rejected_before_any_match(tmp_path):
    with pytest.raises(ValueError,match='Only validation'):
        evaluate_job(dict(output=str(tmp_path/'final-forbidden'),partition='final_id'))
    assert not list(tmp_path.rglob('matches.jsonl'))


def test_existing_output_and_selection_are_preserved(tmp_path):
    lock=tmp_path/'selection.json';write_json(lock,dict(selected='old'),exclusive=True)
    with pytest.raises(FileExistsError):write_json(lock,dict(selected='new'),exclusive=True)
    assert json.loads(lock.read_text())['selected']=='old'
    with pytest.raises(FileExistsError):train_run(tmp_path,'full',source_hashes(),'test',time()+60,verification=True)


def test_failed_training_preserves_failure_record_and_no_bad_checkpoint(tmp_path,monkeypatch):
    import phase6.study_training as module
    original=module.Phase6Env
    class Broken(original):
        def step(self,action):
            result=super().step(action)
            self.match.game.metrics['navigation_failures']=1
            return result
    monkeypatch.setattr(module,'Phase6Env',Broken)
    out=tmp_path/'bad'
    with pytest.raises(ValueError,match='navigation'):
        train_run(out,'full',source_hashes(),'test',time()+120,verification=True)
    assert (out/'failure.json').is_file() and not list(out.glob('*.pt'))


def test_training_reproducibility_includes_ppo_and_saved_inference(tmp_path):
    from phase6.study import preflight
    result=preflight(tmp_path,source_hashes(),'test',time()+180)
    assert result['training_state_and_transition_equal'] and not result['final_test_opened']
    assert result['development_rows'][0]==result['development_rows'][1]
    assert {r['method'] for r in result['validation_worker_probe_rows']}=={'idle','random'}
    assert all(r['seed']==1220000 for r in result['validation_worker_probe_rows'])


def test_stage_orchestrates_only_authorized_models_and_validation(tmp_path,monkeypatch):
    import phase6.study as study
    import phase6.study_report as report
    calls=[];jobs_seen=[]
    monkeypatch.setattr(study,'current_revision',lambda:'test')
    monkeypatch.setattr(study,'preflight',lambda *args:None)
    def fake_train(output,condition,*args):
        calls.append(condition);Path(output).mkdir()
        checkpoints=[]
        for step in CHECKPOINTS:
            p=Path(output)/f'checkpoint-{step}.pt';p.write_bytes(b'fixture')
            checkpoints.append(dict(decisions=step,name=p.name,sha256=hashlib.sha256(b'fixture').hexdigest()))
        return dict(initial_state_sha256='same',checkpoints=checkpoints)
    monkeypatch.setattr(study,'train_run',fake_train)
    def fake_jobs(jobs):
        jobs_seen.extend(jobs);rows=[]
        for job in jobs:
            p=Path(job['output']);p.mkdir(parents=True)
            subset=[dict(method=job['method'],family='ID',seed=seed,crew_win=False,
                         own_tasks=0,illegal_actions=0,navigation_failures=0)
                    for seed in validation_manifest(job['count'])]
            (p/'matches.jsonl').write_text('\n'.join(json.dumps(r) for r in subset))
            rows.extend(subset)
        return rows
    monkeypatch.setattr(study,'parallel_jobs',fake_jobs)
    monkeypatch.setattr(report,'build_report',lambda *args:dict(status='fixture',seconds=0,final_test_opened=False))
    out=tmp_path/'stage';study.run(out)
    assert calls==list(CONDITIONS)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['initialization_seeds']==[17] and manifest['decisions_per_model']==8192
    assert len(jobs_seen)==19 and sum(j['count'] for j in jobs_seen)==1640
    assert all(j['partition']=='validation' for j in jobs_seen)
    assert (out/'selection.json').exists()
