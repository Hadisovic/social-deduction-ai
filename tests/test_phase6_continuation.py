import json
from pathlib import Path
from time import time, perf_counter
import pytest
from phase6.config import source_hashes
from phase6.study_training import train_run
from phase6.study_continuation import verify_pilot, budget_projection, CEILING_SECONDS, SEEDS


def test_pilot_integrity_and_frozen_scientific_sources():
    sources=source_hashes();proof=verify_pilot(sources)
    assert proof['original_manifest']['initialization_seeds']==[17]
    assert not proof['final_test_opened']
    assert len(proof['preserved_sha256'])>50
    sources=dict(sources);sources['phase6/env.py']='changed'
    with pytest.raises(ValueError,match='scientific sources'):verify_pilot(sources)


def test_pilot_runtime_and_checkpoint_tamper_fail_closed(monkeypatch):
    import phase6.study_continuation as c
    original=c.runtime_versions
    monkeypatch.setattr(c,'runtime_versions',lambda:{'changed':True})
    with pytest.raises(ValueError,match='runtime versions'):verify_pilot(source_hashes())
    monkeypatch.setattr(c,'runtime_versions',original)
    monkeypatch.setattr(c,'digest',lambda p:'corrupt')
    with pytest.raises(ValueError,match='checkpoint changed'):verify_pilot(source_hashes())


def test_validation_rejects_wrong_initialization_before_match(tmp_path,monkeypatch):
    import phase6.study_validation as v
    from phase6.config import Experiment
    from phase6.study_support import runtime_versions
    monkeypatch.setattr(v,'load_policy',lambda p:(None,Experiment(),dict(smoke_only=False,
        initialization_seed=43,runtime_versions=runtime_versions())))
    with pytest.raises(ValueError,match='matching authorized'):
        v.evaluate_job(dict(output=str(tmp_path/'bad'),partition='validation',count=1,method='full',
            initialization_seed=29,condition='full',checkpoint='fixture',sources=source_hashes(),deadline=time()+60))
    assert not list(tmp_path.rglob('matches.jsonl'))


def test_selection_match_and_training_core_ast_are_preserved():
    import ast,subprocess
    def tree(path,original=False):
        text=subprocess.check_output(['git','show','8187d51:'+path],text=True) if original else Path(path).read_text()
        return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    old=tree('phase6/study_validation.py',True);new=tree('phase6/study_validation.py')
    for name in ('select_checkpoint','play_match','validation_manifest'):
        assert ast.dump(old[name])==ast.dump(new[name])
    old=tree('phase6/study_training.py',True)['train_run'];new=tree('phase6/study_training.py')['train_run']
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):return ast.copy_location(ast.Constant(17),node) if node.id=='initialization_seed' else node
    idx=next(i for i,a in enumerate(new.args.kwonlyargs) if a.arg=='initialization_seed')
    new.args.kwonlyargs.pop(idx);new.args.kw_defaults.pop(idx)
    new.body=[n for n in new.body if not (isinstance(n,ast.If) and 'Unregistered initialization seed' in ast.unparse(n))]
    new=Normalize().visit(new)
    assert ast.dump(old)==ast.dump(new)


@pytest.mark.parametrize('seed',[0,True,19,'29',None])
def test_unapproved_initialization_fails_before_creating_output(tmp_path,seed):
    output=tmp_path/'forbidden'
    with pytest.raises(ValueError,match='initialization seed'):
        train_run(output,'full',source_hashes(),'test',time()+60,initialization_seed=seed)
    assert not output.exists()


def test_budget_projection_stops_before_unaffordable_work(tmp_path):
    pilot={'details':{'full':{'training':{'seconds':100}}}}
    budget_projection(tmp_path,perf_counter(),{},[(29,'full')],pilot,100)
    assert json.loads((tmp_path/'budget.json').read_text())['projected_total_seconds']<CEILING_SECONDS
    with pytest.raises(TimeoutError,match='approval'):
        budget_projection(tmp_path,perf_counter(),{},[(29,'full')],pilot,CEILING_SECONDS)


@pytest.mark.parametrize('seed',SEEDS)
def test_actual_seed_reproduction_and_metadata(tmp_path,seed):
    sources=source_hashes();results=[]
    for suffix in ('a','b'):
        results.append(train_run(tmp_path/suffix,'full',sources,'test',time()+180,
            verification=True,initialization_seed=seed))
        metadata=json.loads((tmp_path/suffix/'config.json').read_text())
        assert metadata['initialization_seed']==seed and metadata['smoke_only']
        assert metadata['seed_partition']=='smoke_train'
    for key in ('initial_state_sha256','final_state_sha256','transition_trace_sha256','simulated_seconds'):
        assert results[0][key]==results[1][key]


def test_continuation_orchestrates_only_six_and_locked_validation(tmp_path,monkeypatch):
    import phase6.study_continuation as c
    import phase6.study_comparison as report
    import hashlib
    calls=[];jobs_seen=[];preflight=[]
    monkeypatch.setattr(c,'current_revision',lambda:'test')
    monkeypatch.setattr(c,'preflight',lambda *args,**kw:preflight.append(kw['initialization_seed']))
    def fake_train(output,condition,*args,**kwargs):
        seed=kwargs['initialization_seed'];calls.append((seed,condition));Path(output).mkdir()
        checkpoints=[]
        for step in (2048,4096,6144,8192):
            p=Path(output)/f'checkpoint-{step}.pt';p.write_bytes(b'fixture')
            checkpoints.append(dict(decisions=step,name=p.name,sha256=hashlib.sha256(b'fixture').hexdigest()))
        return dict(seconds=1.,initial_state_sha256=str(seed),checkpoints=checkpoints)
    monkeypatch.setattr(c,'train_run',fake_train)
    def fake_jobs(jobs):
        jobs_seen.extend(jobs);data=[]
        if jobs[0]['count']==200:assert (tmp_path/'run/selection.json').exists()
        for job in jobs:
            p=Path(job['output']);p.mkdir(parents=True)
            records=[dict(method=job['method'],family='ID',seed=seed,crew_win=False,own_tasks=0,
                     illegal_actions=0,navigation_failures=0) for seed in range(1220000,1220000+job['count'])]
            (p/'matches.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            data+=records
        return data
    monkeypatch.setattr(c,'parallel_jobs',fake_jobs)
    monkeypatch.setattr(report,'build_comparison',lambda *args:dict(status='fixture',seconds=0,final_test_opened=False))
    c.run(tmp_path/'run')
    assert calls==[(seed,condition) for seed in (29,43) for condition in c.CONDITIONS]
    assert preflight==[29,43]
    assert len(jobs_seen)==30 and sum(j['count'] for j in jobs_seen)==1680
    assert all(j['initialization_seed'] in (29,43) and j['partition']=='validation' for j in jobs_seen)
    assert (tmp_path/'run/results.json').exists()
