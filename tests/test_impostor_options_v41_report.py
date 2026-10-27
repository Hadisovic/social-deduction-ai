"""Synthetic offline graph/aggregation fixtures; never real performance data."""
import importlib.util
import json
from pathlib import Path
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v4_offline_report',ROOT/'tools/report_impostor_options_v41.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_negative_reward_components_and_zero_all_win_intervals():
    a=report.wilson(0,20);b=report.wilson(20,20)
    assert a[0]==pytest.approx(0) and a[1]>.15
    assert b[1]==pytest.approx(1) and b[0]<.85
    row=dict(impostor_win=False,reward_components=dict(cooldown_ready=-2.,terminal=-10.,kill=3.),reason='TASKS_COMPLETED')
    assert report.aggregate([row,row])['rewards']==dict(cooldown_ready=-4.,terminal=-20.,kill=6.)


@pytest.mark.parametrize('wins',[False,True])
def test_complete_synthetic_graph_report(tmp_path,wins):
    source=tmp_path/'source';source.mkdir();target=tmp_path/'report'
    checkpoint=tmp_path/'synthetic-policy.bin';checkpoint.write_bytes(b'SYNTHETIC FIXTURE ONLY, NOT A TRAINED POLICY')
    def write(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')
    started=time.time()-1000;locked=time.time()-100
    study=dict(schema='impostor-options-v41',steps=8192,initializations=[71,83],budget_seconds=14400,started_epoch=started,
        original_started_epoch=started-1000,deadline_epoch=started-1000+14400,
        record=dict(core=dict(source_commit='SYNTHETIC-FIXTURE-NO-TRAINING')))
    base=dict(impostor_win=wins,kills=1 if wins else 0,witnessed_kills=0,impostor_ejected=False,
        fake_starts=2,fake_completions=1,fake_selections=4,fake_selections_no_visible=3,max_fake_selection_streak=4,
        fake_activity_seconds=4.,ready_faking_seconds=2.,measured_cooldown_ready_seconds=10.,pursuit_positive_total=.01,
        reward_components=dict(terminal=10. if wins else -10.,kill=3. if wins else 0.,ejection=0.,cooldown_ready=-.2,pursuit=0.),
        actions=dict(fake_task=4,go_to_room=2),reason='PARITY' if wins else 'TASKS_COMPLETED',illegal_actions=0,
        navigation_failures=0,simulated_seconds=100.,seconds=10.,return_=12.8 if wins else -10.2,
        crew_reports=0,impostor_reports=0,correct_crew_votes=0,incorrect_crew_votes=0,innocent_team_ejections=0)
    candidates=[]
    for seed in (71,83):
        for step in (4096,8192):
            method=f'seed{seed}-step{step}'
            candidates.append(dict(seed=seed,step=step,method=method,path=str(checkpoint),sha256=report.digest(checkpoint),
                wins=20 if wins else 0,kills=20 if wins else 0,timeouts=0))
            for scenario in range(3200140,3200160):write(source/'selection'/f'{method}-{scenario}.json',dict(base,seed=scenario,method=method))
        episode=dict(base,decisions=8192)
        update=dict(decisions=8192,policy_loss=.1,value_loss=1.,entropy=1.,approximate_kl=.01,clip_fraction=.1,
            explained_variance=None,kill_probability_when_available=.2,kill_selections=1,kill_available_decisions=5,
            actor_gradient_norm=.2,critic_gradient_norm=.3,collection_seconds=10.,optimization_seconds=1.)
        for name,obj in [('episodes',[episode]),('updates',[update]),('training',dict(decisions=8192,wall_seconds=100.)),('start',dict(epoch=started))]:
            write(source/f'seed-{seed}'/f'{name}.json',obj)
    selected=candidates[1];write(source/'study.json',study)
    write(source/'selection-lock.json',dict(selected=selected,locked_epoch=locked))
    confirmation=[]
    for method in ('selected_v41','unchanged_v2'):
        for scenario in range(3200160,3200180):
            row=dict(base,seed=scenario,method=method);confirmation.append(row)
            write(source/'confirmation'/f'{method}-{scenario}.json',row)
    diagnostic=[dict(base,seed=s,method='selected_sampled') for s in range(3200140,3200145)]
    write(source/'results.json',dict(status='complete',selected=selected,selection=candidates,confirmation=confirmation,
        diagnostic=diagnostic,seconds=1000.,training_wall_seconds=100.,final_tests_opened=False,frozen_weights_unchanged=True))
    summary=report.generate(source,target)
    assert len(list((target/'figures').glob('*.png')))==11
    assert summary['confirmation']['selected_v41']['wins']==20*int(wins)
    assert summary['paired']['win_rate_difference']==0.
    assert summary['confirmation']['selected_v41']['rewards']['cooldown_ready']==pytest.approx(-4.)
    assert 'SYNTHETIC-FIXTURE' in (target/'README.md').read_text()
    with pytest.raises(FileExistsError):report.generate(source,target)


def test_curator_keeps_locked_identity_and_never_overwrites(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('v4_fixture_curator',ROOT/'tools/release_impostor_options_v41.py')
    curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    monkeypatch.setattr(curator,'ROOT',tmp_path)
    source=tmp_path/'raw';summary_dir=tmp_path/'report';source.mkdir();summary_dir.mkdir()
    def write(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
    for seed in (71,83):
        folder=source/f'seed-{seed}';folder.mkdir()
        for step in (0,2048,4096,8192):(folder/f'checkpoint-{step:05d}.pt').write_text(f'{seed}:{step}')
    chosen=source/'seed-71/checkpoint-08192.pt'
    selected=dict(seed=71,step=8192,path=chosen.relative_to(tmp_path).as_posix(),sha256=curator.digest(chosen))
    write(source/'study.json',dict(deadline_epoch=1.,record=dict(original_study_sha256='synthetic',core=dict(source_commit='SYNTHETIC-FIXTURE'))))
    write(source/'selection-lock.json',dict(selected=selected))
    write(source/'results.json',dict(status='complete',final_tests_opened=False,confirmation=[{}]*40,selected=selected))
    write(summary_dir/'summary.json',dict(selected=selected,confirmation=dict(selected_v41=dict(wins=0)),paired={}))
    import impostor_research.frozen as frozen
    import impostor_options_v41.learning as learning
    class Models:
        def assert_unchanged(self):pass
    monkeypatch.setattr(frozen,'FrozenModels',Models)
    def fake_load(path,models):
        seed,step=map(int,Path(path).read_text().split(':'))
        return None,dict(seed=seed,decisions=step,development=False,schema='impostor-options-v41')
    monkeypatch.setattr(learning,'load',fake_load)
    bundle=tmp_path/'curated/study';manifest=curator.curate(source,summary_dir,bundle)
    assert manifest['checkpoint_sha256']==curator.digest(chosen)
    assert (bundle.parent/'release/policy.pt').read_bytes()==chosen.read_bytes()
    assert len(list(bundle.glob('seed-*/*.pt')))==8
    assert manifest['quality']=='insufficient_winning_strategy'
    with pytest.raises(ValueError,match='overwrite'):curator.curate(source,summary_dir,bundle)
