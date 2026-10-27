"""Offline report checks with synthetic records; no game/learning executions."""
import json
import pytest
from impostor_research.report import generate,wilson


def test_wilson_extremes():
    low,high=wilson(0,20);assert low==pytest.approx(0) and high>0
    low,high=wilson(20,20);assert low<1 and high==pytest.approx(1)


def test_graphs_and_missing_opportunities_are_finite(tmp_path):
    source=tmp_path/'source';source.mkdir();rows=[]
    for s in (71,83):
        directory=source/f'seed-{s}';directory.mkdir()
        training=dict(decisions=4096,seconds=1.)
        (directory/'training.json').write_text(json.dumps(training))
        update=dict(decisions=4096,simulated_seconds=100.,seconds=1.,collection_seconds=.8,optimization_seconds=.2,
            actions={'continue':4096},policy_loss=0.,value_loss=1.,entropy=1.,approximate_kl=0.,clip_fraction=0.,gradient_norm=1.)
        (directory/'updates.json').write_text(json.dumps([update]))
        episode=dict(decisions=4096,return_=-10.,impostor_win=False,kills=0,simulated_seconds=100.)
        (directory/'episodes.json').write_text(json.dumps([episode]))
        for i in range(20):
            rows.append(dict(seed=3200000+i,initialization_seed=s,impostor_win=s==83,kills=0,
                witnessed_kills=0,reason='TASKS_COMPLETED',simulated_seconds=100.,impostor_reports=0,crew_reports=0,
                impostor_ejected=False,illegal_actions=0,navigation_failures=0,seconds=1.,first_legal_kill=None))
    (source/'results.json').write_text(json.dumps(dict(status='completed_pilot_not_final_acceptance',validation=rows,
        seconds=40.,frozen_weights_unchanged=True)))
    out=tmp_path/'report';summary=generate(source,out)
    assert summary['validation']['71']['mean_first_legal_kill'] is None
    assert len(list((out/'figures').glob('*.png')))==6
    assert 'NaN' not in (out/'summary.json').read_text()
    with pytest.raises(FileExistsError):generate(source,out)
