import json
import shutil
from phase6.study_comparison import build_comparison
from phase6.study_continuation import PILOT
from phase6.study_training import CONDITIONS


def test_nine_model_report_uses_paired_scenarios_not_fake_extra_seeds(tmp_path):
    pilot=json.loads((PILOT/'results.json').read_text());training={};selection={}
    # Repeated fixture records exercise aggregation only; no new model inference.
    for seed in (29,43):
        for condition in CONDITIONS:
            key=f'{condition}-{seed}';dest=tmp_path/'validation'/key;dest.mkdir(parents=True)
            shutil.copyfile(PILOT/'validation'/condition/'matches.jsonl',dest/'matches.jsonl')
            training[key]=dict(pilot['details'][condition]['training'],seconds=1.)
            selection[key]=pilot['details'][condition]['selected_checkpoint']
    result=build_comparison(tmp_path,training,selection,pilot,12.,2.,3.)
    assert result['initialization_seeds']==[17,29,43] and len(result['metrics'])==13
    assert result['total_training_decisions']==73728 and result['additional_validation_executions']==1680
    assert result['aggregate']['full']['crew_win_rate']['sample_sd']==0
    assert result['aggregate']['full']['votes_correct']==3*159
    assert result['aggregate']['full']['votes_incorrect']==3*8
    assert result['aggregate']['current_only']['voting_accuracy'] is None
    assert all(m['ID']['matches']==200 for m in result['metrics'].values())
    paired=result['paired_condition_comparisons']['full']['current_only']['crew_win']
    assert paired['differences_by_initialization']==[0.,0.,0.]
    assert len(paired['paired_scenario_bootstrap95'])==2 and not result['final_test_opened']
    assert (tmp_path/'nine_model_comparison.png').exists()
