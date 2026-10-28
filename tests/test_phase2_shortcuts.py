from crewmate_v2.phase2.shortcuts import assay


def test_public_classifier_holds_out_whole_scenarios_and_no_roles_in_features():
    rows=[]
    for seed in (2760078,2760079):
        for pid,role,kind in [('a','crewmate','defend_player'),('b','crewmate','defend_player'),('i','impostor','suspect_player')]:
            rows.append(dict(seed=seed,arm='baseline',meeting='m1',actor=pid,
                features=(str(role=='impostor'),kind),label=role))
    result=assay(rows)['baseline']
    assert result['scenario_clusters']==2 and result['held_out_actor_meetings']==6
    assert result['balanced_accuracy']==1 and result['meeting_macro_top1']==1
    assert 'tiny seed folds' in result['limits']


def test_uninformative_public_behavior_remains_a_tie_not_perfect_detection():
    rows=[dict(seed=seed,arm='mixture',meeting='m1',actor=pid,features=('False','none'),label=role)
          for seed in (2760078,2760079) for pid,role in [('a','crewmate'),('b','crewmate'),('i','impostor')]]
    result=assay(rows)['mixture']
    assert abs(result['meeting_macro_top1']-1/3)<1e-10
