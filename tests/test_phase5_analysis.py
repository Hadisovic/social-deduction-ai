import pytest
from phase5_analysis import summarize


def match(method,seed,win,correct=0,cast=0):
    return dict(method=method,seed=seed,family='ID',crew_win=win,own_tasks=int(win),
                survival=10.,votes_cast=cast,votes_correct=correct,skips=int(not cast),
                illegal_actions=0,navigation_failures=0,reason='TASKS_COMPLETED' if win else 'PARITY',
                correct_ejections=0,innocent_ejections=0)


def test_perfect_results_retain_uncertainty_and_pair_by_seed_not_row_order():
    rows=[match('learned',s,True,1,1) for s in range(20)]
    rows += [match('scripted',s,False) for s in reversed(range(20))]
    r=summarize(rows)
    measured=r['metrics']['learned']['ID']
    assert measured['crew_win_rate']==1
    assert .8 < measured['win_rate_wilson95'][0] < 1
    assert measured['votes_cast']==20 and measured['voting_accuracy']==1
    assert r['metrics']['scripted']['ID']['voting_accuracy'] is None
    paired=r['paired']['learned']['ID']['scripted']
    assert paired['win_rate_difference']==1 and paired['bootstrap95']==[1.,1.]


def test_duplicate_matches_cannot_artificially_inflate_sample_size():
    row=match('learned',17,True)
    with pytest.raises(ValueError,match='Duplicate'):summarize([row,row])
