import json
from pathlib import Path

from crewmate_v2.phase2.social import audit_replay, audit_library


def fixture(tmp_path,*,different=False,new_meeting=False):
    actor='a';voter='b' if different else actor
    rows=[dict(record='header',scenario_seed=2790000,agent_roles={'a':'crewmate','b':'crewmate','i':'impostor'},config=dict(context='ordinary',training_condition='shaped_v2')),
          dict(record='event',kind='meeting_started',meeting_id='m1',time=1),
          dict(record='neural',player_id=actor,sim_time=2,action_kind='claim_defend',action_target='i',selected_action_index=0,
               action_probabilities={'00: claim_defend / i':.6,'01: claim_suspect / i':.4},suspicion_probabilities={'i':.95,'b':.05}),
          dict(record='event',kind='claim',claim_kind='defend_player',speaker_id=actor,subject_id='i',channel='public',time=2)]
    if new_meeting:rows.append(dict(record='event',kind='meeting_started',meeting_id='m2',time=3))
    rows.extend([dict(record='neural',player_id=voter,sim_time=4,action_kind='vote',action_target='i',suspicion_probabilities={'i':.95,'b':.05}),
                 dict(record='event',kind='vote_submitted',voter_id=voter,target_id='i',meeting_id='m2' if new_meeting else 'm1',time=4)])
    path=tmp_path/'case.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows));return path


def test_same_owner_meeting_unchanged_belief_is_not_called_a_belief_reversal(tmp_path):
    result=audit_replay(fixture(tmp_path));case=result['cases'][0]
    assert case['target_is_impostor'] and case['target_belief_at_defense']==.95
    assert case['max_belief_change']==0 and case['defense_was_argmax']
    assert not case['intervening_public_events'] and 'Not recorded' in case['private_observations']


def test_different_agents_and_meetings_cannot_be_same_agent_reversals(tmp_path):
    result=audit_replay(fixture(tmp_path,different=True))
    assert not result['cases'] and result['counters']['different_agent_later_vote_links']==1
    result=audit_replay(fixture(tmp_path,new_meeting=True))
    assert not result['cases'] and result['counters']['same_agent_later_meeting_links']==1


def test_training_and_validation_use_partition_not_file_name(tmp_path):
    path=fixture(tmp_path);text=path.read_text();path.write_text(text.replace('2790000','2780001'))
    groups,raw=audit_library(tmp_path)
    assert list(groups)==['training_sample/shaped_v2/ordinary']
