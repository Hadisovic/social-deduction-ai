import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from spectator.v2.data import ReplayIndex
from spectator.v2.preferences import Preferences


def recording():
    def tick(t,phase='roaming'):
        return dict(tick=t,sim_time=float(t),phase=phase,players={p:dict(position=[0.,0.],velocity=[0.,0.],role='impostor' if p=='i' else 'crewmate',status='dead' if p=='g' else 'alive',tasks_total=None if p=='i' else 4,tasks_completed=1) for p in ('a','b','i','g')},bodies={})
    return dict(header={'schema':'spectator-v1.0','training_session':'s','scenario_seed':7,'game_schema':'v2','source_version':'frozen','config':{'context':'ordinary','policy_version':2,'training_condition':'original'}},ticks=[tick(0),tick(1,'discussion'),tick(2,'voting'),tick(3,'voting')],events=[
        dict(kind='meeting_started',meeting_id='old',time=0.,tick=0,participants=['a','b','i']),
        dict(kind='vote_submitted',meeting_id='old',time=0.,tick=0,voter_id='a',target_id='i',channel='truth'),
        dict(kind='meeting_started',meeting_id='now',time=1.,tick=1,participants=['a','b','i','g']),
        dict(kind='vote_submitted',meeting_id='now',time=2.,tick=2,voter_id='a',target_id='i',channel='truth'),
        dict(kind='vote_submitted',meeting_id='now',time=2.,tick=2,voter_id='b',target_id=None,channel='truth'),
        dict(kind='vote_submitted',meeting_id='now',time=2.,tick=2,voter_id='g',target_id='a',channel='truth'),
        dict(kind='vote',meeting_id='now',time=3.,tick=3,voter_id='a',target_id='i',channel='public')],
        neural=[dict(player_id='a',sim_time=0.,action_kind='wait',action_probabilities={'wait':1.},suspicion_probabilities={'i':.7}),dict(player_id='b',sim_time=1.,suspicion_probabilities={'a':.99}),dict(player_id='a',sim_time=3.,action_kind='vote',action_probabilities={'vote':1.},suspicion_probabilities={'i':.9})],
        rewards=[dict(player_id='a',sim_time=1.,reward=1.,cumulative_reward=1.,components={'alive_tasks':1.}),dict(player_id='a',sim_time=3.,reward=-.1,cumulative_reward=.9,components={'time':-.1})],result={'sim_time':3.,'reason':'X'})


def test_meeting_scope_ghosts_missing_skips_and_explicit_truth():
    x=ReplayIndex(recording());x.player.seek_to_time(2.)
    m=x.meeting();assert m['eligible']==['a','b','i']
    assert [b['state'] for b in m['ballots']]==['hidden','hidden','missing']
    assert all(b['target'] is None for b in m['ballots'])
    assert x.meeting(omniscient=False,reveal=True)['ballots'][0]['state']=='hidden'
    m=x.meeting(omniscient=True,reveal=True)
    assert m['ballots'][0]['target']=='i' and m['ballots'][1]['state']=='skip'
    x.player.seek_to_time(3.)
    assert x.meeting()['ballots'][0]['target']=='i'


def test_owner_and_time_isolation_rewards_and_private_gap():
    x=ReplayIndex(recording());x.player.seek_to_time(2.)
    assert x.neural('a')['sim_time']==0.
    assert x.neural('a')['suspicion_probabilities']=={'i':.7}
    assert x.reward('a')['components']=={'alive_tasks':1.}
    assert x.evidence('a')['private'] is None
    x.player.seek_to_time(3.)
    assert x.reward('a')['components']=={'alive_tasks':1.,'time':-.1}
    assert x.reward('a')['latest']['cumulative_reward']==.9


def test_missing_task_data_never_becomes_a_fake_percentage():
    d=recording();d['ticks'][0]['players']['a'].pop('tasks_total')
    x=ReplayIndex(d);assert x.task_progress('a') is None and x.team_tasks() is None


def test_hidden_ballot_setting_is_independent_of_omniscient_event_log():
    x=ReplayIndex(recording());x.player.seek_to_time(2.)
    rows=x.filtered_events(group='Votes',debug=True)
    assert all('target hidden' in r['text'] for r in rows)


def test_pairing_requires_real_session_environment_context_and_version():
    x=ReplayIndex(recording());other=json.loads(json.dumps(x.header))
    other['config']['training_condition']='shaped_v2';assert x.matching_header(other)
    other['scenario_seed']=8;assert not x.matching_header(other)
    other['scenario_seed']=7;other['config']['context']='body';assert not x.matching_header(other)
    other['config']['context']='ordinary';other['config']['policy_version']=3;assert not x.matching_header(other)


def test_settings_validate_persist_and_reset_without_touching_replays(tmp_path):
    path=tmp_path/'prefs.json';p=Preferences(path)
    assert not p.set('speed',float('nan')) and not p.set('omniscient','yes')
    p.set('reveal_ballots',True);p.set('scale',9);p.favorite('replay.jsonl');p.bookmark('replay.jsonl',1.2)
    q=Preferences(path);assert q['reveal_ballots'] and q['scale']==1.3 and len(q['bookmarks'])==1
    q.reset();assert not Preferences(path)['reveal_ballots']


def test_corrupt_preferences_fail_closed(tmp_path):
    p=tmp_path/'bad.json';p.write_text('not json')
    q=Preferences(p);assert q.error and not q['omniscient'] and not q['reveal_ballots']


def test_invalid_telemetry_is_flagged_and_never_becomes_a_prediction():
    d=recording();d['neural'].append(dict(player_id='a',sim_time=1.,suspicion_probabilities={'i':float('nan')}))
    d['events'].append(dict(kind='claim',time='bad'))
    x=ReplayIndex(d);x.player.seek_to_time(2.)
    assert len(x.issues)==2 and x.neural('a')['sim_time']==0.


def test_unlogged_cumulative_baseline_is_not_assigned_to_an_invented_component():
    d=recording();d['rewards'][0]['cumulative_reward']=1.5
    x=ReplayIndex(d);x.player.seek_to_time(1.)
    r=x.reward('a');assert r['components']=={'alive_tasks':1.}
    assert r['unattributed_balance']==.5 and r['latest']['cumulative_reward']==1.5
