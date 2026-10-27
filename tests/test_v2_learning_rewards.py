from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from among_us_map import get_map
from navigation_service import NavigationPlanner
from crewmate_v2.config import LobbySettings
from crewmate_v2.runtime import CrewmateV2Env
from crewmate_v2.learning.config import RewardConfig
from crewmate_v2.learning.rewards import RewardLedger
from crewmate_v2.learning.env import LearningEnv
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import Intent,IntentKind
from phase3_runner import canonical_bytes

ROLES={'f':Role.CREWMATE,'i':Role.IMPOSTOR,'c':Role.CREWMATE}


def ledger(config=None):return RewardLedger('f',ROLES,config or RewardConfig())
def report(actor='f',body='body'):return dict(kind='report',reporter_id=actor,victim_id=body)
def vote(target='i',actor='f',meeting='m1'):return dict(kind='vote_submitted',voter_id=actor,target_id=target,meeting_id=meeting)


def test_exact_report_once_only_accepted_focal_action():
    x=ledger();a=Intent(IntentKind.REPORT,'body')
    assert x.consume([report()],active=True,intent=a,legal=True).components['reports']==1.25
    assert 'reports' not in x.consume([report()],active=True,intent=a,legal=True).components
    for events,intent,legal in (([],a,True),([report('c')],a,True),([report()],a,False),
                                ([report()],Intent(IntentKind.GO_TO_BODY,'body'),True)):
        y=ledger();y.consume(events,active=True,intent=intent,legal=legal)
        assert y.counts['reports']==0 and y.components['reports']==0


@pytest.mark.parametrize('target,expected,kind',[('i',2.,'correct_votes'),('c',-1.,'incorrect_votes'),(None,0.,'skips')])
def test_vote_reward_and_single_ballot_accounting(target,expected,kind):
    x=ledger();a=Intent(IntentKind.SKIP) if target is None else Intent(IntentKind.VOTE,target)
    frame=x.consume([vote(target)],active=True,intent=a,legal=True)
    assert frame.components[kind]==expected
    x.consume([vote(target)],active=True,intent=a,legal=True)
    assert x.counts[kind]==1 and x.components[kind]==expected


def test_automatic_skip_invalid_teammate_votes_and_ghost_task_do_not_shape():
    x=ledger();x.consume([vote(None),vote('i','c')],active=True)
    assert x.counts['automatic_skips']==1 and x.counts['correct_votes']==x.counts['skips']==0
    x.consume([dict(kind='elimination',victim_id='f'),dict(kind='task_completed',player_id='f',task_id='t')],active=False)
    assert x.components['ghost_tasks']==0 and x.counts['ghost_tasks']==1 and x.components['death']==-5
    x.consume([dict(kind='task_completed',player_id='f',task_id='t')],active=False,terminal=True,winner=Role.CREWMATE)
    assert x.counts['ghost_tasks']==1 and x.components['death']==-5 and x.components['win']==10
    x.consume([],active=False,terminal=True,winner=Role.CREWMATE)
    assert x.counts['win']==1


def test_event_order_alive_completion_before_death_and_ejected_ghost_exclusion():
    x=ledger();x.consume([dict(kind='task_completed',player_id='f',task_id='a'),
        dict(kind='ejection',player_id='f'),dict(kind='task_completed',player_id='f',task_id='b')],active=False)
    assert x.components['alive_tasks']==1 and x.components['ghost_tasks']==0
    old=ledger(RewardConfig.legacy());old.consume([dict(kind='elimination',victim_id='f'),
        dict(kind='task_completed',player_id='f',task_id='b')],active=False)
    assert old.components['ghost_tasks']==1


def test_physical_time_discount_and_separate_option_anchors():
    x=ledger();x.consume([],active=True)
    x.consume([report()],active=True,intent=Intent(IntentKind.REPORT,'body'),legal=True)
    first=x.rollup(0);second=x.rollup(1)
    assert first['discounted_components']['reports']==pytest.approx(1.25*.99**(.2/.6))
    assert first['discount']==pytest.approx(.99**(.4/.6))
    assert second['discounted_components']['reports']==1.25
    assert first['undiscounted_reward']==pytest.approx(1.25-.005*.4/.6)


@pytest.fixture(scope='module')
def planner():return NavigationPlanner(get_map(),cache_size=96)


def quiet(env):
    env.match.controllers={pid:SimpleNamespace(decide=lambda obs:Intent(IntentKind.WAIT)) for pid in env.match.controllers}


def voting(env):
    g=env.match.game;g.phase=Phase.VOTING;g.phase_started_tick=g.tick;g.meeting_id='fixture'
    g.participants=tuple(pid for pid,p in g.players.items() if p.active);g._invalidate();env._observe();env._packet()


@pytest.mark.parametrize('correct,expected',[(True,2.),(False,-1.)])
def test_real_accepted_votes_reward_precisely(planner,correct,expected):
    env=LearningEnv(rewards=RewardConfig(),planner=planner,lobby=LobbySettings(timeout=2.));env.reset(seed=2730000);quiet(env);voting(env)
    g=env.match.game
    target=next(pid for pid,p in g.players.items() if pid!=env.focal and (p.role is Role.IMPOSTOR)==correct)
    _,_,_,_,info=env.step_scripted(Intent(IntentKind.VOTE,target))
    key='correct_votes' if correct else 'incorrect_votes'
    assert info['reward_components'][key]==expected and g.votes[env.focal]==target
    with pytest.raises(ValueError):env.step_scripted(Intent(IntentKind.VOTE,target))
    assert env.match.ledger.counts[key]==1


def test_real_report_once_no_approach_credit_and_reset_isolation(planner):
    env=LearningEnv(rewards=RewardConfig(),planner=planner,lobby=LobbySettings(timeout=3.));env.reset(seed=2730001);quiet(env)
    g=env.match.game;killer=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
    victim=next(p for pid,p in g.players.items() if p.role is Role.CREWMATE and pid!=env.focal)
    killer.motion.spawn((-.7,-2.8));victim.motion.spawn((-1.4,-2.8));g.players[env.focal].motion.spawn((-.7,-3.0))
    g._eliminate(killer,victim);g._invalidate();env._observe();env._packet()
    _,_,_,_,info=env.step_scripted(Intent(IntentKind.REPORT,victim.player_id))
    assert info['reward_components']['reports']==1.25 and g.metrics['reports']==1
    with pytest.raises(ValueError):env.step_scripted(Intent(IntentKind.REPORT,victim.player_id))
    old=env.match.ledger;env.reset(seed=2730002)
    assert env.match.ledger is not old and not env.match.ledger.reports and not env.match.ledger.frames


def test_legacy_disabled_reward_and_actor_packets_reproduce(planner):
    a=CrewmateV2Env(planner=planner,lobby=LobbySettings(timeout=3.))
    b=LearningEnv(planner=planner,lobby=LobbySettings(timeout=3.))
    pa,_=a.reset(seed=2730003);pb,_=b.reset(seed=2730003)
    while True:
        assert all(np.array_equal(pa[k],pb[k]) for k in pa)
        pa,ra,da,_,ia=a.step(0);pb,rb,db,_,ib=b.step(0)
        assert ra==pytest.approx(rb,abs=1e-10) and da==db
        assert ia['undiscounted_reward']==pytest.approx(ib['undiscounted_reward'])
        if da:break
    assert canonical_bytes(a.match.game.event_log)==canonical_bytes(b.match.game.event_log)


def test_forced_wait_coalescing_exact_discount_rewards_events_and_memory(planner):
    envs=[LearningEnv(rewards=RewardConfig(),coalesce=c,planner=planner,lobby=LobbySettings(timeout=2.)) for c in (False,True)]
    for env in envs:
        env.reset(seed=2730004);quiet(env);voting(env)
        env.match.game._vote(env.focal,None);env.match.game._invalidate();env._observe();env._packet()
        assert len(env.options)==1
    a,b=envs;total=0.;discount=1.
    while True:
        _,r,done,_,info=a.step(0);total+=discount*r;discount*=info['discount']
        if done:break
    _,reward,done,_,info=b.step(0)
    assert done and info['coalesced_forced_windows']>0
    assert reward==pytest.approx(total) and info['discount']==pytest.approx(discount)
    assert canonical_bytes(a.match.game.event_log)==canonical_bytes(b.match.game.event_log)
    assert a.memory.events==b.memory.events


def test_reward_truth_cannot_enter_actor_packet_or_force_a_vote(planner):
    env=LearningEnv(rewards=RewardConfig(),planner=planner,lobby=LobbySettings(timeout=1.));packet,_=env.reset(seed=2730005);quiet(env);voting(env)
    _,_,_,_,info=env.step_scripted(Intent(IntentKind.SKIP))
    assert env.match.game.votes[env.focal] is None and info['reward_counts']['skips']==1
    from crewmate_v2.learning.policy import check_packet
    with pytest.raises(ValueError):check_packet({**packet,'reward_counts':info['reward_counts']})
    x=ledger();y=RewardLedger('f',{'f':Role.CREWMATE,'i':Role.CREWMATE,'c':Role.IMPOSTOR},RewardConfig())
    a=Intent(IntentKind.VOTE,'i')
    assert x.consume([vote()],active=True,intent=a,legal=True).components['correct_votes']==2
    assert y.consume([vote()],active=True,intent=a,legal=True).components['incorrect_votes']==-1
    # Both scorers see the same actor data; changing trainer truth did not mutate it.
    assert set(packet)=={'global','players','actions','targets','mask','own_tasks_v2','task_targets'}


@pytest.mark.parametrize('shaped',[False,True])
def test_real_ghost_completion_and_death_settlement_keep_team_progress(planner,shaped):
    env=LearningEnv(rewards=RewardConfig() if shaped else None,planner=planner,lobby=LobbySettings(timeout=1.))
    env.reset(seed=2730013);quiet(env);g=env.match.game;focal=g.players[env.focal]
    task=min(focal.tasks,key=lambda t:t.task_id);task.stage=len(task.plan.stages)-1
    task.work=task.duration-.1;task.ready_at=0.
    focal.motion.spawn(g._ghost_goals[task.console_id]);g._invalidate();env._observe();env._packet()
    killer=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
    original=g.step;first=True
    def controlled(actions=None):
        nonlocal first
        if first:g._eliminate(killer,focal);first=False
        return original(actions)
    g.step=controlled
    _,_,done,_,info=env.step(0)
    assert done and info['episode']['ghost_tasks']==1 and info['episode']['alive_tasks']==0
    assert env.match.ledger.components['ghost_tasks']==(0. if shaped else 1.)
    assert info['episode']['reward_counts']['death']==1 and task.progress==1.
    assert g.observe_v2(next(pid for pid,p in g.players.items() if p.active)).task_bar.completed>=1


def test_real_simultaneous_report_loser_has_no_reward(planner):
    env=LearningEnv(rewards=RewardConfig(),planner=planner,lobby=LobbySettings(timeout=2.))
    for seed in range(2730014,2730020):
        env.reset(seed=seed)
        if env.focal!='p0':break
    assert env.focal!='p0';quiet(env);g=env.match.game
    killer=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
    victim=next(p for pid,p in g.players.items() if p.role is Role.CREWMATE and pid not in (env.focal,'p0'))
    victim.motion.spawn((-1.4,-2.8));killer.motion.spawn((-.7,-2.8));g._eliminate(killer,victim)
    for pid in (env.focal,'p0'):g.players[pid].motion.spawn((-.7,-3.0))
    g._invalidate();env._observe();env._packet();original=g.step;first=True
    def simultaneous(actions=None):
        nonlocal first
        merged=dict(actions or {})
        if first:merged['p0']=Intent(IntentKind.REPORT,victim.player_id);first=False
        return original(merged)
    g.step=simultaneous
    _,_,_,_,info=env.step_scripted(Intent(IntentKind.REPORT,victim.player_id))
    assert g.metrics['reports']==1 and g.metrics['illegal_actions']==0
    assert env.match.ledger.counts['reports']==0 and info['reward_components'].get('reports',0)==0
