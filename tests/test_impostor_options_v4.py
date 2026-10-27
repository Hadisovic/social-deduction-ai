"""V4 boundaries, private-state isolation, fake options and bounded shaping.

Only development fixtures and tiny collection/replay checks; no PPO optimizer steps
or performance studies are invoked by this suite.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import pytest
import numpy as np
import torch
from social_deduction.actor import Phase,Role,Point,PlayerSeen
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.learning import packet_hash
from impostor_research.learning_v2 import load as load_v2
from impostor_research.strategic_v2 import StrategicEnv,FamilyPolicy
from impostor_options_v4.config import Rewards,ready_seconds
from impostor_options_v4.env import OptionsEnv
from impostor_options_v4.shaping import PursuitPotential
from impostor_options_v4.learning import Session,load,verify


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return FrozenModels()


def env_at_fake(models,cls=OptionsEnv):
    env=cls(models=models,coalesce=False);env.reset(seed=3500010)
    game=env.match.game;own=game.players[env.match.focal]
    destination=next(d for d in game.map.destinations if d.category=='task')
    own.motion.spawn(destination.standing);game._invalidate();env._observe()
    return env,next(c.intent.target for c in env.options if c.kind=='fake_task')


def choose(env,kind,target=None):
    return next(i for i,c in enumerate(env.options) if c.kind==kind and (target is None or c.intent.target==target))


@pytest.mark.parametrize('cooldown,elapsed,expected',[(45,.2,0),(.3,.2,0),(.2,.2,0),(.1,.2,.1),(0,.2,.2),(0,0,0)])
def test_exact_ready_overlap(cooldown,elapsed,expected):
    assert ready_seconds(elapsed,cooldown,roaming=True,active=True)==pytest.approx(expected)


@pytest.mark.parametrize('flags',[dict(roaming=False,active=True),dict(roaming=True,active=False),
                                    dict(roaming=True,active=True,entered_meeting=True)])
def test_no_ready_penalty_during_meeting_or_after_elimination(flags):
    assert ready_seconds(.2,0,**flags)==0.


def test_no_ready_charge_on_successful_kill_reset():
    assert ready_seconds(.2,0,roaming=True,active=True,killed=True)==0.


@pytest.mark.parametrize('cooldown,expected',[(45,0),(.3,-.006),(0,-.012)])
def test_real_simulated_reward_without_nearby_crew(models,cooldown,expected):
    env=OptionsEnv(models=models,coalesce=False);env.reset(seed=3500010);g=env.match.game
    own=g.players[env.match.focal];own.motion.spawn((-.7,-2.8));own.cooldown=cooldown
    for p in g.players.values():
        if p.role is Role.CREWMATE:p.motion.spawn(next(d.standing for d in g.map.destinations if d.room=='Reactor'))
    g._invalidate();env._observe();assert not env.envelope.actor.visible_players
    _,r,done,_,info=env.step(0)
    assert not done and r==pytest.approx(expected)
    assert info['elapsed_seconds']==pytest.approx(.6)
    assert r==pytest.approx(sum(info['reward_components'].values()))


def test_real_meeting_pauses_timer_and_resumption_boundary(models):
    env=OptionsEnv(models=models,coalesce=False);env.reset(seed=3500010);g=env.match.game
    own=g.players[env.match.focal];own.cooldown=.1
    g._start_meeting(own);g._invalidate();env._observe()
    _,r,_,_,info=env.step(0)
    assert info['cooldown_ready_seconds']==0 and r==0 and own.cooldown==.1
    g.phase=Phase.VOTING;g.phase_started_tick=g.tick-math.ceil(g.config.voting_seconds/g.config.dt)+1
    for pid in g.participants:g._vote(pid,None)
    g._invalidate();env._observe();_,r,_,_,info=env.step(0)
    # Meeting resolution tick is excluded; only the following two roaming ticks count.
    assert g.phase is Phase.ROAMING
    assert info['cooldown_ready_seconds']==pytest.approx(.3)
    assert r==pytest.approx(-.006)


def test_kill_reward_and_45_second_engine_cooldown(models):
    env=OptionsEnv(models=models);env.reset(seed=3500010);g=env.match.game
    own=g.players[env.match.focal];crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
    for p,room in zip(crew[1:],('Reactor','Navigation','Communications')):
        p.motion.spawn(next(d.standing for d in g.map.destinations if d.room==room))
    own.motion.spawn((-.7,-2.8));crew[0].motion.spawn((-1.4,-2.8));own.cooldown=0
    g._invalidate();env._observe();_,r,done,_,info=env.step(choose(env,'kill'))
    assert not done and r==pytest.approx(3.) and info['kill_bonus']==3
    assert own.cooldown==pytest.approx(45-info['elapsed_seconds'])
    assert g.config.kill_cooldown==45 and info['cooldown_ready_seconds']==0


def test_ejection_extra_and_terminal_once(models):
    env=OptionsEnv(models=models,coalesce=False);env.reset(seed=3500010);g=env.match.game
    own=g.players[env.match.focal];own.cooldown=0
    g._start_meeting(own);g.phase=Phase.VOTING
    g.phase_started_tick=g.tick-math.ceil(g.config.voting_seconds/g.config.dt)+1
    for pid in g.participants:g._vote(pid,env.match.focal)
    g._invalidate();env._observe();_,r,done,_,info=env.step(0)
    assert done and g.result.reason=='IMPOSTOR_EJECTED'
    assert r==pytest.approx(-16.)
    assert info['reward_components']==dict(terminal=-10.,kill=0.,ejection=-6.,cooldown_ready=0.,pursuit=0.)
    with pytest.raises(RuntimeError):env.step(0)


@pytest.mark.parametrize('winner,reason,expected',[(None,'TIMEOUT',-12.),(Role.CREWMATE,'TASKS_COMPLETED',-10.),(Role.IMPOSTOR,'PARITY',10.)])
def test_terminal_reward_accounting_without_elapsed_time(models,monkeypatch,winner,reason,expected):
    env=OptionsEnv(models=models);env.reset(seed=3500010);g=env.match.game
    g.players[env.match.focal].cooldown=0
    monkeypatch.setattr(env.match,'tick',lambda intent:g._finish(winner,reason))
    _,r,done,_,info=env.step(0)
    assert done and r==expected and info['elapsed_seconds']==0 and info['cooldown_ready_seconds']==0


def test_fake_repeated_selection_completes_and_can_restart_deliberately(models):
    env,target=env_at_fake(models)
    remaining=[]
    for _ in range(7):
        env.step(choose(env,'fake_task',target))
        remaining.append(env.match.game.players[env.match.focal].interaction_remaining)
        assert env.observation_space.contains(env.packet)
    summary=env.summary()
    assert remaining[:6]==pytest.approx([3.4,2.8,2.2,1.6,1.,.4])
    assert summary['fake_starts']==1 and summary['fake_completions']==1
    assert summary['fake_repeated_continuations']==6
    assert env.envelope.actor.interaction_task is None
    env.step(choose(env,'fake_task',target))
    assert env.summary()['fake_starts']==2


@pytest.mark.parametrize('exit_kind',['cancel_navigation','go_to_room'])
def test_neural_fake_abandonment(models,exit_kind):
    env,target=env_at_fake(models);env.step(choose(env,'fake_task',target))
    env.step(choose(env,exit_kind))
    assert env.envelope.actor.interaction_task is None
    assert env.summary()['fake_interruptions']==1 and env.summary()['fake_completions']==0


def test_meeting_interrupts_fake_and_does_not_resume_it_automatically(models):
    env,target=env_at_fake(models);env.step(choose(env,'fake_task',target))
    g=env.match.game;g._start_meeting(g.players[env.match.focal]);g._invalidate();env._observe()
    assert env.envelope.actor.interaction_task is None
    assert env.summary()['fake_interruptions']==1
    env.step(0);assert env.envelope.actor.interaction_task is None


def test_fake_repeat_replay_and_continue_have_identical_physics(models):
    records=[]
    for repeat in (True,False,True):
        env,target=env_at_fake(models);rewards=[]
        for j in range(7):
            _,r,_,_,info=env.step(choose(env,'fake_task',target) if j==0 or repeat else 0)
            rewards.append((r,info['reward_components']))
        records.append((packet_hash(env.packet),env.match.trace.hexdigest(),rewards,
                        env.summary()['event_hash'],env.summary()['fake_completions']))
    assert records[0]==records[1]==records[2]


def sight(base,t,x,positions,phase=Phase.ROAMING):
    return replace(base,time_seconds=float(t),tick=round(t/.2),own=replace(base.own,position=Point(x,0.)),
        context=replace(base.context,phase=phase),visible_players=tuple(PlayerSeen(pid,Point(pos,0.),'Cafeteria',Point(0.,0.),False) for pid,pos in positions.items()))


def test_pursuit_meaningful_progress_and_terminal_telescoping(models):
    env=OptionsEnv(models=models);env.reset(seed=3500010);base=env.envelope.actor
    potential=PursuitPotential();payments=[]
    for t,x in enumerate([0.,.4,.8,1.2,1.6,2.,2.4,2.8,3.2,3.6,4.]):
        payments.append(potential.advance(sight(base,t,x,{'crew':5.})))
    assert any(v>0 for v in payments) and sum(payments)<=.05+1e-12
    assert potential.advance(sight(base,12,4.,{'crew':5.}))==0
    payments.append(potential.advance(sight(base,13,4.,{},Phase.FINISHED)))
    assert sum(payments)==pytest.approx(0,abs=1e-12)


def test_pursuit_cannot_farm_oscillation_switching_or_visibility(models):
    env=OptionsEnv(models=models);env.reset(seed=3500010);base=env.envelope.actor
    p=PursuitPotential();paid=[]
    for t in range(200):
        visible={'a':5.} if t%4<2 else {'b':5.}
        paid.append(p.advance(sight(base,t,.5*(t%2),visible)))
    first_total=sum(paid)
    assert 0<=first_total<=.05
    for t in range(200,400):
        paid.append(p.advance(sight(base,t,.5*(t%2),{'a':5.})))
    assert sum(paid)<=.05+1e-12 and p.positive_total<=.05+1e-12
    # A meeting cannot replenish the lifetime credit or proximity high-water mark.
    before=(p.credit,p.best,p.positive_total)
    p.advance(sight(base,400,0.,{},Phase.DISCUSSION))
    assert before==(p.credit,p.best,p.positive_total)
    assert p.advance(sight(base,401,0.,{'new':1.}))==0
    assert p.advance(sight(base,402,0.,{'new':.5}))==0  # target moves, actor does not
    assert p.advance(sight(base,403,0.,{}))==0
    assert p.advance(sight(base,404,0.,{'new':.5}))==0


def test_pursuit_small_motion_is_not_paid(models):
    env=OptionsEnv(models=models);env.reset(seed=3500010);base=env.envelope.actor;p=PursuitPotential()
    assert p.advance(sight(base,0,0.,{'a':5.}))==0
    assert p.advance(sight(base,1,.01,{'a':5.}))==0


def test_unseen_hidden_positions_do_not_affect_features_or_shaping(models):
    env=OptionsEnv(models=models);env.reset(seed=3500012);g=env.match.game
    own=g.players[env.match.focal];own.motion.spawn((-.7,-2.8))
    crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
    for p in crew:p.motion.spawn(next(d.standing for d in g.map.destinations if d.room=='Reactor'))
    g._invalidate();env._observe();before={k:v.copy() for k,v in env.packet.items()}
    potential=(env.pursuit.credit,env.pursuit.best,env.pending_pursuit)
    crew[0].motion.spawn(next(d.standing for d in g.map.destinations if d.room=='Navigation'))
    g._invalidate();env._observe()
    for k in before:np.testing.assert_array_equal(before[k],env.packet[k])
    assert potential==(env.pursuit.credit,env.pursuit.best,env.pending_pursuit)


def test_historical_v2_fake_behavior_and_checkpoints_unchanged(models):
    env,target=env_at_fake(models,StrategicEnv)
    for _ in range(7):env.step(choose(env,'fake_task',target))
    events=[e for e in env.match.game.event_log if e.get('player_id')==env.match.focal]
    assert sum(e['kind']=='fake_task_start' for e in events)==7
    assert not any(e['kind']=='fake_task_end' for e in events)
    for path in sorted(Path('artifacts/impostor-v2/study-20261027').glob('seed-*/*.pt')):
        policy,metadata=load_v2(path,models)
        assert sum(v.numel() for v in policy.parameters())==75075
    models.assert_unchanged()


def test_collection_exact_replay_checkpoint_and_architecture_no_optimizer_steps(models,tmp_path):
    a=Session(71,models=models);before=weight_hash(a.policy);frozen=models.hashes()
    a.collect(4)
    assert before==weight_hash(a.policy) and not a.updates
    path=tmp_path/'resume.pt';a.save(path)
    b=Session.restore(path,models)
    assert packet_hash(a.packet)==packet_hash(b.packet)
    assert a.env.summary()==b.env.summary()
    a.collect(4);b.collect(4)
    assert a.episode_actions==b.episode_actions and packet_hash(a.packet)==packet_hash(b.packet)
    assert a.env.summary()==b.env.summary() and models.hashes()==frozen
    assert sum(v.numel() for v in a.policy.parameters())==75075
    assert a.policy.state_dict().keys()==FamilyPolicy().state_dict().keys()
    with pytest.raises(FileExistsError):a.save(path)
    inference=tmp_path/'policy.pt';a.save(inference,False);policy,_=load(inference,models)
    assert policy.act(a.packet)==a.policy.act(a.packet)
    changed=deepcopy(a.study_record);changed['options']['version']='unversioned'
    with pytest.raises(ValueError,match='version'):verify(changed)
    with pytest.raises(PermissionError):Session(71,models=models,development=False)


def test_actual_old_checkpoint_new_viewer_transfer_label(models,tmp_path,monkeypatch,capsys):
    from run_impostor_options import main
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    assert main(['--max-frames','2','--capture',str(tmp_path/'capture')])==0
    assert 'TRANSFER DEMO' in capsys.readouterr().out
    assert (tmp_path/'capture/impostor.png').is_file()


def test_reserved_final_viewer_partition_rejected():
    from run_impostor_options import main
    with pytest.raises(SystemExit):main(['--seed','3300000'])
