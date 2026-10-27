"""Prelaunch telemetry equivalence, genuine PPO gradients and exact V4 resume."""
from pathlib import Path
import pytest
import torch
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.strategic_v2 import StrategicEnv
from impostor_research.learning import packet_hash
from impostor_options_v4.env import OptionsEnv
from impostor_options_v4.diagnostics import DiagnosticEnv,LegacyEnv
from impostor_options_v4.study import TrainingSession,choose,SEEDS,SCREEN,CONFIRM,BUDGET,registered_record


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return FrozenModels()


def fake_fixture(cls,models):
    env=cls(models=models,coalesce=False);env.reset(seed=3500010);g=env.match.game
    point=next(d.standing for d in g.map.destinations if d.category=='task')
    g.players[env.match.focal].motion.spawn(point);g.players[env.match.focal].cooldown=0
    g._invalidate();env._observe()
    return env


@pytest.mark.parametrize('base,audit',[(OptionsEnv,DiagnosticEnv),(StrategicEnv,LegacyEnv)])
def test_telemetry_does_not_change_game_or_actor(base,audit,models):
    results=[]
    for cls in (base,audit):
        env=fake_fixture(cls,models);rewards=[]
        for _ in range(9):
            action=next(i for i,c in enumerate(env.options) if c.kind=='fake_task')
            _,r,_,_,_=env.step(action);rewards.append(r)
        results.append((packet_hash(env.packet),env.actor_trace.hexdigest(),env.match.trace.hexdigest(),
                        env.summary()['event_hash'],rewards,env.summary()))
    assert results[0][:5]==results[1][:5]
    for k,v in results[0][5].items():assert results[1][5][k]==v
    assert results[1][5]['fake_selections']==9 and results[1][5]['max_fake_selection_streak']==9
    assert results[1][5]['fake_activity_seconds']>4.
    assert results[1][5]['ready_faking_seconds']==pytest.approx(results[1][5]['fake_activity_seconds'])


def test_real_v4_ppo_gradient_and_resume(models,tmp_path):
    a=TrainingSession(71,models=models,development=True)
    before=weight_hash(a.policy);frozen=models.hashes();row=a.update(8)
    assert row['actor_gradient_norm']>0 and row['critic_gradient_norm']>0
    assert weight_hash(a.policy)!=before and models.hashes()==frozen
    path=tmp_path/'resume.pt';a.save(path)
    b=TrainingSession.restore(path,models)
    assert a.env.summary()==b.env.summary()
    a.update(8);b.update(8)
    assert weight_hash(a.policy)==weight_hash(b.policy)
    assert packet_hash(a.packet)==packet_hash(b.packet) and a.episode_actions==b.episode_actions
    assert a.env.summary()==b.env.summary() and models.hashes()==frozen


def test_preregistered_selection_and_partitions(models):
    assert BUDGET==14400 and SEEDS==(71,83)
    assert len(SCREEN)==len(CONFIRM)==20 and not set(SCREEN)&set(CONFIRM)
    assert all(3200100<=s<=3200139 for s in (*SCREEN,*CONFIRM))
    candidates=[dict(wins=1,kills=2,timeouts=0,step=8192,seed=71),
                dict(wins=2,kills=1,timeouts=0,step=4096,seed=83),
                dict(wins=2,kills=1,timeouts=0,step=4096,seed=71)]
    assert choose(candidates)==candidates[-1]
    record=registered_record(models)
    assert record['approval']['message']=='go on' and record['approval']['budget_seconds']==14400
    assert record['initialization'].startswith('scratch') and not record['final_tests_opened']
