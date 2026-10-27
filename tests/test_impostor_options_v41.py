"""Real-spawn regression missed by V4, bounded arming and strict PPO replay."""
from dataclasses import replace
import pytest
import torch
from social_deduction.actor import Phase,Point,PlayerSeen,Role
from social_deduction.phase3_api import Intent,IntentKind
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.learning import packet_hash
from impostor_options_v41.env import OptionsEnv
from impostor_options_v41.shaping import PursuitPotential
from impostor_options_v41.learning import Session,load


def synthetic_evaluation_worker(output,study,method,partition,checkpoint):
    """Spawn/IO fixture only; no simulator, model or actual evaluation cases."""
    import json
    from pathlib import Path
    assert study=={'synthetic_fixture':True}
    folder=Path(output)/partition;folder.mkdir(exist_ok=True)
    rows=[dict(method=method,fixture_index=i,synthetic=True) for i in range(20)]
    for i,row in enumerate(rows):(folder/f'{method}-fixture-{i}.json').write_text(json.dumps(row))
    return rows


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return FrozenModels()


def sight(base,t,x,players,cooldown=0.,phase=Phase.ROAMING):
    return replace(base,time_seconds=float(t),tick=round(t/.2),kill_cooldown=float(cooldown),
        own=replace(base.own,position=Point(x,0.)),context=replace(base.context,phase=phase),
        visible_players=tuple(PlayerSeen(pid,Point(pos,0.),'Cafeteria',Point(0.,0.),False) for pid,pos in players.items()))


def test_crowded_cooldown_cannot_saturate_pursuit(models):
    env=OptionsEnv(models=models);env.reset(seed=3500010);obs=env.envelope.actor
    assert obs.visible_players and obs.kill_cooldown==45.
    assert env.pursuit.best==env.pursuit.credit==0.
    p=PursuitPotential();assert p.advance(sight(obs,0,0.,{'a':.7},45))==0 and p.best==0
    assert p.advance(sight(obs,44,0.,{'a':.7},1))==0 and p.best==0
    assert p.advance(sight(obs,45,0.,{'a':5.}))==0
    assert p.advance(sight(obs,46,.5,{'a':5.}))>0


def test_original_45_second_spawn_then_real_visible_approach(models):
    env=OptionsEnv(models=models,coalesce=False);env.reset(seed=3500010)
    # Stationary crew are a test fixture only: no trained policy/accuracy claim.
    class Wait:
        inference_calls=0
        def decide(self,obs):return Intent(IntentKind.WAIT)
        def _observe(self,obs):pass
    env.match.controllers={pid:Wait() for pid in env.match.controllers}
    reactor=next(i for i,c in enumerate(env.options) if c.kind=='go_to_room' and c.intent.target=='Reactor')
    env.step(reactor)
    while env.match.game.time<45.-1e-8:env.step(0)
    assert env.envelope.actor.kill_cooldown==0. and env.pursuit.credit==0.
    # A room goal stops at its boundary, which need not reveal anyone inside.
    # A public Admin console route passes through the occupied Cafeteria corridor.
    route=next(i for i,c in enumerate(env.options) if c.kind=='go_to_task' and c.region=='Admin')
    env.step(route)
    while env.match.game.time<110 and env.pursuit.positive_total==0:
        choices=[i for i,c in enumerate(env.options) if c.kind=='go_to_location']
        env.step(choices[0] if choices else 0)
    assert 0<env.pursuit.positive_total<=.05+1e-12
    assert env.match.game.metrics['eliminations']==0  # shaping does not force a kill
    assert env.observation_space.contains(env.packet)
    # Controlled terminal verifies debit of all credited potential, not a real win.
    before=sum(env.reward_totals[k] for k in ('pursuit',))
    env.match.game._finish(Role.CREWMATE,'TASKS_COMPLETED')
    _,_,done,_,info=env.step(0)
    assert done and env.reward_totals['pursuit']==pytest.approx(0,abs=1e-12)


def test_budget_not_refilled_by_cooldown_switching_oscillation_or_meeting(models):
    env=OptionsEnv(models=models);env.reset(seed=3500010);base=env.envelope.actor;p=PursuitPotential()
    paid=[]
    for t,x in enumerate([0.,.5,1.,1.5,2.,2.5,3.,3.5]):paid.append(p.advance(sight(base,t,x,{'a':5.})))
    credit=p.credit;total=p.positive_total
    p.advance(sight(base,20,0.,{'a':.5},45));assert p.credit==credit and p.positive_total==total
    for t in range(21,221):
        pos={'a':5.} if t%4<2 else {'b':5.}
        paid.append(p.advance(sight(base,t,.5*(t%2),pos)))
    assert p.positive_total<=.05+1e-12
    prior=(p.credit,p.best,p.positive_total)
    p.advance(sight(base,222,0.,{},phase=Phase.DISCUSSION));assert prior==(p.credit,p.best,p.positive_total)
    assert p.advance(sight(base,223,0.,{'c':1.}))==0
    assert p.advance(sight(base,224,0.,{'c':.5}))==0
    p.advance(sight(base,225,0.,{},phase=Phase.FINISHED))
    assert p.credit==0.


def test_new_schema_gradients_exact_resume_and_old_checkpoint(models,tmp_path):
    a=Session(71,models=models);initial=weight_hash(a.policy);frozen=models.hashes()
    row=a.update(8);assert row['actor_gradient_norm']>0 and weight_hash(a.policy)!=initial
    path=tmp_path/'resume.pt';a.save(path);b=Session.restore(path,models)
    assert a.env.summary()==b.env.summary()
    a.update(8);b.update(8)
    assert weight_hash(a.policy)==weight_hash(b.policy) and packet_hash(a.packet)==packet_hash(b.packet)
    assert a.env.summary()==b.env.summary() and models.hashes()==frozen
    policy_only=tmp_path/'policy.pt';a.save(policy_only,False);policy,metadata=load(policy_only,models)
    assert metadata['schema']=='impostor-options-v41' and policy.act(a.packet)==a.policy.act(a.packet)
    from impostor_options_v4.learning import Session as OldSession
    old_session=OldSession(71,models=models,development=True)
    old_path=tmp_path/'old-v4.pt';old_session.save(old_path,False)
    old,_=load(old_path,models)
    assert sum(p.numel() for p in old.parameters())==75075
    models.assert_unchanged()


def test_two_isolated_cpu_worker_preflight(models,tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from time import time
    from impostor_options_v41.learning import record_for
    from impostor_options_v41.study import train_seed
    record=record_for(models)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_seed,str(tmp_path),record,time()+180,s,False,True,8) for s in (71,83)]
        rows=[f.result(timeout=180) for f in futures]
    assert {r['seed'] for r in rows}=={71,83}
    assert all(r['decisions']==8 and r['development'] and r['policy_changed'] for r in rows)
    assert all(r['frozen_weights_unchanged'] for r in rows)


def test_parallel_evaluation_schedule_collects_all_methods_before_return(tmp_path,monkeypatch):
    import impostor_options_v41.study as study
    monkeypatch.setattr(study,'evaluate_worker',synthetic_evaluation_worker)
    methods=[(f'candidate{i}',None) for i in range(4)]
    result=study.evaluate_parallel(tmp_path,{'synthetic_fixture':True},methods,'selection')
    assert set(result)=={m for m,_ in methods}
    assert all(len(rows)==20 for rows in result.values())
    assert len(list((tmp_path/'selection').glob('*.json')))==80
