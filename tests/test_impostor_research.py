"""Development-only fixtures: no optimizer, training or reserved evaluation seeds."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from shapely.geometry import LineString, Point as ShapePoint
from among_us_map import get_map
from navigation_service import NavigationPlanner
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.engine import CrewmateV2Game
from crewmate_v2.runtime import PublicMemory
from phase3_bots import ScriptedController
from phase3_engine import Body
from phase5_features import choices, encode
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import Intent, IntentKind, WitnessedElimination, intent_allowed
from impostor_research.frozen import FrozenModels, FrozenCrew
from impostor_research.opponents import OpponentGuard, Safeguards
from impostor_research.env import ImpostorEnv, RewardConfig
from impostor_research.features import candidates, encode as impostor_encode
from impostor_research.memory import ImpostorMemory
from impostor_research.policy import ImpostorPolicy


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1)
    return FrozenModels()


@pytest.fixture(scope='module')
def planner(): return NavigationPlanner(get_map(),cache_size=96)


def position(g,p,xy):
    p.motion.spawn(xy); g._invalidate()


def scene(planner, witnessed=True, far=False):
    g = CrewmateV2Game(3500000,planner=planner,mixture=MatchMixture(1.))
    killer = next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
    victim,witness,other,last = [p for p in g.players.values() if p.role is Role.CREWMATE]
    for p,room in ((witness,'Reactor'),(other,'Navigation'),(last,'Communications')):
        position(g,p,next(d.standing for d in g.map.destinations if d.room==room))
    position(g,killer,(-.7,-2.8)); position(g,victim,(-1.4,-2.8)); killer.cooldown=0.
    if witnessed: position(g,witness,(-1.02,-4.8) if far else (-2.,-2.8))
    g._invalidate(); g.step({killer.player_id:Intent(IntentKind.KILL,victim.player_id)})
    if not witnessed: position(g,witness,(-1.02,-4.8) if far else (-2.,-2.8))
    return g,killer,victim,witness,other,last


def proposed(controller, envelope):
    return controller.decide(envelope) if isinstance(controller,FrozenCrew) else controller.decide(envelope.actor)


@pytest.mark.parametrize('learned',[True,False])
@pytest.mark.parametrize('witnessed',[True,False])
def test_real_controller_body_approach_report_and_witness_vote(models,planner,learned,witnessed):
    g,k,v,w,_,_ = scene(planner,witnessed,True)
    controller = FrozenCrew(models) if learned else ScriptedController(31)
    guard = OpponentGuard()
    first = guard.apply(g.observe_v2(w.player_id),proposed(controller,g.observe_v2(w.player_id)))
    assert first.kind is IntentKind.GO_TO_BODY
    g.step({w.player_id:first})
    for _ in range(40):
        envelope = g.observe_v2(w.player_id)
        a = guard.apply(envelope,proposed(controller,envelope))
        assert a.kind is IntentKind.WAIT or intent_allowed(envelope.actor,a)
        g.step({w.player_id:a})
        if g.phase is Phase.DISCUSSION: break
    assert g.phase is Phase.DISCUSSION and g.metrics['reports']==1
    assert bool(guard.witnesses)==witnessed
    while g.phase is Phase.DISCUSSION: g.step()
    envelope = g.observe_v2(w.player_id)
    original = proposed(controller,envelope)
    a = guard.apply(envelope,original)
    assert a.kind is IntentKind.WAIT or intent_allowed(envelope.actor,a)
    if witnessed: assert a == Intent(IntentKind.VOTE,k.player_id)
    else: assert a == original  # Discovery did not invent the killer.
    g.step({w.player_id:a})
    if witnessed: assert g.votes[w.player_id]==k.player_id
    models.assert_unchanged()


def test_frozen_raw_probe_ignores_legal_report_and_adapter_matches_legacy(models,planner):
    g,_,_,w,_,_ = scene(planner)
    c = FrozenCrew(models); envelope = g.observe_v2(w.player_id)
    assert any(a.kind is IntentKind.REPORT for a in envelope.actor.actions)
    a = c.decide(envelope)
    assert a.kind is not IntentKind.REPORT
    expected = encode(envelope.actor,c.memory,models.belief.predict(c.memory),choices(envelope.actor,c.memory))
    for key in expected: np.testing.assert_array_equal(c.last_packet[key],expected[key])
    assert all(not p.requires_grad and p.grad is None for p in models.policy.parameters())
    models.assert_unchanged()


@pytest.mark.parametrize('bodies,eyewitness',[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('witnessed',[False,True])
def test_all_four_guard_controls(planner,bodies,eyewitness,witnessed):
    g,k,v,w,_,_ = scene(planner,witnessed)
    guard = OpponentGuard(Safeguards(bodies,eyewitness)); wait = Intent(IntentKind.WAIT)
    action = guard.apply(g.observe_v2(w.player_id),wait)
    assert (action.kind is IntentKind.REPORT) == (bodies or eyewitness and witnessed)
    assert bool(guard.witnesses)==witnessed
    if guard.records: assert guard.records[-1]['proposed']==wait


def test_other_reports_first_ineligible_killer_and_death(planner):
    g,k,v,w,other,_ = scene(planner)
    guard = OpponentGuard(); guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.WAIT))
    position(g,other,(-2.,-2.8))
    g.step({other.player_id:Intent(IntentKind.REPORT,v.player_id)})
    while g.phase is Phase.DISCUSSION: g.step()
    assert guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.SKIP)).target==k.player_id
    g.participants = tuple(pid for pid in g.participants if pid != k.player_id); g._invalidate()
    assert guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.SKIP)).kind is IntentKind.SKIP
    w.status='dead';g._make_ghost(w);g._invalidate()
    before=dict(guard.witnesses)
    assert guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.SKIP)).kind is IntentKind.WAIT
    assert guard.witnesses==before


def test_wall_blocks_body_and_forced_action(planner):
    g,k,v,w,_,_=scene(planner,False)
    for record in g.map.data['walls']:
        points=record['points']
        found=False
        for a,b in zip(points,points[1:]+points[:1]):
            length=math.dist(a,b)
            if length<.5: continue
            center=tuple((x+y)/2 for x,y in zip(a,b)); normal=(-(b[1]-a[1])/length,(b[0]-a[0])/length)
            for offset in (.5,1.,1.5,2.):
                p=tuple(x+offset*n for x,n in zip(center,normal)); q=tuple(x-offset*n for x,n in zip(center,normal))
                if all(g.map.center_domain.covers(ShapePoint(x)) for x in (p,q)) and g.map.wall_lines.intersects(LineString((p,q))):
                    found=True;break
            if found: break
        if found: break
    assert found
    position(g,w,p);g.bodies[v.player_id]=Body(v.player_id,q,g.tick,k.player_id);g._invalidate()
    envelope=g.observe_v2(w.player_id)
    assert not envelope.actor.visible_bodies
    guard=OpponentGuard(); original=Intent(IntentKind.WAIT)
    assert guard.apply(envelope,original)==original and not guard.records


def test_no_evidence_original_action_and_memories_independent(models,planner):
    g=CrewmateV2Game(3500001,planner=planner)
    crew=[p.player_id for p in g.players.values() if p.role is Role.CREWMATE]
    a,b=FrozenCrew(models),FrozenCrew(models)
    action=a.decide(g.observe_v2(crew[0])); guard=OpponentGuard()
    assert guard.apply(g.observe_v2(crew[0]),action)==action
    assert not b.memory.events and b.memory is not a.memory
    with pytest.raises(ValueError): guard.apply(g.observe_v2(crew[1]),action)


def test_body_disappears_cancels_only_guard_pursuit(planner):
    g,_,v,w,_,_=scene(planner,False,True); guard=OpponentGuard()
    assert guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.WAIT)).kind is IntentKind.GO_TO_BODY
    g.bodies.clear();g._invalidate()
    assert guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.WAIT)).kind is IntentKind.CANCEL_NAVIGATION
    assert guard.apply(g.observe_v2(w.player_id),Intent(IntentKind.WAIT)).kind is IntentKind.WAIT


def test_impostor_interface_private_memory_mask_rng_and_inference(models,planner):
    env=ImpostorEnv(models=models,planner=planner); packet,_=env.reset(seed=3500002)
    assert env.observation_space.contains(packet)
    state=torch.random.get_rng_state().clone(); policy=ImpostorPolicy()
    assert torch.equal(state,torch.random.get_rng_state())
    assert policy.act(packet)==policy.act(packet)
    with pytest.raises(ValueError): policy.act(packet,deterministic=False)
    for option in env.options:
        assert option.intent is None or intent_allowed(env.envelope.actor,option.intent)
    bad=int(np.flatnonzero(packet['mask']==0)[0])
    with pytest.raises(ValueError):env.step(bad)
    a=ImpostorMemory(); b=ImpostorMemory(); a.update(env.envelope.actor)
    assert not b.events
    with pytest.raises(ValueError): a.update(env.match.game.observe(env.match.learned))
    models.assert_unchanged()


def test_hidden_positions_and_ghost_secrets_do_not_change_impostor_packet(models,planner):
    env=ImpostorEnv(models=models,planner=planner);env.reset(seed=3500003);g=env.match.game
    own=g.players[env.match.focal];position(g,own,(-.7,-2.8))
    crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
    remote=next(d.standing for d in g.map.destinations if d.room=='Reactor')
    for p in crew:position(g,p,remote)
    obs=g.observe_v2(own.player_id); assert not obs.actor.visible_players
    m=ImpostorMemory();m.update(obs.actor);pub=PublicMemory();pub.update(obs)
    before=impostor_encode(obs,m,pub,candidates(obs,m))
    position(g,crew[0],next(d.standing for d in g.map.destinations if d.room=='Navigation'))
    after_obs=g.observe_v2(own.player_id);assert obs==after_obs
    g._record('private_fixture_secret',killer='not-an-input')
    after=impostor_encode(after_obs,m,pub,candidates(after_obs,m))
    for key in before:np.testing.assert_array_equal(before[key],after[key])


@pytest.mark.parametrize('coalesce',[False,True])
def test_reset_determinism_and_terminal_absorbing(models,planner,coalesce):
    env=ImpostorEnv(models=models,planner=planner,lobby=LobbySettings(timeout=1.2),coalesce=coalesce)
    summaries=[]
    for _ in range(2):
        packet,_=env.reset(seed=3500004)
        while not env.finished: packet,r,done,_,info=env.step(0)
        summaries.append(env.summary())
        assert packet['mask'].sum()==1 and np.count_nonzero(packet['global_'])==0
        assert info['episode']['reason']=='TIMEOUT'
        with pytest.raises(RuntimeError):env.step(0)
    assert summaries[0]==summaries[1]


def test_forced_meeting_coalescing_preserves_ticks_observations_and_discount(models,planner):
    results=[]
    for coalesce in (False,True):
        env=ImpostorEnv(models=models,planner=planner,coalesce=coalesce,lobby=LobbySettings(timeout=20.))
        env.reset(seed=3500005);g=env.match.game
        g.phase=Phase.VOTING;g.meeting_id='fixture';g.participants=tuple(g.players);g.phase_started_tick=0
        g.votes={env.match.focal:None};g._invalidate();env._observe()
        r=0.;discount=1.
        while not env.finished:
            _,reward,_,_,info=(env.step(0) if coalesce else env.step_scripted(Intent(IntentKind.WAIT)))
            r+=discount*reward;discount*=info['discount']
        results.append((env.summary(),r,discount))
    for key in ('event_hash','action_hash','actor_hash','simulated_seconds','scheduled_windows'):
        assert results[0][0][key]==results[1][0][key],key
    assert results[0][1:]==pytest.approx(results[1][1:])


def test_kill_fake_report_claim_and_vote_are_legitimate_candidates(planner):
    g,k,v,w,_,_=scene(planner)
    def options():
        envelope=g.observe_v2(k.player_id)
        memory=ImpostorMemory();memory.update(envelope.actor)
        result=candidates(envelope,memory)
        assert all(c.intent is None or intent_allowed(envelope.actor,c.intent) for c in result)
        return result
    assert any(c.kind=='report' for c in options())
    k.cooldown=0.;position(g,w,(-1.4,-2.8));g._invalidate()
    assert any(c.kind=='kill' and c.target==w.player_id for c in options())
    console=next(d for d in g.map.destinations if d.category=='task')
    position(g,k,console.standing)
    assert any(c.kind=='fake_task' for c in options())
    emergency=next(d for d in g.map.destinations if d.name=='EmergencyConsole')
    position(g,k,emergency.standing)
    g.step({k.player_id:Intent(IntentKind.CALL_MEETING,emergency.id)})
    while g.phase is Phase.DISCUSSION and g.observe(k.player_id).context.speaker_id!=k.player_id:g.step()
    assert any(c.kind.startswith('claim_') for c in options())
    while g.phase is Phase.DISCUSSION:g.step()
    assert any(c.kind=='vote' for c in options()) and any(c.kind=='skip' for c in options())


def test_v2_only_information_does_not_enter_frozen_network(models,planner):
    from crewmate_v2.api import TaskBar
    g=CrewmateV2Game(3500006,planner=planner)
    pid=next(p.player_id for p in g.players.values() if p.role is Role.CREWMATE)
    obs=g.observe_v2(pid)
    a,_=FrozenCrew(models).packet(obs)
    b,_=FrozenCrew(models).packet(replace(obs,task_bar=TaskBar(1,16)))
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_registered_partitions_are_separate_from_historical_partitions():
    from impostor_research.protocol import PARTITIONS,development_seeds
    from phase6.config import PARTITIONS as old
    intervals=list(PARTITIONS.values())
    for i,(low,high) in enumerate(intervals):
        for a,b in intervals[i+1:]+list(old.values()):assert high<a or low>b
    assert development_seeds(2)==(3500000,3500001)
    with pytest.raises(ValueError):development_seeds(21)


def test_false_kill_claim_is_not_direct_witness(planner):
    from social_deduction.actor import Provenance
    from social_deduction.phase3_api import EventView,StructuredClaim,ClaimKind
    g,k,_,w,_,_=scene(planner,False)
    envelope=g.observe_v2(w.player_id)
    claim=EventView(g.tick,Provenance.CLAIM,StructuredClaim(ClaimKind.SAW_ELIMINATION,k.player_id,
          k.player_id,'Cafeteria',g.tick,g.tick))
    envelope=replace(envelope,actor=replace(envelope.actor,evidence=(*envelope.actor.evidence,claim)))
    guard=OpponentGuard(Safeguards(False,True))
    assert guard.apply(envelope,Intent(IntentKind.WAIT)).kind is IntentKind.WAIT
    assert not guard.witnesses


def test_candidate_network_is_equivariant_to_player_row_order(models,planner):
    env=ImpostorEnv(models=models,planner=planner);packet,_=env.reset(seed=3500007)
    policy=ImpostorPolicy().eval()
    perm=np.asarray([2,0,3,1]);inverse=np.argsort(perm)
    other={k:v.copy() for k,v in packet.items()};other['players']=packet['players'][perm]
    valid=other['targets']>=0;other['targets'][valid]=inverse[other['targets'][valid]]
    with torch.inference_mode():
        a,av=policy({k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()})
        b,bv=policy({k:torch.as_tensor(v).unsqueeze(0) for k,v in other.items()})
    torch.testing.assert_close(a.probs,b.probs);torch.testing.assert_close(av,bv)

