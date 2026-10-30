"""Run in the isolated Jane runtime; no full study, final sets or old model updates."""
from dataclasses import replace
import os
from pathlib import Path
import pytest

SOURCE = os.environ.get('JANE_UPSTREAM_REVIEW_ROOT')
if not SOURCE:
    pytest.skip('Explicit pinned upstream root required in an isolated invocation',allow_module_level=True)
from jane.impostor.v1.bootstrap import activate,experiment,frozen_crew
activate(SOURCE)
import numpy as np
import torch
torch.set_num_threads(1)
from patrick_jane.a2_environment import A2Game
from patrick_jane.memory import Memory
from social_deduction.phase3_api import Intent,IntentKind,intent_allowed,legal_actions
from navigation_service import NavigationPlanner
from jane.impostor.v1.features import OwnFeedback,Packet,Candidate,encode,KIND_FAMILY
from jane.impostor.v1.model import ImpostorPolicy
from jane.impostor.v1.session import Actor,Transition,run_match
from jane.impostor.v1.session import resolve_team_contention
from jane.impostor.v1.rewards import ready_seconds
from jane.impostor.v1.ppo import advantages,update
from jane.impostor.v1.run import tensor_hash,save_checkpoint,load_checkpoint

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    settings,config=experiment(ROOT/'jane/configs/patrick_jane_a2_12p.json')
    game=A2Game(9300010,experiment=config)
    owner=next(pid for pid,p in game.players.items() if p.role.value=='impostor')
    return settings,config,game,owner,NavigationPlanner(game.base_map)


def feedback(obs,game):
    p=game.players[obs.actor.own.player_id];task=p.interaction_task
    return OwnFeedback(p.player_id,obs.actor.match_id,obs.actor.tick,float(obs.actor.kill_cooldown),
        float(p.interaction_remaining) if task and task.startswith('fake:') else 0.,
        task[5:] if task and task.startswith('fake:') else None)


def packet(fixture,owner=None):
    settings,config,game,pid,planner=fixture;pid=owner or pid
    obs=game.observe_pj(pid)
    return encode(obs,feedback(obs,game),Memory(pid),planner,settings)


def test_real_packet_has_only_legal_roaming_candidates(fixture):
    game=fixture[2];pid=fixture[3];obs=game.observe_pj(pid)
    p=packet(fixture)
    assert p.self_features.shape==(192,) and p.players.shape==(14,24)
    assert p.actions.shape[1]==48 and len(p.candidates)>1
    for c in p.candidates:
        if type(c.payload) is Intent: assert intent_allowed(obs.actor,c.payload),(c.kind,c.key)
    assert all(c.key not in obs.teammates for c in p.candidates if c.kind=='kill')


def test_own_feedback_and_memory_cannot_change_actor(fixture):
    settings,_,game,pid,planner=fixture;obs=game.observe_pj(pid);f=feedback(obs,game)
    for broken in (replace(f,owner='other'),replace(f,tick=f.tick+1),replace(f,cooldown=f.cooldown+1)):
        with pytest.raises(ValueError): encode(obs,broken,Memory(pid),planner,settings)
    with pytest.raises(ValueError): encode(obs,f,Memory('other'),planner,settings)


def test_hidden_positions_are_not_read_into_actor_features(fixture):
    settings,_,game,pid,planner=fixture
    # Controlled fixture: move a non-ally outside the current view, then vary only
    # that unseen world position. No policy receives the world or this manipulation.
    other=next(p for p in game.players if p!=pid and game.players[p].role.value=='crewmate')
    import math
    own=game.players[pid].position
    positions=[(d.standing.x,d.standing.y) for d in game.observe_pj(pid).actor.destinations
               if math.dist(own,(d.standing.x,d.standing.y))>9]
    assert len(positions)>2
    game.players[other].motion.spawn(positions[0]);game._invalidate()
    obs1=game.observe_pj(pid)
    assert other not in {p.player_id for p in obs1.actor.visible_players}
    p1=encode(obs1,feedback(obs1,game),Memory(pid),planner,settings)
    game.players[other].motion.spawn(positions[-1]);game._invalidate()
    obs2=game.observe_pj(pid)
    assert other not in {p.player_id for p in obs2.actor.visible_players}
    p2=encode(obs2,feedback(obs2,game),Memory(pid),planner,settings)
    for key,value in p1.tensors().items(): assert np.array_equal(value,p2.tensors()[key]),key


def test_arrived_destination_is_masked_but_wait_and_revisit_remain(fixture):
    game=fixture[2];pid=fixture[3]
    destination=next(d for d in game.observe_pj(pid).actor.destinations if d.category=='task')
    game.players[pid].motion.spawn((destination.standing.x,destination.standing.y));game._invalidate()
    p=packet(fixture)
    assert p.candidates[0].kind=='wait'
    assert not any(c.kind=='destination' and c.key==destination.id for c in p.candidates)
    game.players[pid].motion.spawn((-2.,-3.));game._invalidate()
    assert any(c.kind=='destination' and c.key==destination.id for c in packet(fixture).candidates)


def test_fake_task_continues_finishes_and_can_be_cancelled(fixture):
    game=fixture[2];pid=fixture[3]
    destination=next(d for d in game.observe_pj(pid).actor.destinations if d.category=='task')
    game.players[pid].motion.spawn((destination.standing.x,destination.standing.y));game._invalidate()
    action=next(c.payload for c in packet(fixture).candidates if c.kind=='fake' and c.key==destination.id)
    game.step({pid:action})
    remaining=game.players[pid].interaction_remaining
    for _ in range(4):
        p=packet(fixture)
        assert not any(c.kind=='fake' and c.key==destination.id for c in p.candidates)
        assert any(c.kind=='cancel' for c in p.candidates)
        game.step({pid:Intent(IntentKind.WAIT)})
    assert game.players[pid].interaction_remaining<remaining
    for _ in range(20): game.step({pid:Intent(IntentKind.WAIT)})
    assert game.players[pid].interaction_task is None
    assert any(c.kind=='fake' and c.key==destination.id for c in packet(fixture).candidates)
    game.step({pid:action});game.step({pid:Intent(IntentKind.CANCEL_NAVIGATION)})
    assert game.players[pid].interaction_task is None


def test_private_rngs_accept_native_large_seeds_and_memories_are_separate(fixture):
    settings,_,game,pid,planner=fixture;model=ImpostorPolicy()
    ids=[p for p in game.players if game.players[p].role.value=='impostor']
    actors=[Actor(p,model,settings,planner,2**120+i) for i,p in enumerate(ids)]
    assert actors[0].memory is not actors[1].memory
    assert actors[0].hidden.data_ptr()!=actors[1].hidden.data_ptr()
    assert actors[0].rng is not actors[1].rng
    actor=actors[0];obs=game.observe_pj(ids[0]);command,row,_=actor.choose(obs,feedback(obs,game))
    assert row.packet.owner==ids[0] and torch.isfinite(actor.hidden).all()


def synthetic():
    af=np.zeros((2,48),np.float32);af[0,0]=1;af[1,8]=1
    return Packet(np.zeros(192,np.float32),np.zeros((14,24),np.float32),np.zeros(14,bool),
        np.zeros((64,16),np.float32),np.zeros(64,bool),af,np.array([0,KIND_FAMILY['kill']]),
        (Candidate('wait'),Candidate('kill')),'p0',0,0.,'roaming')


def test_empty_attention_has_finite_normalized_probabilities_and_gradients():
    model=ImpostorPolicy();p=synthetic();d,v,h=model([p],torch.zeros(1,256))
    assert model.architecture()['parameters']==1036561
    assert torch.isfinite(h).all() and torch.isfinite(v).all()
    assert float(d[0].probs.sum())==pytest.approx(1.)
    loss=-d[0].log_prob(torch.tensor(1))+v.square().mean();loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)


def test_elapsed_gae_allows_zero_duration_terminal():
    p=synthetic();q=replace(p,time=.6,tick=3)
    rows=[Transition(p,0,0,0,0),Transition(q,0,0,0,1)]
    adv,ret=advantages(rows,.6)
    assert ret[1]==pytest.approx(1.)
    assert adv[0]==pytest.approx(.995**.6)


def test_recurrent_optimizer_updates_only_the_new_policy():
    model=ImpostorPolicy();p=synthetic();q=replace(p,time=.6,tick=3)
    with torch.no_grad():
        d,v,_=model([p,q],torch.zeros(1,256),sequence=True)
    rows=[Transition(p,0,float(d[0].log_prob(torch.tensor(0))),float(v[0]),0),
          Transition(q,1,float(d[1].log_prob(torch.tensor(1))),float(v[1]),1)]
    before=tensor_hash(model);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    logs=update(model,optimizer,[rows],1.2,epochs=1,sequence=1)
    assert len(logs)==1 and tensor_hash(model)!=before
    assert all(np.isfinite(v) for row in logs for v in row.values())


def test_new_checkpoint_strict_load_rejects_different_settings(tmp_path,fixture):
    model=ImpostorPolicy();path=tmp_path/'fresh.pt';settings=fixture[0]
    save_checkpoint(path,model,None,settings,{},731)
    other=ImpostorPolicy(947);load_checkpoint(path,other,settings)
    assert tensor_hash(model)==tensor_hash(other)
    with pytest.raises(ValueError,match='mismatch'):
        load_checkpoint(path,other,replace(settings,profile_id='different'))


def test_bounded_real_match_replay_and_frozen_crew(fixture,tmp_path):
    settings,config,_,_,_=fixture
    crew=frozen_crew(Path(SOURCE));before=tensor_hash(crew);model=ImpostorPolicy()
    progress=[]
    summary,rows=run_match(model,crew,settings,config,9300011,output=tmp_path/'bounded.jsonl',
                         max_observations=6,max_wall_seconds=45,status=progress.append)
    assert summary['truncated'] and summary['winner'] is None
    assert summary['metrics']['illegal_actions']==0
    assert tensor_hash(crew)==before and all(not p.requires_grad for p in crew.parameters())
    assert sum(len(r) for r in rows)>=6
    assert all(v in ('frozen-patrick-jane-102','Lizbon-v1') for v in summary['controllers'].values())
    import json
    records=[json.loads(line) for line in (tmp_path/'bounded.jsonl').read_text().splitlines()]
    assert records[0]['record']=='header' and records[-1]['record']=='result'
    assert records[-1]['winner'] is None


@pytest.mark.parametrize('elapsed,cooldown,expected',[(.6,0,.6),(.6,.2,.4),(.6,1,0),(0,0,0)])
def test_ready_penalty_uses_simulated_overlap(elapsed,cooldown,expected):
    assert ready_seconds(elapsed,cooldown)==pytest.approx(expected)


@pytest.mark.parametrize('field',['active_before','active_after','roaming_before','roaming_after'])
def test_ready_penalty_excludes_meetings_elimination_and_terminal_intervals(field):
    assert ready_seconds(.6,0,**{field:False})==0


def test_ready_penalty_excludes_vent_pause_and_kill_reset():
    assert ready_seconds(.6,0,vented_after=True)==0
    assert ready_seconds(.6,0,vented_before=True)==0
    assert ready_seconds(.6,0,killed=True)==0
    assert ready_seconds(46,0,killed=True)==1
    assert ready_seconds(.6,0,transition_before=True)==0
    assert ready_seconds(.6,0,transition_after=True)==0


def test_team_contention_preserves_native_priority_without_scripted_targets(fixture):
    from patrick_jane.api import Action
    game=fixture[2]
    for _ in range(51):game.step({})
    ids=sorted(pid for pid,p in game.players.items() if p.role.value=='impostor')
    room=game.observe_pj(ids[0]).door_actions[0]
    requested={pid:Action('doors',room) for pid in ids}
    resolved,declined=resolve_team_contention(requested)
    assert len(declined)==1 and all(a.kind=='doors' for a in requested.values())
    game.step(resolved)
    assert game.metrics['illegal_actions']==0
    assert game.a2_metrics['door_sabotages_accepted']==1
    mixed={ids[0]:Action('sabotage','lights'),ids[1]:Action('doors',room)}
    resolved,declined=resolve_team_contention(mixed)
    assert ids[0] in declined and resolved[ids[1]]==mixed[ids[1]]


@pytest.mark.parametrize('phase',['discussion','voting'])
def test_meeting_candidates_are_legal_and_not_truth_filtered(fixture,phase):
    from social_deduction.actor import Phase
    settings,_,game,pid,planner=fixture;obs=game.observe_pj(pid)
    context=replace(obs.actor.context,phase=Phase(phase),speaker_id=pid,
                    participants=tuple(game.players),voted_ids=())
    actor=replace(obs.actor,context=context)
    actor=replace(actor,actions=legal_actions(actor))
    obs=replace(obs,actor=actor)
    p=encode(obs,feedback(obs,game),Memory(pid),planner,settings)
    assert len(p.candidates)>1
    for c in p.candidates:
        # Native Patrick Jane filters WAIT before generic intent validation;
        # this is its own silence/no-submission operation in meetings.
        if type(c.payload) is Intent and c.payload.kind is not IntentKind.WAIT:
            assert intent_allowed(actor,c.payload)
    if phase=='discussion':
        assert any(c.kind=='claim' and c.key==pid for c in p.candidates)
        assert any(c.kind=='claim' and getattr(c.payload,'claim',None) and
                   c.payload.claim.kind.value=='saw_elimination' for c in p.candidates)
    else:
        assert not any(c.kind=='vote' and c.key==pid for c in p.candidates)
        assert any(c.kind=='skip' for c in p.candidates)


def test_dead_or_terminal_actor_cannot_act(fixture):
    from social_deduction.actor import Phase
    settings,_,game,pid,planner=fixture;obs=game.observe_pj(pid)
    dead=replace(obs,actor=replace(obs.actor,own=replace(obs.actor.own,active=False)))
    with pytest.raises(ValueError):encode(dead,feedback(obs,game),Memory(pid),planner,settings)
    ended=replace(obs,actor=replace(obs.actor,context=replace(obs.actor.context,phase=Phase.FINISHED)))
    with pytest.raises(ValueError):encode(ended,feedback(obs,game),Memory(pid),planner,settings)


@pytest.mark.parametrize('phase',['discussion','voting'])
def test_native_meeting_wait_is_a_legal_no_submission(fixture,phase):
    from social_deduction.actor import Phase
    game,pid=fixture[2:4]
    game._start_meeting(game.players[pid])
    game.phase=Phase(phase);game._invalidate()
    before=game.metrics['illegal_actions']
    game.step({pid:Intent(IntentKind.WAIT)})
    assert game.metrics['illegal_actions']==before


def test_expired_optimizer_budget_preserves_weights_and_discards_gradients():
    model=ImpostorPolicy();p=synthetic();before=tensor_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    logs=update(model,optimizer,[[Transition(p,0,0,0,1)]],.6,deadline=0)
    assert logs==[] and tensor_hash(model)==before
    assert all(x.grad is None for x in model.parameters())


@pytest.mark.parametrize('sabotage,ready',[('lights',0),(None,10)])
def test_emergency_capabilities_respect_native_extension_restrictions(fixture,sabotage,ready):
    settings,_,game,pid,planner=fixture;obs=game.observe_pj(pid)
    # The base phase3 family alone is insufficient: Jane adds these restrictions.
    actor=replace(obs.actor,actions=obs.actor.actions+(Intent(IntentKind.CALL_MEETING,'EmergencyConsole'),))
    obs=replace(obs,actor=actor,sabotage=sabotage,emergency_ready_in=ready)
    p=encode(obs,feedback(obs,game),Memory(pid),planner,settings)
    assert not any(c.kind=='emergency' for c in p.candidates)


def test_seeded_neural_match_replays_are_deterministic(fixture,tmp_path):
    settings,config=fixture[:2];crew=frozen_crew(Path(SOURCE));model=ImpostorPolicy()
    results=[]
    for i in range(2):
        summary,trajectories=run_match(model,crew,settings,config,9300012,
            output=tmp_path/f'repeat-{i}.jsonl',max_observations=4,max_wall_seconds=45)
        results.append((summary,trajectories))
    assert results[0][0]['trace']==results[1][0]['trace']
    assert results[0][0]['impostor_actions']==results[1][0]['impostor_actions']
    for left,right in zip(results[0][1],results[1][1]):
        assert len(left)==len(right)
        for a,b in zip(left,right):
            assert (a.action,a.logp,a.value,a.reward)==(b.action,b.logp,b.value,b.reward)
            assert np.array_equal(a.packet.self_features,b.packet.self_features)
