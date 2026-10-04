from dataclasses import replace
import hashlib
import numpy as np
import pygame
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from shapely.geometry import LineString
from among_us_map import AmongUsMap,get_map,ASSET_DIR
from phase3_assets import resolve_red_palette
from phase5_env import CrewmateStrategicEnv
from phase5_features import choices,encode,mechanical_intent
from phase5_policy import StrategicPolicy,tensors
from phase5_training import time_aware_gae,update
from social_deduction.actor import Phase,PublicContext
from social_deduction.phase3_api import legal_actions,IntentKind,intent_allowed


@pytest.fixture(scope='module')
def env():
    torch.set_num_threads(1)
    return CrewmateStrategicEnv()


def test_conservative_buffers_cover_exact_circle_and_keep_legacy_available():
    m=get_map()
    assert m.buffer_radius>m.radius and not m.closed_doors
    # Every precomputed navigation edge must respect the exact circle offset.
    import shapely
    gy,gx=np.nonzero(m.grid)
    from among_us_map import DIRECTIONS
    for k,(dx,dy) in enumerate(DIRECTIONS):
        valid=m.edges[gy,gx,k];x,y=gx[valid],gy[valid]
        a=np.c_[m.xs[x],m.ys[y]];b=np.c_[m.xs[x+dx],m.ys[y+dy]]
        distances=shapely.distance(shapely.linestrings(np.stack([a,b],axis=1)),m.barriers)
        assert distances.min()>=m.radius-1e-9
    old=AmongUsMap(clearance_policy='legacy_polygon')
    assert old.buffer_radius==old.radius


def test_builder_is_reproducible_without_editing_committed_blueprint(tmp_path):
    from tools.build_among_us_map import build
    path=tmp_path/'map.json';build(path)
    assert path.read_bytes()==(ASSET_DIR/'among_us_map.json').read_bytes()


def test_rgb_red_mask_palette_does_not_mutate_source_alpha_or_neutral_bone():
    s=pygame.Surface((4,1),pygame.SRCALPHA)
    for i,c in enumerate(((255,0,0,255),(0,255,0,128),(0,0,255,255),(240,240,230,200))):s.set_at((i,0),c)
    original=pygame.image.tobytes(s,'RGBA');r=resolve_red_palette(s)
    assert tuple(r.get_at((0,0)))==(198,17,17,255)
    assert tuple(r.get_at((1,0)))==(148,201,219,128)
    assert tuple(r.get_at((2,0)))==(122,8,56,255)
    assert r.get_at((3,0))==s.get_at((3,0))
    assert pygame.image.tobytes(s,'RGBA')==original


def test_gym_contract_seed_replay_and_bad_action(env):
    check_env(env,skip_render_check=True)
    runs=[]
    for _ in range(2):
        packet,_=env.reset(seed=600000);result=[]
        for a in (0,1,0,2):
            packet,r,d,t,i=env.step(a);assert env.observation_space.contains(packet)
            result.append((packet,r,d,t,i))
        runs.append(result)
    for a,b in zip(*runs):
        assert a[1:]==b[1:]
        assert all(np.array_equal(a[0][k],b[0][k]) for k in a[0])
    assert env.step(95)[4]['masked_selection']
    with pytest.raises(ValueError):env.step(96)


def test_four_candidate_votes_self_skip_claims_and_no_entropy_threshold(env):
    env.reset(seed=600000);obs=env.observation
    ctx=PublicContext(Phase.VOTING,'m1',tuple(i.player_id for i in obs.roster))
    voting=replace(obs,context=ctx);voting=replace(voting,actions=legal_actions(voting))
    options=choices(voting,env.memory)
    assert len([c for c in options if c.kind=='vote'])==5
    assert len([c for c in options if c.kind=='skip'])==1
    ctx=replace(ctx,phase=Phase.DISCUSSION,speaker_id=obs.own.player_id)
    speaking=replace(obs,context=ctx);speaking=replace(speaking,actions=legal_actions(speaking))
    for c in choices(speaking,env.memory):
        if c.intent:assert intent_allowed(speaking,c.intent)
    assert sum(c.kind=='claim_suspect' for c in choices(speaking,env.memory))==4


def test_room_actions_are_distinguishable_from_public_geometry(env):
    packet,_=env.reset(seed=600000)
    rows=[packet['actions'][i].tobytes() for i,c in enumerate(env.options) if c.kind=='room']
    assert len(rows)==14 and len(set(rows))==14


def test_mask_sampling_and_candidate_permutation_equivariance(env):
    packet,_=env.reset(seed=600000);torch.manual_seed(7);p=StrategicPolicy()
    distribution,value=p(tensors(packet))
    assert torch.all(distribution.probs[0][~torch.tensor(packet['mask'],dtype=torch.bool)]==0)
    for _ in range(30):assert packet['mask'][p.act(packet)[0]]
    perm=np.array([2,0,3,1]);changed={k:v.copy() for k,v in packet.items()}
    changed['players']=changed['players'][perm]
    inverse=np.argsort(perm);mask=changed['targets']>=0;changed['targets'][mask]=inverse[changed['targets'][mask]]
    d2,v2=p(tensors(changed))
    assert torch.allclose(distribution.probs,d2.probs,atol=1e-7)
    assert torch.allclose(value,v2,atol=1e-7)


def test_time_aware_gae_terminal_boundary_and_long_duration():
    r=np.array([1.,2.]);v=np.zeros(2);nextv=np.array([0.,99.]);done=np.array([False,True])
    a,_=time_aware_gae(r,v,nextv,np.array([.99,.99**10]),done,np.array([.6,6.]))
    assert a[1]==2 and a[0]==pytest.approx(1+.99*.95*2)
    a,_=time_aware_gae(r,v,nextv,np.array([.99**10,.99]),done,np.array([6.,.6]))
    assert a[0]==pytest.approx(1+(.99*.95)**10*2)


def test_policy_checkpoint_roundtrip_and_update_preserves_frozen_belief(env,tmp_path):
    packet,_=env.reset(seed=600000);torch.manual_seed(1);policy=StrategicPolicy()
    frozen={k:v.clone() for k,v in env.belief_model.network.state_dict().items()}
    before={k:v.clone() for k,v in policy.state_dict().items()}
    rollout={k:[] for k in ('packets','actions','logp','values','next_values','rewards','discounts','dones','durations')}
    for j in range(8):
        action,lp,value=policy.act(packet)
        for k,v in zip(rollout,(packet,action,lp,value,0.,float(j%2),.99,j==7,.6)):rollout[k].append(v)
    loss=update(policy,torch.optim.Adam(policy.parameters(),lr=.001),rollout,epochs=1)
    assert np.isfinite(loss) and any(not torch.equal(before[k],v) for k,v in policy.state_dict().items())
    assert all(torch.equal(frozen[k],v) for k,v in env.belief_model.network.state_dict().items())
    assert not any(p.requires_grad for p in env.belief_model.network.parameters())
    path=tmp_path/'policy.pt';policy.save(path);loaded,_=StrategicPolicy.load(path)
    assert policy.act(packet,True)==loaded.act(packet,True)


def test_complete_random_match_settles_focal_elimination(env):
    packet,_=env.reset(seed=600000);rng=np.random.default_rng(7)
    for _ in range(500):
        packet,r,done,_,info=env.step(int(rng.choice(np.flatnonzero(packet['mask']))))
        assert np.isfinite(r) and env.observation_space.contains(packet)
        if done:break
    assert done and env.match.game.is_terminal
    assert info['episode']['illegal_actions']==0
    with pytest.raises(RuntimeError):env.step(0)
