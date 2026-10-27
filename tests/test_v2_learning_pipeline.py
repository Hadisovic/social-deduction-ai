from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pytest
import torch
from among_us_map import get_map
from navigation_service import NavigationPlanner
from crewmate_v2.config import LobbySettings
from crewmate_v2.learning.config import RewardConfig,PARTITIONS,match_seed,pilot_plan
from crewmate_v2.learning.env import LearningEnv
from crewmate_v2.learning.opponents import ReporterController
from crewmate_v2.learning.policy import V2Policy,tensors,check_packet
from crewmate_v2.learning.provenance import runtime_lock,digest,plan_hash,validate_approval
from crewmate_v2.learning.evaluate import probability_metrics,validation_contexts,forecast,evaluate_game
from crewmate_v2.learning.train import pilot,state_hash
from crewmate_v2.learning.comparison import summarize,compare
from phase3_bots import BotStyle
from social_deduction.actor import Phase


@pytest.fixture(scope='module')
def env():
    torch.set_num_threads(1)
    result=LearningEnv(planner=NavigationPlanner(get_map(),cache_size=96),lobby=LobbySettings(timeout=2.))
    result.reset(seed=2730010)
    return result


def test_versioned_profile_finite_validation_and_partition_disjointness(tmp_path):
    path=tmp_path/'rewards.json';path.write_text(json.dumps(RewardConfig().to_dict()))
    assert RewardConfig.from_file(path)==RewardConfig()
    with pytest.raises(ValueError):RewardConfig(report=float('nan'))
    with pytest.raises(ValueError):RewardConfig(schema='historical')
    ranges=list(PARTITIONS.values())
    for i,(lo,hi) in enumerate(ranges):
        assert lo>2630109
        assert all(hi<a or lo>b for a,b in ranges[i+1:])
    with pytest.raises(ValueError):match_seed('final',0)
    with pytest.raises(ValueError):match_seed('validation',4)


def test_v2_task_action_alignment_and_input_schema(env):
    packet=env._packet();check_packet(packet)
    assert env.observation_space.contains(packet)
    ordered=sorted(env.observation.own.tasks,key=lambda t:t.task_id)
    for i,choice in enumerate(env.options):
        if choice.kind=='task':assert ordered[packet['task_targets'][i]].task_id==choice.task_id
    assert packet['players'].shape==(4,29) and packet['own_tasks_v2'].shape==(4,8)
    policy=V2Policy()
    for _ in range(10):assert packet['mask'][policy.act(packet)[0]]
    with pytest.raises(ValueError):policy(tensors({**packet,'hidden_roles':np.zeros(5)}))
    broken=dict(packet,players=packet['players'][:,:26])
    with pytest.raises(ValueError):check_packet(broken)


def test_policy_permutation_equivariance_candidate_and_task_rows(env):
    p=env._packet();policy=V2Policy().eval()
    permutation=np.asarray([2,0,3,1]);inverse=np.argsort(permutation)
    q={k:v.copy() for k,v in p.items()}
    for matrix,key in (('players','targets'),('own_tasks_v2','task_targets')):
        q[matrix]=q[matrix][permutation]
        valid=q[key]>=0;q[key][valid]=inverse[q[key][valid]]
    a,v=policy(tensors(p));b,w=policy(tensors(q))
    assert torch.allclose(a.probs,b.probs,atol=1e-7) and torch.allclose(v,w,atol=1e-7)


def test_exclusive_checkpoint_roundtrip_source_and_schema_rejection(tmp_path,env):
    policy=V2Policy();path=tmp_path/'new.pt';metadata=dict(runtime=runtime_lock(LobbySettings()))
    policy.save(path,metadata);loaded,_=V2Policy.load(path)
    assert state_hash(policy)==state_hash(loaded)
    with pytest.raises(FileExistsError):policy.save(path,metadata)
    damaged=torch.load(path,weights_only=True);damaged['metadata']['runtime']['learning_sources']={}
    other=tmp_path/'drift.pt';torch.save(damaged,other)
    with pytest.raises(ValueError,match='runtime'):V2Policy.load(other)
    torch.save(dict(schema='historical'),tmp_path/'old.pt')
    with pytest.raises(ValueError,match='schema'):V2Policy.load(tmp_path/'old.pt')


def test_three_style_rotations_cover_each_style_without_role_selection():
    styles={ReporterController(17,BotStyle(),r).reporting_style for r in (0,1,2)}
    assert styles=={'prompt','inspect','bypass'}
    for r in (0,1,2):
        assert ReporterController(17,BotStyle(),r).reporting_style==ReporterController(17,BotStyle('social','patient','cautious'),r).reporting_style


def test_owner_gate_refuses_before_creating_output(tmp_path):
    output=tmp_path/'not-authorized'
    with pytest.raises(PermissionError):pilot(output,dict(approved=False),{})
    assert not output.exists()
    lock=runtime_lock(LobbySettings())
    with pytest.raises(PermissionError):validate_approval(dict(approved=True,plan_sha256='wrong',runtime_sha256=digest(lock)),lock)


def test_pre_vote_forecast_excludes_self_and_has_no_trainer_label_input(env):
    g=env.match.game;g.phase=Phase.VOTING;g.phase_started_tick=g.tick;g.meeting_id='forecast'
    g.participants=tuple(g.players);g._invalidate();env._observe();packet=env._packet()
    policy=V2Policy();d,_=policy(tensors(packet))
    result=forecast(packet,env.options,env.belief,env.observation,d)
    assert env.focal not in result['candidates']
    assert sum(result['policy_vote_ranking'])==pytest.approx(1.)
    assert sum(result['frozen_belief'])==pytest.approx(1.)
    assert 0<=result['self_vote_probability']<1


def test_calibration_metrics_and_exact_validation_matrix():
    m=probability_metrics([([.8,.2],0),([.3,.7],1)])
    assert m['top1']==1. and m['brier']==pytest.approx((.08+.18)/2)
    assert m['nll']==pytest.approx(-np.log(.8*.7)/2) and m['ece']==pytest.approx(.25)
    contexts=list(validation_contexts());assert len(contexts)==32
    assert len({c['seed'] for c in contexts})==4
    assert len([c for c in contexts if c['opponents']=='varied'])==24


def test_deadline_and_actor_choice_survive_wrapper(env):
    env.reset(seed=2730011);env.deadline=perf_counter()-1;env.match.deadline=env.deadline
    with pytest.raises(TimeoutError):env.step(0)
    env.deadline=None


def test_bounded_frozen_inference_validation_path_no_training():
    result=evaluate_game(V2Policy().eval(),seed=2730012,mode=1.,deadline=perf_counter()+30,lobby=LobbySettings(timeout=1.2))
    assert result['reason']=='TIMEOUT' and result['pre_vote_count']==0
    assert result['illegal_actions']==result['navigation_failures']==0
    summary=summarize([result])
    assert summary['reporting_rate'] is None and summary['legal_ballot_opportunities']==0
    paired=compare([dict(arm='original',games=[result]),dict(arm='shaped',games=[result])])
    assert paired['paired_bootstrap']['win_rate']['mean']==0


def test_direct_training_helper_does_not_bypass_owner_gate(tmp_path):
    from crewmate_v2.learning.train import run_one
    target=tmp_path/'not-authorized'
    with pytest.raises(PermissionError):run_one(target,'original',17,1024,perf_counter()+30,expected_lock=runtime_lock(LobbySettings()))
    assert not target.exists()
