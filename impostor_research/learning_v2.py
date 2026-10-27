"""Registered family-policy PPO with dense outcome feedback and isolated gradients."""
from dataclasses import dataclass,asdict
import hashlib,json
from pathlib import Path
from time import perf_counter,time
import uuid
import numpy as np
import torch
from .strategic_v2 import FamilyPolicy,StrategicEnv,SCHEMA,FAMILIES,EXTRA_COUNT,ACTION_COUNT_V2
from .features import KINDS,GLOBAL_COUNT,PLAYER_COUNT,MAX_ACTIONS
from .protocol import ROOT,manifest
from .frozen import weight_hash
from .learning import clean,gae,packet_hash,check_packet


@dataclass(frozen=True)
class Config:
    decisions:int=8192
    rollout:int=256
    epochs:int=4
    minibatch:int=64
    learning_rate:float=.0003
    lambda_per_second:float=.995
    clip:float=.2
    value_coefficient:float=.5
    actor_grad_norm:float=.5
    critic_grad_norm:float=1.
    family_entropy_start:float=.02
    family_entropy_end:float=.005
    conditional_entropy:float=.001


def record_for(env):
    record=manifest(env)
    for name in ('tools/run_impostor_v2.py','run_impostor.py'):
        p=ROOT/name;record['sources'][name]=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    record.update(schema=SCHEMA,config=asdict(Config()),training_authorized=True,engineering_only=False,
        owner_approval='2026-10-27: fix policies/rewards/NN, train stronger impostor and provide runnable viewer',
        network=dict(parameters=sum(p.numel() for p in FamilyPolicy().parameters()),families=list(FAMILIES),
            global_count=GLOBAL_COUNT,players=PLAYER_COUNT,extra=EXTRA_COUNT,actions=ACTION_COUNT_V2),
        shaping=dict(successful_kill=2.,own_ejection=-2.,no_witness_truth_reward=True,travel_option_seconds=3.),
        partitions=dict(train=[3100100,3199999],development=[3500010,3500019],selection=[3200020,3200039],
                        confirmation=[3200040,3200059],demo=[3600000,3600999]),
        candidate_checkpoints=[4096,8192],selection_rule='wins, then kills, then fewer timeouts, then earlier step and initialization',
        confirmation_rule='locked selected policy and old pilot-83, scripted hunter, untrained family policy; 20 paired fresh matches each',
        conditional_entropy_normalized=True,critic_encoder_independent=True)
    candidates=json.loads((ROOT/'docs/benchmarks/phase6/remaining-six/final-candidates-proposal.json').read_text())['candidates']
    record['frozen_checkpoints']={c['path']:c['sha256'] for c in candidates}
    return clean(record)


def verify(record,models=None,deadline=None):
    if record.get('schema')!=SCHEMA or record['config']!=asdict(Config()):raise ValueError('V2 config/schema mismatch')
    if deadline is not None and time()>deadline:raise TimeoutError('Follow-up 3-hour ceiling exceeded')
    for name,digest in {**record['historical_sources'],**record['sources']}.items():
        data=(ROOT/name).read_bytes()
        if name.endswith('.py') or name in record['sources']:data=data.replace(b'\r\n',b'\n')
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('V2 runtime changed: '+name)
    from crewmate_v2.config import LobbySettings
    if record['lobby']!=LobbySettings.from_file(ROOT/'configs/crewmate-v2-lobby.json').to_dict() or record['mixture']!={'deduction_frequency':0.}:
        raise ValueError('Canonical ordinary lobby changed')
    if record['safeguards']!={'bodies':True,'eyewitness':True}:raise ValueError('Opponent safeguards changed')
    if record['reward']!={'win':10.,'loss':-10.,'timeout':-12.,'per_second':-.002,'gamma_per_second':1.}:
        raise ValueError('Reward configuration mismatch')
    if record['shaping']!={'successful_kill':2.,'own_ejection':-2.,'no_witness_truth_reward':True,'travel_option_seconds':3.}:
        raise ValueError('Shaping configuration mismatch')
    for name,digest in record['frozen_checkpoints'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Historical checkpoint changed')
    if models:
        models.assert_unchanged()
        if models.hashes()!=record['frozen_tensors']:raise ValueError('Opponent tensor mismatch')


class Session:
    def __init__(self,seed,*,models=None,record=None,development=False):
        if seed not in (71,83):raise ValueError('Unregistered initialization')
        self.seed=seed;self.development=development;self.env=StrategicEnv(models=models)
        self.record=record or record_for(self.env);verify(self.record,self.env.models)
        self.policy=FamilyPolicy(seed).train()
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=Config().learning_rate,eps=1e-5)
        self.generator=torch.Generator().manual_seed(seed^0xB6710);self.rng=np.random.default_rng(seed^0x6A17)
        self.episode=0;self.decisions=0;self.simulated_seconds=0.;self.elapsed_seconds=0.
        self.episodes=[];self.updates=[];self.episode_actions=[];self.action_counts={}
        self.kill_available=0;self.kill_selected=0;self.kill_probability_sum=0.
        self.initial_hash=weight_hash(self.policy);self.packet,_=self.env.reset(seed=self.scenario_seed)

    @property
    def scenario_seed(self):
        start=3500010 if self.development else 3100100
        if start+self.episode>(3500019 if self.development else 3199999):raise ValueError('Training partition exhausted')
        return start+self.episode

    def collect(self,count,deadline=None):
        rollout={k:[] for k in ('packets','actions','logp','values','next_values','rewards','discounts','dones','durations')}
        for _ in range(count):
            if deadline and time()>deadline:raise TimeoutError('Training ceiling reached')
            check_packet(self.packet)
            action,lp,value=self.policy.act(self.packet,generator=self.generator,deterministic=False)
            choice=self.env.options[action]
            if self.packet['extra'][7]:
                with torch.no_grad():
                    _,_,families,_,_=self.policy.evaluate({k:torch.as_tensor(v).unsqueeze(0) for k,v in self.packet.items()})
                self.kill_available+=1;self.kill_selected+=choice.kind=='kill'
                self.kill_probability_sum+=float(families.probs[0,FAMILIES.index('kill')])
            self.action_counts[choice.kind]=self.action_counts.get(choice.kind,0)+1
            old=self.packet;self.packet,r,done,_,info=self.env.step(action)
            if self.env.match.game.metrics['illegal_actions'] or self.env.match.game.metrics['navigation_failures']:
                raise ValueError('Illegal action/navigation failure')
            with torch.no_grad():next_value=0. if done else float(self.policy({k:torch.as_tensor(v).unsqueeze(0) for k,v in self.packet.items()})[1])
            for key,item in zip(rollout,(old,action,lp,value,next_value,r,info['discount'],done,info['elapsed_seconds'])):rollout[key].append(item)
            self.episode_actions.append(action);self.decisions+=1;self.simulated_seconds+=info['elapsed_seconds']
            if done:
                self.episodes.append(dict(decisions=self.decisions,**info['episode']))
                self.episode+=1;self.episode_actions=[];self.packet,_=self.env.reset(seed=self.scenario_seed)
        return rollout

    def update(self,count=256,deadline=None):
        verify(self.record,self.env.models,deadline)
        started=perf_counter();rollout=self.collect(count,deadline);collection=perf_counter()-started
        data={k:torch.as_tensor(np.stack([p[k] for p in rollout['packets']])) for k in rollout['packets'][0]}
        actions=torch.tensor(rollout['actions']);old=torch.tensor(rollout['logp'])
        adv,returns=gae(*[rollout[k] for k in ('rewards','values','next_values','discounts','dones','durations')],Config().lambda_per_second)
        adv=torch.from_numpy(adv);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8);returns=torch.from_numpy(returns)
        actor_params=self.policy.actor_parameters();critic_params=self.policy.critic_parameters()
        if set(map(id,actor_params))&set(map(id,critic_params)):raise ValueError('Critic and actor must be independent')
        actual=[p for g in self.optimizer.param_groups for p in g['params']]
        if set(map(id,actual))!=set(map(id,self.policy.parameters())):raise ValueError('Wrong optimizer ownership')
        coefficient=.02+(.005-.02)*min(self.decisions/8192,1.)
        stats=[];optim_start=perf_counter()
        for _ in range(4):
            for ids in np.array_split(self.rng.permutation(count),max(1,int(np.ceil(count/64)))):
                distribution,value,families,_,conditional=self.policy.evaluate({k:v[ids] for k,v in data.items()})
                ratio=(distribution.log_prob(actions[ids])-old[ids]).exp()
                actor=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                critic=(value-returns[ids]).square().mean()
                actor_objective=actor-coefficient*families.entropy().mean()-.001*conditional.mean()
                if not torch.isfinite(actor_objective+critic):raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True);actor_objective.backward();(.5*critic).backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in actual):raise FloatingPointError('Nonfinite gradient')
                an=torch.nn.utils.clip_grad_norm_(actor_params,.5);cn=torch.nn.utils.clip_grad_norm_(critic_params,1.)
                self.optimizer.step()
                stats.append((float(actor.detach()),float(critic.detach()),float(an),float(cn)))
        self.env.models.assert_unchanged()
        with torch.no_grad():
            distribution,value,families,_,conditional=self.policy.evaluate(data)
            logratio=distribution.log_prob(actions)-old;ratio=logratio.exp()
            variance=returns.var(unbiased=False);explained=float(1-(returns-value).var(unbiased=False)/variance) if variance>1e-8 else None
        if any(not torch.isfinite(p).all() for p in self.policy.parameters()):raise FloatingPointError('Nonfinite weights')
        self.elapsed_seconds+=perf_counter()-started
        means=np.asarray(stats).mean(0)
        row=dict(decisions=self.decisions,simulated_seconds=self.simulated_seconds,seconds=self.elapsed_seconds,
            completed_episodes=len(self.episodes),actions=dict(self.action_counts),policy_loss=float(means[0]),value_loss=float(means[1]),
            actor_gradient_norm=float(means[2]),critic_gradient_norm=float(means[3]),entropy=float(families.entropy().mean()),
            joint_entropy=float(distribution.entropy().mean()),entropy_coefficient=coefficient,
            approximate_kl=float(((ratio-1)-logratio).mean()),clip_fraction=float(((ratio-1).abs()>.2).float().mean()),
            explained_variance=explained,collection_seconds=collection,optimization_seconds=perf_counter()-optim_start,
            kill_available_decisions=self.kill_available,kill_selections=self.kill_selected,
            kill_probability_when_available=self.kill_probability_sum/self.kill_available if self.kill_available else None)
        self.updates.append(row);return row

    def save(self,path,resume=True):
        path=Path(path)
        if path.exists():raise FileExistsError('Do not overwrite checkpoints')
        self.env.models.assert_unchanged()
        state=dict(schema=SCHEMA,record=self.record,seed=self.seed,decisions=self.decisions,policy=self.policy.state_dict(),
            policy_hash=weight_hash(self.policy),initial_hash=self.initial_hash,development=self.development)
        if resume:state.update(optimizer=self.optimizer.state_dict(),generator=self.generator.get_state(),numpy_rng=self.rng.bit_generator.state,
            episode=self.episode,episode_actions=self.episode_actions,episodes=self.episodes,updates=self.updates,
            simulated_seconds=self.simulated_seconds,elapsed_seconds=self.elapsed_seconds,actions=self.action_counts,
            kill_available=self.kill_available,kill_selected=self.kill_selected,kill_probability_sum=self.kill_probability_sum,
            packet_hash=packet_hash(self.packet),actor_hash=self.env.actor_trace.hexdigest(),opponent_hash=self.env.match.trace.hexdigest())
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');torch.save(state,tmp);tmp.rename(path)

    @classmethod
    def restore(cls,path,models=None,deadline=None):
        p=torch.load(path,map_location='cpu',weights_only=True);verify(p['record'],models,deadline)
        if p['schema']!=SCHEMA or 'optimizer' not in p:raise ValueError('Wrong checkpoint type')
        obj=cls(p['seed'],models=models,record=p['record'],development=p['development']);obj.episode=p['episode']
        obj.packet,_=obj.env.reset(seed=obj.scenario_seed)
        for action in p['episode_actions']:
            if deadline and time()>deadline:raise TimeoutError('Resume deadline')
            obj.packet,_,done,_,_=obj.env.step(action)
            if done:raise ValueError('Resume journal crosses terminal')
        if packet_hash(obj.packet)!=p['packet_hash'] or obj.env.actor_trace.hexdigest()!=p['actor_hash'] or obj.env.match.trace.hexdigest()!=p['opponent_hash']:
            raise ValueError('Resume replay mismatch')
        obj.policy.load_state_dict(p['policy'],strict=True);obj.optimizer.load_state_dict(p['optimizer'])
        if weight_hash(obj.policy)!=p['policy_hash']:raise ValueError('Tensor mismatch')
        obj.generator.set_state(p['generator']);obj.rng.bit_generator.state=p['numpy_rng']
        for key in ('decisions','episode_actions','episodes','updates','simulated_seconds','elapsed_seconds','initial_hash',
                    'kill_available','kill_selected','kill_probability_sum'):setattr(obj,key,p[key])
        obj.action_counts=p['actions'];return obj


def load(path,models=None,deadline=None):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p.get('schema')!=SCHEMA:raise ValueError('Not a family-policy checkpoint')
    verify(p['record'],models,deadline)
    policy=FamilyPolicy(p['seed']);policy.load_state_dict(p['policy'],strict=True);policy.eval().requires_grad_(False)
    if weight_hash(policy)!=p['policy_hash'] or any(not torch.isfinite(v).all() for v in policy.state_dict().values()):raise ValueError('Policy tensor hash/finite mismatch')
    return policy,p
