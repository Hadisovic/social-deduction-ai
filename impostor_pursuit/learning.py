"""Strict continuation using the unchanged, tested V2 PPO/physics implementation."""
import hashlib
from pathlib import Path
from time import time
import uuid
import numpy as np
import torch
from impostor_research.learning_v2 import Session as BaseSession,record_for as core_record,verify as verify_core,load as load_v2
from impostor_research.learning import clean,packet_hash
from impostor_research.strategic_v2 import StrategicEnv
from impostor_research.protocol import ROOT
from impostor_research.frozen import weight_hash
from .policy import PursuitPolicy,SCHEMA,FAMILIES


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_for(models,parent,development=False):
    parent=Path(parent).resolve();_,p=load_v2(parent,models)
    if not development and (p['development'] or p['decisions'] not in (4096,8192) or not parent.is_relative_to(ROOT)):
        raise ValueError('Production continuation requires a locked, in-repository V2 parent')
    paths=sorted([*ROOT.glob('impostor_pursuit/*.py'),ROOT/'tools/run_impostor_pursuit.py',ROOT/'run_impostor_ai.py'])
    return clean(dict(schema=SCHEMA,core=core_record(StrategicEnv(models=models)),
        parent=dict(path=parent.relative_to(ROOT).as_posix() if parent.is_relative_to(ROOT) else str(parent),sha256=digest(parent),seed=p['seed'],decisions=p['decisions']),
        development=development,network=dict(parameters=sum(v.numel() for v in PursuitPolicy().parameters()),families=list(FAMILIES)),
        change='Only go_to_location (visible-player goals) moves from travel to a separate learned pursuit family; warm-start own V2 impostor, new family embedding copies travel.',
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths},
        partitions=dict(train=[3101000,3199999],selection=[3200060,3200079],confirmation=[3200080,3200099]),
        budget='Uses the original V2 study deadline; no extension',candidate_steps=[4096,8192],
        selection_rule='wins, kills, fewer timeouts, prefer unchanged parent on ties, earlier step',
        condition='Run only if V2 locked confirmation has fewer than 5/20 wins and >=3600 seconds remain',
        final_tests_opened=False))


def verify(record,models=None,deadline=None):
    if record.get('schema')!=SCHEMA or record['final_tests_opened']:raise ValueError('Pursuit schema/protocol mismatch')
    verify_core(record['core'],models,deadline)
    if record['network']!=dict(parameters=75091,families=list(FAMILIES)):raise ValueError('Pursuit architecture mismatch')
    for name,expected in record['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('Pursuit source changed: '+name)
    if digest(ROOT/record['parent']['path'])!=record['parent']['sha256']:raise ValueError('Frozen V2 parent changed')


class Session(BaseSession):
    def __init__(self,seed,*,models=None,record=None,parent=None,development=False):
        if record is None:
            from impostor_research.frozen import FrozenModels
            models=models or FrozenModels();record=record_for(models,parent,development)
        verify(record,models)
        if seed!=record['parent']['seed'] or development!=record['development']:raise ValueError('Continuation identity mismatch')
        super().__init__(seed,models=models,record=record['core'],development=development)
        self.study_record=record
        parent_policy,_=load_v2(ROOT/record['parent']['path'],self.env.models)
        self.policy=PursuitPolicy(seed);self.policy.warm_start(parent_policy);self.policy.train()
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=.0003,eps=1e-5)
        self.generator=torch.Generator().manual_seed(seed^0xAB31);self.rng=np.random.default_rng(seed^0xFA17)
        self.initial_hash=weight_hash(self.policy)

    @property
    def scenario_seed(self):
        start=3500010 if self.development else 3101000
        if start+self.episode>(3500019 if self.development else 3199999):raise ValueError('Pursuit training range exhausted')
        return start+self.episode

    def update(self,count=256,deadline=None):
        verify(self.study_record,self.env.models,deadline)
        return super().update(count,deadline)

    def save(self,path,resume=True):
        path=Path(path)
        if path.exists():raise FileExistsError('Never overwrite pursuit checkpoints')
        verify(self.study_record,self.env.models)
        state=dict(schema=SCHEMA,record=self.study_record,seed=self.seed,decisions=self.decisions,policy=self.policy.state_dict(),
            policy_hash=weight_hash(self.policy),initial_hash=self.initial_hash,development=self.development)
        if resume:state.update(optimizer=self.optimizer.state_dict(),generator=self.generator.get_state(),numpy_rng=self.rng.bit_generator.state,
            episode=self.episode,episode_actions=self.episode_actions,episodes=self.episodes,updates=self.updates,
            simulated_seconds=self.simulated_seconds,elapsed_seconds=self.elapsed_seconds,actions=self.action_counts,
            kill_available=self.kill_available,kill_selected=self.kill_selected,kill_probability_sum=self.kill_probability_sum,
            packet_hash=packet_hash(self.packet),actor_hash=self.env.actor_trace.hexdigest(),opponent_hash=self.env.match.trace.hexdigest())
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');torch.save(state,temporary);temporary.rename(path)

    @classmethod
    def restore(cls,path,models=None,deadline=None):
        p=torch.load(path,map_location='cpu',weights_only=True);verify(p['record'],models,deadline)
        if p['schema']!=SCHEMA or 'optimizer' not in p:raise ValueError('Wrong pursuit resume schema')
        obj=cls(p['seed'],models=models,record=p['record'],development=p['development']);obj.episode=p['episode']
        obj.packet,_=obj.env.reset(seed=obj.scenario_seed)
        for action in p['episode_actions']:
            if deadline and time()>deadline:raise TimeoutError('Original deadline reached')
            obj.packet,_,done,_,_=obj.env.step(action)
            if done:raise ValueError('Resume journal crosses terminal')
        if packet_hash(obj.packet)!=p['packet_hash'] or obj.env.actor_trace.hexdigest()!=p['actor_hash'] or obj.env.match.trace.hexdigest()!=p['opponent_hash']:
            raise ValueError('Pursuit resume replay mismatch')
        obj.policy.load_state_dict(p['policy'],strict=True);obj.optimizer.load_state_dict(p['optimizer'])
        if weight_hash(obj.policy)!=p['policy_hash']:raise ValueError('Pursuit tensor mismatch')
        obj.generator.set_state(p['generator']);obj.rng.bit_generator.state=p['numpy_rng']
        for k in ('decisions','episode_actions','episodes','updates','simulated_seconds','elapsed_seconds','initial_hash','kill_available','kill_selected','kill_probability_sum'):
            setattr(obj,k,p[k])
        obj.action_counts=p['actions'];return obj


def load(path,models=None,deadline=None):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p.get('schema')!='impostor-pursuit-v3':return load_v2(path,models,deadline)
    verify(p['record'],models,deadline)
    policy=PursuitPolicy(p['seed']);policy.load_state_dict(p['policy'],strict=True);policy.eval().requires_grad_(False)
    if weight_hash(policy)!=p['policy_hash'] or any(not torch.isfinite(v).all() for v in policy.state_dict().values()):raise ValueError('Pursuit finite/tensor mismatch')
    return policy,p
