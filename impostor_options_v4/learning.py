"""Same V2 network/PPO, with a strict separate environment/checkpoint schema.

No study runner is launched by importing or constructing a development session.
Production construction requires an explicit approval flag from a future protocol.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
from time import time
import uuid
import torch
from impostor_research.learning_v2 import Session as BaseSession,record_for as core_record,verify as verify_core,load as load_v2
from impostor_research.learning import clean,packet_hash
from impostor_research.strategic_v2 import StrategicEnv,FamilyPolicy,FAMILIES
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.protocol import ROOT
from . import SCHEMA
from .config import Rewards,Options
from .env import OptionsEnv


def record_for(models,rewards=Rewards(),options=Options()):
    paths=sorted([*ROOT.glob('impostor_options_v4/*.py'),ROOT/'run_impostor_options.py'])
    return clean(dict(schema=SCHEMA,core=core_record(StrategicEnv(models=models)),
        rewards=asdict(rewards),options=asdict(options),
        network=dict(parameters=75075,families=list(FAMILIES)),
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths},
        study_status='Implementation/preflight only; a new full study needs explicit owner approval',
        final_tests_opened=False))


def verify(record,models=None,deadline=None):
    if record.get('schema')!=SCHEMA or record.get('final_tests_opened') is not False:
        raise ValueError('V4 schema/protocol mismatch')
    verify_core(record['core'],models,deadline)
    Rewards(**record['rewards']);Options(**record['options'])
    if record['network']!=dict(parameters=75075,families=list(FAMILIES)):
        raise ValueError('V4 architecture mismatch')
    for name,expected in record['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise ValueError('V4 runtime changed: '+name)


class Session(BaseSession):
    def __init__(self,seed,*,models=None,record=None,development=True,training_approved=False,parent=None):
        if not development and not training_approved:
            raise PermissionError('New full PPO study requires explicit owner approval and preregistration')
        models=models or FrozenModels();record=record or record_for(models)
        verify(record,models)
        super().__init__(seed,models=models,record=record['core'],development=development)
        self.study_record=record;self.training_approved=training_approved
        self.env=OptionsEnv(models=models,rewards=Rewards(**record['rewards']),options=Options(**record['options']))
        self.packet,_=self.env.reset(seed=self.scenario_seed)
        self.parent_hash=None
        if parent is not None:
            previous,_=load_v2(parent,models)
            self.policy.load_state_dict(previous.state_dict(),strict=True)
            self.parent_hash=hashlib.sha256(Path(parent).read_bytes()).hexdigest()
        self.initial_hash=weight_hash(self.policy)
        self.generator=torch.Generator().manual_seed(seed^0xC401)
        import numpy as np
        self.rng=np.random.default_rng(seed^0xC402)

    @property
    def scenario_seed(self):
        start,end=(3500010,3500019) if self.development else (3110000,3199999)
        if start+self.episode>end:raise ValueError('V4 partition exhausted')
        return start+self.episode

    def update(self,count=256,deadline=None):
        verify(self.study_record,self.env.models,deadline)
        return super().update(count,deadline)

    def save(self,path,resume=True):
        path=Path(path)
        if path.exists():raise FileExistsError('Never overwrite V4 or historical checkpoints')
        verify(self.study_record,self.env.models)
        state=dict(schema=SCHEMA,record=self.study_record,seed=self.seed,decisions=self.decisions,
            policy=self.policy.state_dict(),policy_hash=weight_hash(self.policy),initial_hash=self.initial_hash,
            development=self.development,training_approved=self.training_approved,parent_sha256=self.parent_hash)
        if resume:
            state.update(optimizer=self.optimizer.state_dict(),generator=self.generator.get_state(),numpy_rng=self.rng.bit_generator.state,
                episode=self.episode,episode_actions=self.episode_actions,episodes=self.episodes,updates=self.updates,
                simulated_seconds=self.simulated_seconds,elapsed_seconds=self.elapsed_seconds,actions=self.action_counts,
                kill_available=self.kill_available,kill_selected=self.kill_selected,kill_probability_sum=self.kill_probability_sum,
                packet_hash=packet_hash(self.packet),actor_hash=self.env.actor_trace.hexdigest(),opponent_hash=self.env.match.trace.hexdigest(),
                environment_summary=clean(self.env.summary()))
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        torch.save(state,temporary);temporary.rename(path)

    @classmethod
    def restore(cls,path,models=None,deadline=None,training_approved=False):
        p=torch.load(path,map_location='cpu',weights_only=True)
        verify(p['record'],models,deadline)
        if p['schema']!=SCHEMA or 'optimizer' not in p:raise ValueError('Wrong V4 resume schema')
        obj=cls(p['seed'],models=models,record=p['record'],development=p['development'],training_approved=training_approved)
        obj.episode=p['episode'];obj.packet,_=obj.env.reset(seed=obj.scenario_seed)
        for action in p['episode_actions']:
            if deadline and time()>deadline:raise TimeoutError('Resume deadline')
            obj.packet,_,done,_,_=obj.env.step(action)
            if done:raise ValueError('Resume journal crosses terminal')
        if (packet_hash(obj.packet)!=p['packet_hash'] or obj.env.actor_trace.hexdigest()!=p['actor_hash']
            or obj.env.match.trace.hexdigest()!=p['opponent_hash'] or clean(obj.env.summary())!=p['environment_summary']):
            raise ValueError('V4 replay/reward/option mismatch')
        obj.policy.load_state_dict(p['policy'],strict=True);obj.optimizer.load_state_dict(p['optimizer'])
        if weight_hash(obj.policy)!=p['policy_hash']:raise ValueError('V4 tensor mismatch')
        obj.generator.set_state(p['generator']);obj.rng.bit_generator.state=p['numpy_rng']
        for key in ('decisions','episode_actions','episodes','updates','simulated_seconds','elapsed_seconds','initial_hash',
                    'kill_available','kill_selected','kill_probability_sum'):setattr(obj,key,p[key])
        obj.parent_hash=p['parent_sha256'];obj.action_counts=p['actions']
        return obj


def load(path,models=None,deadline=None):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p.get('schema')!=SCHEMA:return load_v2(path,models,deadline)
    verify(p['record'],models,deadline)
    policy=FamilyPolicy(p['seed']);policy.load_state_dict(p['policy'],strict=True)
    policy.eval().requires_grad_(False)
    if weight_hash(policy)!=p['policy_hash'] or any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
        raise ValueError('V4 tensor hash/finite mismatch')
    return policy,p
