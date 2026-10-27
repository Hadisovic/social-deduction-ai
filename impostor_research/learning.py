"""Pilot PPO, strict checkpoints and deterministic replay-based resume."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, time
import uuid
import numpy as np
import torch
from .env import ImpostorEnv
from .env import RewardConfig
from .opponents import Safeguards
from crewmate_v2.config import LobbySettings
from .features import GLOBAL_COUNT, PLAYER_COUNT, ACTION_COUNT, MAX_ACTIONS, KINDS
from .frozen import FrozenModels, weight_hash
from .policy import ImpostorPolicy
from .protocol import ROOT, manifest, PARTITIONS

CHECKPOINT_SCHEMA = 'impostor-ppo-pilot-v1'


@dataclass(frozen=True)
class PilotConfig:
    decisions: int = 4096
    rollout: int = 256
    minibatch: int = 64
    epochs: int = 4
    learning_rate: float = .0003
    clip: float = .2
    value_coefficient: float = .5
    max_grad_norm: float = .5
    lambda_per_second: float = .99
    entropy_start: float = .02
    entropy_end: float = .005

    def __post_init__(self):
        if self.decisions != 4096 or self.rollout != 256 or self.minibatch != 64 or self.epochs != 4:
            raise ValueError('This driver supports only the approved fixed pilot')
        if (self.learning_rate,self.clip,self.value_coefficient,self.max_grad_norm,self.lambda_per_second,
            self.entropy_start,self.entropy_end) != (.0003,.2,.5,.5,.99,.02,.005):
            raise ValueError('Pilot hyperparameters are locked')


def clean(value):
    return json.loads(json.dumps(value,allow_nan=False))


def write_json(path, value, *, exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open('x',encoding='utf-8') as f: json.dump(clean(value),f,indent=2); f.write('\n')
    else:
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(clean(value),indent=2)+'\n',encoding='utf-8')
        os.replace(temporary,path)


@contextmanager
def exclusive_worker(directory):
    """OS lock releases on crashes; a second healthy writer cannot enter."""
    path=Path(directory)/'worker.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size==0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle,fcntl.LOCK_UN)


def packet_hash(packet):
    h=hashlib.sha256()
    for key in sorted(packet): h.update(key.encode());h.update(packet[key].tobytes())
    return h.hexdigest()


def pilot_manifest(env):
    record=manifest(env)
    entry=ROOT/'tools/run_impostor_pilot.py'
    record['sources'][entry.relative_to(ROOT).as_posix()]=hashlib.sha256(entry.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    candidates=json.loads((ROOT/'docs/benchmarks/phase6/remaining-six/final-candidates-proposal.json').read_text())['candidates']
    proposal=ROOT/'docs/benchmarks/phase6/remaining-six/final-candidates-proposal.json'
    record['sources'][proposal.relative_to(ROOT).as_posix()]=hashlib.sha256(proposal.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    record.update(training_authorized=True,engineering_only=False,pilot_schema=CHECKPOINT_SCHEMA,
        owner_approval='2026-10-27: go on; two 4096-decision pilots and 40 validation matches',
        config=asdict(PilotConfig()),evaluation=dict(seeds=list(range(3200000,3200020)),
            checkpoint=4096,action_selection='greedy',checkpoint_search=False,final_tests=False),
        frozen_checkpoints={c['path']:c['sha256'] for c in candidates},
        network=dict(parameters=48738,global_count=GLOBAL_COUNT,player_count=PLAYER_COUNT,
            action_count=ACTION_COUNT,max_actions=MAX_ACTIONS,kinds=list(KINDS)))
    return clean(record)


def verify_record(record, *, models=None, deadline=None):
    if deadline is not None and time()>deadline: raise TimeoutError('Approved 3-hour pilot budget exhausted')
    expected=dict(record['historical_sources']);expected.update(record['sources'])
    for name,digest in expected.items():
        data=(ROOT/name).read_bytes()
        if name.endswith('.py') or name in record['sources']: data=data.replace(b'\r\n',b'\n')
        if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('Pilot source mismatch: '+name)
    for name,digest in record['frozen_checkpoints'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Frozen checkpoint changed: '+name)
    if record['config']!=asdict(PilotConfig()): raise ValueError('Pilot config mismatch')
    if (record['lobby']!=LobbySettings.from_file(ROOT/'configs/crewmate-v2-lobby.json').to_dict()
            or record['reward']!=asdict(RewardConfig()) or record['safeguards']!=asdict(Safeguards())
            or record['mixture']!={'deduction_frequency':0.}):
        raise ValueError('Pilot lobby/reward/opponent condition mismatch')
    candidates=json.loads((ROOT/'docs/benchmarks/phase6/remaining-six/final-candidates-proposal.json').read_text())['candidates']
    if record['frozen_checkpoints']!={c['path']:c['sha256'] for c in candidates}:
        raise ValueError('Frozen checkpoint registration mismatch')
    if record['network']!=dict(parameters=48738,global_count=GLOBAL_COUNT,player_count=PLAYER_COUNT,
            action_count=ACTION_COUNT,max_actions=MAX_ACTIONS,kinds=list(KINDS)):
        raise ValueError('Impostor network/schema mismatch')
    if models is not None:
        models.assert_unchanged()
        if models.hashes()!=record['frozen_tensors']: raise ValueError('Frozen tensor mismatch')


def check_packet(packet):
    if any(not np.isfinite(v).all() for v in packet.values()): raise FloatingPointError('Nonfinite actor packet')
    if not packet['mask'].any(): raise ValueError('Empty actor mask')


def gae(rewards,values,next_values,discounts,dones,durations,lambda_per_second=.99):
    arrays=[np.asarray(a) for a in (rewards,values,next_values,discounts,dones,durations)]
    if len({a.shape for a in arrays})!=1 or arrays[0].ndim!=1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid rollout arrays')
    if np.any(arrays[5]<=0) or not 0<lambda_per_second<=1: raise ValueError('Invalid elapsed time/lambda')
    advantage=np.zeros(len(rewards),np.float64);carry=0.
    for i in reversed(range(len(rewards))):
        live=1.-float(dones[i])
        delta=rewards[i]+discounts[i]*next_values[i]*live-values[i]
        carry=delta+discounts[i]*(lambda_per_second**durations[i])*live*carry
        advantage[i]=carry
    return advantage.astype(np.float32),(advantage+values).astype(np.float32)


def ppo_update(policy,optimizer,rollout,rng,entropy,config=PilotConfig()):
    # The optimizer is allowed exactly the learner parameters, no frozen models.
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if len(actual)!=len(list(policy.parameters())) or {id(p) for p in actual}!={id(p) for p in policy.parameters()}:
        raise ValueError('Optimizer must contain exactly impostor parameters')
    packet={k:torch.as_tensor(np.stack([p[k] for p in rollout['packets']])) for k in rollout['packets'][0]}
    actions=torch.as_tensor(rollout['actions'],dtype=torch.long)
    old=torch.as_tensor(rollout['logp'],dtype=torch.float32)
    a,returns=gae(*[rollout[k] for k in ('rewards','values','next_values','discounts','dones','durations')],config.lambda_per_second)
    advantage=torch.from_numpy(a);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    targets=torch.from_numpy(returns)
    losses=[];norms=[];actor_losses=[];value_losses=[]
    for _ in range(config.epochs):
        for ids in np.array_split(rng.permutation(len(actions)),max(1,int(np.ceil(len(actions)/config.minibatch)))):
            distribution,value=policy({k:v[ids] for k,v in packet.items()})
            ratio=(distribution.log_prob(actions[ids])-old[ids]).exp()
            actor=-torch.minimum(ratio*advantage[ids],ratio.clamp(1-config.clip,1+config.clip)*advantage[ids]).mean()
            critic=(value-targets[ids]).square().mean()
            loss=actor+config.value_coefficient*critic-entropy*distribution.entropy().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in actual):
                raise FloatingPointError('Nonfinite impostor gradient')
            norm=torch.nn.utils.clip_grad_norm_(actual,config.max_grad_norm)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in actual): raise FloatingPointError('Nonfinite learner weights')
            losses.append(float(loss.detach()));norms.append(float(norm));actor_losses.append(float(actor.detach()));value_losses.append(float(critic.detach()))
    with torch.no_grad():
        distribution,value=policy(packet);log_ratio=distribution.log_prob(actions)-old;ratio=log_ratio.exp()
        variance=targets.var(unbiased=False)
        explained=float(1-(targets-value).var(unbiased=False)/variance) if variance>1e-8 else None
    return dict(loss=float(np.mean(losses)),policy_loss=float(np.mean(actor_losses)),value_loss=float(np.mean(value_losses)),
        gradient_norm=float(np.mean(norms)),entropy=float(distribution.entropy().mean()),entropy_coefficient=float(entropy),
        approximate_kl=float(((ratio-1)-log_ratio).mean()),clip_fraction=float(((ratio-1).abs()>config.clip).float().mean()),
        explained_variance=explained,return_mean=float(np.mean(returns)),return_std=float(np.std(returns)))


class LearnerSession:
    def __init__(self,seed,*,models=None,record=None,development=False):
        if seed not in (71,83): raise ValueError('Unapproved initialization seed')
        self.seed=seed;self.development=development
        self.env=ImpostorEnv(models=models)
        self.record=record or pilot_manifest(self.env)
        verify_record(self.record,models=self.env.models)
        self.policy=ImpostorPolicy(seed).train()
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=PilotConfig().learning_rate)
        self.generator=torch.Generator().manual_seed(seed^0xA6710)
        self.rng=np.random.default_rng(seed^0x5A17)
        self.episode=0;self.decisions=0;self.simulated_seconds=0.;self.elapsed_seconds=0.
        self.episodes=[];self.updates=[];self.episode_actions=[];self.action_counts={}
        self.initial_hash=weight_hash(self.policy)
        self.packet,_=self.env.reset(seed=self.scenario_seed)

    @property
    def scenario_seed(self):
        start,end=PARTITIONS['development' if self.development else 'train']
        if start+self.episode>end: raise ValueError('Registered training partition exhausted')
        return start+self.episode

    def collect(self,count,deadline=None):
        rollout={k:[] for k in ('packets','actions','logp','values','next_values','rewards','discounts','dones','durations')}
        for _ in range(count):
            if deadline is not None and time()>deadline: raise TimeoutError('Pilot deadline exceeded during collection')
            check_packet(self.packet)
            action,logp,value=self.policy.act(self.packet,generator=self.generator,deterministic=False)
            if not self.packet['mask'][action]: raise ValueError('Policy selected masked action')
            kind=self.env.options[action].kind;self.action_counts[kind]=self.action_counts.get(kind,0)+1
            old_packet=self.packet
            self.packet,reward,done,truncated,info=self.env.step(action)
            if truncated: raise ValueError('Engine timeout must remain a scored terminal outcome')
            check_packet(self.packet)
            if self.env.match.game.metrics['illegal_actions'] or self.env.match.game.metrics['navigation_failures']:
                raise ValueError('Illegal action/navigation failure during pilot')
            with torch.no_grad():
                next_value=0. if done else float(self.policy({k:torch.as_tensor(v).unsqueeze(0) for k,v in self.packet.items()})[1].item())
            for key,item in zip(rollout,(old_packet,action,logp,value,next_value,reward,info['discount'],done,info['elapsed_seconds'])):
                rollout[key].append(item)
            self.episode_actions.append(action);self.decisions+=1;self.simulated_seconds+=info['elapsed_seconds']
            if done:
                self.episodes.append(dict(decisions=self.decisions,**info['episode']))
                self.episode+=1;self.episode_actions=[]
                self.packet,_=self.env.reset(seed=self.scenario_seed)
        return rollout

    def update(self,count=256,deadline=None):
        verify_record(self.record,models=self.env.models,deadline=deadline)
        start=perf_counter();rollout=self.collect(count,deadline)
        collection_seconds=perf_counter()-start;optimization_started=perf_counter()
        entropy=.02+(.005-.02)*min(self.decisions/4096,1.)
        diagnostics=ppo_update(self.policy,self.optimizer,rollout,self.rng,entropy)
        diagnostics.update(collection_seconds=collection_seconds,optimization_seconds=perf_counter()-optimization_started)
        self.env.models.assert_unchanged();self.elapsed_seconds+=perf_counter()-start
        row=dict(decisions=self.decisions,simulated_seconds=self.simulated_seconds,seconds=self.elapsed_seconds,
                 completed_episodes=len(self.episodes),actions=dict(self.action_counts),**diagnostics)
        self.updates.append(row)
        return row

    def save(self,path,*,resume=True):
        path=Path(path)
        if path.exists(): raise FileExistsError('Checkpoint overwrite prohibited')
        self.env.models.assert_unchanged()
        payload=dict(schema=CHECKPOINT_SCHEMA,record=self.record,initialization_seed=self.seed,
            decisions=self.decisions,policy=self.policy.state_dict(),initial_hash=self.initial_hash,
            policy_hash=weight_hash(self.policy),development=self.development)
        if resume:
            payload.update(optimizer=self.optimizer.state_dict(),generator=self.generator.get_state(),
                numpy_rng=self.rng.bit_generator.state,torch_rng=torch.random.get_rng_state(),
                episode=self.episode,episode_actions=self.episode_actions,episodes=self.episodes,updates=self.updates,
                simulated_seconds=self.simulated_seconds,elapsed_seconds=self.elapsed_seconds,actions=self.action_counts,
                packet_hash=packet_hash(self.packet),actor_hash=self.env.actor_trace.hexdigest(),
                opponent_action_hash=self.env.match.trace.hexdigest())
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        torch.save(payload,temporary)
        # Windows rename refuses an existing destination. One OS-locked worker owns the directory.
        if path.exists(): raise FileExistsError('Concurrent checkpoint writer')
        temporary.rename(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def restore(cls,path,*,models=None,deadline=None):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=CHECKPOINT_SCHEMA or 'optimizer' not in payload:
            raise ValueError('Not a resumable impostor checkpoint')
        verify_record(payload['record'],models=models,deadline=deadline)
        obj=cls(payload['initialization_seed'],models=models,record=payload['record'],development=payload['development'])
        obj.episode=payload['episode'];obj.packet,_=obj.env.reset(seed=obj.scenario_seed)
        for action in payload['episode_actions']:
            if deadline is not None and time()>deadline: raise TimeoutError('Deadline while reconstructing resume state')
            obj.packet,_,done,_,_=obj.env.step(action)
            if done: raise ValueError('Resume journal crosses an episode boundary')
        if (packet_hash(obj.packet)!=payload['packet_hash'] or obj.env.actor_trace.hexdigest()!=payload['actor_hash']
            or obj.env.match.trace.hexdigest()!=payload['opponent_action_hash']):
            raise ValueError('Resume replay/memory mismatch')
        obj.policy.load_state_dict(payload['policy'],strict=True)
        if weight_hash(obj.policy)!=payload['policy_hash']: raise ValueError('Checkpoint tensor digest mismatch')
        obj.optimizer.load_state_dict(payload['optimizer']);obj.generator.set_state(payload['generator'])
        if any(group['lr']!=PilotConfig().learning_rate for group in obj.optimizer.param_groups):
            raise ValueError('Checkpoint optimizer config mismatch')
        if any(not torch.isfinite(p).all() for p in obj.policy.parameters()):
            raise ValueError('Nonfinite resumed weights')
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for state in obj.optimizer.state.values() for v in state.values()):
            raise ValueError('Nonfinite resumed optimizer')
        obj.rng.bit_generator.state=payload['numpy_rng'];torch.random.set_rng_state(payload['torch_rng'])
        for field in ('decisions','episode_actions','episodes','updates','simulated_seconds','elapsed_seconds','initial_hash'):
            setattr(obj,field,payload[field])
        obj.action_counts=payload['actions']
        return obj


def load_inference(path,*,models=None,deadline=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!=CHECKPOINT_SCHEMA: raise ValueError('Wrong impostor checkpoint schema')
    verify_record(payload['record'],models=models,deadline=deadline)
    policy=ImpostorPolicy(payload['initialization_seed'])
    policy.load_state_dict(payload['policy'],strict=True);policy.eval().requires_grad_(False)
    if weight_hash(policy)!=payload['policy_hash'] or any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError('Invalid impostor checkpoint tensors')
    return policy,payload
