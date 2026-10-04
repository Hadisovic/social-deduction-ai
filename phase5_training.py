"""Small reference PPO implementation with masked updates and elapsed-time GAE."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase5_env import CrewmateStrategicEnv
from phase5_policy import StrategicPolicy,tensors

ROOT=Path(__file__).resolve().parent


def time_aware_gae(rewards,values,next_values,discounts,dones,durations,lam=.95):
    advantage=np.zeros(len(rewards),np.float32);carry=0.
    for i in reversed(range(len(rewards))):
        live=1-float(dones[i])
        delta=rewards[i]+discounts[i]*next_values[i]*live-values[i]
        carry=delta+discounts[i]*(lam**(durations[i]/.6))*live*carry
        advantage[i]=carry
    return advantage,advantage+values


def update(policy,optimizer,rollout,*,epochs=4,batch_size=64,entropy=.02,rng=None):
    rng=rng or np.random.default_rng(0)
    packets={k:torch.as_tensor(np.stack([p[k] for p in rollout['packets']])) for k in rollout['packets'][0]}
    actions=torch.tensor(rollout['actions'],dtype=torch.long)
    old=torch.tensor(rollout['logp'],dtype=torch.float32)
    adv,returns=time_aware_gae(*[np.asarray(rollout[k]) for k in ('rewards','values','next_values','discounts','dones','durations')])
    adv=torch.from_numpy(adv);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    returns=torch.from_numpy(returns.astype(np.float32))
    losses=[]
    for _ in range(epochs):
        for ids in np.array_split(rng.permutation(len(actions)),max(1,math_ceil(len(actions)/batch_size))):
            distribution,value=policy({k:v[ids] for k,v in packets.items()})
            ratio=(distribution.log_prob(actions[ids])-old[ids]).exp()
            clipped=torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids])
            loss=-clipped.mean()+.5*(value-returns[ids]).square().mean()-entropy*distribution.entropy().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def math_ceil(value):return int(np.ceil(value))


def train(directory,steps=32768,seed=17,ablation=False,quick=False):
    if steps<2:raise ValueError('At least two transitions required')
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'policy.pt').exists():raise FileExistsError('Choose a fresh output directory; existing checkpoint is preserved')
    config=json.loads((ROOT/'artifacts/phase5/config.json').read_text())
    config.update(seed=seed,ablation=ablation,steps=steps,quick=quick)
    sources={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
             for name in ('phase5_features.py','phase5_env.py','phase5_policy.py','phase5_runner.py','phase5_training.py','among_us_map.py')}
    config['sources_sha256']=sources
    (directory/'run-config.json').write_text(json.dumps(config,indent=2)+'\n')
    env=CrewmateStrategicEnv(ablation=ablation);policy=StrategicPolicy()
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    rng=np.random.default_rng(seed);episodes=[];history=[];began=perf_counter()
    baseline={k:v.clone() for k,v in env.belief_model.network.state_dict().items()}
    initial={k:v.clone() for k,v in policy.state_dict().items()}
    match_base=960000 if quick else 600000
    # Fixed, pre-registered match seeds shared across model/ablation conditions.
    episode_index=0;packet,_=env.reset(seed=match_base)
    completed=0
    while completed<steps:
        rollout={k:[] for k in ('packets','actions','logp','values','next_values','rewards','discounts','dones','durations')}
        for _ in range(min(config['rollout_steps'],steps-completed)):
            action,logp,value=policy.act(packet)
            new,reward,done,_,info=env.step(action)
            with torch.no_grad():next_value=float(policy(tensors(new))[1].item()) if not done else 0.
            for k,v in zip(rollout,(packet,action,logp,value,next_value,reward,info['discount'],done,info['elapsed_seconds'])):
                rollout[k].append(v)
            completed+=1
            if done:
                episodes.append({'match_seed':match_base+episode_index,'training_step':completed,**info['episode']})
                with (directory/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(episodes[-1])+'\n')
                episode_index+=1
                if episode_index >= (100 if quick else 20000):raise ValueError('Training split exhausted')
                packet,_=env.reset(seed=match_base+episode_index)
            else:packet=new
        entropy=.02+(.001-.02)*completed/steps
        loss=update(policy,optimizer,rollout,entropy=entropy,rng=rng)
        row={'steps':completed,'episodes':len(episodes),'loss':loss,'seconds':perf_counter()-began}
        history.append(row);print(json.dumps(row),flush=True)
        with (directory/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if not quick and completed%8192==0 and completed<steps:
            policy.save(directory/f'checkpoint-{completed}.pt',{**config,'steps':completed,
                'features_sha256':sources['phase5_features.py'],'geometry_sha256':sources['among_us_map.py']})
    assert all(torch.equal(v,baseline[k]) for k,v in env.belief_model.network.state_dict().items())
    changed=any(not torch.equal(v,initial[k]) for k,v in policy.state_dict().items())
    if not changed:raise AssertionError('Training did not change policy weights')
    metadata={**config,'belief_frozen_verified':True,'weights_changed':changed,
        'features_sha256':sources['phase5_features.py'],
        'geometry_sha256':sources['among_us_map.py']}
    policy.save(directory/'policy.pt',metadata)
    result={'config':metadata,'history':history,'episodes':episodes,'seconds':perf_counter()-began,
            'checkpoint_sha256':hashlib.sha256((directory/'policy.pt').read_bytes()).hexdigest(),
            'status':'smoke_only' if quick else 'trained_not_evaluated'}
    (directory/'training.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
