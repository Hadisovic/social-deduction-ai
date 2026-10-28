"""Truncated recurrent PPO, elapsed-time GAE, independent complete actor episodes."""
from time import perf_counter
import numpy as np
import torch
from phase5_training import time_aware_gae
from .network import collate
from .actions import evaluate_batch


def prepare(trajectories,experiment):
    sequences=[];all_adv=[]
    for key,trajectory in trajectories.items():
        if not trajectory:continue
        if not trajectory[-1].done:raise ValueError('PPO requires complete match settlement')
        values=np.asarray([t.value for t in trajectory],np.float32)
        durations=np.asarray([t.duration for t in trajectory],np.float32)
        # GAE helper uses a .6-second reference, identical to the locked lobby default.
        if experiment.lobby.decision_interval!=.6:raise ValueError('Elapsed-time reference change requires explicit protocol registration')
        adv,returns=time_aware_gae(np.asarray([t.reward for t in trajectory],np.float32),values,
            np.asarray([t.next_value for t in trajectory],np.float32),experiment.gamma**(durations/.6),
            np.asarray([t.done for t in trajectory]),durations,experiment.gae_lambda)
        for start in range(0,len(trajectory),experiment.sequence_length):
            end=min(start+experiment.sequence_length,len(trajectory))
            sequences.append((key,trajectory[start:end],adv[start:end],returns[start:end]));all_adv.extend(adv[start:end])
    mean=float(np.mean(all_adv));std=float(np.std(all_adv))
    return [(key,tr,(adv-mean)/(std+1e-8),ret) for key,tr,adv,ret in sequences]


def update(policy,optimizer,trajectories,experiment,*,rng=None,device='cpu',sequences_per_batch=4):
    started=perf_counter();rng=rng or np.random.default_rng(17)
    sequences=prepare(trajectories,experiment)
    if not sequences:raise ValueError('No actor experience for PPO')
    metrics=[]
    for epoch in range(experiment.epochs):
        order=rng.permutation(len(sequences))
        for offset in range(0,len(order),sequences_per_batch):
            optimizer.zero_grad(set_to_none=True);batch=order[offset:offset+sequences_per_batch]
            total=sum(len(sequences[i][1]) for i in batch);loss_value=0.;entropy_value=0.;kl_value=0.
            for index in batch:
                key,tr,advantages,returns=sequences[index]
                hidden=torch.as_tensor(tr[0].hidden,device=device).unsqueeze(0)
                sequence_output,hidden=policy.forward_sequence(collate([t.packet for t in tr],device),hidden)
                logp,entropy=evaluate_batch(sequence_output,[t.packet for t in tr],[t.choice for t in tr])
                old_logp=torch.as_tensor([t.logp for t in tr],device=device)
                ratio=(logp-old_logp).clamp(-20,20).exp();advantage=torch.as_tensor(advantages,device=device)
                actor=-torch.minimum(ratio*advantage,ratio.clamp(1-experiment.clip,1+experiment.clip)*advantage)
                value=(sequence_output['value']-torch.as_tensor(returns,device=device))**2
                loss=actor+.5*value-experiment.entropy*entropy
                loss_value+=float(loss.detach().sum())/total;entropy_value+=float(entropy.detach().sum())/total
                kl_value+=float((old_logp-logp).detach().sum())/total
                (loss.sum()/total).backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),experiment.max_grad_norm)
            if not torch.isfinite(norm) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.parameters()):
                optimizer.zero_grad(set_to_none=True);raise FloatingPointError('Nonfinite PPO gradients; update aborted')
            optimizer.step()
            metrics.append(dict(loss=loss_value,entropy=entropy_value,approx_kl=kl_value,grad_norm=float(norm)))
    return dict(updates=len(metrics),epochs=experiment.epochs,sequences=len(sequences),
        decisions=sum(len(tr) for _,tr,_,_ in sequences),elapsed=perf_counter()-started,
        **{key:float(np.mean([m[key] for m in metrics])) for key in metrics[0] if key!='updates'})
