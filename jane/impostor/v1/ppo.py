"""Complete-match recurrent PPO; each epoch reconstructs private memory in order."""
import math
from time import perf_counter
import numpy as np
import torch


def advantages(trajectory, end_time, lam=.995):
    values = np.array([r.value for r in trajectory],np.float64)
    rewards = np.array([r.reward for r in trajectory],np.float64)
    adv = np.zeros(len(trajectory),np.float64); carry = 0.
    for i in reversed(range(len(trajectory))):
        next_value = values[i+1] if i+1<len(trajectory) else 0.
        next_time = trajectory[i+1].packet.time if i+1<len(trajectory) else end_time
        elapsed = max(0.,next_time-trajectory[i].packet.time)
        delta = rewards[i]+next_value-values[i]
        carry = delta + lam**elapsed*carry
        adv[i] = carry
    return adv.astype(np.float32),(adv+values).astype(np.float32)


def update(model, optimizer, trajectories, end_time, epochs=4, sequence=128, deadline=None):
    prepared = []
    for trajectory in trajectories:
        if trajectory:
            adv, returns = advantages(trajectory,end_time)
            prepared.append((trajectory,adv,returns))
    if not prepared: raise ValueError('No completed trajectories')
    all_adv = np.concatenate([a for _,a,_ in prepared]); mean,std = all_adv.mean(),max(all_adv.std(),1e-8)
    logs = []; model.train()
    # No optimizer step between chunks of one trajectory: recurrent states in
    # that pass are generated with one unchanged parameter version, not stale h.
    for epoch in range(epochs):
        for trajectory,adv,returns in prepared:
            optimizer.zero_grad(); hidden = torch.zeros(1,256); parts=[]
            for start in range(0,len(trajectory),sequence):
                if deadline is not None and perf_counter() >= deadline:
                    optimizer.zero_grad(); model.eval()
                    return logs  # Preserve only completed optimizer passes.
                rows = trajectory[start:start+sequence]
                distributions,values,hidden = model([r.packet for r in rows],hidden.detach(),sequence=True)
                new_lp = torch.stack([d.log_prob(torch.tensor(r.action)) for d,r in zip(distributions,rows)])
                old_lp = torch.tensor([r.logp for r in rows])
                ratio = (new_lp-old_lp).exp()
                advantage = torch.tensor((adv[start:start+len(rows)]-mean)/std)
                target = torch.tensor(returns[start:start+len(rows)])
                policy = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                value = .5*(values-target).square().mean()
                entropy = torch.stack([d.entropy() for d in distributions]).mean()
                loss = policy+.5*value-.01*entropy
                if not torch.isfinite(loss): raise ValueError('Nonfinite PPO loss')
                (loss*len(rows)/len(trajectory)).backward()
                parts.append(dict(policy=float(policy.detach()),value=float(value.detach()),
                                  entropy=float(entropy.detach()),kl=float((old_lp-new_lp).mean().detach())))
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),.5))
            if not math.isfinite(norm): raise ValueError('Nonfinite PPO gradients')
            if deadline is not None and perf_counter() >= deadline:
                optimizer.zero_grad(); model.eval()
                return logs
            optimizer.step()
            logs.append({**{k:float(np.mean([p[k] for p in parts])) for k in parts[0]},'gradient_norm':norm,'epoch':epoch})
    model.eval()
    return logs
