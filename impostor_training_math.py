"""Versioned candidate math for the NEXT trainer; frozen V4.1 does not import it.

Instantaneous terminal kills legitimately consume zero simulator time. This GAE
accepts that case without fabricating elapsed time. Wire it into a separately
versioned/registered learner before any future full study; do not patch old sources.
"""
import numpy as np

SCHEMA='impostor-terminal-time-gae-v2'


def gae(rewards,values,next_values,discounts,dones,durations,lambda_per_second=.995):
    arrays=[np.asarray(a) for a in (rewards,values,next_values,discounts,dones,durations)]
    if len({a.shape for a in arrays})!=1 or arrays[0].ndim!=1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid rollout arrays')
    r,v,nv,g,done,dt=arrays
    if (np.any((done!=0)&(done!=1)) or np.any(dt<0) or np.any((dt==0)&(done==0))
        or not np.isfinite(lambda_per_second) or not 0<lambda_per_second<=1):
        raise ValueError('Zero duration is allowed only for a terminal transition')
    advantage=np.zeros(len(r),np.float64);carry=0.
    for i in reversed(range(len(r))):
        live=1.-float(done[i])
        delta=r[i]+g[i]*nv[i]*live-v[i]
        carry=delta+g[i]*(lambda_per_second**dt[i])*live*carry
        advantage[i]=carry
    return advantage.astype(np.float32),(advantage+v).astype(np.float32)
