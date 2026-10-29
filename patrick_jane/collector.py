"""Independent CPU simulations, one batched policy, no per-worker optimizers."""
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase3_engine import stream_seed
from .worker import serve,Match
from .network import collate
from .actions import choose


class Pool:
    def __init__(self,count,processes=True):
        self.count=count;self.processes=processes;self.connections=[];self.children=[];self.matches={}
        self.sync_seconds=0.;self.worker_critical_seconds=0.;self.worker_cpu_seconds=0.;self.timeout=180.
        if processes:
            os.environ['PYTHONHASHSEED']='0'  # Stable process-level tie ordering; never actor input.
            ctx=mp.get_context('spawn')
            for _ in range(count):
                parent,child=ctx.Pipe();p=ctx.Process(target=serve,args=(child,),daemon=True);p.start();child.close()
                self.connections.append(parent);self.children.append(p)

    def call(self,commands):
        began=perf_counter();result={}
        if self.processes:
            for i,command in commands.items():self.connections[i].send(command)
            for i in commands:
                timeout=min(self.timeout,max(0.,getattr(self,'deadline',float('inf'))-perf_counter()))
                if not self.connections[i].poll(timeout):raise TimeoutError('Simulation worker exceeded bounded RPC deadline')
                status,value=self.connections[i].recv()
                if status!='ok':raise RuntimeError(value)
                result[i]=value
        else:
            for i,(command,payload) in commands.items():
                local_started=perf_counter()
                if command=='reset':self.matches[i]=Match(**payload);result[i]=self.matches[i].prepare()
                else:result[i]=self.matches[i].advance(payload)
                result[i]['worker_simulation_seconds']=perf_counter()-local_started
        times=[r['worker_simulation_seconds'] for r in result.values()]
        self.worker_critical_seconds+=max(times,default=0.);self.worker_cpu_seconds+=sum(times)
        self.sync_seconds+=perf_counter()-began
        return result

    def close(self):
        for c in self.connections:
            try:c.send(('close',None))
            except (OSError,EOFError):pass
        for p in self.children:
            p.join(3.)
            if p.is_alive():p.terminate();p.join(3.)
        for c in self.connections:c.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


@dataclass
class Transition:
    packet:object
    choice:object
    hidden:np.ndarray
    logp:float
    value:float
    time:float
    reward:float=0.
    duration:float=0.
    next_value:float=0.
    done:bool=False


def collect(policy,pool,experiment,seeds,*,device='cpu',update=0,training_step=0,replay_dir=None,
            record_every=0,held_out=False,wall_limit=180.,status_callback=None,checkpoint=None,monitor_dir=None,decision_limit=None):
    if not 0<len(seeds)<=pool.count:raise ValueError('One independent seed per active worker required')
    began=perf_counter();simulation_started=perf_counter()
    initial_cpu=pool.worker_cpu_seconds;initial_critical=pool.worker_critical_seconds;initial_sync=pool.sync_seconds
    pool.deadline=began+wall_limit
    responses=pool.call({i:('reset',dict(seed=seed,experiment=experiment,held_out=held_out,
        live_dir=str(monitor_dir) if monitor_dir else None,
        replay=str(Path(replay_dir)/f'env-{i:02d}-seed-{seed}-update-{update}.jsonl') if replay_dir and record_every and i%record_every==0 else None,
        metadata=dict(session='patrick-jane',environment=i,update=update,checkpoint=checkpoint or f'fresh-initialization-seed17/update-{update}',
                      training_step=training_step))) for i,seed in enumerate(seeds)})
    simulation_seconds=perf_counter()-simulation_started
    trajectories=defaultdict(list);pending={};hidden={};generators={};completed={};counts=0
    initial_rewards=defaultdict(float)
    inference_seconds=0.;last_notice=began;reward_components=defaultdict(lambda:defaultdict(float))
    def pay(response,i):
        for time,pid,components in response['rewards']:
            for name,value in components.items():reward_components[(i,pid)][name]+=value
            tr=pending.get((i,pid))
            if tr is not None:
                tr.reward+=experiment.gamma**(max(0.,time-tr.time)/experiment.lobby.decision_interval)*sum(components.values())
            else:initial_rewards[(i,pid)]+=sum(components.values())
    while True:
        if perf_counter()-began>wall_limit:raise TimeoutError('Bounded development collector reached wall limit; no partial match used for PPO')
        for i,r in responses.items():pay(r,i)
        keys=[];packets=[]
        for i,r in responses.items():
            if r['terminal']:
                completed[i]=r['result']
                for key in tuple(pending):
                    if key[0]==i:
                        tr=pending.pop(key);tr.done=True;tr.duration=r['time']-tr.time;trajectories[key].append(tr)
                continue
            for pid,packet in r['packets'].items():keys.append((i,pid));packets.append(packet)
        if len(completed)==len(seeds):break
        commands={i:{} for i in responses if i not in completed}
        if packets:
            started=perf_counter()
            hs=[]
            for key in keys:
                if key not in hidden:hidden[key]=torch.zeros(256)
                if key not in generators:
                    g=torch.Generator();g.manual_seed(stream_seed(seeds[key[0]],'pj:policy:'+key[1])%(2**63-1));generators[key]=g
                hs.append(hidden[key])
            with torch.no_grad():output,new_hidden=policy(collate(packets,device),torch.stack(hs).to(device))
            # One transfer per tensor batch. Per-actor GPU categorical calls caused
            # hundreds of device synchronizations; CPU streams remain independent.
            output={k:v.cpu() for k,v in output.items()};new_hidden=new_hidden.cpu()
            for j,(key,packet) in enumerate(zip(keys,packets)):
                i,pid=key;out={k:v[j] for k,v in output.items()}
                choice,lp,en,probs=choose(out,packet,generator=generators[key])
                if key in pending:
                    tr=pending.pop(key);tr.duration=responses[i]['time']-tr.time;tr.next_value=float(out['value']);trajectories[key].append(tr)
                pending[key]=Transition(packet,choice,hidden[key].detach().cpu().numpy().copy(),float(lp),float(out['value']),responses[i]['time'])
                pending[key].reward+=initial_rewards.pop(key,0.)
                hidden[key]=new_hidden[j].detach();counts+=1
                if decision_limit is not None and counts>decision_limit:
                    raise RuntimeError('Absolute decision ceiling reached; partial matches withheld from PPO')
                commands[i][pid]=dict(choice=choice,logp=float(lp),value=float(out['value']),entropy=float(en),
                    probabilities=probs if (replay_dir and record_every and i%record_every==0 or responses[i].get('monitoring')) else {},
                    update=update,training_step=training_step,collection_decisions=counts)
            inference_seconds+=perf_counter()-started
        started=perf_counter();responses=pool.call({i:('advance',decisions) for i,decisions in commands.items()});simulation_seconds+=perf_counter()-started
        if status_callback and perf_counter()-last_notice>10:
            status_callback(dict(decisions=counts,complete=len(completed),environments=len(seeds),
                elapsed=perf_counter()-began,update=update,streams=[dict(environment=i,seed=seeds[i],
                    simulation_seconds=completed[i]['simulation_seconds'] if i in completed else responses.get(i,{}).get('time',0),
                    complete=i in completed,
                    phase='finished' if i in completed else responses.get(i,{}).get('phase'),
                    snapshot=str(Path(monitor_dir)/f'env-{i:02d}.json') if monitor_dir else None,
                    replay=str(Path(replay_dir)/f'env-{i:02d}-seed-{seeds[i]}-update-{update}.jsonl')
                    if replay_dir and record_every and i%record_every==0 else None) for i in range(len(seeds))]));last_notice=perf_counter()
    summaries=[completed[i] for i in range(len(seeds))]
    elapsed=perf_counter()-began
    return dict(trajectories=dict(trajectories),matches=summaries,individual_decisions=counts,
        elapsed=elapsed,decisions_per_second=counts/elapsed,games_per_second=len(summaries)/elapsed,
        inference_seconds=inference_seconds,environment_and_sync_seconds=simulation_seconds,
        recording_seconds=sum(m['recording_seconds'] for m in summaries),
        worker_cpu_seconds=pool.worker_cpu_seconds-initial_cpu,worker_critical_seconds=pool.worker_critical_seconds-initial_critical,
        rpc_sync_overhead_seconds=max(0.,pool.sync_seconds-initial_sync-(pool.worker_critical_seconds-initial_critical)),
        reward_components={f'{i}/{p}':dict(v) for (i,p),v in reward_components.items()})
