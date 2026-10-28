"""Atomic observer telemetry shared by engineering and approved training runs."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter


def atomic_json(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(path)


class Progress:
    def __init__(self,folder,workers,resources,*,target=0,mirror=None):
        self.folder=Path(folder);self.workers=workers;self.resources=resources;self.target=target
        self.mirror=Path(mirror) if mirror else None;self.started=perf_counter();self.matches=[]
        self.components=Counter();self.ppo={};self.checkpoint=None;self.stage='preparing';self.history=[]
        self.counters=dict(individual_decisions=0,complete_matches=0,update=0)
        self.monitor_dir=(self.folder/'live').resolve();self.active={}

    def publish(self,active=None,**extra):
        if active is not None:self.active=active
        elapsed=perf_counter()-self.started
        decisions=self.counters['individual_decisions']+(self.active.get('decisions',0) if self.stage in ('collecting','benchmark','PPO','PPO benchmark') else 0)
        rate=decisions/max(1,elapsed)
        record=dict(schema='patrick-jane-progress-v1',stage=self.stage,counters=dict(self.counters),
            workers=self.workers,elapsed=elapsed,target_decisions=self.target,overall_decisions=decisions,
            throughput=rate,eta_seconds=max(0,self.target-decisions)/rate if rate and self.target else 0,
            checkpoint=str(self.checkpoint) if self.checkpoint else None,monitor_dir=str(self.monitor_dir),
            environment_progress=self.active,resources=self.resources.summary(),ppo=self.ppo,
            outcomes=dict(games=len(self.matches),wins=sum(m['winner']=='crewmate' for m in self.matches),
                losses=sum(m['winner']=='impostor' for m in self.matches),draws=sum(m['winner'] is None for m in self.matches)),
            reward_components=dict(self.components),learning_curve=self.history[-100:],**extra)
        atomic_json(self.folder/'status.json',record)
        if self.mirror:atomic_json(self.mirror,record)

    def settle(self,rollout):
        self.matches.extend(rollout['matches'])
        for components in rollout['reward_components'].values():self.components.update(components)
        self.active={}

    def final_streams(self,matches,update,record_every):
        return dict(streams=[dict(environment=i,seed=m['seed'],simulation_seconds=m['simulation_seconds'],
            phase='finished',complete=True,snapshot=str(self.monitor_dir/f'env-{i:02d}.json'),
            replay=str((self.folder/'replays'/f'env-{i:02d}-seed-{m["seed"]}-update-{update}.jsonl').resolve())
            if record_every and i%record_every==0 else None) for i,m in enumerate(matches)])
