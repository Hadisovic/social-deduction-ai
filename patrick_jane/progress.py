"""Atomic observer telemetry shared by engineering and approved training runs."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter,sleep


def duration_text(seconds):
    if seconds is None:return 'measuring'
    seconds=max(0,int(seconds));hours,rest=divmod(seconds,3600);minutes,seconds=divmod(rest,60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes}m {seconds:02d}s'


def progress_line(record):
    c=record.get('counters',{});decisions=record.get('overall_decisions',c.get('individual_decisions',0));target=record.get('target_decisions',0)
    percent=min(100.,max(0.,100*decisions/target)) if target else 0.
    return (f"PATRICK JANE | {percent:.1f}% | {decisions:,} / {target:,} decisions | {record.get('workers',0)} workers | "
        f"{c.get('update',0)} PPO updates | {record.get('throughput',0):.1f} decisions/s | ETA {duration_text(record.get('eta_seconds'))}")


def atomic_json(path,record,*,attempts=8,compact=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record,indent=None if compact else 2,
        separators=(',',':') if compact else None,allow_nan=False)+'\n')
    for attempt in range(attempts):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt+1==attempts:raise
            sleep(.003)  # Windows readers can briefly deny rename; never spin indefinitely.


class Progress:
    def __init__(self,folder,workers,resources,*,target=0,mirror=None):
        self.folder=Path(folder);self.workers=workers;self.resources=resources;self.target=target
        self.mirror=Path(mirror) if mirror else None;self.started=perf_counter();self.matches=[]
        self.components=Counter();self.ppo={};self.checkpoint=None;self.stage='preparing';self.history=[]
        self.counters=dict(individual_decisions=0,complete_matches=0,update=0)
        self.monitor_dir=(self.folder/'live').resolve();self.active={}
        self.console=False;self.last_console=-1e20

    def publish(self,active=None,**extra):
        if active is not None:self.active=active
        elapsed=perf_counter()-self.started
        decisions=self.counters['individual_decisions']+(self.active.get('decisions',0) if self.stage in ('collecting','benchmark','PPO','PPO benchmark') else 0)
        rate=decisions/max(1,elapsed)
        record=dict(schema='patrick-jane-progress-v1',stage=self.stage,counters=dict(self.counters),
            workers=self.workers,elapsed=elapsed,target_decisions=self.target,overall_decisions=decisions,
            throughput=rate,eta_seconds=max(0,self.target-decisions)/rate if rate and self.target else None,
            checkpoint=str(self.checkpoint) if self.checkpoint else None,monitor_dir=str(self.monitor_dir),
            environment_progress=self.active,resources=self.resources.summary(),ppo=self.ppo,
            outcomes=dict(games=len(self.matches),wins=sum(m['winner']=='crewmate' for m in self.matches),
                losses=sum(m['winner']=='impostor' for m in self.matches),
                truncations=sum(m.get('truncated',False) for m in self.matches),
                legitimate_games=sum(m['winner'] in ('crewmate','impostor') for m in self.matches),
                draws=sum(m['winner'] is None and not m.get('truncated') for m in self.matches)),
            reward_components=dict(self.components),learning_curve=self.history[-100:],**extra)
        record.update(progress_percent=min(100.,max(0.,100*decisions/self.target)) if self.target else 0.,
            checkpointed_additional_decisions=self.counters['individual_decisions'],
            collected_additional_decisions=decisions,
            training_state=self.stage if self.stage in ('complete','interrupted','failed') else 'running')
        record['progress_line']=progress_line(record)
        atomic_json(self.folder/'status.json',record)
        if self.mirror:atomic_json(self.mirror,record)
        if self.console and (perf_counter()-self.last_console>=5 or self.stage in ('complete','interrupted','failed')):
            print(record['progress_line']+f" | elapsed {duration_text(elapsed)} | games {record['outcomes']['legitimate_games']} | "
                f"truncated {record['outcomes']['truncations']} | state {record['training_state']} | stage {self.stage}",flush=True)
            print('LATEST CHECKPOINT: '+str(record['checkpoint']),flush=True);self.last_console=perf_counter()

    def settle(self,rollout):
        self.matches.extend(rollout['matches'])
        for components in rollout['reward_components'].values():self.components.update(components)
        self.active={}

    def final_streams(self,matches,update,record_every):
        return dict(streams=[dict(environment=i,seed=m['seed'],simulation_seconds=m['simulation_seconds'],
            phase='finished',complete=True,snapshot=str(self.monitor_dir/f'env-{i:02d}.json'),
            replay=str((self.folder/'replays'/f'env-{i:02d}-seed-{m["seed"]}-update-{update}.jsonl').resolve())
            if record_every and i%record_every==0 else None) for i,m in enumerate(matches)])
