"""Trainer-only event ledger. Never passed to actor observation or policy encoding."""
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from social_deduction.actor import Role
from social_deduction.phase3_api import IntentKind


@dataclass(frozen=True)
class Frame:
    components:dict
    counts:dict


class RewardLedger:
    def __init__(self,focal,roles,config,dt=.2):
        self.focal=focal;self.roles=MappingProxyType(dict(roles));self.config=config;self.dt=dt
        self.frames=[];self.tasks=set();self.reports=set();self.ballots=set()
        self.alive=True;self.death_paid=False;self.terminal_paid=False
        self.components=Counter();self.counts=Counter()

    def consume(self,events,*,active,terminal=False,winner=None,intent=None,legal=False):
        c=Counter(time=self.config.time_per_decision*self.dt/.6);n=Counter()
        for e in events:
            kind=e['kind']
            if kind=='elimination' and e['victim_id']==self.focal:self.alive=False
            if kind=='ejection' and e.get('player_id')==self.focal:self.alive=False
            if kind=='task_completed' and e['player_id']==self.focal and e['task_id'] not in self.tasks:
                self.tasks.add(e['task_id']);key='alive_tasks' if self.alive else 'ghost_tasks'
                n[key]+=1;c[key]+=self.config.alive_task if self.alive else self.config.ghost_task
            if kind=='report' and e['reporter_id']==self.focal and legal and intent is not None:
                if intent.kind is IntentKind.REPORT and intent.target==e['victim_id'] and e['victim_id'] not in self.reports:
                    self.reports.add(e['victim_id']);n['reports']+=1;c['reports']+=self.config.report
            if kind=='vote_submitted' and e['voter_id']==self.focal and legal and intent is not None:
                key=e['meeting_id'];target=e['target_id']
                selected=((intent.kind is IntentKind.VOTE and intent.target==target) or
                          (intent.kind is IntentKind.SKIP and target is None))
                if selected and key not in self.ballots:
                    self.ballots.add(key)
                    category='skips' if target is None else 'correct_votes' if self.roles[target] is Role.IMPOSTOR else 'incorrect_votes'
                    n[category]+=1;c[category]+=getattr(self.config,{'skips':'skip','correct_votes':'correct_vote','incorrect_votes':'incorrect_vote'}[category])
            if kind=='vote_submitted' and e['voter_id']==self.focal and e['target_id'] is None and e['meeting_id'] not in self.ballots:
                self.ballots.add(e['meeting_id']);n['automatic_skips']+=1
        self.alive=bool(active)
        if not active and not self.death_paid:
            self.death_paid=True;n['death']+=1;c['death']+=self.config.death
        if terminal and not self.terminal_paid:
            self.terminal_paid=True
            category='win' if winner is Role.CREWMATE else 'loss' if winner is Role.IMPOSTOR else 'draw'
            n[category]+=1;c[category]+=getattr(self.config,category)
        frame=Frame(dict(c),dict(n));self.frames.append(frame);self.components.update(c);self.counts.update(n)
        return frame

    def rollup(self,start,gamma=.99):
        frames=self.frames[start:];raw=Counter();discounted=Counter();counts=Counter()
        for i,frame in enumerate(frames):
            raw.update(frame.components);counts.update(frame.counts)
            for key,value in frame.components.items():discounted[key]+=gamma**(i*self.dt/.6)*value
        return dict(reward=sum(discounted.values()),undiscounted_reward=sum(raw.values()),
            components=dict(raw),discounted_components=dict(discounted),counts=dict(counts),
            elapsed_seconds=len(frames)*self.dt,discount=gamma**(len(frames)*self.dt/.6))
