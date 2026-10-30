"""Exact PJ coefficients, event uniqueness, private bonuses and delayed team credit."""
from collections import defaultdict
from social_deduction.actor import Role

COEFFICIENTS=dict(win=10.,loss=-10.,task=1.,ghost_task=0.,repair=3.,impostor_ejection=4.,
                  innocent_ejection=-2.,report=1.25,exploration=.02,new_player=.05,death=-2.,idle=-.01,
                  invalid=0.,vote=0.,accuse=0.,defend=0.)


class RewardLedger:
    def __init__(self,game):
        self.coefficients={**COEFFICIENTS,'task':getattr(getattr(game,'experiment',None),'living_task_reward',COEFFICIENTS['task'])}
        self.crew=tuple(pid for pid,p in game.players.items() if p.role is Role.CREWMATE)
        self.cursor=0;self.credited=set();self.exploration=defaultdict(float);self.players_seen=defaultdict(set)
        self.idle=defaultdict(float);self.entries=[];self.terminal=False

    def collect(self,game):
        result={pid:defaultdict(float) for pid in self.crew}
        def add(pid,kind,key):
            family='task_completion' if kind in ('task','ghost_task') else kind
            unique=(None if kind=='report' else pid,family,key)
            if pid in result and unique not in self.credited:
                self.credited.add(unique);result[pid][kind]+=self.coefficients[kind]
        for e in game.event_log[self.cursor:]:
            kind=e['kind']
            if kind=='task_completed':
                pid=e['player_id'];ghost=e.get('ghost',False)
                add(pid,'ghost_task' if ghost else 'task',e['task_id'])
            elif kind=='report':add(e['reporter_id'],'report',e['victim_id'])
            elif kind=='elimination':add(e['victim_id'],'death',e['victim_id'])
            elif kind=='sabotage_repaired':
                for pid in e['participants']:add(pid,'repair',e['instance'])
            elif kind=='ejection' and e.get('player_id'):
                role=game.players[e['player_id']].role
                reward='impostor_ejection' if role is Role.IMPOSTOR else 'innocent_ejection'
                for pid in self.crew:add(pid,reward,e['meeting_id'])
        self.cursor=len(game.event_log)
        if game.is_terminal and not self.terminal:
            self.terminal=True
            reward='win' if game.result.winner is Role.CREWMATE else 'loss' if game.result.winner is Role.IMPOSTOR else None
            if reward:
                for pid in self.crew:add(pid,reward,'terminal')
        self._record(game.time,result)
        return {pid:dict(v) for pid,v in result.items()}

    def observation_bonus(self,pid,obs,fresh,nav,memory):
        result=defaultdict(float)
        # Once per newly seen area batch, capped. Revisits have zero new coverage.
        if fresh and self.exploration[pid]<.3-1e-9:
            amount=min(COEFFICIENTS['exploration'],.3-self.exploration[pid])
            self.exploration[pid]+=amount;result['exploration']=amount
        for p in obs.actor.visible_players:
            if p.player_id not in self.players_seen[pid]:
                self.players_seen[pid].add(p.player_id)
                if len(self.players_seen[pid])<=14:result['new_player']+=.05
        return dict(result)

    def waiting(self,pid,elapsed,avoidable):
        if not avoidable:self.idle[pid]=0.;return {}
        self.idle[pid]+=elapsed
        count=int((self.idle[pid]+1e-9)//10.)
        self.idle[pid]-=count*10.
        return {'idle':-.01*count} if count else {}

    def _record(self,time,result):
        for pid,components in result.items():
            if components:self.entries.append(dict(time=time,player_id=pid,components=dict(components)))
