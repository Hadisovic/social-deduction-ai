"""Seeded tactical scripts accepting only detached actor-local observations."""
from random import Random
from collections import deque
import math
from social_deduction.actor import Phase
from social_deduction.phase3_api import Intent,IntentKind
from .api import Action,Observation
from .opponents import Opponent
from .a2_geometry import vents,door_groups


class TacticalOpponent:
    def __init__(self,seed,public_geometry):
        self.rng=Random(seed);self.social=Opponent(seed,allowed_sabotages=('lights',))
        # Continuous private preferences, not a public archetype label.
        self.aggression=self.rng.uniform(.45,.95);self.caution=self.rng.random()
        self.trapping=self.rng.uniform(.15,.65);self.venting=self.rng.uniform(.15,.6)
        self.last_seen={};self.visited=deque(maxlen=8);self.goal=None;self.next_change=0.;self.rest_until=0.
        self.vents=vents();self.groups=door_groups();self.geometry=public_geometry
        self.hiding_until=0.;self.last_vent=None

    def decide(self,obs):
        if type(obs) is not Observation:raise TypeError('Opponent accepts detached observations only')
        a=obs.actor;now=a.time_seconds;pid=a.own.player_id
        if not a.own.active:return Intent(IntentKind.WAIT)
        if a.context.phase is not Phase.ROAMING:return self.social.decide(obs)
        if not obs.vent_transition_ready:return Intent(IntentKind.WAIT)
        visible=[p for p in a.visible_players if p.player_id not in obs.teammates]
        for p in visible:self.last_seen[p.player_id]=((p.position.x,p.position.y),now)
        self.last_seen={k:v for k,v in self.last_seen.items() if now-v[1]<35.}
        if obs.vent_state:
            if now<self.hiding_until:return Intent(IntentKind.WAIT)
            if obs.vent_links and self.rng.random()<.55:
                choices=[v for v in obs.vent_links if v!=self.last_vent] or list(obs.vent_links)
                self.last_vent=obs.vent_state;self.hiding_until=now+self.rng.uniform(.8,3.)
                return Action('vent_move',self.rng.choice(choices))
            if obs.vent_can_exit:return Action('vent_exit',obs.vent_state)
            return Intent(IntentKind.WAIT)
        legal=list(a.actions)
        kills=[x for x in legal if x.kind is IntentKind.KILL and x.target not in obs.teammates and x.target!=pid]
        witnesses=max(0,len(visible)-1)
        if kills and self.rng.random()<self.aggression*(1.-self.caution*.7 if witnesses else 1.):
            self.rest_until=now+self.rng.uniform(.6,2.4)
            return self.rng.choice(kills)
        reports=[x for x in legal if x.kind is IntentKind.REPORT]
        if reports and self.rng.random()<.08:return self.rng.choice(reports)
        if obs.door_actions and visible and self.rng.random()<self.trapping*.12:
            # Only close near self/observed players. No hidden victim selection.
            rooms={p.room for p in visible}|{a.own.room}
            targets=[r for r in obs.door_actions if r in rooms]
            if targets:return Action('doors',self.rng.choice(targets))
        if obs.can_sabotage and self.rng.random()<(.12 if visible else .025):return Action('sabotage','lights')
        if obs.vent_actions and self.rng.random()<self.venting*(.4 if now<self.rest_until else .08):
            self.hiding_until=now+self.rng.uniform(.8,3.);self.goal=None
            return Action('vent_enter',self.rng.choice(obs.vent_actions))
        if now<self.rest_until:return Intent(IntentKind.WAIT)
        if a.interaction_task is not None:return Intent(IntentKind.WAIT)
        fake=[x for x in legal if x.kind is IntentKind.FAKE_TASK]
        if fake and a.kill_cooldown and a.kill_cooldown>10 and self.rng.random()<.025:return self.rng.choice(fake)
        if a.navigation_status=='moving' and now<self.next_change:return Intent(IntentKind.WAIT)
        if visible and (not a.kill_cooldown or a.kill_cooldown<12) and self.rng.random()<.7:
            target=min(visible,key=lambda p:math.dist((a.own.position.x,a.own.position.y),(p.position.x,p.position.y)))
            point=(target.position.x,target.position.y)
        elif self.last_seen and self.rng.random()<.35:
            point=self.rng.choice(list(self.last_seen.values()))[0]
        elif self.rng.random()<self.venting:
            candidates=[(k,v) for k,v in self.vents.items() if k not in self.visited] or list(self.vents.items())
            key,v=self.rng.choice(candidates);point=v['position'];self.visited.append(key)
        else:
            destinations=[d for d in a.destinations if d.id not in self.visited] or list(a.destinations)
            d=self.rng.choice(destinations);point=(d.standing.x,d.standing.y);self.visited.append(d.id)
        self.goal=point;self.next_change=now+self.rng.uniform(4.,12.)
        from social_deduction.actor import Point
        return Intent(IntentKind.GO_TO_LOCATION,position=Point(*map(float,point)))
