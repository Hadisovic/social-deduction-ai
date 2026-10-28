"""Actor-local stochastic opponents. This module never receives a game or role table."""
from dataclasses import dataclass
import hashlib
import math
from random import Random

from social_deduction.actor import Phase
from social_deduction.phase3_api import ClaimDraft, ClaimKind, Intent, IntentKind, intent_allowed


@dataclass(frozen=True)
class Profile:
    kill: float
    report_hazard: float  # per second of legal report exposure, not per physics tick
    pursue: float
    fake: float
    accuse: float
    defend: float
    skip: float


# Overlapping preferences: a profile is never an always/never-report signature.
PROFILES = {
    'aggressive': Profile(.90, .10, .80, .10, .45, .20, .15),
    'stealthy': Profile(.55, .045, .35, .60, .25, .25, .35),
    'self_report': Profile(.70, .36, .50, .35, .40, .20, .20),
    'evasive': Profile(.70, .025, .35, .25, .25, .20, .40),
    'social': Profile(.65, .14, .45, .40, .50, .30, .15),
}


def stream_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'phase2-v1:{seed}:{name}'.encode()).digest()[:8], 'big')


class MixtureController:
    """One owner, independent streams, body-level hazard and bounded mode changes.

    The runner supplies a behavior seed independently of role assignment. Inputs
    are exactly the immutable actor API. Target candidates come from local legal
    actions, visible players or public meeting participants. False social claims
    are permitted game messages, never altered observation evidence.
    """
    def __init__(self, seed, weights=None, switch_seconds=(12., 24.)):
        weights = dict(weights or {name: 1. for name in PROFILES})
        if not weights or set(weights)-set(PROFILES) or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in weights.values()) or not sum(weights.values()) > 0:
            raise ValueError('Invalid opponent mixture weights')
        if len(switch_seconds)!=2 or not 1 <= switch_seconds[0] <= switch_seconds[1] <= 120:
            raise ValueError('Invalid mode interval')
        self.weights=weights;self.switch_seconds=switch_seconds
        self.streams={name: Random(stream_seed(seed,name)) for name in ('mode','report','kill','movement','social')}
        self.profile=None;self.next_switch=-1.;self.mode_history=[];self.owner=None
        self.body_exposure={};self.last_time=None;self.last_move=-math.inf;self.last_destination=None
        self.last_kill=-math.inf;self.last_claim=None;self.last_room=None;self.last_room_tick=0

    @staticmethod
    def available(obs,kind):
        return sorted((a for a in obs.actions if a.kind is kind),key=lambda a:a.target or '')

    def _mode(self,obs):
        if self.profile is None or obs.time_seconds >= self.next_switch:
            rng=self.streams['mode'];names=sorted(self.weights)
            self.profile=rng.choices(names,weights=[self.weights[n] for n in names])[0]
            self.next_switch=obs.time_seconds+rng.uniform(*self.switch_seconds)
            self.mode_history.append((obs.time_seconds,self.profile))
        return PROFILES[self.profile]

    def _report(self,obs,p,elapsed):
        legal=self.available(obs,IntentKind.REPORT)
        for a in legal:
            state=self.body_exposure.get(a.target)
            if state is None:
                state=[-math.log(max(1e-12,1-self.streams['report'].random())),0.]
                self.body_exposure[a.target]=state
            # The first legal opportunity contributes one decision interval only.
            state[1]+=p.report_hazard*min(max(elapsed,.001),.6)
            if state[1]>=state[0]:return a
        return None

    def _claim(self,obs,p):
        rng=self.streams['social'];others=sorted(set(obs.context.participants)-{self.owner})
        region=self.last_room or obs.own.room;tick=min(self.last_room_tick,obs.tick)
        draw=rng.random()
        if others and draw<p.accuse:
            # A framing attempt from public IDs, with no hidden victim/role lookup.
            draft=ClaimDraft(ClaimKind.SUSPECT_PLAYER,rng.choice(others),region,tick)
        elif others and draw<p.accuse+p.defend:
            draft=ClaimDraft(ClaimKind.DEFEND_PLAYER,rng.choice(others),region,tick)
        else:
            draft=ClaimDraft(ClaimKind.WAS_IN_REGION,self.owner,region,tick)
        return Intent(IntentKind.CLAIM,claim=draft)

    def decide(self,obs):
        if self.owner is None:self.owner=obs.own.player_id
        if self.owner!=obs.own.player_id:raise ValueError('Controller cannot share owners')
        if self.last_time is not None and obs.time_seconds<self.last_time:raise ValueError('Controller cannot travel backwards')
        elapsed=.6 if self.last_time is None else obs.time_seconds-self.last_time
        self.last_time=obs.time_seconds
        if not obs.own.active or not obs.actions:return Intent(IntentKind.WAIT)
        p=self._mode(obs)
        if obs.context.phase is Phase.DISCUSSION:
            if self.available(obs,IntentKind.CLAIM) and self.last_claim!=obs.context.meeting_id:
                self.last_claim=obs.context.meeting_id
                return self._claim(obs,p)
            return Intent(IntentKind.WAIT)
        if obs.context.phase is Phase.VOTING:
            votes=[a for a in self.available(obs,IntentKind.VOTE) if a.target!=self.owner]
            skip=self.available(obs,IntentKind.SKIP)
            if skip and (not votes or self.streams['social'].random()<p.skip):return skip[0]
            return self.streams['social'].choice(votes) if votes else Intent(IntentKind.WAIT)
        if obs.context.phase is not Phase.ROAMING:return Intent(IntentKind.WAIT)
        self.last_room=obs.own.room;self.last_room_tick=obs.tick
        report=self._report(obs,p,elapsed)
        kills=self.available(obs,IntentKind.KILL)
        # Density is what this actor can see; it cannot query actual witness packets.
        risk=1. if len(obs.visible_players)<=1 else .45
        if kills and self.streams['kill'].random()<p.kill*(risk if self.profile in ('stealthy','evasive') else 1.):
            seen={x.player_id:x for x in obs.visible_players}
            ranked=sorted(kills,key=lambda a:(math.dist((obs.own.position.x,obs.own.position.y),(seen[a.target].position.x,seen[a.target].position.y)),a.target))
            # All ranks retain nonzero probability; nearest victim is not guaranteed.
            choice=self.streams['kill'].choices(ranked,weights=[1/(i+1) for i in range(len(ranked))])[0]
            self.last_kill=obs.time_seconds;return choice
        if report is not None:return report
        rng=self.streams['movement']
        if obs.interaction_task is not None:return Intent(IntentKind.WAIT)
        # A real legal fake-task action, rather than fabricated task completion.
        fakes=self.available(obs,IntentKind.FAKE_TASK)
        if fakes and rng.random()<p.fake:return rng.choice(fakes)
        if obs.navigation_status=='moving' and obs.time_seconds-self.last_move<3.:return Intent(IntentKind.WAIT)
        visible=list(obs.visible_players)
        if visible and (obs.kill_cooldown or 0)<=3 and rng.random()<p.pursue:
            target=rng.choice(sorted(visible,key=lambda x:x.player_id))
            intent=Intent(IntentKind.GO_TO_LOCATION,position=target.position)
        else:
            choices=[a for k in (IntentKind.GO_TO_ROOM,IntentKind.GO_TO_TASK) for a in self.available(obs,k) if a.target!=self.last_destination]
            if not choices:return Intent(IntentKind.WAIT)
            intent=rng.choice(choices);self.last_destination=intent.target
        self.last_move=obs.time_seconds
        if not intent_allowed(obs,intent):raise ValueError('Mixture produced an illegal local intention')
        return intent
