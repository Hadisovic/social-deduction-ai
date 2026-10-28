"""Goal-retaining opponent v2; v1 and canonical scripts remain untouched."""
import math
from social_deduction.actor import Phase,Provenance
from social_deduction.phase3_api import Intent,IntentKind,ClaimDraft,ClaimKind,StructuredClaim,WitnessedElimination,intent_allowed
from crewmate_v2.phase2.opponents import MixtureController


class CoherentMixture(MixtureController):
    version='phase21-opponent-v2'

    def __init__(self,seed,weights=None,speech=(.55,.25,.20),pursuit_interval=2.4):
        super().__init__(seed,weights)
        if len(speech)!=3 or any(not math.isfinite(v) or v<0 for v in speech) or abs(sum(speech)-1)>1e-9:raise ValueError('Invalid speech mixture')
        if not 1<=pursuit_interval<=10:raise ValueError('Invalid pursuit interval')
        self.speech=speech;self.pursuit_interval=pursuit_interval;self.chase=None;self.last_chase=-math.inf
        self.last_sighting=None;self.retained_goals=0;self.chase_updates=0

    def _claim(self,obs,p):
        others=sorted(set(obs.context.participants)-{self.owner});rng=self.streams['social']
        room=self.last_room or obs.own.room;tick=min(self.last_room_tick,obs.tick);draw=rng.random()
        if others and draw<self.speech[0]:draft=ClaimDraft(ClaimKind.DEFEND_PLAYER,rng.choice(others),room,tick)
        elif others and draw<sum(self.speech[:2]):draft=ClaimDraft(ClaimKind.SUSPECT_PLAYER,rng.choice(others),room,tick)
        elif self.last_sighting and rng.random()<.5:
            pid,region,seen_tick=self.last_sighting;draft=ClaimDraft(ClaimKind.SAW_PLAYER,pid,region,seen_tick)
        else:draft=ClaimDraft(ClaimKind.WAS_IN_REGION,self.owner,room,tick)
        return Intent(IntentKind.CLAIM,claim=draft)

    def decide(self,obs):
        if obs.context.phase is not Phase.ROAMING:return super().decide(obs)
        if self.owner is None:self.owner=obs.own.player_id
        if self.owner!=obs.own.player_id:raise ValueError('Controller cannot change owners')
        if self.last_time is not None and obs.time_seconds<self.last_time:raise ValueError('Time cannot reverse')
        elapsed=.6 if self.last_time is None else obs.time_seconds-self.last_time;self.last_time=obs.time_seconds
        if not obs.own.active or not obs.actions:return Intent(IntentKind.WAIT)
        p=self._mode(obs);self.last_room=obs.own.room;self.last_room_tick=obs.tick
        if obs.visible_players:
            seen=sorted(obs.visible_players,key=lambda x:x.player_id)[0];self.last_sighting=(seen.player_id,seen.room,obs.tick)
        report=self._report(obs,p,elapsed);kills=self.available(obs,IntentKind.KILL)
        risk=1. if len(obs.visible_players)<=1 else .45
        if kills and self.streams['kill'].random()<p.kill*(risk if self.profile in ('stealthy','evasive') else 1.):
            seen={x.player_id:x for x in obs.visible_players}
            ranked=sorted(kills,key=lambda a:(math.dist((obs.own.position.x,obs.own.position.y),(seen[a.target].position.x,seen[a.target].position.y)),a.target))
            self.last_kill=obs.time_seconds
            return self.streams['kill'].choices(ranked,weights=[1/(i+1) for i in range(len(ranked))])[0]
        if report is not None:return report
        rng=self.streams['movement'];visible={x.player_id:x for x in obs.visible_players}
        if obs.interaction_task is not None:return Intent(IntentKind.WAIT)
        if visible and (obs.kill_cooldown or 0)<=3 and obs.time_seconds-self.last_chase>=self.pursuit_interval:
            if self.chase not in visible:self.chase=rng.choice(sorted(visible))
            if rng.random()<p.pursue:
                self.last_chase=obs.time_seconds;self.chase_updates+=1
                intent=Intent(IntentKind.GO_TO_LOCATION,position=visible[self.chase].position)
                if not intent_allowed(obs,intent):raise ValueError('Illegal pursuit')
                return intent
        # Finish a public navigation goal instead of replacing it every three seconds.
        if obs.navigation_status=='moving':self.retained_goals+=1;return Intent(IntentKind.WAIT)
        self.chase=None
        fakes=self.available(obs,IntentKind.FAKE_TASK)
        if fakes and rng.random()<p.fake:return rng.choice(fakes)
        choices=[a for k in (IntentKind.GO_TO_ROOM,IntentKind.GO_TO_TASK) for a in self.available(obs,k) if a.target!=self.last_destination]
        if not choices:return Intent(IntentKind.WAIT)
        action=rng.choice(choices);self.last_destination=action.target;self.last_move=obs.time_seconds
        return action


def local_proxy(obs):
    """Explicit script confidence proxy, NOT the crew-only frozen belief model.

    Same local evidence rules can be called for either own role. Public/received
    evidence only, no engine, hidden roles or other memory. Exact confidence stays
    private; ordering experiments must report estimator heterogeneity.
    """
    ids=sorted(set(obs.context.participants)-{obs.own.player_id});scores={p:0. for p in ids}
    for e in obs.evidence:
        c=e.payload
        if type(c) is WitnessedElimination and e.provenance is Provenance.DIRECT and c.killer_id in scores:scores[c.killer_id]+=6
        elif type(c) is StructuredClaim and c.speaker_id!=obs.own.player_id and c.subject_id in scores:
            scores[c.subject_id]+=1 if c.kind is ClaimKind.SUSPECT_PLAYER else -.5 if c.kind is ClaimKind.DEFEND_PLAYER else 0
    if not scores:return {}
    top=max(scores.values());weights={p:math.exp(v-top) for p,v in scores.items()};total=sum(weights.values())
    return {p:v/total for p,v in weights.items()}
