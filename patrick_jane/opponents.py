"""Local scripts with separate held-out mixtures; partner neural project untouched."""
from random import Random
from phase3_engine import stream_seed
from social_deduction.actor import Phase
from social_deduction.phase3_api import Intent,IntentKind,ClaimDraft,ClaimKind
from crewmate_v2.phase21.opponents import CoherentMixture
from .api import Action


class Opponent:
    def __init__(self,seed,held_out=False):
        weights=dict(aggressive=.5,stealthy=2.,self_report=.5,evasive=2.,social=1.) if held_out else None
        self.script=CoherentMixture(seed,weights=weights)
        self.rng=Random(stream_seed(seed,'pj:sabotage'))
        self.social=Random(stream_seed(seed,'pj:deception'))
        self.sabotage_probability=.65 if held_out else .35
        self.patient_until=0.;self.held_out=held_out

    def decide(self,obs):
        a=obs.actor
        # Own-role teammate knowledge is legitimate. No hidden crew-role table is read.
        from dataclasses import replace
        a=replace(a,visible_players=tuple(p for p in a.visible_players if p.player_id not in obs.teammates),
                  actions=tuple(x for x in a.actions if not (x.kind in (IntentKind.KILL,IntentKind.VOTE) and x.target in obs.teammates)))
        if a.context.phase is Phase.ROAMING:
            if obs.can_sabotage and self.rng.random()<self.sabotage_probability:
                return Action('sabotage',self.rng.choice(('lights','oxygen','reactor')))
            # Bounded ambush/patient periods, not privileged pursuit through walls.
            if a.time_seconds<self.patient_until:return Intent(IntentKind.WAIT)
            if a.kill_cooldown and a.kill_cooldown>5 and self.rng.random()<.015:
                self.patient_until=a.time_seconds+self.rng.uniform(2.,5.)
                return Intent(IntentKind.WAIT)
        if a.context.phase is Phase.DISCUSSION and a.context.speaker_id==a.own.player_id:
            addressed=[e for e in obs.events if e.provenance=='claim' and e.subject==a.own.player_id and e.source!=a.own.player_id]
            if addressed and self.social.random()<.6:
                last=addressed[-1]
                kind=ClaimKind.DEFEND_PLAYER if last.kind=='accuse' else ClaimKind.WAS_IN_REGION
                if last.kind in ('accuse','question'):
                    return Intent(IntentKind.CLAIM,claim=ClaimDraft(kind,a.own.player_id,
                        self.script.last_room or a.own.room,min(self.script.last_room_tick,a.tick)))
        result=self.script.decide(a)
        # False location alibis are claims, not fabricated observation records.
        if result.kind is IntentKind.CLAIM and self.social.random()<.25:
            return Intent(IntentKind.CLAIM,claim=ClaimDraft(ClaimKind.WAS_IN_REGION,a.own.player_id,
                self.social.choice(a.map.rooms),max(0,a.tick-self.social.randrange(0,50))))
        return result
