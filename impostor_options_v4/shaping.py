"""A bounded potential on actor-observable history, never on engine truth."""
import math
from social_deduction.actor import Phase


def distance(a,b):
    return math.hypot(a.x-b.x,a.y-b.y)


class PursuitPotential:
    """At most scale positive credit in an entire match; no reset on kill/meeting.

    Only sustained visible IDs and actual own movement toward the previous visible
    position can increase credit. New targets establish a non-paying high-water
    baseline. Moving targets cannot contribute more than own displacement progress.
    Potential is credited progress in [0,scale], and zero at terminal. With gamma=1,
    F=Phi(next)-Phi(previous) telescopes to zero over a complete match (initial Phi=0).
    Terminal correction removes an incentive to stall instead of achieving outcomes.
    """
    def __init__(self,scale=.05,min_progress=.05):
        self.scale=scale;self.min_progress=min_progress
        self.credit=0.;self.best=0.;self.previous={};self.own=None;self.last_time=None
        self.positive_total=0.

    def advance(self,obs):
        if obs.context.phase is Phase.FINISHED or not obs.own.active:
            correction=-self.credit;self.credit=0.;self.previous={};self.own=None
            return correction
        if obs.context.phase is not Phase.ROAMING:
            self.previous={};self.own=None;self.last_time=obs.time_seconds
            return 0.
        current={p.player_id:p.position for p in obs.visible_players}
        own=obs.own.position
        span=max(obs.settings.sight_range-obs.settings.kill_range,1e-9)
        scores={pid:max(0.,min(1.,1.-(distance(own,pos)-obs.settings.kill_range)/span))
                for pid,pos in current.items()}
        increment=0.
        if self.own is not None and self.last_time is not None and obs.time_seconds > self.last_time:
            for pid in current.keys() & self.previous.keys():
                before=distance(self.own,self.previous[pid])
                own_gain=before-distance(own,self.previous[pid])
                net_gain=before-distance(own,current[pid])
                if own_gain >= self.min_progress and net_gain >= self.min_progress:
                    increment=max(increment,min(max(0.,scores[pid]-self.best),own_gain/span))
        # Every new observation raises the baseline without paying for visibility
        # appearance, switching targets, target motion alone, or reacquisition.
        self.best=max(self.best,max(scores.values(),default=0.))
        payment=min(self.scale-self.credit,self.scale*increment)
        payment=max(0.,payment)
        self.credit+=payment;self.positive_total+=payment
        self.previous=current;self.own=own;self.last_time=obs.time_seconds
        return payment
