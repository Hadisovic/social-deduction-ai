"""Per-owner credit; dead actors cannot receive individual report/vote credit."""
from crewmate_v2.learning.rewards import RewardLedger


class ActorLedger(RewardLedger):
    def consume(self,events,*,active,**kwargs):
        if not self.alive or not active:
            events=[e for e in events if not (
                e['kind']=='report' and e.get('reporter_id')==self.focal or
                e['kind']=='vote_submitted' and e.get('voter_id')==self.focal)]
        return super().consume(events,active=active,**kwargs)
