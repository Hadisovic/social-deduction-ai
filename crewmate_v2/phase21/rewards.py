"""Versioned pre-step actor eligibility; coefficients and old ledgers unchanged."""
from crewmate_v2.learning.rewards import RewardLedger

SCHEMA='crewmate-phase21-accounting-v1'


class EligibleLedger(RewardLedger):
    schema=SCHEMA

    def consume(self,events,*,active,pre_active=None,**kwargs):
        # self.alive is the preceding step's status, never the post-resolution one.
        before=self.alive if pre_active is None else bool(pre_active)
        if not before:
            events=[e for e in events if not (e['kind']=='vote_submitted' and e.get('voter_id')==self.focal
                     or e['kind']=='report' and e.get('reporter_id')==self.focal)]
        return super().consume(events,active=active,**kwargs)
