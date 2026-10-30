"""Trusted retrospective reward ledger. No private credit is an observation."""
from .lights_setup import LightsLedger
from social_deduction.actor import Role


class A2Ledger(LightsLedger):
    def __init__(self,game):
        super().__init__(game);self.ballot_cursor=0;self.game=game

    def collect(self,game):
        # The inherited unique event ledger awards one terminal family only.
        # Set its coefficient before collection, never stack a second victory.
        if game.is_terminal and game.result.winner is Role.CREWMATE:
            self.coefficients['win']=5. if game.result.reason=='TASKS_COMPLETED' else 10.
        result=super().collect(game)
        if game.experiment.reward_regime!='investigation-a2-v1':return result
        credits={}
        for ballot in game.accepted_ballots[self.ballot_cursor:]:
            pid=ballot['voter_id']
            if pid not in self.crew:continue
            # Includes a valid ballot whose voter was ejected in this same step.
            amount=ballot['reward']
            if amount:
                result[pid]['vote_accuracy']=result[pid].get('vote_accuracy',0.)+amount
                credits.setdefault(pid,{})['vote_accuracy']=credits.get(pid,{}).get('vote_accuracy',0.)+amount
        self.ballot_cursor=len(game.accepted_ballots)
        self._record(game.time,credits)
        return result
