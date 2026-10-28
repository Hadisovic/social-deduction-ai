"""Actor-only forecasts first; hidden scoring labels are inspected only after play."""
from collections import Counter,defaultdict
import numpy as np
from torch.distributions import Categorical
from social_deduction.actor import Role
from social_deduction.phase3_api import WitnessedElimination
from crewmate_v2.learning.evaluate import forecast,probability_metrics


class DecisionAudit:
    def __init__(self):self.seen=set();self.predictions=[];self.claims=[]

    def __call__(self,pid,observation,memory,belief,options,probabilities,packet,selected):
        key=(pid,observation.context.meeting_id)
        distribution=Categorical(probs=probabilities.unsqueeze(0))
        f=forecast(packet,options,belief,observation,distribution)
        if f is not None and key not in self.seen:
            self.seen.add(key);self.predictions.append(dict(actor=pid,meeting_id=observation.context.meeting_id,**f,
                direct_witness=any(type(e.payload) is WitnessedElimination for e in memory.events)))
        choice=options[selected]
        if choice.kind=='claim_suspect':self.claims.append((pid,choice.target_player))

    def score(self,game,controlled_meeting=None):
        # Scoring happens after collection. These roles never return to inference.
        belief=[];policy=[];heuristic=Counter();strata=defaultdict(list);witness=defaultdict(list)
        nonreporter=defaultdict(list);nonreporter_heuristic=defaultdict(Counter)
        for f in self.predictions:
            targets=[i for i,pid in enumerate(f['candidates']) if game.players[pid].role is Role.IMPOSTOR]
            if len(targets)!=1:raise ValueError('Offline scoring needs one eligible impostor')
            target=targets[0];belief.append((f['frozen_belief'],target));policy.append((f['policy_vote_ranking'],target))
            reporter=f['reporter'];role=game.players[reporter].role.value if reporter is not None else 'emergency'
            strata[role].append((f['policy_vote_ranking'],target));witness[str(f['direct_witness'])].append((f['policy_vote_ranking'],target))
            for name,weights in f['reporter_heuristics'].items():heuristic[name]+=weights[f['candidates'][target]]
            if reporter!=f['actor'] and reporter is not None and (controlled_meeting is None or f['meeting_id']==controlled_meeting):
                nonreporter[role].append((f['policy_vote_ranking'],target))
                for name,weights in f['reporter_heuristics'].items():nonreporter_heuristic[role][name]+=weights[f['candidates'][target]]
        n=len(policy)
        return dict(pre_ballot_actor_meetings=n,
            controlled_prefix_actor_meetings=sum(f['meeting_id']==controlled_meeting for f in self.predictions) if controlled_meeting else 0,
            frozen_belief=probability_metrics(belief),
            policy_vote_ranking=probability_metrics(policy),reporter_only_accuracy={k:v/n for k,v in heuristic.items()} if n else {},
            policy_by_reporter_role={k:probability_metrics(v) for k,v in strata.items()},
            nonreporter_observer_policy={k:probability_metrics(v) for k,v in nonreporter.items()},
            nonreporter_reporter_only={role:{k:v/len(nonreporter[role]) for k,v in values.items()} for role,values in nonreporter_heuristic.items()},
            policy_by_direct_witness={k:probability_metrics(v) for k,v in witness.items()},
            selected_suspect_claims=len(self.claims),selected_incorrect_suspect_claims=sum(game.players[target].role is Role.CREWMATE for actor,target in self.claims),
            limits='Policy vote-target preferences are not identity beliefs. Prefix reports earn no learning credit. Selected claims are distinct from accepted public claims.')
