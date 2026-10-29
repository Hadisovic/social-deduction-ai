import unittest,tempfile,json
from pathlib import Path
from copy import deepcopy
import torch,numpy as np
from patrick_jane.reward_transition import import_update11,APPROVED
from patrick_jane.network import PatrickJane
from patrick_jane.rewards import RewardLedger,COEFFICIENTS
from patrick_jane.health import audit_rollout
from patrick_jane.worker import Match
from patrick_jane.lights_setup import LightsExperiment
from patrick_jane.config import ENGINEERING
from social_deduction.actor import Role

class RewardTransitionTests(unittest.TestCase):
    def test_verified_weights_reset_adam_and_correct_counts(self):
        torch.set_num_threads(1);p=PatrickJane();o=torch.optim.Adam(p.parameters(),lr=.0003)
        c,info=import_update11(p,o,np.random.default_rng(17))
        self.assertEqual(c['individual_decisions'],764660);self.assertEqual(c['run_decisions'],509577)
        self.assertEqual(info['policy_tensors_verified'],52);self.assertEqual(len(o.state),0)
        self.assertEqual(COEFFICIENTS,APPROVED)

    def test_death_minus_two_valid_duplicate_and_ghost_pay_rejected(self):
        r=dict(matches=[dict(individual_decisions=1,neural_actions={'travel':1},metrics={'illegal_actions':0},winner=None,crew_reports=1)],reward_components={'0/p0':{'death':-2.,'report':1.25,'ghost_task':0.}})
        audit_rollout(r)
        for key,value in [('death',-1.),('death',-4.),('ghost_task',1.),('report',2.5)]:
            bad=deepcopy(r);bad['reward_components']['0/p0'][key]=value
            with self.assertRaises(RuntimeError):audit_rollout(bad)

    def test_real_ghost_task_progress_without_direct_neural_reward(self):
        m=Match(ENGINEERING[0],LightsExperiment());g=m.game
        crew=next(p for p in g.players.values() if p.role is Role.CREWMATE)
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        g._eliminate(imp,crew);m.ledger.collect(g)
        before=g.metrics['task_completions'];paid=0.
        for _ in range(3000):
            g.step({})
            paid+=m.ledger.collect(g)[crew.player_id].get('ghost_task',0)
            if g.metrics['task_completions']>before:break
        self.assertGreater(g.metrics['task_completions'],before);self.assertEqual(paid,0.)
        self.assertTrue(any(t.progress==1 for t in crew.tasks))

if __name__=='__main__':unittest.main()
