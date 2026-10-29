import unittest
from copy import deepcopy
from patrick_jane.health import audit_rollout,severe_paired_regression,severe_speed_regression


class HealthTests(unittest.TestCase):
    def test_paired_regression_requires_substantial_and_unlikely_decline(self):
        reference=[dict(seed=i,winner='crewmate' if i<50 else 'impostor') for i in range(100)]
        unchanged=deepcopy(reference)
        self.assertFalse(severe_paired_regression(reference,unchanged)['stop'])
        for row in unchanged[:30]:row['winner']='impostor'
        self.assertTrue(severe_paired_regression(reference,unchanged)['stop'])
        with self.assertRaises(ValueError):severe_paired_regression(reference,unchanged[:-1])

    def test_speed_regression_is_sustained_not_one_slow_game_batch(self):
        history=[dict(rollout=dict(decisions_per_second=n)) for n in (50,48,52,10,9,12)]
        self.assertTrue(severe_speed_regression(history));history[-1]['rollout']['decisions_per_second']=45
        self.assertFalse(severe_speed_regression(history))

    def test_reward_audit_detects_duplicate_bonuses_and_wrong_owner_settlement(self):
        rollout=dict(matches=[dict(individual_decisions=1,neural_actions={'travel':1},metrics={'illegal_actions':0},winner='impostor',crew_reports=0)],
            reward_components={'0/p0':{'loss':-10.,'exploration':.3,'ghost_task':0.}})
        audit_rollout(rollout)
        for key,value in (('exploration',.32),('ghost_task',5.),('loss',0.)):
            bad=deepcopy(rollout);bad['reward_components']['0/p0'][key]=value
            with self.assertRaises(RuntimeError):audit_rollout(bad)


if __name__=='__main__':unittest.main()
