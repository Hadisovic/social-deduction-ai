"""No neural training. Small fixtures plus exact baseline comparison."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import unittest
import numpy as np
import torch
from crewmate_v2.config import LobbySettings
from crewmate_v2.fourcrew.match import FourCrewMatch
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.phase21.rewards import EligibleLedger
from crewmate_v2.fourcrew.rewards import ActorLedger
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import Intent,IntentKind
from frozen_handoff.models import load_model
from frozen_handoff.match import FrozenCrewMatch
from frozen_handoff.controller import LegalImpostor


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.policy,_=load_model('B')

    def test_both_selected_files_load_strictly_and_are_frozen(self):
        for name in ('A','B'):
            p,m=load_model(name);self.assertFalse(any(x.requires_grad for x in p.parameters()))
            self.assertEqual(m['arm'],'original' if name=='A' else 'shaped_v2')

    def test_new_coordinator_matches_original_without_reinterpreting_policy(self):
        lobby=LobbySettings(timeout=3.)
        original=FourCrewMatch(2760000,lobby=lobby,deadline=perf_counter()+60)
        handoff=FrozenCrewMatch(2760000,lobby=lobby,deadline=perf_counter()+60)
        a,sa=original.collect(self.policy);b,sb=handoff.collect(self.policy)
        self.assertEqual(sa,sb)
        for key in a:
            if key=='packets':
                for x,y in zip(a[key],b[key]):self.assertTrue(all(np.array_equal(x[k],y[k]) for k in x))
            else:self.assertEqual(a[key],b[key])

    def test_private_memories_and_safe_external_role_envelope(self):
        seen=[]
        class Controller:
            def decide(self,obs):seen.append(obs);return Intent(IntentKind.WAIT)
        m=FrozenCrewMatch(2760000,lobby=LobbySettings(timeout=1.8),controller_factory=lambda **kw:Controller())
        self.assertEqual(len({id(v.memory) for v in m.views.values()}),4)
        self.assertEqual(len({id(v.public_memory) for v in m.views.values()}),4)
        m.collect(self.policy);self.assertTrue(seen)
        for obs in seen:
            self.assertIs(obs.own.role,Role.IMPOSTOR)
            self.assertFalse(hasattr(obs,'players'));self.assertFalse(hasattr(obs,'roles'))
            self.assertTrue(all(not hasattr(p,'role') for p in obs.visible_players))

    def test_external_invalid_actions_fail_closed(self):
        m=FrozenCrewMatch(2760000);obs=m.game.observe(m.impostor)
        class Controller:
            def decide(self,obs):return Intent(IntentKind.KILL,'not-a-player')
        with self.assertRaises(ValueError):LegalImpostor(Controller(),m.impostor).decide(obs)
        with self.assertRaises(ValueError):LegalImpostor(Controller(),'other').decide(obs)

    def test_versioned_ledger_repairs_only_same_step_eligibility(self):
        roles={'a':Role.CREWMATE,'i':Role.IMPOSTOR};config=Rewards()
        events=[dict(kind='vote_submitted',voter_id='a',target_id='i',meeting_id='m'),dict(kind='ejection',player_id='a')]
        kwargs=dict(active=False,legal=True,intent=Intent(IntentKind.VOTE,'i'))
        old=ActorLedger('a',roles,config);new=EligibleLedger('a',roles,config)
        old.consume(events,**kwargs);new.consume(events,**kwargs)
        self.assertEqual(old.counts['correct_votes'],0);self.assertEqual(new.counts['correct_votes'],1)
        self.assertEqual(old.components['death'],new.components['death'])

    def test_explicit_accounting_and_opponents_not_silently_enabled(self):
        baseline=FrozenCrewMatch(2760000);self.assertIs(type(baseline.ledgers[baseline.crew[0]]),ActorLedger)
        opt=FrozenCrewMatch(2760000,accounting='eligible-v1',opponent='v2')
        self.assertIs(type(opt.ledgers[opt.crew[0]]),EligibleLedger)
        self.assertEqual(opt.handoff_options['opponent'],'v2')
        for seed in (1230000,1240000,2780000,2790000):
            with self.assertRaises(ValueError):FrozenCrewMatch(seed)
        m=FrozenCrewMatch(3600000);self.assertEqual(m.seed,3600000)


if __name__=='__main__':unittest.main()
