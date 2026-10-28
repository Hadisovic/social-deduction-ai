import unittest
from dataclasses import replace
import numpy as np
import torch
from among_us_map import get_map
from navigation_service import NavigationPlanner
from social_deduction.actor import Role,Phase
from patrick_jane.api import Event
from patrick_jane.config import Experiment,Population
from patrick_jane.environment import PatrickJaneGame
from patrick_jane.memory import Memory
from patrick_jane.navigation import NavigationMemory,Goal
from patrick_jane.features import encode,model_packet,target_mask
from patrick_jane.network import PatrickJane,collate
from patrick_jane.actions import choose,evaluate_batch
from patrick_jane.rewards import RewardLedger


class NetworkMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(17);cls.policy=PatrickJane()
        cls.planner=NavigationPlanner(get_map(),cache_size=128)

    def setup_actor(self,pop=Population()):
        g=PatrickJaneGame(2920001,planner=self.planner,experiment=Experiment(population=pop))
        pid=next(p for p,x in g.players.items() if x.role is Role.CREWMATE)
        obs=g.observe_pj(pid);m=Memory(pid);m.update(obs);nav=NavigationMemory(g.map,17);nav.update(obs)
        return g,pid,obs,m,nav,encode(obs,m,nav)

    def test_locked_shapes_parameter_count_five_heads(self):
        g,pid,obs,m,nav,p=self.setup_actor();o,h=self.policy(collate([p]),torch.zeros(1,256))
        self.assertEqual(h.shape,(1,256));self.assertEqual(self.policy.architecture()['parameters'],1031070)
        for name,width in (('navigation',3),('interaction',5),('communication',6),('voting',15)):
            self.assertEqual(o[name].shape,(1,width))
        self.assertEqual(o['target'].shape,(1,len(p.targets)));self.assertEqual(o['value'].shape,(1,))

    def test_masks_remove_player_event_padding_influence(self):
        _,_,_,_,_,p=self.setup_actor(Population(4,1));batch=collate([p]);h=torch.zeros(1,256)
        expected,hidden=self.policy(batch,h)
        batch['players'][~batch['player_mask']]=10000;batch['events'][~batch['event_mask']]=-10000
        got,new=self.policy(batch,h)
        self.assertTrue(torch.equal(hidden,new))
        for name in expected:self.assertTrue(torch.equal(expected[name],got[name]))

    def test_empty_attention_finite_and_target_padding(self):
        _,_,_,_,_,p=self.setup_actor();b=collate([p]);b['player_mask'][:]=False;b['event_mask'][:]=False
        o,h=self.policy(b,torch.zeros(1,256))
        self.assertTrue(torch.isfinite(h).all());self.assertTrue(all(torch.isfinite(v).all() for v in o.values()))

    def test_hidden_states_independent_and_sequence_equivalence(self):
        _,_,_,_,_,p=self.setup_actor();h=torch.zeros(1,256);outs=[]
        for _ in range(3):o,h=self.policy(collate([p]),h);outs.append(o['value'][0])
        seq,hs=self.policy.forward_sequence(collate([p,p,p]),torch.zeros(1,256))
        self.assertTrue(torch.allclose(seq['value'],torch.stack(outs),atol=1e-6))
        self.assertTrue(torch.allclose(h,hs,atol=1e-6))
        batch=collate([p,p]);h0=torch.zeros(2,256);h0[1]=1.
        _,different=self.policy(batch,h0);self.assertFalse(torch.equal(different[0],different[1]))

    def test_all_phases_log_probability_reconstruction_and_gradients(self):
        _,_,_,_,_,p=self.setup_actor();m=model_packet(p)
        for phase in (0,1,2):
            m.phase=phase
            o,h=self.policy(collate([m]),torch.zeros(1,256));out={k:v[0] for k,v in o.items()}
            choice,lp,en,_=choose(out,m);_,again,_,_=choose(out,m,choice=choice)
            self.assertTrue(torch.equal(lp,again));self.assertTrue(torch.isfinite(lp));self.assertTrue(torch.isfinite(en))
            self.policy.zero_grad();(lp+out['value']).backward()
            self.assertIsNotNone(self.policy.gru.weight_ih_l0.grad)
            self.assertTrue(all(x.grad is None or torch.isfinite(x.grad).all() for x in self.policy.parameters()))

    def test_vectorized_conditional_likelihood_matches_scalar(self):
        _,_,_,_,_,p=self.setup_actor();packets=[model_packet(p) for _ in range(3)]
        for phase,packet in enumerate(packets):packet.phase=phase
        outputs,_=self.policy(collate(packets),torch.zeros(3,256));choices=[];expected=[];entropies=[]
        for i,p in enumerate(packets):
            c,lp,en,_=choose({k:v[i] for k,v in outputs.items()},p);choices.append(c);expected.append(lp);entropies.append(en)
        lp,en=evaluate_batch(outputs,packets,choices)
        self.assertTrue(torch.allclose(lp,torch.stack(expected),atol=1e-6))
        self.assertTrue(torch.allclose(en,torch.stack(entropies),atol=1e-6))

    def test_existing_map_astar_produces_collision_clear_route(self):
        from patrick_jane.navigation import PatrickPlanner
        from navigation_service import NavTarget,NavStatus
        planner=PatrickPlanner(self.planner.map)
        start=(-.7,-2.8)
        for i in (0,10000,40000,60000):
            goal=tuple(planner.map.nodes[i]);plan=planner.plan(start,NavTarget.location(goal))
            self.assertIn(plan.status,(NavStatus.MOVING,NavStatus.SUCCESS))
            self.assertTrue(all(planner.map.segment_clear(a,b) for a,b in zip(plan.route,plan.route[1:])))

    def test_conditional_masks_never_select_unrelated_targets(self):
        _,_,_,_,_,p=self.setup_actor()
        for kind,choice in (('communication',2),('communication',1),('navigation',2)):
            mask=target_mask(p,kind,choice)
            expected=('player',) if choice==2 and kind=='communication' else ('evidence',) if kind=='communication' else ('location','task','explore','body','repair')
            self.assertTrue(all(t.kind in expected for t,valid in zip(p.candidates,mask) if valid))

    def test_memory_provenance_own_statements_and_capacity(self):
        _,pid,obs,m,_,_=self.setup_actor();m.time=20.
        direct=Event('sighting','p2',room='Electrical',observed=2.,delivered=2.,source=pid,provenance='direct')
        claim=Event('strategic_claim','p2',room='MedBay',observed=2.,delivered=20.,source='p2',provenance='claim')
        m.append(direct);m.append(claim);m.append(replace(claim,source=pid));self.assertEqual(m.own_statements[-1].provenance,'claim')
        self.assertIn(direct,m.events);self.assertIn(claim,m.events)
        for i in range(100):m.append(Event('question',observed=float(i/10),delivered=20.,detail=str(i)))
        self.assertEqual(len(m.events),64)

    def test_memory_rejects_other_owner_future_and_dead_ingestion(self):
        _,pid,obs,m,_,_=self.setup_actor()
        with self.assertRaises(ValueError):Memory('someone').update(obs)
        with self.assertRaises(ValueError):m.append(Event('witness',observed=1000.,delivered=1000.))
        count=len(m.events);dead=replace(obs,actor=replace(obs.actor,own=replace(obs.actor.own,active=False)),
                                      events=(Event('witness',observed=0.,delivered=0.),))
        m.update(dead);self.assertEqual(count,len(m.events));self.assertFalse(Memory(pid).events)

    def test_navigation_detour_resume_and_goal_replacement(self):
        _,_,obs,_,nav,_=self.setup_actor();points=nav.candidates((obs.actor.own.position.x,obs.actor.own.position.y))
        first=Goal('one',points[0][1]);second=Goal('two',points[1][1])
        nav.command(2,first,0.,obs);nav.command(1,second,3.,obs);self.assertEqual(nav.original,first)
        nav.command(0,None,12.,obs);self.assertEqual(nav.goal,first);self.assertIsNone(nav.original)
        nav.command(2,second,13.,obs);self.assertEqual(nav.goal,second)

    def test_bonus_caps_revisits_and_unavoidable_wait(self):
        g,pid,obs,m,nav,_=self.setup_actor();ledger=RewardLedger(g)
        total=0
        for _ in range(30):total+=sum(ledger.observation_bonus(pid,obs,1,nav,m).values())
        self.assertLessEqual(total,.3+.7+1e-9)
        self.assertEqual(nav.update(obs),0);self.assertEqual(ledger.waiting(pid,100.,False),{})
        self.assertEqual(ledger.waiting(pid,10.,True),{'idle':-.01})


if __name__=='__main__':unittest.main()
