"""Engineering correctness only: no PPO optimizer updates or reserved partitions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch.distributions import Categorical
from frozen_handoff.match import FrozenCrewMatch
from frozen_handoff.models import load_model,MODELS,ROOT,sha256
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import IntentKind
from impostor_shared.controller import Controller,ActorContext
from impostor_shared.coordinator import ImpostorLedger
from impostor_shared.features import KINDS
from impostor_shared.weights import load_transfer,verify_parity,tensor_hash,ASSETS

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.policy,cls.receipt=load_transfer()
        cls.crews={name:load_model(name)[0] for name in ('A','B')}

    def context(self,game):
        pid=next(pid for pid,p in game.players.items() if p.role is Role.IMPOSTOR)
        envelope=game.observe_v2(pid);own=game.players[pid]
        fake=envelope.actor.interaction_task and envelope.actor.interaction_task.startswith('fake:')
        return ActorContext(envelope,float(own.interaction_remaining) if fake else 0.)

    def test_strict_crew_hashes_and_no_gradients(self):
        for name,policy in self.crews.items():
            self.assertEqual(sha256(ROOT/MODELS[name][0]),MODELS[name][1])
            self.assertEqual(sum(p.numel() for p in policy.parameters()),51778)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in policy.parameters()))

    def test_exact_tensor_transfer_and_cross_runtime_parity(self):
        result=verify_parity(self.policy,self.receipt)
        self.assertEqual(result['cases'],3)
        self.assertEqual(tensor_hash(self.policy),self.receipt['source_policy_hash'])
        self.assertEqual(sum(p.numel() for p in self.policy.parameters()),75075)

    def test_only_impostor_gradients_no_optimizer_updates(self):
        policy,_=load_transfer(trainable=True);before=tensor_hash(policy)
        with np.load(ASSETS/'synthetic-parity.npz',allow_pickle=False) as data:
            packet={k:torch.as_tensor(data['0_'+k]).unsqueeze(0) for k in ('global_','extra','players','actions','targets','mask')}
        distribution,value=policy(packet);loss=-distribution.log_prob(torch.tensor([0])).mean()+value.square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.actor_parameters()))
        self.assertEqual(tensor_hash(policy),before)
        self.assertTrue(all(p.grad is None for crew in self.crews.values() for p in crew.parameters()))

    def test_private_memories_and_stale_context_rejection(self):
        match=FrozenCrewMatch(2760003);game=match.game;ctx=self.context(game);obs=ctx.envelope.actor
        self.assertEqual(len({id(v.memory) for v in match.views.values()}),4)
        self.assertEqual(len({id(v.public_memory) for v in match.views.values()}),4)
        actor=Controller(self.policy,seed=17,player_id=obs.own.player_id)
        actor.supply(ctx);actor.decide(obs)
        with self.assertRaises(ValueError):ctx.checked(replace(obs,tick=obs.tick+1))
        with self.assertRaises((AttributeError,TypeError)):ctx.game=game
        self.assertFalse(hasattr(actor,'game'));self.assertFalse(hasattr(actor,'recorder'))

    def test_hidden_other_position_does_not_change_neural_packet(self):
        match=FrozenCrewMatch(2760004);game=match.game;ctx=self.context(game);pid=match.crew[-1]
        destinations=ctx.envelope.actor.destinations
        points=[next(d.standing for d in destinations if d.room==room) for room in ('Reactor','Navigation')]
        packets=[]
        for point in points:
            game.players[pid].motion.spawn((point.x,point.y));game._invalidate()
            ctx=self.context(game);obs=ctx.envelope.actor
            self.assertNotIn(pid,{p.player_id for p in obs.visible_players})
            actor=Controller(self.policy,seed=17,player_id=obs.own.player_id)
            actor.supply(ctx);actor.decide(obs);packets.append(actor.last_decision['packet'])
        for key in packets[0]:np.testing.assert_array_equal(packets[0][key],packets[1][key])

    def test_fixed_seed_sampled_actor_is_reproducible(self):
        match=FrozenCrewMatch(2760005);ctx=self.context(match.game);obs=ctx.envelope.actor
        actors=[Controller(self.policy,seed=43,player_id=obs.own.player_id,stochastic=True) for _ in range(2)]
        actions=[]
        for actor in actors:actor.supply(ctx);actions.append(actor.decide(obs))
        self.assertEqual(actions[0],actions[1]);self.assertIsNot(actors[0].memory,actors[1].memory)
        self.assertIsNot(actors[0].generator,actors[1].generator)

    def test_real_handoff_factory_seed_fits_private_generator(self):
        from phase3_runner import behavior_seed
        seed=behavior_seed(2760002,'p1')
        actor=Controller(self.policy,seed=seed,player_id='p1')
        repeat=Controller(self.policy,seed=seed,player_id='p1')
        self.assertLess(actor.generator.initial_seed(),1<<63)
        self.assertEqual(actor.generator.initial_seed(),repeat.generator.initial_seed())

    def test_repeated_fake_finishes_without_restart(self):
        class FakePolicy:
            def evaluate(self,packet):
                mask=packet['mask'].bool();distribution=Categorical(logits=torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,-1e9))
                return distribution,torch.zeros(1),Categorical(logits=torch.zeros((1,12))),torch.zeros_like(mask,dtype=torch.long),torch.zeros(1)
            def act(self,packet,**kwargs):
                indices=np.flatnonzero(packet['mask'] & (packet['actions'][:,KINDS.index('fake_task')]>0))
                index=int(indices[0]);distribution,*_=self.evaluate({k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()})
                return index,float(distribution.log_prob(torch.tensor([index]))),0.
        match=FrozenCrewMatch(2760006);game=match.game;pid=match.impostor;initial=self.context(game)
        for d in initial.envelope.actor.destinations:
            game.players[pid].motion.spawn((d.standing.x,d.standing.y));game._invalidate();ctx=self.context(game)
            if any(a.kind is IntentKind.FAKE_TASK for a in ctx.envelope.actor.actions):break
        else:self.fail('No legitimate fake console found')
        actor=Controller(FakePolicy(),seed=1,player_id=pid);actor.supply(ctx)
        game.step({pid:actor.decide(ctx.envelope.actor)})
        remaining=game.players[pid].interaction_remaining
        for _ in range(30):
            ctx=self.context(game)
            if not ctx.envelope.actor.interaction_task:break
            actor.supply(ctx);action=actor.decide(ctx.envelope.actor)
            self.assertEqual(action.kind,IntentKind.WAIT)
            game.step({pid:action})
            self.assertLess(game.players[pid].interaction_remaining,remaining+1e-8)
            remaining=game.players[pid].interaction_remaining
        self.assertIsNone(game.players[pid].interaction_task);self.assertGreater(actor.fake_repeats,0)

    def test_simulated_reward_boundaries_and_zero_time_terminal(self):
        for label,phase,cooldown,dt,winner,ejected,kills,expected in (
            ('boundary',Phase.ROAMING,.1,.2,None,False,0,-.002),
            ('meeting',Phase.DISCUSSION,0.,.2,None,False,0,0.),
            ('parity',Phase.ROAMING,0.,0.,Role.IMPOSTOR,False,1,13.),
            ('ejection',Phase.VOTING,0.,.2,Role.CREWMATE,True,0,-16.)):
            with self.subTest(label=label):
                own=SimpleNamespace(active=True,cooldown=cooldown,status='alive')
                game=SimpleNamespace(time=0.,tick=0,phase=phase,players={'p0':own},metrics={'eliminations':0},
                    config=SimpleNamespace(kill_cooldown=45.),is_terminal=False,result=None,observe=lambda pid:None)
                def step(actions):
                    game.time+=dt;game.tick+=int(dt>0);game.metrics['eliminations']+=kills
                    if winner is not None:
                        game.is_terminal=True;game.phase=Phase.FINISHED;game.result=SimpleNamespace(winner=winner)
                    if ejected:own.active=False;own.status='ejected'
                game.step=step
                controller=SimpleNamespace(player_id='p0',pending_pursuit=0.,pursuit=SimpleNamespace(advance=lambda obs:0.,positive_total=0.))
                ledger=ImpostorLedger(game,controller)
                ledger.decision(dict(packet={},index=0,logp=0.,value=0.))
                ledger.step({});self.assertAlmostEqual(ledger.return_,expected)
                if winner is not None:
                    credit=ledger.check();self.assertEqual(credit['terminal_bootstrap'],0.)
                    self.assertTrue(credit['finite_advantages'])
                    before=ledger.return_;ledger.step({})
                    self.assertEqual(ledger.return_,before)

if __name__=='__main__':unittest.main()
