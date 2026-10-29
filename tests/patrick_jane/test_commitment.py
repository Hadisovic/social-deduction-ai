import unittest,tempfile,json
from pathlib import Path
from dataclasses import replace
from random import Random
import numpy as np
import torch
from navigation_service import NavStatus
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import Intent,IntentKind
from crewmate_v2.tasks import make_plan,task_catalog
from patrick_jane.worker import Match
from patrick_jane.lights_setup import LightsExperiment
from patrick_jane.config import ENGINEERING
from patrick_jane.api import Choice
from patrick_jane.features import target_mask,model_packet,goal_for
from patrick_jane.actions import choose,evaluate_batch
from patrick_jane.network import PatrickJane,collate
from patrick_jane.collector import collect,Pool


class CommitmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,kind='UploadData',enabled=True):
        m=Match(ENGINEERING[0],LightsExperiment(task_commitment_mode=enabled));m.opponents={}
        pid=m.crew[0];g=m.game;p=g.players[pid];t=p.tasks[0]
        t.plan=make_plan(pid,kind,'Long',task_catalog(g.map),Random(17));t.stage=0;t.work=0.;t.ready_at=0.
        d=next(d for d in g.map.destinations if d.id==t.console_id);p.motion.spawn(tuple(d.standing));g._invalidate()
        m.prepare();return m,pid,t

    def step(self,m,pid,choice=None):
        packet=m.packets[pid]
        if choice is None:
            if packet.interaction_mask[1]:choice=Choice(interaction=1,target=int(np.flatnonzero(target_mask(packet,'interaction',1))[0]))
            elif packet.navigation_mask[0]:choice=Choice(interaction=0,navigation=0)
            else:choice=Choice(interaction=0,navigation=2,target=int(np.flatnonzero(target_mask(packet,'navigation',2))[0]))
        return m.advance({pid:dict(choice=choice,logp=0.,entropy=0.,value=0.,probabilities={})})

    def test_multistage_only_final_reward_no_scripted_executor(self):
        m,pid,t=self.fixture();self.assertIsNone(m.nav[pid].commitment.task)
        self.step(m,pid);self.assertEqual(m.nav[pid].commitment.task,t.task_id)
        saw_stage=False;paid=0.
        for _ in range(1800):
            if m.game.is_terminal:break
            response=self.step(m,pid)
            paid+=sum(v.get('task',0) for _,p,v in response['rewards'] if p==pid)
            if t.stage==1:saw_stage=True;self.assertEqual(paid,0.)
            if t.progress==1:break
        self.assertTrue(saw_stage);self.assertEqual(t.progress,1.);self.assertEqual(paid,1.)
        self.assertIsNone(m.nav[pid].commitment.task)
        stats=m.summary()['learning_diagnostics'];self.assertGreater(stats['forced_decisions'],0)
        self.assertEqual(stats['forced_decisions']+stats['voluntary_decisions'],m.decisions)

    def test_continue_cannot_cancel_active_work_even_when_disabled(self):
        for enabled in (False,True):
            m,pid,t=self.fixture(enabled=enabled);self.step(m,pid)
            nav=m.nav[pid];nav.goal=goal_for(next(x for x in m.packets[pid].candidates if x.kind=='task' and x.key==t.task_id))
            a=nav.command(0,None,m.game.time,m.game.observe_pj(pid));self.assertIs(a.kind,IntentKind.WAIT)
            self.assertEqual(m.game.players[pid].interaction_task,t.task_id)

    def test_incubation_wait_then_final_stage(self):
        m,pid,t=self.fixture('InspectSample');self.step(m,pid)
        for _ in range(20):
            self.step(m,pid)
            if t.stage==1:break
        self.assertEqual(t.stage,1);self.assertGreater(t.ready_at,m.game.time)
        for _ in range(10):
            packet=m.packets[pid];self.assertFalse(packet.interaction_mask[1]);self.assertTrue(packet.navigation_mask[0])
            r=self.step(m,pid);self.assertFalse(any(v.get('idle',0) for _,p,v in r['rewards'] if p==pid))
        self.assertEqual(t.stage,1);self.assertIsNotNone(m.nav[pid].commitment.task)
        for _ in range(150):
            self.step(m,pid)
            if t.progress==1:break
        self.assertEqual(t.progress,1.);self.assertIsNone(m.nav[pid].commitment.task)

    def test_meeting_preserves_task_and_rebuilds_route_death_releases(self):
        m,pid,t=self.fixture();self.step(m,pid);g=m.game
        g._start_meeting(next(p for p in g.players.values() if p.active));m.prepare()
        self.assertEqual(m.nav[pid].commitment.task,t.task_id)
        g.phase=Phase.VOTING
        for voter in g.participants:g._vote(voter,None)
        g._resolve_meeting();m.prepare();self.assertIs(g.phase,Phase.ROAMING)
        self.assertIsNone(m.nav[pid].goal);self.assertEqual(m.nav[pid].commitment.task,t.task_id)
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g._eliminate(imp,g.players[pid]);m._commitment_events([],g.time)
        self.assertIsNone(m.nav[pid].commitment.task)

    def test_body_and_lights_pauses_return_no_emergency_or_detour(self):
        from phase3_engine import Body
        m,pid,t=self.fixture();self.step(m,pid);g=m.game
        victim=next(p for p in g.players.values() if p.player_id!=pid and p.role is Role.CREWMATE)
        g.bodies[victim.player_id]=Body(victim.player_id,g.players[pid].position,g.tick,'offline-test');g._invalidate();m.prepare()
        packet=m.packets[pid];self.assertTrue(packet.interaction_mask[2]);self.assertFalse(packet.interaction_mask[4]);self.assertFalse(packet.navigation_mask[1])
        target=int(np.flatnonzero(target_mask(packet,'interaction',2))[0]);self.step(m,pid,Choice(interaction=2,target=target))
        self.assertIs(g.phase,Phase.DISCUSSION);self.assertEqual(m.nav[pid].commitment.task,t.task_id)
        g.phase=Phase.VOTING
        for voter in g.participants:g._vote(voter,None)
        g._resolve_meeting();g._begin_sabotage('lights')
        key=g.utilities['lights'][0];d=next(d for d in g.map.destinations if d.id==key)
        g._interrupt(g.players[pid],'fixture');g.players[pid].motion.spawn(tuple(d.standing));g._invalidate();m.prepare()
        packet=m.packets[pid];self.assertTrue(packet.interaction_mask[3]);target=int(np.flatnonzero(target_mask(packet,'interaction',3))[0])
        self.step(m,pid,Choice(interaction=3,target=target));self.assertIsNone(g.sabotage)
        self.assertEqual(m.nav[pid].commitment.task,t.task_id);self.assertIsNone(m.nav[pid].commitment.pause)

    def test_impossible_route_is_truncated_without_outcome_reward(self):
        m,pid,t=self.fixture();self.step(m,pid);c=m.nav[pid].commitment
        for _ in range(3):c.failed_route()
        m.prepare();r=self.step(m,pid)
        self.assertTrue(r['terminal']);self.assertEqual(r['result']['reason'],'TRUNCATED')
        self.assertIsNone(r['result']['winner']);self.assertEqual(r['result']['truncation']['cause'],'commitment_route_failures')
        self.assertFalse(any(v.get('win',0) or v.get('loss',0) for _,_,v in r['rewards']))

    def test_masks_roundtrip_singleton_and_ppo_log_probability(self):
        m,pid,t=self.fixture();self.step(m,pid);p=PatrickJane()
        packet=model_packet(m.packets[pid]);output,_=p(collate([packet],'cpu'),torch.zeros(1,256))
        choice,lp,entropy,_=choose({k:v[0] for k,v in output.items()},packet)
        logp,ent=evaluate_batch(output,[packet],[choice]);self.assertAlmostEqual(float(lp),float(logp[0]),places=6)
        self.assertEqual(float(entropy),0.);self.assertEqual(float(ent[0]),0.);self.assertEqual(float(lp),0.)
        self.assertEqual(sum(x.numel() for x in p.parameters()),1031070)

    def test_no_progress_guard_and_default_off(self):
        self.assertFalse(LightsExperiment().task_commitment_mode)
        m,pid,t=self.fixture();self.step(m,pid);c=m.nav[pid].commitment
        c.sync(m.game.observe_pj(pid),m.nav[pid]);c.last_progress=-100.
        c.sync(m.game.observe_pj(pid),m.nav[pid]);self.assertEqual(c.failure,'commitment_no_progress')

    def test_other_actor_repair_clears_exception_and_stale_route(self):
        m,pid,t=self.fixture();self.step(m,pid);g=m.game;g._begin_sabotage('lights');m.prepare()
        nav=m.nav[pid];c=nav.commitment;target=next(x for x in m.packets[pid].candidates if x.kind=='repair')
        c.pause=('repair',target.key);nav.goal=goal_for(target)
        g.sabotage=None;g._invalidate();m.prepare()
        self.assertIsNone(c.pause);self.assertIsNone(nav.goal);self.assertFalse(c.route_allowed)

    def test_actual_rejected_routes_trip_bounded_guard(self):
        from unittest.mock import patch
        m,pid,t=self.fixture();self.step(m,pid);t.stage=1;t.work=0.
        m.game._interrupt(m.game.players[pid],'fixture');m.game._invalidate();m.prepare()
        with patch.object(m.game.players[pid].navigator,'navigate',return_value=NavStatus.NO_ROUTE):
            for _ in range(3):r=self.step(m,pid)
        self.assertTrue(r['terminal']);self.assertEqual(r['result']['truncation']['cause'],'commitment_route_failures')

    def test_real_recurrent_ppo_update_strict_checkpoint_and_flag_rejection(self):
        from patrick_jane.ppo import update
        from patrick_jane.checkpoint import save,load
        from patrick_jane.learning_run import validate_pointer
        ex=replace(LightsExperiment(task_commitment_mode=True),technical_max_seconds=12.)
        p=PatrickJane();o=torch.optim.Adam(p.parameters(),lr=ex.learning_rate);rng=np.random.default_rng(17)
        with Pool(1,processes=False) as pool:r=collect(p,pool,ex,[ENGINEERING[1]],wall_limit=40)
        update(p,o,r['trajectories'],ex,rng=rng)
        with tempfile.TemporaryDirectory(dir=Path('local-runs')) as directory:
            folder=Path(directory);h=save(folder/'checkpoint.pt',p,o,ex,{'update':1},rng=rng)
            q=PatrickJane();opt=torch.optim.Adam(q.parameters(),lr=ex.learning_rate)
            load(folder/'checkpoint.pt',q,opt,rng=np.random.default_rng(2))
            self.assertTrue(all(torch.equal(v,q.state_dict()[k]) for k,v in p.state_dict().items()))
            with self.assertRaises(ValueError):validate_pointer(folder,{'checkpoint':'checkpoint.pt','sha256':h},q,opt,rng,replace(ex,task_commitment_mode=False))

    def test_bootstrap_syncs_the_final_private_commitment_before_closing(self):
        from unittest.mock import patch
        m,pid,t=self.fixture();m.game.experiment=replace(m.game.experiment,technical_max_seconds=.2)
        c=m.nav[pid].commitment
        with patch.object(c,'sync',wraps=c.sync) as sync:r=self.step(m,pid)
        self.assertTrue(r['terminal']);self.assertGreater(sync.call_count,0)
        self.assertEqual(r['bootstrap_packets'][pid].self_features[188],1.)
        self.assertIsNone(c.task)

if __name__=='__main__':unittest.main()
