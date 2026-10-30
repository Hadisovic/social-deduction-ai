"""Bounded A2 mechanics/privacy/credit checks; no experimental training campaign."""
import unittest,tempfile,json,shutil,math
from pathlib import Path
from dataclasses import replace
from collections import Counter
import numpy as np
import torch
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import Intent,IntentKind
from navigation_service import NavStatus,NavTarget
from patrick_jane.api import Action,Choice
from patrick_jane.a2_config import A2Experiment
from patrick_jane.worker import Match
from patrick_jane.a2_geometry import geometry,vents
from patrick_jane.a2_rewards import A2Ledger
from patrick_jane.features import encode,model_packet,event_code,KINDS,target_mask
from patrick_jane.network import PatrickJane,collate
from patrick_jane.actions import choose,evaluate_batch
from patrick_jane.config import ENGINEERING

ROOT=Path(__file__).resolve().parents[2]


class A2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def fixture(self,**kw):
        m=Match(ENGINEERING[0],A2Experiment(**kw));m.opponents={};return m,m.game
    def voting(self,g):
        pid=next(p.player_id for p in g.players.values() if p.active)
        g._start_meeting(g.players[pid]);g.phase=Phase.VOTING;g._invalidate()
        crew=[p for p in g.players if g.players[p].role is Role.CREWMATE];imps=[p for p in g.players if g.players[p].role is Role.IMPOSTOR]
        return crew,imps

    def test_accepted_ballot_correct_wrong_skip_duplicate_self_rejection(self):
        m,g=self.fixture();crew,imps=self.voting(g);ledger=A2Ledger(g)
        self.assertTrue(g._vote(crew[0],imps[0]));self.assertTrue(g._vote(crew[1],crew[2]));self.assertTrue(g._vote(crew[3],None))
        self.assertFalse(g._vote(crew[0],imps[1]));self.assertFalse(g._vote(crew[4],crew[4]))
        credits=ledger.collect(g);self.assertAlmostEqual(credits[crew[0]]['vote_accuracy'],.2*(1-2/11))
        self.assertAlmostEqual(credits[crew[1]]['vote_accuracy'],-.2*2/11)
        self.assertEqual(credits[crew[3]].get('vote_accuracy',0),0);self.assertFalse(any(v.get('vote_accuracy',0) for v in ledger.collect(g).values()))
        obs=g.observe_pj(crew[4]);eligible={a.target for a in obs.actor.actions if a.kind is IntentKind.VOTE}
        self.assertEqual(eligible,set(g._eligible_voters())-{crew[4]})
        self.assertAlmostEqual((2/11)*.2*(1-2/11)+(9/11)*(-.2*2/11),0.)

    def test_same_step_ejected_voter_keeps_accepted_ballot_credit(self):
        m,g=self.fixture();crew,imps=self.voting(g);ledger=A2Ledger(g)
        self.assertTrue(g._vote(crew[0],imps[0]))
        for pid in g.participants:
            if pid!=crew[0]:g._vote(pid,crew[0])
        g._resolve_meeting();self.assertFalse(g.players[crew[0]].active)
        self.assertAlmostEqual(ledger.collect(g)[crew[0]]['vote_accuracy'],.2*(1-2/11))

    def test_vote_snapshot_one_impostor_zero_targets_and_automatic_skip(self):
        m,g=self.fixture();crew,imps=self.voting(g);g.players[imps[1]].status='ejected'
        g._vote(crew[0],imps[0]);self.assertAlmostEqual(g.accepted_ballots[-1]['base_rate'],1/10)
        g._vote(crew[1],None);self.assertEqual(g.accepted_ballots[-1]['reward'],0);self.assertFalse(g.accepted_ballots[-1]['explicit'])
        g.participants=(crew[2],);g._vote(crew[2],None);self.assertEqual(g.accepted_ballots[-1]['candidate_count'],0)

    def test_vote_truth_private_no_observation_or_memory_feedback(self):
        m,g=self.fixture();crew,imps=self.voting(g);g._vote(crew[0],imps[0]);obs=g.observe_pj(crew[1])
        self.assertFalse(any(e.kind=='a2_ballot_credit' for e in obs.events))
        m.memory[crew[1]].update(obs)
        self.assertFalse(any(e.kind=='a2_ballot_credit' for e in m.memory[crew[1]].events))
        self.assertTrue(any(e['channel']=='training-private' for e in g.event_log if e['kind']=='a2_ballot_credit'))

    def test_terminal_rewards_nonstacking_and_exact_task_ghost(self):
        for reason,expected in [('TASKS_COMPLETED',5.),('IMPOSTOR_EJECTED',10.),('PARITY',-10.)]:
            m,g=self.fixture();ledger=A2Ledger(g);g._finish(Role.IMPOSTOR if reason=='PARITY' else Role.CREWMATE,reason)
            credits=ledger.collect(g)
            for v in credits.values():self.assertEqual(v.get('win',0)+v.get('loss',0),expected)
            self.assertFalse(any(v for v in ledger.collect(g).values()))
        m,g=self.fixture();ledger=A2Ledger(g);pid=m.crew[0]
        for kind in ('task_start','task_stage_complete'):g._record(kind,player_id=pid,task_id='t')
        self.assertEqual(ledger.collect(g)[pid],{})
        g._record('task_completed',player_id=pid,task_id='t',ghost=False);self.assertEqual(ledger.collect(g)[pid]['task'],.5)
        g._record('task_completed',player_id=pid,task_id='t2',ghost=True);self.assertEqual(ledger.collect(g)[pid]['ghost_task'],0)

    def test_door_geometry_swept_collision_and_reopening_shared_cooldown(self):
        m,g=self.fixture();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        g.tick=50;self.assertTrue(g.close_doors(imp.player_id,'Electrical'))
        key=g.door_groups['Electrical'][0];self.assertIn(key,g.map.closed_doors)
        self.assertFalse(g.observe_pj(imp.player_id).can_sabotage)
        old=geometry(());closed=g.map
        from shapely.geometry import Polygon
        door=next(d for d in old.data['doors'] if d['id']==key);poly=Polygon(door['points']);cx,cy=poly.centroid.coords[0]
        pairs=[pair for v in (.8,1.2,1.5,2.) for pair in (((cx,cy-v),(cx,cy+v)),((cx-v,cy),(cx+v,cy)))]
        a,b=next((a,b) for a,b in pairs if old.contains(a) and old.contains(b) and old.segment_clear(a,b))
        self.assertFalse(closed.segment_clear(a,b));pos,_=closed.move(a,((b[0]-a[0]),(b[1]-a[1])))
        self.assertGreater(math.dist(pos,b),.1)
        self.assertEqual(g.door_ready['Electrical'],40.)
        g.tick=100;g._update_doors();self.assertFalse(g.door_until);self.assertFalse(g.map.closed_doors)
        self.assertTrue(g.map.segment_clear(a,b));self.assertEqual(g.sabotage_ready,50.)

    def test_doors_multiple_rooms_no_lights_overlap_and_meeting_opens(self):
        m,g=self.fixture();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g.tick=50
        self.assertTrue(g.close_doors(imp.player_id,'Electrical'));self.assertTrue(g.close_doors(imp.player_id,'MedBay'))
        self.assertFalse(g._extension_allowed(imp.player_id,Action('sabotage','lights'),g.observe_pj(imp.player_id)))
        g._start_meeting(g.players[m.crew[0]]);self.assertFalse(g.map.closed_doors);self.assertFalse(g.door_until)
        g.phase=Phase.ROAMING;g._begin_sabotage('lights');self.assertFalse(g.can_close(imp.player_id,'Storage'))

    def test_goal_temporarily_inside_closed_door_is_not_illegal(self):
        from shapely.geometry import Polygon
        from social_deduction.actor import Point
        m,g=self.fixture();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g.tick=50
        g.close_doors(imp.player_id,'Electrical');door=next(d for d in g.base_map.data['doors'] if d['room']=='Electrical')
        center=Polygon(door['points']).centroid;point=(float(center.x),float(center.y))
        self.assertTrue(g.base_map.contains(point));self.assertFalse(g.map.contains(point))
        g.step({m.crew[0]:Intent(IntentKind.GO_TO_LOCATION,position=Point(*point))})
        self.assertEqual(g.metrics['illegal_actions'],0)

    def test_closed_door_blocks_actor_vision_but_never_exposes_remote_state(self):
        from shapely.geometry import Polygon
        m,g=self.fixture();crew=m.crew[:2];door=next(d for d in g.base_map.data['doors'] if d['room']=='Electrical');cx,cy=Polygon(door['points']).centroid.coords[0]
        g.players[crew[0]].motion.spawn((cx,cy-.8));g.players[crew[1]].motion.spawn((cx,cy+.8));g._invalidate()
        self.assertIn(crew[1],{p.player_id for p in g.observe_pj(crew[0]).actor.visible_players})
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g.tick=50;g.close_doors(imp.player_id,'Electrical')
        obs=g.observe_pj(crew[0]);self.assertNotIn(crew[1],{p.player_id for p in obs.actor.visible_players});self.assertEqual(obs.door_actions,())
        self.assertFalse(hasattr(obs,'door_until'));self.assertFalse(hasattr(obs,'accepted_ballots'))

    def test_vent_graph_entry_hiding_travel_exit_cooldown_and_illegal_actions(self):
        m,g=self.fixture();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);key='3'
        imp.motion.spawn(g.vents[key]['position']);imp.cooldown=20.
        self.assertFalse(g.vent_action(m.crew[0],'vent_enter',key));self.assertTrue(g.vent_action(imp.player_id,'vent_enter',key))
        self.assertFalse(g.vent_action(imp.player_id,'vent_move','0'));self.assertFalse(g.observe_pj(imp.player_id).can_sabotage)
        self.assertFalse(g.observe_pj(imp.player_id).actor.visible_players);self.assertFalse(g.observe_pj(imp.player_id).actor.visible_bodies)
        for _ in range(4):g.step({})
        self.assertEqual(imp.cooldown,20.)
        other=g.vents[key]['links'][0];self.assertTrue(g.vent_action(imp.player_id,'vent_move',other))
        for _ in range(4):g.step({})
        self.assertTrue(g.vent_action(imp.player_id,'vent_exit',other));self.assertNotIn(imp.player_id,g.vented)
        self.assertLess(math.dist(imp.position,g.vents[other]['position']),1e-8)

    def test_vent_evidence_firsthand_only_no_hidden_travel_destination(self):
        m,g=self.fixture();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);point=g.vents['3']['position']
        witness=m.crew[0];far=m.crew[1]
        imp.motion.spawn(point);g.players[witness].motion.spawn(point);g.players[far].motion.spawn(g.vents['12']['position']);g._invalidate()
        self.assertTrue(g.vent_action(imp.player_id,'vent_enter','3'))
        events=g.observe_pj(witness).events;self.assertTrue(any(e.kind=='vent_enter' and e.subject==imp.player_id and e.provenance=='direct' for e in events))
        self.assertFalse(any(e.kind.startswith('vent_') for e in g.observe_pj(far).events))
        self.assertFalse(any(p.player_id==imp.player_id for p in g.observe_pj(witness).actor.visible_players))
        g.tick+=4;g.vent_action(imp.player_id,'vent_move','5')
        self.assertFalse(any(e.kind=='vent_move' for e in g.observe_pj(witness).events))
        g._start_meeting(g.players[witness]);self.assertFalse(g.vented)
        self.assertFalse(any(e.kind=='vent_exit' for e in g.observe_pj(witness).events))

    def test_vented_observer_gets_no_deferred_visual_task_proof(self):
        from crewmate_v2.tasks import TaskPlan,Stage
        m,g=self.fixture();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        imp.motion.spawn(g.vents['3']['position']);g.vent_action(imp.player_id,'vent_enter','3')
        crew=g.players[m.crew[0]];crew.motion.spawn(g.vents['3']['position']);t=crew.tasks[0]
        t.plan=TaskPlan(crew.player_id,'SubmitScan','Long',(Stage(t.console_id,1.),),True)
        g._completed(crew,t)
        self.assertFalse(g.v2_direct[imp.player_id])

    def test_features_keep_original_codes_and_shapes_no_forced_votes(self):
        for i,k in enumerate(KINDS):self.assertEqual(event_code(k,True),(i+1)/len(KINDS))
        self.assertEqual(event_code('vent_enter',False),0);self.assertGreater(event_code('vent_enter',True),1)
        m,g=self.fixture();m.prepare();p=PatrickJane();packet=model_packet(m.packets[m.crew[0]])
        out,h=p(collate([packet],'cpu'),torch.zeros(1,256));c,lp,en,_=choose({k:v[0] for k,v in out.items()},packet)
        a,b=evaluate_batch(out,[packet],[c]);self.assertAlmostEqual(float(lp.detach()),float(a[0].detach()),places=6)
        self.assertEqual(h.shape,(1,256));self.assertEqual(p.architecture()['parameters'],1031070)
        crew,imps=self.voting(g);m.prepare();packet=m.packets[crew[0]];self.assertTrue(packet.vote_mask[14]);self.assertGreater(packet.vote_mask.sum(),2)

    def test_original_regime_and_opponent_source_still_selectable(self):
        from patrick_jane.lights_setup import LightsExperiment
        from patrick_jane.opponents import Opponent
        old=Match(ENGINEERING[0],LightsExperiment(task_commitment_mode=True,task_pursuit_mode=True,living_task_reward=2.))
        self.assertFalse(hasattr(old.game,'accepted_ballots'));self.assertTrue(all(type(o) is Opponent for o in old.opponents.values()))
        self.assertEqual(old.ledger.coefficients['task'],2.)

    def test_temporary_door_wait_recovers_committed_task_without_failure(self):
        from crewmate_v2.tasks import TaskPlan,Stage
        from patrick_jane.features import goal_for
        m,g=self.fixture();pid=m.crew[0];p=g.players[pid];t=p.tasks[0]
        destination=next(d for d in g.base_map.destinations if d.room=='Electrical' and d.category=='task')
        t.plan=TaskPlan(pid,'DiagnosticElectrical','Short',(Stage(destination.id,1.),))
        # Engineering fixture: committed travel toward the next owned stage.
        p.motion.spawn(g.base_map.nearest_node((-9.5,-14.8)));m.nav[pid].commitment.start(t.task_id,g.time)
        m.prepare();target=next(x for x in m.packets[pid].candidates if x.kind=='task' and x.key==t.task_id)
        m.nav[pid].goal=goal_for(target);g._navigate(p,NavTarget.task(destination.id),destination.id)
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g.tick=50;g.close_doors(imp.player_id,'Electrical')
        saw_wait=False
        for _ in range(220):
            m.prepare();packet=m.packets[pid]
            if packet.interaction_mask[1]:choice=Choice(interaction=1,target=int(np.flatnonzero(target_mask(packet,'interaction',1))[0]))
            elif packet.navigation_mask[0]:choice=Choice(interaction=0,navigation=0)
            else:choice=Choice(interaction=0,navigation=2,target=int(np.flatnonzero(target_mask(packet,'navigation',2))[0]))
            m.advance({pid:dict(choice=choice,logp=0.,value=0.,entropy=0.,probabilities={})})
            saw_wait|=pid in g.door_waiting
            if t.progress>=1:break
        self.assertTrue(saw_wait);self.assertEqual(t.progress,1.);self.assertEqual(g.metrics['navigation_failures'],0)
        self.assertEqual(m.nav[pid].commitment.stats.get('commitment_technical_failures',0),0)

    def test_upgraded_opponent_private_state_repeatability_and_legal_actions(self):
        from patrick_jane.a2_opponents import TacticalOpponent
        m,g=self.fixture(opponent_regime='upgraded-a2-v1');imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        one=TacticalOpponent(17,g.base_map);two=TacticalOpponent(17,g.base_map);obs=g.observe_pj(imp.player_id)
        self.assertEqual(one.decide(obs),two.decide(obs));self.assertIsNot(one.last_seen,two.last_seen)
        self.assertEqual(set(obs.teammates),{p.player_id for p in g.players.values() if p.role is Role.IMPOSTOR}-{imp.player_id})
        with self.assertRaises(TypeError):one.decide(g)
        for _ in range(70):
            obs=g.observe_pj(imp.player_id);action=one.decide(obs);g.step({imp.player_id:action})
        self.assertEqual(g.metrics['illegal_actions'],0)

    def test_origin102_all_weights_adam_and_new_source_checkpoint_resume(self):
        from patrick_jane.a2_transition import import_update102,SHA
        from patrick_jane.checkpoint import save,load
        tmp=ROOT/'local-runs/a2-engineering-tmp';tmp.mkdir(parents=True,exist_ok=True)
        old=ROOT/'artifacts/patrick-jane/task-pursuit-v1-update102'
        original_checkpoint=old/'checkpoint.pt'
        if not original_checkpoint.exists():self.skipTest('Published update102 artifacts are unavailable')
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            d=Path(d);shutil.copyfile(original_checkpoint,d/'update-000102.pt');shutil.copyfile(old/'source-snapshot.zip',d/'source-snapshot.zip')
            policy=PatrickJane();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003);rng=np.random.default_rng(17)
            c,info=import_update102(d,policy,optimizer,rng);self.assertEqual(c['individual_decisions'],2813879);self.assertEqual(info['policy_tensors_verified'],52)
            r=torch.load(d/'update-000102.pt',weights_only=True,map_location='cpu')
            for k,v in policy.state_dict().items():self.assertTrue(torch.equal(v,r['state_dict'][k]))
            for key,values in optimizer.state_dict()['state'].items():
                for name,v in values.items():self.assertTrue(torch.equal(v.cpu(),r['optimizer']['state'][key][name].cpu()))
            save(d/'a2-origin.pt',policy,optimizer,A2Experiment(),c,rng=rng)
            restored=PatrickJane();load(d/'a2-origin.pt',restored,restore_rng=False)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items()))

    def test_short_engineering_recording_on_off_and_ppo_finite(self):
        from patrick_jane.collector import Pool,collect
        from patrick_jane.ppo import update
        tmp=ROOT/'local-runs/a2-engineering-tmp';tmp.mkdir(parents=True,exist_ok=True)
        ex=replace(A2Experiment(opponent_regime='upgraded-a2-v1'),technical_max_seconds=3.)
        torch.manual_seed(17);policy=PatrickJane()
        with tempfile.TemporaryDirectory(dir=tmp) as d,Pool(1,processes=False) as pool:
            recorded=collect(policy,pool,ex,[ENGINEERING[2]],wall_limit=60,replay_dir=Path(d)/'replays',record_every=1)
            plain=collect(policy,pool,ex,[ENGINEERING[2]],wall_limit=60)
            self.assertEqual(recorded['matches'][0]['trace'],plain['matches'][0]['trace'])
            self.assertEqual(recorded['reward_components'],plain['reward_components'])
            self.assertEqual(recorded['matches'][0]['reason'],'TRUNCATED')
            optimizer=torch.optim.Adam(policy.parameters(),lr=ex.learning_rate)
            learning=update(policy,optimizer,recorded['trajectories'],ex,rng=np.random.default_rng(17))
            self.assertGreater(learning['updates'],0)
            for key in ('loss','approx_kl','entropy','grad_norm'):self.assertTrue(math.isfinite(learning[key]))
            from spectator.replay_format import parse_replay
            replay=parse_replay(next((Path(d)/'replays').glob('*.jsonl')))
            self.assertTrue(any(e['kind']=='a2_world_state' for e in replay['events']))

if __name__=='__main__':unittest.main()
