"""Bounded correctness only: no complete-game evaluation or optimizer steps."""
import json,tempfile,unittest,shutil,zipfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from patrick_jane.worker import Match
from patrick_jane.config import ENGINEERING
from patrick_jane.duration_run import experiment
from patrick_jane.duration_budget import ActiveBudget
from patrick_jane.lights_setup import LightsExperiment
from patrick_jane.rewards import COEFFICIENTS,RewardLedger
from patrick_jane.health import audit_rollout
from patrick_jane.api import Choice
from patrick_jane.features import target_mask,model_packet
from patrick_jane.decoder import execute
from patrick_jane.network import PatrickJane,collate
from patrick_jane.actions import choose,evaluate_batch
from social_deduction.actor import Role
from social_deduction.phase3_api import IntentKind
from portable_support import safe_path,import_results,export_results,digest

ROOT=Path(__file__).resolve().parents[2]
TEST_TEMP=ROOT/'local-runs/portable-correctness-tmp'


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);TEST_TEMP.mkdir(parents=True,exist_ok=True)

    def test_historical_defaults_and_unchanged_architecture_ppo(self):
        old=LightsExperiment();new=experiment()
        self.assertFalse(old.task_pursuit_mode);self.assertFalse(old.task_commitment_mode);self.assertEqual(old.living_task_reward,1)
        self.assertTrue(new.task_commitment_mode);self.assertTrue(new.task_pursuit_mode);self.assertEqual(new.living_task_reward,2)
        self.assertEqual(COEFFICIENTS['task'],1.)
        for k in ('learning_rate','gamma','gae_lambda','clip','entropy','epochs','max_grad_norm','sequence_length'):self.assertEqual(getattr(old,k),getattr(new,k))
        self.assertEqual(sum(p.numel() for p in PatrickJane().parameters()),1031070)

    def test_distant_task_selection_navigates_without_start_or_lock(self):
        m=Match(ENGINEERING[0],experiment());m.prepare();pid=m.crew[0];packet=m.packets[pid]
        index=next(i for i,t in enumerate(packet.candidates) if t.kind=='task' and t.detail.console_id not in packet.observation.actor.interactable_destinations)
        self.assertTrue(target_mask(packet,'interaction',1)[index])
        choice=Choice(interaction=1,target=index);intent=execute(choice,packet,m.nav[pid])
        self.assertIs(intent.kind,IntentKind.GO_TO_TASK)
        self.assertIsNone(m.nav[pid].commitment.task);self.assertIsNone(m.game.players[pid].interaction_task)
        self.assertFalse(any(e['kind']=='task_start' for e in m.game.event_log))

    def test_pursuit_sampling_and_recurrent_likelihood_masks_match(self):
        m=Match(ENGINEERING[1],experiment());m.prepare();packet=model_packet(m.packets[m.crew[0]])
        p=PatrickJane();out,_=p(collate([packet],'cpu'),torch.zeros(1,256));single={k:v[0] for k,v in out.items()}
        index=int(np.flatnonzero(target_mask(packet,'interaction',1))[0]);choice=Choice(interaction=1,target=index)
        chosen,lp,en,_=choose(single,packet,choice=choice);a,b=evaluate_batch(out,[packet],[chosen])
        self.assertAlmostEqual(float(lp.detach()),float(a[0].detach()),places=6);self.assertAlmostEqual(float(en.detach()),float(b[0].detach()),places=6)

    def test_full_task_only_two_points_once_and_zero_ghost(self):
        game=SimpleNamespace(experiment=experiment(),players={'p0':SimpleNamespace(role=Role.CREWMATE)},event_log=[],time=0.,is_terminal=False)
        ledger=RewardLedger(game)
        for kind in ('task_start','task_stage_completed','task_interrupted'):
            game.event_log.append(dict(kind=kind,player_id='p0',task_id='t0'));self.assertEqual(ledger.collect(game)['p0'],{})
        game.event_log.append(dict(kind='task_completed',player_id='p0',task_id='t0'));self.assertEqual(ledger.collect(game)['p0']['task'],2.)
        game.event_log.append(dict(kind='task_completed',player_id='p0',task_id='t0'));self.assertEqual(ledger.collect(game)['p0'],{})
        game.event_log.append(dict(kind='task_completed',player_id='p0',task_id='t1',ghost=True));self.assertEqual(ledger.collect(game)['p0']['ghost_task'],0.)
        self.assertEqual({k:v for k,v in ledger.coefficients.items() if k!='task'},{k:v for k,v in COEFFICIENTS.items() if k!='task'})

    def test_reward_bound_uses_registered_coefficient_and_owned_count(self):
        match=dict(individual_decisions=1,neural_actions={'travel':1},metrics={'illegal_actions':0},winner=None,crew_reports=0,
            task_reward_coefficient=2.,assigned_tasks_per_actor={'p0':4})
        r=dict(matches=[match],reward_components={'0/p0':{'task':8.,'ghost_task':0.}});audit_rollout(r)
        r['reward_components']['0/p0']['task']=10.
        with self.assertRaises(RuntimeError):audit_rollout(r)
        match.update(monitor_error='observer failed',observer_errors_fatal=False);r['reward_components']['0/p0']['task']=8.;audit_rollout(r)

    def test_pursuit_commitment_multistage_incubation_and_final_reward(self):
        from crewmate_v2.tasks import make_plan,task_catalog
        from random import Random
        for kind in ('UploadData','InspectSample'):
            m=Match(ENGINEERING[0],experiment());m.opponents={};pid=m.crew[0];g=m.game;t=g.players[pid].tasks[0]
            t.plan=make_plan(pid,kind,'Long',task_catalog(g.map),Random(17));t.stage=0;t.work=0.;t.ready_at=0.
            dest=next(d for d in g.map.destinations if d.id==t.console_id);g.players[pid].motion.spawn(tuple(dest.standing));g._invalidate();m.prepare()
            paid=0.;incubation_seen=False;stage_seen=False
            for _ in range(1800):
                packet=m.packets[pid]
                if packet.interaction_mask[1]:choice=Choice(interaction=1,target=int(np.flatnonzero(target_mask(packet,'interaction',1))[0]))
                elif packet.navigation_mask[0]:choice=Choice(interaction=0,navigation=0)
                else:choice=Choice(interaction=0,navigation=2,target=int(np.flatnonzero(target_mask(packet,'navigation',2))[0]))
                response=m.advance({pid:dict(choice=choice,logp=0.,value=0.,entropy=0.,probabilities={})})
                paid+=sum(v.get('task',0) for _,owner,v in response['rewards'] if owner==pid)
                if t.stage>0:stage_seen=True
                if t.ready_at>g.time:
                    incubation_seen=True;self.assertFalse(m.packets[pid].interaction_mask[1])
                if t.progress<1:self.assertEqual(paid,0.)
                else:break
            self.assertEqual(t.progress,1.);self.assertEqual(paid,2.);self.assertTrue(stage_seen);self.assertIsNone(m.nav[pid].commitment.task)
            if kind=='InspectSample':self.assertTrue(incubation_seen)

    def test_budget_counts_only_active_work_and_resumes_and_extends(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as d:
            now=[0.];b=ActiveBudget(d,1,clock=lambda:now[0]);now[0]=300;self.assertEqual(b.value(),0)
            b.begin();now[0]+=100;b.persist();b.end();now[0]+=900;self.assertEqual(b.value(),100)
            resumed=ActiveBudget(d,1,clock=lambda:now[0]);self.assertEqual(resumed.value(),100)
            extended=ActiveBudget(d,2,clock=lambda:now[0]);self.assertEqual(extended.target,7200)
            self.assertEqual(len(json.loads((Path(d)/'duration-budget.json').read_text())),2)
            with self.assertRaises(ValueError):ActiveBudget(d,1,clock=lambda:now[0])

    def test_update18_exact_policy_and_adam_transfer(self):
        from patrick_jane.pursuit_transition import import_update18,SHA
        old=ROOT/'local-runs/patrick-jane/task-commitment-v1'
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as d:
            shutil.copyfile(old/'checkpoints/attempt-0002-update-000018.pt',Path(d)/'update-000018.pt')
            shutil.copyfile(old/'source-snapshot.zip',Path(d)/'source-snapshot.zip')
            p=PatrickJane();o=torch.optim.Adam(p.parameters(),lr=.0003)
            c,info=import_update18(d,p,o,np.random.default_rng(17))
            original=torch.load(Path(d)/'update-000018.pt',map_location='cpu',weights_only=True)
            self.assertEqual(c['individual_decisions'],1518786);self.assertEqual(c['next_seed'],0);self.assertEqual(c['run_decisions'],0)
            self.assertEqual(len(o.state),52);self.assertEqual(info['checkpoint_sha256'],SHA)
            for k,v in p.state_dict().items():self.assertTrue(torch.equal(v.cpu(),original['state_dict'][k]))
            transferred=o.state_dict()
            for k,state in original['optimizer']['state'].items():
                for name,v in state.items():self.assertTrue(torch.equal(transferred['state'][k][name].cpu(),v.cpu()) if torch.is_tensor(v) else transferred['state'][k][name]==v)

    def test_import_checks_hashes_paths_and_never_overwrites(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as d:
            root=Path(d);payload=root/'weights.pt';payload.write_bytes(b'unchanged-model')
            zpath=root/'export.zip';record={'schema':'patrick-jane-results-export-v1','files':{'run/weights.pt':{'sha256':digest(payload),'size':payload.stat().st_size}}}
            with zipfile.ZipFile(zpath,'w') as z:z.write(payload,'run/weights.pt');z.writestr('export-manifest.json',json.dumps(record))
            out=import_results(zpath,root/'new');self.assertEqual((out/'run/weights.pt').read_bytes(),payload.read_bytes())
            with self.assertRaises(FileExistsError):import_results(zpath,root/'new')
            for name in ('../escape','C:/escape','/absolute','..\\escape'):
                with self.assertRaises(ValueError):safe_path(root,name)

    def test_actual_export_roundtrip_manifest_hashes_archived_bytes(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as d:
            root=Path(d);run=root/'results';run.mkdir();(root/'module.py').write_text('portable_source=True\n')
            (run/'latest.json').write_text(json.dumps({'counters':{'update':19,'individual_decisions':1519000}}))
            (run/'model.pt').write_bytes(b'preserved-model');(run/'viewer-state.json').write_text('{"heartbeat":1}')
            (root/'portable.manifest.json').write_text(json.dumps({'files':{'module.py':{'sha256':digest(root/'module.py'),'size':(root/'module.py').stat().st_size}}}))
            archive=export_results(root,run);destination=import_results(archive,root/'imported')
            self.assertEqual((destination/'run/model.pt').read_bytes(),b'preserved-model');self.assertTrue(archive.with_suffix('.zip.sha256').exists())

    def test_duration_runner_complete_wave_then_update_and_resume_without_learning(self):
        import contextlib
        import patrick_jane.duration_run as r
        from patrick_jane.progress import atomic_json
        clock=[0.];waves=[];updates=[]
        class FakePolicy(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
            def to(self,*a,**k):return self
            def architecture(self):return {'fake':True}
        def fake_save(path,*a,**k):path.parent.mkdir(exist_ok=True);path.write_bytes(b'verified');return digest(path)
        def fake_collect(policy,pool,exp,seeds,**kw):
            waves.append(seeds);clock[0]+=31
            return dict(individual_decisions=60,elapsed=31.,trajectories={},reward_components={},matches=[dict(seed=s,simulation_seconds=10.,winner='impostor') for s in seeds])
        def fake_update(*a,**k):updates.append(1);clock[0]+=9;return dict(updates=1,elapsed=9.,approx_kl=0.,loss=0.)
        def fake_recover(folder,history,*a):
            pointer=json.loads((folder/'latest.json').read_text());return pointer['counters'],pointer,history,[]
        def report(folder,*a,**k):(folder/'TRAINING_REPORT.md').write_text('mocked boundary test; no learning')
        counters=dict(individual_decisions=1518786,complete_matches=549,update=18,next_seed=0,run_decisions=0,run_matches=0,run_updates=0)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as d,contextlib.ExitStack() as stack:
            root=Path(d);folder=root/'results'
            overrides={'ROOT':root,'PatrickJane':FakePolicy,'source_lock':lambda:{'s':'h'},'freeze_sources':lambda f:{},
                'ActiveBudget':lambda f,h:ActiveBudget(f,h,clock=lambda:clock[0]),'Pool':lambda n:contextlib.nullcontext(object()),
                'Resources':lambda:contextlib.nullcontext(SimpleNamespace(summary=lambda:{})),'save':fake_save,'collect':fake_collect,'update':fake_update,
                'audit_lights':lambda r:None,'generate':report,'recover':fake_recover,'import_update18':lambda *a:(dict(counters),{})}
            for name,v in overrides.items():stack.enter_context(patch.object(r,name,v))
            stack.enter_context(patch('torch.cuda.is_available',return_value=True));stack.enter_context(patch('torch.optim.Adam',return_value=object()))
            stack.enter_context(patch('portable_support.verify',return_value={}));stack.enter_context(patch('portable_support.export_results',return_value='mock-export.zip'))
            self.assertEqual(r.train(folder,root,hours=.01),0)
            self.assertEqual(len(waves),1);self.assertEqual(len(updates),1);self.assertEqual(clock[0],40)
            self.assertEqual(r.train(folder,root,hours=.02),0)
            self.assertEqual(len(waves),2);self.assertEqual(waves[0],list(range(3100000,3100006)));self.assertEqual(waves[1],list(range(3100006,3100012)))
            latest=json.loads((folder/'latest.json').read_text());self.assertEqual(latest['counters']['run_decisions'],120)
            self.assertEqual(latest['counters']['update'],20)
            failed_folder=root/'failed-results'
            def failed_collection(*a,**kw):clock[0]+=31;raise RuntimeError('mock crash, no learner execution')
            with patch.object(r,'collect',failed_collection):self.assertEqual(r.train(failed_folder,root,hours=.02),1)
            self.assertEqual(r.train(failed_folder,root,hours=.02),0)
            allocation=json.loads((failed_folder/'seed-allocation.json').read_text())
            self.assertEqual(allocation[0]['state'],'allocated')
            self.assertEqual([a['seeds'][0] for a in allocation],[3100000,3100006,3100012])
            latest=json.loads((failed_folder/'latest.json').read_text());self.assertEqual(latest['counters']['run_decisions'],120)

if __name__=='__main__':unittest.main()
