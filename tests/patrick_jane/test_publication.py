"""Publication/orchestration only: synthetic waves, no policy learning or games."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from patrick_jane.progress import atomic_json,recorded_metrics
from patrick_jane.duration_budget import ActiveBudget
from portable_support import export_results,import_results,digest

ROOT=Path(__file__).resolve().parents[2]


class PublicationTests(unittest.TestCase):
    def test_real_prepared_report_has_version_and_no_training_claim(self):
        from patrick_jane.a2_report import generate
        tmp=ROOT/'local-runs/publication-tests';tmp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            folder=Path(d)
            atomic_json(folder/'registration.json',dict(source_version={'revision':'synthetic-test-only'},import_provenance={'checkpoint_sha256':'test-only'}))
            result=generate(folder,'prepared',active_seconds=0.)
            self.assertEqual(result['source_version']['revision'],'synthetic-test-only')
            self.assertEqual(result['additional_decisions'],0);self.assertFalse(result['validation_performed'])
            self.assertTrue((folder/'TRAINING_REPORT.md').exists())

    def test_terminal_counters_use_recorded_complete_matches(self):
        matches=[dict(reason='TASKS_COMPLETED',crew=10,metrics={'task_completions':40},ghost_tasks=5,
            neural_actions={'correct_vote':2,'incorrect_vote':1,'skip':4}),
            dict(reason='PARITY',crew=10,metrics={'task_completions':12},ghost_tasks=7,neural_actions={})]
        m=recorded_metrics(matches)
        self.assertEqual((m['task_victories'],m['living_tasks'],m['ghost_tasks'],m['assigned_tasks']),(1,40,12,80))
        self.assertEqual((m['correct_votes'],m['incorrect_votes'],m['skips']),(2,1,4))

    def run_fixture(self,root,*,export_failure=False):
        import patrick_jane.a2_run as runner
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
            def to(self,*a,**k):return self
            def architecture(self):return {'synthetic_test_only':True}
        now=[0.];calls=[]
        counters=dict(update=102,individual_decisions=2813879,complete_matches=1053,next_seed=0,run_decisions=0,run_matches=0,run_updates=0)
        def collect(*a,**k):
            calls.append('existing collector invoked');now[0]+=3
            return dict(individual_decisions=60,elapsed=3.,trajectories={},matches=[dict(seed=3200000+i,winner='impostor',reason='PARITY',truncated=False,
                simulation_seconds=1.,crew=10,metrics={'task_completions':0},ghost_tasks=0,neural_actions={}) for i in range(6)],reward_components={})
        def update(*a,**k):calls.append('mock PPO invoked; no optimizer steps');now[0]+=1;return dict(updates=0,loss=0.,approx_kl=0.,elapsed=1.)
        def save(path,*a,**k):path.parent.mkdir(exist_ok=True);path.write_bytes(b'synthetic checkpoint');return digest(path)
        def report(folder,state,**k):
            # Explicit synthetic artifacts exercise export/failure paths only.
            (folder/'TRAINING_REPORT.md').write_text('SYNTHETIC ORCHESTRATION TEST: '+state)
            atomic_json(folder/'training_results.json',dict(state=state,synthetic_test_only=True))
            (folder/'learning_curves.png').write_bytes(b'synthetic-placeholder-not-a-research-figure')
        def recover(folder,history,*a):
            pointer=json.loads((folder/'latest.json').read_text());return pointer['counters'],pointer,history,[]
        (root/'module.py').write_text('synthetic_test_only=True')
        atomic_json(root/'portable.manifest.json',dict(files={'module.py':dict(sha256=digest(root/'module.py'),size=(root/'module.py').stat().st_size)}))
        with contextlib.ExitStack() as stack:
            overrides=dict(ROOT=root,PatrickJane=Policy,source_lock=lambda:{'s':'h'},freeze_sources=lambda f:{},
                ActiveBudget=lambda f,h,**kw:ActiveBudget(f,h,clock=lambda:now[0]),Pool=lambda n:contextlib.nullcontext(object()),
                Resources=lambda:contextlib.nullcontext(SimpleNamespace(summary=lambda:{})),save=save,collect=collect,update=update,
                audit_lights=lambda r:None,generate=report,recover=recover,import_update102=lambda *a:(dict(counters),{}))
            for name,value in overrides.items():stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch('torch.cuda.is_available',return_value=True));stack.enter_context(patch('torch.optim.Adam',return_value=object()))
            stack.enter_context(patch('portable_support.verify',return_value={}))
            if export_failure:stack.enter_context(patch('portable_support.export_results',side_effect=OSError('synthetic export failure')))
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture):
                code=runner.train(root/'results',root,hours=.001)
                if not export_failure:resumed=runner.train(root/'results',root,hours=.002)
            if not export_failure:self.assertEqual(resumed,0);self.assertEqual(len(calls),4)
        return code,capture.getvalue(),root/'results'

    def test_combined_existing_runner_success_resume_and_real_export(self):
        tmp=ROOT/'local-runs/publication-tests';tmp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            root=Path(d);code,console,folder=self.run_fixture(root)
            self.assertEqual(code,0);self.assertIn('TRAINING COMPLETED',console)
            self.assertIn('FINAL WEIGHTS:',console);self.assertIn('RESULTS EXPORT:',console)
            self.assertEqual(json.loads((folder/'latest.json').read_text())['counters']['update'],104)
            archive=sorted((root/'exports').glob('*.zip'))[-1];imported=import_results(archive,root/'returned')
            for name in ('registration.json','latest.json','history.json','TRAINING_REPORT.md','training_results.json','final-model.json','active-time.json','duration-budget.json','seed-allocation.json'):
                self.assertTrue((imported/'run'/name).exists(),name)
            self.assertTrue(list((imported/'run/checkpoints').glob('*.pt')))
            self.assertTrue(list((imported/'run/weights').glob('*.pt')))

    def test_required_export_failure_never_announces_completion(self):
        tmp=ROOT/'local-runs/publication-tests';tmp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            code,console,folder=self.run_fixture(Path(d),export_failure=True)
            self.assertEqual(code,1);self.assertNotIn('TRAINING COMPLETED',console)
            self.assertIn('EXPORT FAILED',console);self.assertTrue((folder/'final-model.json').exists())
            status=json.loads((folder/'completion-status.json').read_text());self.assertEqual(status['state'],'failed');self.assertFalse(status['success'])

if __name__=='__main__':unittest.main()
