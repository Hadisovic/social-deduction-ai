import json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from among_us_map import get_map
from navigation_service import NavigationPlanner
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import Intent,IntentKind
from patrick_jane.environment import PatrickJaneGame
from patrick_jane.lights_setup import LightsExperiment,LightsLedger
from patrick_jane.config import Experiment
from patrick_jane.api import Action
from patrick_jane.rewards import COEFFICIENTS


class LightsLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.planner=NavigationPlanner(get_map(),cache_size=128)
    def game(self,**changes):return PatrickJaneGame(2920000,experiment=replace(LightsExperiment(),**changes),planner=self.planner)

    def test_per_checkpoint_reward_totals_retain_negative_components(self):
        from patrick_jane.learning_report import component_totals
        result=component_totals([{'loss':-10,'lights_unresolved_time':-.1,'task':1},
                                 {'loss':-10,'lights_unresolved_time':-.2}])
        self.assertEqual(result['loss'],-20);self.assertAlmostEqual(result['lights_unresolved_time'],-.3)
        self.assertEqual(result['task'],1)

    def test_profile_enforces_lights_but_retains_original_sabotages(self):
        g=self.game()
        for kind in ('oxygen','reactor'):
            with self.assertRaises(ValueError):g._begin_sabotage(kind)
        g._begin_sabotage('lights');self.assertEqual(g.sabotage,'lights')
        old=PatrickJaneGame(2920000,experiment=Experiment(),planner=self.planner)
        old._begin_sabotage('reactor');self.assertEqual(old.sabotage,'reactor')
        none=self.game(allowed_sabotages=())
        with self.assertRaises(ValueError):none._begin_sabotage('lights')

    def test_penalty_uses_simulated_elapsed_time_once_and_stops_on_repair(self):
        g=self.game();ledger=LightsLedger(g);g._begin_sabotage('lights')
        crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
        for _ in range(50):g.step({})
        result=ledger.collect(g)
        for p in crew:self.assertAlmostEqual(result[p.player_id]['lights_unresolved_time'],-.05)
        self.assertTrue(all(not c for c in ledger.collect(g).values()))
        target=g.utilities['lights'][0];p=crew[0];p.motion.spawn(g.planner.destinations[target].standing);g._invalidate()
        g.step({p.player_id:Action('repair',target)});r=ledger.collect(g)
        self.assertEqual(r[p.player_id]['repair'],COEFFICIENTS['repair'])
        self.assertTrue(all('lights_unresolved_time' not in c for c in r.values()))

    def test_darkness_cost_continues_in_meetings_but_excludes_ghosts(self):
        g=self.game();ledger=LightsLedger(g);crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        dead=crew[-1];g._eliminate(imp,dead);g._start_meeting(crew[0]);g._begin_sabotage('lights')
        before=g.time
        for _ in range(10):g.step({})
        self.assertIs(g.phase,Phase.DISCUSSION);r=ledger.collect(g)
        self.assertNotIn('lights_unresolved_time',r[dead.player_id])
        for p in crew[:-1]:self.assertAlmostEqual(r[p.player_id]['lights_unresolved_time'],-.005*(g.time-before))

    def test_killing_retains_normal_cooldowns_and_is_legal(self):
        g=self.game();imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);crew=next(p for p in g.players.values() if p.role is Role.CREWMATE)
        self.assertEqual(imp.cooldown,10.);self.assertEqual(g.config.kill_cooldown,45.)
        imp.cooldown=0;imp.motion.spawn(crew.position);g._invalidate()
        g.step({imp.player_id:Intent(IntentKind.KILL,crew.player_id)})
        self.assertFalse(crew.active);self.assertEqual(g.metrics['illegal_actions'],0);self.assertAlmostEqual(imp.cooldown,44.8)

    def test_no_network_or_existing_coefficient_change(self):
        from patrick_jane.network import PatrickJane
        self.assertEqual(sum(p.numel() for p in PatrickJane().parameters()),1031070)
        self.assertEqual(COEFFICIENTS['repair'],3.);self.assertEqual(COEFFICIENTS['task'],1.)
        self.assertNotIn('lights_unresolved_time',COEFFICIENTS)

    def test_script_only_selects_permitted_sabotage(self):
        from patrick_jane.opponents import Opponent
        g=self.game()
        for _ in range(51):g.step({})
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        obs=g.observe_pj(imp.player_id);script=Opponent(17,allowed_sabotages=('lights',));seen=[]
        for _ in range(100):
            a=script.decide(obs)
            if isinstance(a,Action) and a.kind=='sabotage':seen.append(a.target)
        self.assertTrue(seen);self.assertEqual(set(seen),{'lights'})

    def test_report_refuses_false_completion_and_preserves_failure_results(self):
        from patrick_jane.learning_report import generate
        parent=Path('local-runs/patrick-jane');parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            folder=Path(temp);self.assertTrue(folder.resolve().is_relative_to(parent.resolve()))
            with self.assertRaises(ValueError):generate(folder,'complete',plots=False)
            result=generate(folder,'interrupted',error='test interruption',plots=False)
            self.assertFalse(result['success']);self.assertEqual(result['additional_decisions'],0)
            self.assertIn('INTERRUPTED',(folder/'TRAINING_REPORT.md').read_text())

    def test_recovery_skips_a_corrupt_latest_pointer(self):
        import torch,numpy as np
        from patrick_jane.network import PatrickJane
        from patrick_jane.checkpoint import save
        from patrick_jane.learning_run import recover
        from patrick_jane.progress import atomic_json
        parent=Path('local-runs/patrick-jane')
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            folder=Path(temp);self.assertTrue(folder.resolve().is_relative_to(parent.resolve()))
            p=PatrickJane();opt=torch.optim.Adam(p.parameters(),lr=.0003);rng=np.random.default_rng(17);ex=LightsExperiment()
            c=dict(update=6,individual_decisions=255083,run_decisions=0,run_matches=0,run_updates=0)
            sha=save(folder/'origin.pt',p,opt,ex,c,rng=rng);good=dict(checkpoint='origin.pt',sha256=sha,counters=c)
            atomic_json(folder/'origin.json',good)
            atomic_json(folder/'latest.json',dict(checkpoint='missing.pt',sha256='0'*64,counters={**c,'update':7}))
            counters,pointer,history,rejected=recover(folder,[],p,opt,rng,ex)
            self.assertEqual(pointer['sha256'],sha);self.assertEqual(counters['run_decisions'],0);self.assertEqual(len(rejected),1)
            (folder/'latest.json').write_text('{truncated')
            counters,pointer,history,rejected=recover(folder,[],p,opt,rng,ex)
            self.assertEqual(pointer['sha256'],sha);self.assertEqual(rejected[0]['checkpoint'],'latest.json')

    def test_startup_configuration_failure_is_visible_and_never_successful(self):
        from patrick_jane.learning_run import train
        from patrick_jane.progress import atomic_json
        parent=Path('local-runs/patrick-jane')
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            folder=Path(temp);self.assertTrue(folder.resolve().is_relative_to(parent.resolve()))
            atomic_json(folder/'registration.json',dict(experiment={}))
            self.assertEqual(train(folder,additional_target=100,workers=1,device='cpu',wall_seconds=20),1)
            status=json.loads((folder/'status.json').read_text());completion=json.loads((folder/'completion-status.json').read_text())
            self.assertEqual(status['stage'],'failed');self.assertFalse(completion['success']);self.assertFalse((folder/'checkpoints').exists())

    def test_running_marker_precedes_collection_and_interrupt_is_truthful(self):
        from unittest.mock import patch
        from patrick_jane.learning_run import train
        parent=Path('local-runs/patrick-jane')
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            folder=Path(temp);self.assertTrue(folder.resolve().is_relative_to(parent.resolve()))
            class InterruptedPool:
                def __init__(self,*a):pass
                def __enter__(self):
                    current=json.loads((folder/'completion-status.json').read_text())
                    self_case.assertEqual(current['state'],'running');self_case.assertFalse(current['success'])
                    raise KeyboardInterrupt('test before collection')
                def __exit__(self,*a):pass
            self_case=self
            with patch('patrick_jane.learning_run.Pool',InterruptedPool):
                self.assertEqual(train(folder,additional_target=100,workers=1,device='cpu',wall_seconds=30,commitment_mode=True),130)
            self.assertEqual(json.loads((folder/'completion-status.json').read_text())['state'],'interrupted')


if __name__=='__main__':unittest.main()
