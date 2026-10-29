import io,json,tempfile,unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
import torch
from among_us_map import get_map
from navigation_service import NavigationPlanner
from social_deduction.actor import Role
from patrick_jane.config import ENGINEERING
from patrick_jane.lights_setup import LightsExperiment
from patrick_jane.environment import PatrickJaneGame
from patrick_jane.collector import collect,Pool
from patrick_jane.network import PatrickJane
from patrick_jane.ppo import prepare
from patrick_jane.progress import Progress
from patrick_jane.learning_report import summarize
from spectator.outcomes import outcome_label


class ConstantCritic(PatrickJane):
    def forward(self,*args,**kwargs):
        output,hidden=super().forward(*args,**kwargs);output['value']=torch.full_like(output['value'],5.)
        return output,hidden


class TechnicalTruncationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.planner=NavigationPlanner(get_map(),cache_size=128)

    def game(self,**changes):return PatrickJaneGame(ENGINEERING[0],planner=self.planner,experiment=replace(LightsExperiment(),**changes))

    def test_normal_lights_match_passes_old_900_second_boundary(self):
        g=self.game();g.tick=4500;g.step({})
        self.assertGreater(g.time,900);self.assertFalse(g.is_terminal);self.assertIsNone(g.truncation_pending)

    def test_real_victory_precedes_technical_limit(self):
        g=self.game(technical_max_seconds=.2)
        for p in g.players.values():
            if p.role is Role.IMPOSTOR:p.status='ejected'
        g.step({});self.assertTrue(g.is_terminal);self.assertIs(g.result.winner,Role.CREWMATE)
        self.assertEqual(g.result.reason,'IMPOSTOR_EJECTED');self.assertIsNone(g.truncation_details)

    def test_stationary_guard_is_explicit_and_nonwinning(self):
        g=self.game(technical_stall_seconds=.4);g.step({});g.step({});g.step({});g.step({})
        self.assertEqual(g.truncation_pending['cause'],'stationary_without_progress')
        g.finalize_truncation();self.assertIsNone(g.result.winner);self.assertEqual(g.result.reason,'TRUNCATED')
        self.assertEqual(g.metrics['technical_truncation']['cause'],'stationary_without_progress')

    def test_live_truncation_bootstraps_without_extra_decisions_or_rewards(self):
        ex=replace(LightsExperiment(),technical_max_seconds=2.4)
        with Pool(1,processes=False) as pool:r=collect(ConstantCritic(),pool,ex,[ENGINEERING[1]],wall_limit=30)
        m=r['matches'][0];self.assertTrue(m['truncated']);self.assertIsNone(m['winner']);self.assertEqual(m['reason'],'TRUNCATED')
        self.assertEqual(m['individual_decisions'],sum(len(t) for t in r['trajectories'].values()))
        self.assertGreater(len(m['bootstrap_values']),0);self.assertLessEqual(len(m['bootstrap_values']),10)
        for tr in r['trajectories'].values():
            last=tr[-1]
            if not last.truncated:
                self.assertTrue(last.done);self.assertEqual(last.next_value,0.);continue
            self.assertFalse(last.done);self.assertEqual(last.next_value,5.)
            target=prepare({0:tr},ex)[-1][3][-1]
            self.assertAlmostEqual(float(target),last.reward+ex.gamma**(last.duration/.6)*5.,places=5)
            terminal=replace(last,done=True,truncated=False,next_value=99.)
            self.assertAlmostEqual(float(prepare({0:[terminal]},ex)[0][3][-1]),terminal.reward,places=5)
        for components in r['reward_components'].values():
            self.assertEqual(components.get('win',0),0);self.assertEqual(components.get('loss',0),0)
        summary=summarize(r['matches']);self.assertEqual(summary['totals']['truncations'],1)
        self.assertEqual(summary['totals']['draws'],0);self.assertEqual(summary['totals']['legitimate_games'],0)
        self.assertIsNone(summary['win_rate'])

    def test_technical_limit_waits_for_ordinary_roaming_boundary(self):
        from social_deduction.actor import Phase
        g=self.game(technical_max_seconds=.2)
        crew=next(p for p in g.players.values() if p.role is Role.CREWMATE);g._start_meeting(crew)
        for _ in range(700):
            g.step({})
            if g.phase is Phase.ROAMING:break
            self.assertIsNone(g.truncation_pending)
        self.assertIs(g.phase,Phase.ROAMING);self.assertIsNotNone(g.truncation_pending)

    def test_replay_label_never_calls_truncation_a_victory_or_draw(self):
        label=outcome_label(dict(winner=None,reason='TRUNCATED',metrics={'technical_truncation':{'cause':'stationary_without_progress'}}))
        self.assertTrue(label.startswith('TRUNCATED'));self.assertNotIn('VICTORY',label);self.assertNotIn('DRAW',label)
        self.assertEqual(outcome_label(dict(winner='crewmate',reason='TASKS_COMPLETED')),'CREWMATE VICTORY')
        self.assertTrue(outcome_label(dict(winner=None,reason='TIMEOUT')).startswith('HISTORICAL TIMEOUT'))

    def test_progress_is_additional_actual_decisions_and_resumes(self):
        class Resources:
            def summary(self):return {}
        parent=Path('local-runs/patrick-jane');parent.mkdir(exist_ok=True,parents=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            folder=Path(temp);self.assertTrue(folder.resolve().is_relative_to(parent.resolve()))
            p=Progress(folder,32,Resources(),target=1000000);p.console=True;p.started-=3600
            p.counters=dict(individual_decisions=426000,complete_matches=61,update=12);p.stage='collecting'
            with redirect_stdout(io.StringIO()) as output:p.publish({'decisions':1000})
            s=json.loads((folder/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(s['progress_percent'],42.7);self.assertEqual(s['overall_decisions'],427000)
            self.assertIn('427,000 / 1,000,000',output.getvalue());self.assertIn('32 workers',output.getvalue());self.assertIn('12 PPO updates',output.getvalue())
            p.stage='validation-entry';p.publish({'decisions':999999})
            s=json.loads((folder/'status.json').read_text(encoding='utf-8'));self.assertEqual(s['overall_decisions'],426000)
            p.stage='checkpoint';p.counters['individual_decisions']=1000100;p.publish({})
            s=json.loads((folder/'status.json').read_text(encoding='utf-8'));self.assertEqual(s['progress_percent'],100.)
            self.assertEqual(s['training_state'],'running')

    def test_emergency_finalizer_does_not_double_count_previous_attempt_time(self):
        from unittest.mock import patch
        from tools.run_pj_learning import main
        parent=Path('local-runs/patrick-jane')
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            folder=Path(temp)
            (folder/'status.json').write_text(json.dumps(dict(elapsed=1000,active_seconds_before_attempt=400)),encoding='utf-8')
            with patch('sys.argv',['run_pj_learning','--folder',str(folder),'--finalize','interrupted']), \
                 patch('patrick_jane.learning_report.generate',return_value={'additional_decisions':10}) as generate, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(),1)
            self.assertEqual(generate.call_args.kwargs['active_seconds'],1000)
            completion=json.loads((folder/'completion-status.json').read_text(encoding='utf-8'))
            self.assertEqual(completion['active_seconds'],1000);self.assertFalse(completion['success'])

    def test_censored_validation_cannot_select_a_champion_as_nonwinning(self):
        from unittest.mock import patch
        from patrick_jane.learning_run import select_uncensored_champion
        with patch('patrick_jane.learning_run.select_champion',return_value=True) as select:
            self.assertFalse(select_uncensored_champion('checkpoint',[{'reason':'TRUNCATED'}],'champions'))
            select.assert_not_called()
            self.assertTrue(select_uncensored_champion('checkpoint',[{'reason':'PARITY','winner':'impostor'}],'champions'))
            select.assert_called_once()


if __name__=='__main__':unittest.main()
