import unittest
from dataclasses import replace,asdict
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import Intent,IntentKind
from navigation_service import NavigationPlanner
from among_us_map import get_map
from patrick_jane.config import Experiment,Population
from patrick_jane.environment import PatrickJaneGame
from patrick_jane.api import Action,Event
from patrick_jane.rewards import RewardLedger,COEFFICIENTS


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.planner=NavigationPlanner(get_map(),cache_size=128)
    def game(self,pop=None):return PatrickJaneGame(2920000,planner=self.planner,experiment=Experiment(population=pop or Population()))

    def test_populations_and_owned_tasks(self):
        for pop in (Population(4,1),Population(),Population(12,3),Population(6,2)):
            g=self.game(pop);g.assert_invariants()
            self.assertEqual(len(g.players),pop.total);self.assertEqual(len(g.tasks),4*pop.crew)
            self.assertFalse(g.deduction_only)

    def test_population_rejects_invalid(self):
        for values in ((3,3),(13,2),(12,4),(0,1),(4,0)):
            with self.assertRaises(ValueError):Population(*values)

    def test_hidden_role_swap_does_not_change_crew_packet(self):
        g=self.game();crew=[p for p in g.players.values() if p.role is Role.CREWMATE];imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        before=asdict(g.observe_pj(crew[0].player_id))
        crew[1].role,imp.role=imp.role,crew[1].role;g._invalidate()
        self.assertEqual(before,asdict(g.observe_pj(crew[0].player_id)))

    def test_lights_dynamic_role_vision(self):
        g=self.game();g._begin_sabotage('lights')
        self.assertAlmostEqual(g.settings_for(Role.CREWMATE).sight_range,.9)
        self.assertAlmostEqual(g.settings_for(Role.IMPOSTOR).sight_range,6.75)
        crew=next(p for p in g.players.values() if p.role is Role.CREWMATE)
        target=g.utilities['lights'][0];crew.motion.spawn(g.planner.destinations[target].standing);g._invalidate()
        g.step({crew.player_id:Action('repair',target)})
        self.assertIsNone(g.sabotage);self.assertAlmostEqual(g.settings_for(Role.CREWMATE).sight_range,4.5)

    def test_oxygen_two_independent_repairs_and_unique_reward(self):
        g=self.game();ledger=RewardLedger(g);g._begin_sabotage('oxygen')
        crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
        for p,key in zip(crew,g.utilities['oxygen']):p.motion.spawn(g.planner.destinations[key].standing)
        g._invalidate();g.step({crew[0].player_id:Action('repair',g.utilities['oxygen'][0])})
        self.assertEqual(g.sabotage,'oxygen');g.step({crew[1].player_id:Action('repair',g.utilities['oxygen'][1])})
        rewards=ledger.collect(g)
        self.assertEqual(rewards[crew[0].player_id]['repair'],3.);self.assertEqual(rewards[crew[1].player_id]['repair'],3.)
        self.assertTrue(all(not x for x in ledger.collect(g).values()))

    def test_reactor_requires_simultaneous_distinct_players(self):
        g=self.game();g._begin_sabotage('reactor');crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
        keys=g.utilities['reactor']
        for p,key in zip(crew,keys):p.motion.spawn(g.planner.destinations[key].standing)
        g._invalidate();g.step({crew[0].player_id:Action('repair',keys[0])});g.step({crew[1].player_id:Action('repair',keys[1])})
        self.assertEqual(g.sabotage,'reactor')
        g.step({p.player_id:Action('repair',key) for p,key in zip(crew,keys)})
        self.assertIsNone(g.sabotage)

    def test_critical_failure_and_last_tick_repair_order(self):
        for kind in ('oxygen','reactor'):
            g=self.game();g._begin_sabotage(kind);g.sabotage_deadline=g.time+g.config.dt
            g.step({});self.assertEqual(g.result.reason,kind.upper()+'_FAILURE')
        g=self.game();g._begin_sabotage('lights');crew=next(p for p in g.players.values() if p.role is Role.CREWMATE)
        key=g.utilities['lights'][0];crew.motion.spawn(g.planner.destinations[key].standing);g._invalidate()
        g.sabotage_deadline=g.time+g.config.dt;g.step({crew.player_id:Action('repair',key)})
        self.assertFalse(g.is_terminal)
        g=self.game();g._begin_sabotage('oxygen');crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
        for p,key in zip(crew,g.utilities['oxygen']):p.motion.spawn(g.planner.destinations[key].standing)
        g._invalidate();g.sabotage_deadline=g.time+g.config.dt
        g.step({p.player_id:Action('repair',key) for p,key in zip(crew,g.utilities['oxygen'])})
        self.assertIsNone(g.sabotage);self.assertFalse(g.is_terminal)

    def test_structured_outputs_are_public_claims_and_timestamp_rounding_safe(self):
        g=self.game();g._start_meeting(next(iter(g.players.values())));g.tick=58;g.phase_started_tick=58;g._invalidate()
        pid=g.context.speaker_id
        for kind in ('share_evidence','accuse','defend','question','strategic_claim'):
            e=Event(kind,subject=pid,room='Electrical',observed=58*.2,delivered=58*.2,source=pid)
            obs=g.observe_pj(pid);self.assertTrue(g._extension_allowed(pid,Action('communication',statement=e),obs))
        g.step({pid:Action('communication',statement=e)})
        for other,p in g.players.items():
            if p.active:
                delivered=g.observe_pj(other).events[-1]
                self.assertEqual(delivered.provenance,'claim');self.assertEqual(delivered.kind,'strategic_claim')

    def test_delayed_team_credit_includes_dead_actors(self):
        g=self.game();ledger=RewardLedger(g);crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
        imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g._eliminate(imp,crew[0])
        ledger.collect(g);g._finish(Role.CREWMATE,'fixture');rewards=ledger.collect(g)
        self.assertEqual(rewards[crew[0].player_id]['win'],10.)
        self.assertTrue(all(r['win']==10. for r in rewards.values()))

    def test_meeting_cancels_critical_but_not_lights(self):
        for kind in ('oxygen','reactor','lights'):
            g=self.game();g._begin_sabotage(kind);p=next(iter(g.players.values()))
            g._start_meeting(p)
            self.assertEqual(g.sabotage,'lights' if kind=='lights' else None)

    def test_impostors_cannot_kill_teammates(self):
        g=self.game();imps=[p for p in g.players.values() if p.role is Role.IMPOSTOR]
        for p in imps:p.cooldown=0.
        g.step({imps[0].player_id:Intent(IntentKind.KILL,imps[1].player_id)})
        self.assertTrue(imps[1].active)
        self.assertNotIn(imps[1].player_id,[a.target for a in g.observe(imps[0].player_id).actions if a.kind is IntentKind.KILL])

    def test_ghost_executor_retains_ownership_and_freezes_meetings(self):
        g=self.game();crew=next(p for p in g.players.values() if p.role is Role.CREWMATE);imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
        tasks=tuple(t.task_id for t in crew.tasks);g._eliminate(imp,crew)
        g.step({});self.assertEqual(tuple(t.task_id for t in crew.tasks),tasks)
        reporter=next(p for p in g.players.values() if p.active);g._start_meeting(reporter)
        position=g.ghosts[crew.player_id].position;progress=[t.progress for t in crew.tasks]
        g.step({});self.assertEqual(position,g.ghosts[crew.player_id].position);self.assertEqual(progress,[t.progress for t in crew.tasks])

    def test_vote_targets_hidden_until_results_and_ejection_rewards(self):
        for role,expected in ((Role.CREWMATE,-2.),(Role.IMPOSTOR,4.)):
            g=self.game();ledger=RewardLedger(g);g._start_meeting(next(iter(g.players.values())))
            g.phase=Phase.VOTING;g._invalidate();target=next(p.player_id for p in g.players.values() if p.role is role)
            pid=g.participants[0];g.step({pid:Intent(IntentKind.VOTE,target)})
            self.assertFalse(any(type(e.payload).__name__=='Vote' for e in g.observe(pid).evidence))
            for voter in g.participants:
                if voter!=pid:g._vote(voter,target)
            g._resolve_meeting();rewards=ledger.collect(g)
            key='innocent_ejection' if role is Role.CREWMATE else 'impostor_ejection'
            self.assertTrue(all(r[key]==expected for r in rewards.values()))

    def test_task_report_ghost_rewards_unique_and_no_vote_reward(self):
        g=self.game();ledger=RewardLedger(g);crew=next(p for p in g.players.values() if p.role is Role.CREWMATE)
        for kind,key in (('task_completed','task'),('report','report')):
            data=dict(player_id=crew.player_id,task_id='one',ghost=True) if kind=='task_completed' else dict(reporter_id=crew.player_id,victim_id='body')
            g._record(kind,**data);g._record(kind,**data)
        r=ledger.collect(g)[crew.player_id];self.assertEqual(r['ghost_task'],1.);self.assertNotIn('task',r);self.assertEqual(r['report'],1.1)
        self.assertEqual(COEFFICIENTS['vote'],0.)


if __name__=='__main__':unittest.main()
