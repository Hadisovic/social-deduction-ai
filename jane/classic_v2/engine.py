"""Versioned shared engine extension, leaving the pinned upstream source intact."""
from collections import Counter
from dataclasses import dataclass,replace
from types import MappingProxyType
from threading import RLock
from .tasks import assignments,family
from . import GEOMETRY_REVISION

_FACTORY_LOCK=RLock()


def experiment(profile):
    from patrick_jane.a2_config import A2Experiment
    from patrick_jane.config import ClassicLobby,Population
    from jane.shared.patrick_jane import reference_review
    import json
    from pathlib import Path
    s=profile.settings;counts=profile.effective_counts

    @dataclass(frozen=True)
    class FlexiblePopulation(Population):
        def __post_init__(self):
            minimum={1:4,2:7,3:9}.get(self.impostors,99)
            if (type(self.crew) is not int or type(self.impostors) is not int
                    or not minimum<=self.total<=15 or self.crew<=self.impostors):
                raise ValueError('Illegal Classic population')

    @dataclass(frozen=True)
    class FlexibleLobby(ClassicLobby):
        def __post_init__(self):
            # All host values have already passed shared immutable validation.
            if self.to_dict()!=expected_lobby:raise ValueError('Lobby/profile mismatch')
        def engine_config(self,radius):
            from phase3_engine import GameConfig
            return GameConfig(dt=self.dt,decision_interval=self.decision_interval,speed=2.5*self.player_speed,
                tasks_per_crew=sum(counts),task_seconds=4.,fake_task_seconds=4.,
                initial_kill_cooldown=self.initial_kill_cooldown,kill_cooldown=self.kill_cooldown,
                kill_range=2*radius*self.kill_distance_character_widths,
                discussion_seconds=max(self.discussion_seconds,self.dt),voting_seconds=self.voting_seconds,
                timeout=self.timeout,rules_version=6,crewmate_sight_range=4.5*self.crewmate_vision,
                impostor_sight_range=4.5*self.impostor_vision)

    @dataclass(frozen=True)
    class ClassicExperiment(A2Experiment):
        host_profile: dict=None

    path=Path(__file__).resolve().parents[1]/'shared/upstream/patrick_jane_a2_default.json'
    values=json.loads(path.read_text(encoding='utf-8'))
    expected_lobby={**values['lobby'],'schema':'jane-classic-lobby-v2',
        'player_speed':s.lobby.player_speed,'confirm_ejects':s.lobby.confirm_ejects,
        'anonymous_votes':s.meetings.anonymous_votes,'task_bar_updates':s.tasks.task_bar_updates,
        'emergency_meetings':s.meetings.emergency_meetings,'emergency_cooldown':float(s.meetings.emergency_cooldown_seconds),
        'discussion_seconds':float(s.meetings.discussion_seconds),'voting_seconds':float(s.meetings.voting_seconds),
        'crewmate_vision':s.lobby.crewmate_vision,'impostor_vision':s.lobby.impostor_vision,
        'common_tasks':s.tasks.common,'long_tasks':s.tasks.long,'short_tasks':s.tasks.short,
        'visual_tasks':s.tasks.visual_tasks,'kill_cooldown':s.lobby.kill_cooldown_seconds,
        'dt':s.simulation.physics_step_seconds,'decision_interval':s.simulation.observation_interval_seconds}
    from patrick_jane.config import SabotageSettings
    values.update(population=FlexiblePopulation(s.crewmates,s.lobby.impostors),lobby=FlexibleLobby(**expected_lobby),
        sabotage=SabotageSettings(**values['sabotage']),allowed_sabotages=tuple(values['allowed_sabotages']),
        technical_max_seconds=s.simulation.match_cap_seconds,technical_stall_seconds=s.simulation.stall_cap_seconds,
        host_profile=profile.to_dict())
    return ClassicExperiment(**values)


def game_class():
    from patrick_jane.a2_environment import A2Game
    from social_deduction.actor import Phase,Role,Vote
    class ClassicGame(A2Game):
        def reset(self,seed=0):
            self._installing=True
            super().reset(seed)
            from .settings import HostProfile
            self.host_profile=HostProfile.from_dict(self.experiment.host_profile)
            plans=assignments(self.map,seed,self.players,self.host_profile.effective_counts)
            self.hypothetical_tasks=plans;self.tasks={}
            for pid,p in self.players.items():
                p.tasks=list(plans[pid]) if p.role is Role.CREWMATE else []
                self.tasks.update({t.task_id:t for t in p.tasks})
            self.initial_task_count=len(self.tasks)
            self._owners=MappingProxyType({t.task_id:t.owner_id for t in self.tasks.values()})
            self._taskbar_visible=(0,self.initial_task_count)
            self._installing=False;self._invalidate();self.assert_invariants()
            self._record('classic_v2_configuration',channel='spectator',profile=self.experiment.host_profile,
                effective_task_counts=self.host_profile.effective_counts,geometry_revision=GEOMETRY_REVISION)
            return self

        def _bind_projection_geometry(self):
            """The selected planner owns the trusted collision/projection map.

            Upstream has independent geometry/planner LRU caches. Equal door
            revisions can therefore refer to different map instances. Repair
            that identity without weakening the projection guard or changing
            any actor's route knowledge, destinations or strategy.
            """
            canonical=self.planner.map
            if canonical.closed_doors!=self.map.closed_doors:
                raise ValueError('Door revision mismatch between geometry and planner')
            changed=self.map is not canonical
            self.map=canonical
            for player in self.players.values():
                if player.motion.map is not canonical:
                    player.motion.map=canonical;changed=True
            if changed:self._invalidate()

        def _update_doors(self):
            super()._update_doors()
            self._bind_projection_geometry()

        def assert_invariants(self):
            if not getattr(self,'_v2_ready',False):return
            assert len(self.players)==self.experiment.population.total
            assert sum(p.role is Role.IMPOSTOR for p in self.players.values())==self.experiment.population.impostors
            assigned=[t.task_id for p in self.players.values() for t in p.tasks]
            assert len(assigned)==len(set(assigned))==self.initial_task_count==len(self.tasks)
            assert all(t.owner_id==p.player_id and self._owners[t.task_id]==p.player_id for p in self.players.values() for t in p.tasks)
            assert all(0<=t.progress<=1 for t in self.tasks.values())
            base=getattr(self,'base_map',self.map);underground=getattr(self,'vented',{})
            assert all((base if not p.active or pid in underground else self.map).contains(p.position)
                       for pid,p in self.players.items())
            assert all(p.status in ('alive','dead','ejected') for p in self.players.values())
            assert all(not self.players[pid].active and self.players[pid].role is Role.CREWMATE for pid in self.ghosts)
            assert set(self.votes)<=set(self.participants)
            assert all(target is None or target in self.participants for target in self.votes.values())
            assert self.is_terminal==(self.phase is Phase.FINISHED)
            if not getattr(self,'_installing',True):
                assert all(len(p.tasks)==sum(self.host_profile.effective_counts) for p in self.players.values() if p.role is Role.CREWMATE)
                assert all(len({family(t.plan.task_type) for t in p.tasks})==len(p.tasks) for p in self.players.values())

        def _start_meeting(self,reporter,body=None):
            result=super()._start_meeting(reporter,body)
            self._bind_projection_geometry()
            self._taskbar_visible=(sum(t.progress>=1 for t in self.tasks.values()),self.initial_task_count)
            if self.lobby.discussion_seconds==0:
                self.phase=Phase.VOTING;self.phase_started_tick=self.tick
                self._record('voting_started',channel='public',meeting_id=self.meeting_id);self._invalidate()
            return result

        def _publish(self,payload,kind,**data):
            if self.lobby.anonymous_votes and type(payload) is Vote:
                # Target counts are public after resolution; voter identity is not.
                self._record('anonymous_vote',channel='public',meeting_id=payload.meeting_id,target_id=payload.target_id)
                return
            return super()._publish(payload,kind,**data)

        def observe_pj(self,pid):
            obs=super().observe_pj(pid)
            mode=self.lobby.task_bar_updates
            if mode=='never':return replace(obs,task_bar=(0,0))
            if mode=='meetings':return replace(obs,task_bar=self._taskbar_visible)
            return obs

    return ClassicGame


def make_match(seed,config,**kwargs):
    """Reuse the native coordinator with a scoped constructor dependency injection.

    Each simulation process owns this binding. Restore even if construction fails;
    no source file, global NN policy or frozen runtime is modified.
    """
    from patrick_jane.worker import Match
    from patrick_jane import a2_environment
    with _FACTORY_LOCK:
        original=a2_environment.A2Game
        cls=game_class()
        try:
            a2_environment.A2Game=cls
            return Match(seed,config,**kwargs)
        finally:a2_environment.A2Game=original
