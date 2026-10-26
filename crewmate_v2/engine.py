"""Opt-in mechanics; original geometry/action contracts and historical engine are unchanged."""
import math
from dataclasses import replace
from types import MappingProxyType
from among_us_map import get_map
from navigation_service import NavigationPlanner, NavStatus, NavTarget
from phase3_engine import Phase3Game
from social_deduction.actor import Ejection, Phase, Provenance, Role, Vote
from social_deduction.phase3_api import Intent, IntentKind, StructuredClaim, intent_allowed, legal_actions
from social_deduction.phase3_observation import visible_between
from .api import (ConfirmedEjection, OwnTaskDetail, TaskBar, V2Event, V2Observation,
                  VisualTaskProof, validate)
from .config import LobbySettings, MatchMixture
from .ghosts import GhostState, advance_ghost
from .tasks import assignments
from . import SCHEMA


class CrewmateV2Game(Phase3Game):
    def __init__(self,seed=0,*,lobby=None,mixture=None,planner=None,fast=True):
        self.lobby=lobby or LobbySettings()
        self.mixture=mixture or MatchMixture()
        if type(fast) is not bool:raise ValueError('Execution flag must be boolean')
        self.fast=fast
        planner=planner or NavigationPlanner(get_map(),cache_size=96)
        super().__init__(seed,self.lobby.engine_config(planner.map.radius),planner)

    def reset(self,seed=0):
        # Base reset's temporary legacy assignments are replaced before any actor can observe.
        self._v2_ready=False
        super().reset(seed)
        plans=assignments(self.map,seed,self.players)
        self.tasks={}
        for pid,p in self.players.items():
            p.tasks=list(plans[pid]) if p.role is Role.CREWMATE else []
            p.meetings_remaining=self.lobby.emergency_meetings
            self.tasks.update({t.task_id:t for t in p.tasks})
        self.initial_task_count=len(self.tasks)
        self._owners=MappingProxyType({t.task_id:t.owner_id for t in self.tasks.values()})
        self.ghosts={}
        self._ghost_goals={d.id:tuple(d.standing) for d in self.map.destinations}
        self.v2_publications=[]
        self.v2_direct={pid:[] for pid in self.players}
        self.deduction_only=self.mixture.select(seed)
        self._record('v2_configuration',lobby=self.lobby.to_dict(),deduction_only=self.deduction_only,
                     deduction_frequency=self.mixture.deduction_frequency)
        self._v2_ready=True
        self._invalidate();self.assert_invariants()
        return self

    def _reassign_publicly_absent_tasks(self):
        # One ownership model: no task ever changes owners, regardless of public absence.
        return

    def observe(self,player_id):
        actor=super().observe(player_id)
        if getattr(self,'_v2_ready',False):
            ready=tuple(tid for tid in actor.interactable_tasks if self.tasks[tid].ready_at<=self.time+1e-9)
            if ready!=actor.interactable_tasks:
                actor=replace(actor,interactable_tasks=ready)
                actor=replace(actor,actions=legal_actions(actor))
                self._observation_cache[player_id]=actor
        return actor

    def observe_v2(self,player_id):
        actor=self.observe(player_id)
        player=self.players[player_id]
        details=tuple(OwnTaskDetail(t.task_id,t.plan.task_type,t.plan.category,t.stage,len(t.plan.stages),
            float(max(0.,t.ready_at-self.time)),t.duration,t.plan.visual and self.lobby.visual_tasks) for t in sorted(player.tasks,key=lambda t:t.task_id))
        events=tuple(sorted((*self.v2_publications,*self.v2_direct[player_id]),key=lambda e:e.tick))
        return validate(V2Observation(SCHEMA,actor,TaskBar(sum(t.progress>=1 for t in self.tasks.values()),
            self.initial_task_count),events,details))

    def _make_ghost(self,player):
        if player.role is Role.CREWMATE and player.player_id not in self.ghosts:
            # No killer, event history, observed role, controller or agent memory is passed.
            self.ghosts[player.player_id]=GhostState(player.player_id,player.position,
                tuple(map(float,self.map.center_domain.bounds)),self.config.speed)

    def _eliminate(self,killer,victim):
        super()._eliminate(killer,victim)
        self._make_ghost(victim)

    def _vote(self,voter_id,target_id):
        if voter_id in self.votes:return
        self.votes[voter_id]=target_id;self.metrics['votes']+=1
        # During voting only voted_ids is public. Target/identity pairs are
        # non-anonymous RESULTS, delivered together at the resolution deadline.
        self._record('vote_submitted',voter_id=voter_id,target_id=target_id,meeting_id=self.meeting_id)
        self._invalidate()

    def _resolve_meeting(self):
        self._record('meeting_resolution',channel='public',meeting_id=self.meeting_id)
        for voter,target in sorted(self.votes.items()):
            self._publish(Vote(self.meeting_id,voter,target),'vote',meeting_id=self.meeting_id,
                          voter_id=voter,target_id=target)
        target=self.tally_votes(self.votes)
        if target is not None:
            player=self.players[target];self._interrupt(player,'ejected');player.status='ejected'
            self._make_ghost(player)
        self._publish(Ejection(self.meeting_id,target),'ejection',meeting_id=self.meeting_id,player_id=target)
        if target is not None and self.lobby.confirm_ejects:
            self.v2_publications.append(V2Event(self.tick,Provenance.PUBLIC,
                ConfirmedEjection(self.meeting_id,target,self.players[target].role)))
        if self._check_win():return
        self.phase=Phase.ROAMING;self.phase_started_tick=self.tick
        self._record('exploration_resumed',channel='public',meeting_id=self.meeting_id)
        self.meeting_id=None;self.participants=();self.votes={};self.claimed=set();self._invalidate()

    def _check_win(self):
        living=[p for p in self.players.values() if p.active]
        crew=sum(p.role is Role.CREWMATE for p in living)
        impostors=sum(p.role is Role.IMPOSTOR for p in living)
        if impostors==0:self._finish(Role.CREWMATE,'IMPOSTOR_EJECTED')
        elif impostors>=crew:self._finish(Role.IMPOSTOR,'PARITY')
        elif not self.deduction_only and self.tasks and all(t.progress>=1 for t in self.tasks.values()):
            self._finish(Role.CREWMATE,'TASKS_COMPLETED')
        return self.is_terminal

    def _completed(self,player,task,*,ghost=False):
        self.metrics['task_completions']+=1
        self._record('task_completed',player_id=player.player_id,task_id=task.task_id,console_id=task.console_id)
        if self.lobby.visual_tasks and task.plan.visual and player.active and not ghost:
            for observer in self.players.values():
                if observer.active and observer.player_id!=player.player_id and visible_between(self.map,
                        observer.position,player.position,self.config.effective_sight_range(observer.role)):
                    self.v2_direct[observer.player_id].append(V2Event(self.tick,Provenance.DIRECT,
                        VisualTaskProof(player.player_id,task.plan.task_type,task.console_id)))

    def _advance_tasks(self):
        for player in self.players.values():
            if not player.active or player.interaction_task is None:continue
            if player.interaction_task.startswith('fake:'):
                console=player.interaction_task[5:]
                if not self.planner.arrival_region(NavTarget.task(console)).contains(player.position):
                    self._interrupt(player,'left_interaction_region');continue
                player.interaction_remaining=max(0.,player.interaction_remaining-self.config.dt)
                if player.interaction_remaining<=1e-9:
                    self._record('fake_task_end',player_id=player.player_id,console_id=console)
                    player.interaction_task=None;player.intention='wait'
                continue
            task=self.tasks[player.interaction_task]
            if task.owner_id!=player.player_id:raise RuntimeError('Task ownership violation')
            if not self.planner.arrival_region(NavTarget.task(task.console_id)).contains(player.position):
                self._interrupt(player,'left_interaction_region');continue
            old_stage=task.stage
            if task.advance(self.config.dt,self.time):self._completed(player,task)
            if task.stage!=old_stage:
                # A completed stage requires a new legitimate interaction at the next console.
                player.interaction_task=None;player.intention='wait'

    def _advance_ghosts(self):
        for pid,state in sorted(self.ghosts.items()):
            player=self.players[pid]
            new,completed=advance_ghost(state,tuple(player.tasks),self._ghost_goals,self.config.dt,self.time)
            self.ghosts[pid]=new
            if completed is not None:self._completed(player,completed,ghost=True)

    def _apply_exploration(self,p,action,observation):
        if action.kind is IntentKind.INTERACT:
            task=self.tasks.get(action.target)
            if task is None or task.owner_id!=p.player_id or self.time+1e-9<task.ready_at:
                # Readiness is own task information, not hidden world state.
                return
        super()._apply_exploration(p,action,observation)

    def step(self,actions=None):
        """Same tick semantics in both paths; fast omits UNUSED validation projections."""
        if self.is_terminal:return self.result
        actions=actions or {};self.metrics['steps']+=1
        submitted={pid:a for pid,a in actions.items() if a is not None and not
                   (isinstance(a,Intent) and a.kind is IntentKind.WAIT)}
        ids=sorted(pid for pid in submitted if pid in self.players) if self.fast else sorted(self.players)
        observations={pid:self.observe(pid) for pid in ids}
        valid={}
        for pid,action in sorted(submitted.items()):
            if pid not in observations or not intent_allowed(observations[pid],action):
                self.metrics['illegal_actions']+=1;self._record('illegal_action',player_id=pid)
            else:valid[pid]=action
        if self.phase is Phase.ROAMING:
            for pid,action in valid.items():
                if action.kind is IntentKind.REPORT:
                    self._start_meeting(self.players[pid],self.bodies[action.target]);break
                if action.kind is IntentKind.CALL_MEETING:
                    self._start_meeting(self.players[pid]);break
            if self.phase is Phase.ROAMING:
                for pid,action in valid.items():
                    if action.kind is IntentKind.KILL:
                        self._eliminate(self.players[pid],self.players[action.target])
                        if self.is_terminal:return self.result
                for pid,action in valid.items():
                    if self.players[pid].active:self._apply_exploration(self.players[pid],action,observations[pid])
        elif self.phase is Phase.DISCUSSION:
            for pid,action in valid.items():
                if action.kind is IntentKind.CLAIM:
                    draft=action.claim
                    claim=StructuredClaim(draft.kind,pid,draft.subject_id,draft.region,self.tick,draft.asserted_tick)
                    self.claimed.add(pid)
                    self._publish(claim,'claim',claim_kind=draft.kind,speaker_id=pid,
                        subject_id=draft.subject_id,region=draft.region,asserted_tick=draft.asserted_tick)
        elif self.phase is Phase.VOTING:
            for pid,action in valid.items():
                if action.kind in (IntentKind.VOTE,IntentKind.SKIP):
                    self._vote(pid,action.target if action.kind is IntentKind.VOTE else None)
        if self.phase is Phase.ROAMING:self._advance_physics()
        self.tick+=1;self._invalidate()
        if self.phase is Phase.ROAMING:self._advance_tasks()
        self._advance_ghosts()
        if not self.is_terminal and self.phase is Phase.DISCUSSION:
            if self.tick-self.phase_started_tick>=math.ceil(self.config.discussion_seconds/self.config.dt):
                self.phase=Phase.VOTING;self.phase_started_tick=self.tick
                self._record('voting_started',channel='public',meeting_id=self.meeting_id)
        elif not self.is_terminal and self.phase is Phase.VOTING:
            if self.tick-self.phase_started_tick>=math.ceil(self.config.voting_seconds/self.config.dt):
                for pid in self.participants:
                    if pid not in self.votes:self._vote(pid,None)
                self._resolve_meeting()
        # Resolve an ejection at its deadline before checking simultaneous task
        # completion, preserving ejection/parity priority over task victory.
        if not self.is_terminal:self._check_win()
        if not self.is_terminal and self.tick>=math.ceil(self.config.timeout/self.config.dt):self._finish(None,'TIMEOUT')
        self._invalidate()
        if self.config.check_invariants:self.assert_invariants()
        return self.result

    def assert_invariants(self):
        super().assert_invariants()
        if not getattr(self,'_v2_ready',False):return
        assert all(t.owner_id==p.player_id and self._owners[t.task_id]==p.player_id for p in self.players.values() for t in p.tasks)
        assert all(not self.players[pid].active and self.players[pid].role is Role.CREWMATE for pid in self.ghosts)
        assert all(g.bounds[0]<=g.position[0]<=g.bounds[2] and g.bounds[1]<=g.position[1]<=g.bounds[3] for g in self.ghosts.values())
