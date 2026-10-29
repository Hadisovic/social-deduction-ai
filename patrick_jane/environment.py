"""Population/sabotage extension using unchanged V2 task, ghost and map rules."""
from dataclasses import replace
import math
from random import Random
from types import MappingProxyType
from collections import Counter,defaultdict
from among_us_map import get_map
from among_us_map_simulation import AmongUsMapEnv
from navigation_service import NavigationPlanner, NavigationService, ControllerConfig, NavStatus, NavTarget
from phase3_engine import Phase3Game, Player, Body, stream_seed
from social_deduction.actor import Role, Phase, Point, Provenance
from social_deduction.truth import assign_identities, assign_roles
from social_deduction.phase3_api import Intent, IntentKind, EventView, WitnessedElimination, intent_allowed
from social_deduction.phase3_observation import visible_between
from crewmate_v2.engine import CrewmateV2Game
from crewmate_v2.tasks import assignments
from .config import Experiment
from .api import Action, Observation, Event


class PatrickJaneGame(CrewmateV2Game):
    def __init__(self, seed=0, *, experiment=None, planner=None):
        from .navigation import PatrickPlanner
        self.experiment = experiment or Experiment()
        self.lobby = self.experiment.lobby
        self.fast = True
        planner = planner or PatrickPlanner(get_map(), cache_size=128)
        Phase3Game.__init__(self, seed, self.lobby.engine_config(planner.map.radius), planner)

    def reset(self, seed=0):
        self._v2_ready = False
        self.seed = int(seed); self.tick = 0; self.phase = Phase.ROAMING
        self.phase_started_tick = 0; self.result = None
        self.players = {}; self.bodies = {}; self.tasks = {}
        self.learning_diagnostics=Counter();self.lights_actor_seconds=defaultdict(float)
        self.truncation_pending=None;self.truncation_details=None;self._safety_progress_time=0.;self._safety_progress_signature=None
        self.publications = []; self.event_log = []; self._observation_cache = {}; self._snapshot_cache = None
        self._meeting_number = 0; self.meeting_id = None; self.participants = ()
        self.votes = {}; self.claimed = set()
        self.metrics = {k:0 for k in ('steps','eliminations','reports','meetings','votes','task_completions',
                                     'illegal_actions','navigation_failures','sabotages','repairs')}
        self.metrics.update({k:0. for k in ('navigation_seconds','visibility_seconds','event_log_seconds','physics_seconds')})
        n = self.experiment.population.total
        identities = assign_identities(n, seed=stream_seed(seed,'identities'))
        roles = assign_roles(n, seed=stream_seed(seed,'roles'), impostors=self.experiment.population.impostors)
        colors = ['Red','Blue','Green','Yellow','Purple','Orange','Black','Pink','White','Brown',
                  'Cyan','Lime','Maroon','Rose','Banana']
        Random(stream_seed(seed,'colors')).shuffle(colors)
        self._record('match_start',seed=seed,config=self.experiment.to_dict(),game_schema='patrick-jane-v1')
        for slot,(identity,role,color) in enumerate(zip(identities,roles,colors)):
            position = self.map.nearest_node((-2.1+(slot%5)*.7, -2.8-(slot//5)*.7))
            motion = AmongUsMapEnv(speed=self.config.speed); motion.map = self.map; motion.spawn(position)
            player = Player(identity.player_id,color,role,motion,
                NavigationService(self.planner,ControllerConfig(speed=self.config.speed)),
                cooldown=self.config.initial_kill_cooldown if role is Role.IMPOSTOR else 0.,
                meetings_remaining=self.lobby.emergency_meetings)
            self.players[player.player_id] = player
            self._record('spawn',player_id=player.player_id,color=color,role=role,position=position,internal_slot=slot)
        self.players = dict(sorted(self.players.items()))
        self.direct_events = {pid:[] for pid in self.players}; self._seen = {pid:set() for pid in self.players}
        plans = assignments(self.map,seed,self.players)
        for pid,p in self.players.items():
            p.tasks = list(plans[pid]) if p.role is Role.CREWMATE else []
            self.tasks.update({t.task_id:t for t in p.tasks})
        self.initial_task_count = len(self.tasks)
        self._owners = MappingProxyType({t.task_id:t.owner_id for t in self.tasks.values()})
        self.ghosts = {}; self._ghost_goals = {d.id:tuple(d.standing) for d in self.map.destinations}
        self.v2_publications = []; self.v2_direct = {pid:[] for pid in self.players}; self.deduction_only = False
        self.pj_public = []; self.sabotage = None; self.sabotage_deadline = None
        self.sabotage_ready = self.experiment.sabotage.initial_cooldown
        self.emergency_ready = 0.  # Retail emergency cooldown starts after meetings.
        self.repair_holds = {}
        self.repaired = set(); self.repair_contributors = set(); self.sabotage_number = 0
        utilities = list(self.map.destinations)
        self.utilities = {
            'lights':tuple(d.id for d in utilities if d.name == 'SwitchConsole'),
            'oxygen':tuple(d.id for d in utilities if d.name == 'NoOxyConsole'),
            'reactor':tuple(d.id for d in utilities if d.name in ('UpperHandConsole','LowerHandConsole'))}
        if any(not v for v in self.utilities.values()): raise ValueError('Required real Skeld utilities absent')
        if len(self.utilities['oxygen']) != 2 or len(self.utilities['reactor']) != 2:
            raise ValueError('Expected two oxygen and reactor utilities')
        self._v2_ready = True; self._invalidate(); self.assert_invariants()
        return self

    def settings_for(self, role):
        settings = super().settings_for(role)
        if role is Role.CREWMATE and self.sabotage == 'lights':
            settings = replace(settings,sight_range=settings.sight_range*self.experiment.sabotage.lights_factor)
        return settings

    def observe(self, player_id):
        obs = super().observe(player_id)
        teammates = {p.player_id for p in self.players.values() if p.role is Role.IMPOSTOR}
        actions = tuple(a for a in obs.actions if not (
            a.kind is IntentKind.KILL and a.target in teammates or
            a.kind is IntentKind.CALL_MEETING and (self.sabotage is not None or self.time < self.emergency_ready)))
        return replace(obs,actions=actions)

    def observe_pj(self, player_id):
        obs = self.observe_v2(player_id)
        actor = obs.actor
        active = actor.own.active
        teammates = tuple(p.player_id for p in self.players.values() if p.role is Role.IMPOSTOR and p.player_id != player_id)
        public = list(self.pj_public)
        # These are already authorized V2 public or personally witnessed events.
        for ev in obs.events:
            p = ev.payload
            if type(p).__name__ == 'ConfirmedEjection':
                public.append(Event('confirmed_role',p.player_id,observed=ev.tick*self.config.dt,
                    delivered=ev.tick*self.config.dt,detail=p.role.value))
            elif type(p).__name__ == 'VisualTaskProof':
                public.append(Event('visual_task',p.player_id,observed=ev.tick*self.config.dt,
                    delivered=ev.tick*self.config.dt,source=player_id,provenance='direct',detail=p.task_type))
        return Observation(actor,obs.own_tasks,(obs.task_bar.completed,obs.task_bar.total),tuple(public),
            self.sabotage,max(0.,self.sabotage_deadline-self.time) if self.sabotage_deadline is not None else None,
            self.utilities.get(self.sabotage,()),tuple(sorted(self.repaired)),
            active and actor.own.role is Role.IMPOSTOR and self.sabotage is None and self.time>=self.sabotage_ready
                and bool(getattr(self.experiment,'allowed_sabotages',('lights','oxygen','reactor'))),
            teammates if actor.own.role is Role.IMPOSTOR else (),
            tuple(self.players[player_id].navigator.route[getattr(self.players[player_id].navigator,'_waypoint',0):]),
            max(0.,self.emergency_ready-self.time),self.repair_holds.get(player_id))

    def _eliminate(self,killer,victim):
        # Preserve V2 witness semantics with dynamic, role-local lights visibility.
        for observer in self.players.values():
            sight = self.settings_for(observer.role).sight_range
            if observer.active and visible_between(self.map,observer.position,killer.position,sight) and visible_between(self.map,observer.position,victim.position,sight):
                self.direct_events[observer.player_id].append(EventView(self.tick,Provenance.DIRECT,
                    WitnessedElimination(killer.player_id,victim.player_id,Point(*victim.position),self.map.region(victim.position))))
                self._record('elimination_witnessed',channel='direct',observer_id=observer.player_id,
                             killer_id=killer.player_id,victim_id=victim.player_id)
        self._interrupt(victim,'eliminated'); victim.status='dead'
        self.bodies[victim.player_id] = Body(victim.player_id,victim.position,self.tick,killer.player_id)
        self._interrupt(killer,'kill_lunge'); killer.motion.spawn(victim.position)
        killer.cooldown=self.config.kill_cooldown; self.metrics['eliminations']+=1
        self._record('elimination',killer_id=killer.player_id,victim_id=victim.player_id,position=victim.position)
        self._make_ghost(victim); self._invalidate(); self._check_win()

    def _completed(self,player,task,*,ghost=False):
        from crewmate_v2.api import V2Event,VisualTaskProof
        self.metrics['task_completions']+=1
        self._record('task_completed',player_id=player.player_id,task_id=task.task_id,
                     console_id=task.console_id,ghost=ghost)
        if self.lobby.visual_tasks and task.plan.visual and player.active and not ghost:
            for observer in self.players.values():
                if observer.active and observer.player_id!=player.player_id and visible_between(self.map,
                        observer.position,player.position,self.settings_for(observer.role).sight_range):
                    self.v2_direct[observer.player_id].append(V2Event(self.tick,Provenance.DIRECT,
                        VisualTaskProof(player.player_id,task.plan.task_type,task.console_id)))

    def _start_meeting(self,reporter,body=None):
        self.repair_holds.clear()
        if self.sabotage in ('oxygen','reactor'):
            self._record('sabotage_cancelled',channel='public',sabotage=self.sabotage)
            self.sabotage=None; self.sabotage_deadline=None
            self.sabotage_ready=self.time+self.experiment.sabotage.cooldown
            self.repaired.clear(); self.repair_contributors.clear()
        return super()._start_meeting(reporter,body)

    def _interrupt(self,player,reason):
        if hasattr(self,'repair_holds'):self.repair_holds.pop(player.player_id,None)
        return super()._interrupt(player,reason)

    def _resolve_meeting(self):
        super()._resolve_meeting()
        if self.phase is Phase.ROAMING:
            self.emergency_ready=self.time+self.lobby.emergency_cooldown
            for slot,player in enumerate(self.players.values()):
                if not player.active:continue
                self._interrupt(player,'meeting_respawn')
                player.motion.spawn(self.map.nearest_node((-2.1+(slot%5)*.7,-2.8-(slot//5)*.7)))
                if player.role is Role.IMPOSTOR:player.cooldown=self.config.kill_cooldown
            self._record('meeting_respawn',channel='public',emergency_ready=self.emergency_ready)
            self._invalidate()

    def _extension_allowed(self,pid,action,obs):
        if type(action) is not Action or not obs.actor.own.active: return False
        base=obs.actor
        if action.kind=='sabotage': return (obs.can_sabotage and action.target in self.utilities
            and action.target in getattr(self.experiment,'allowed_sabotages',('lights','oxygen','reactor')))
        if action.kind=='repair':
            return (base.context.phase is Phase.ROAMING and action.target in obs.repair_locations
                    and action.target not in obs.repaired_locations and action.target in base.interactable_destinations)
        if action.kind=='communication':
            e=action.statement
            return (base.context.phase is Phase.DISCUSSION and base.context.speaker_id==pid and e is not None
                and e.kind in ('share_evidence','accuse','defend','question','strategic_claim')
                and e.subject in ('',*(i.player_id for i in base.roster))
                and e.origin_source in ('',*(i.player_id for i in base.roster))
                and e.room in ('','Hallway',*base.map.rooms) and math.isfinite(e.observed) and 0<=e.observed<=self.time+1e-8)
        return False

    def _begin_sabotage(self,kind):
        if kind not in getattr(self.experiment,'allowed_sabotages',('lights','oxygen','reactor')):
            raise ValueError('Sabotage disabled by this experiment profile')
        self.repair_holds.clear()
        self.sabotage=kind; self.sabotage_number+=1; self.repaired.clear(); self.repair_contributors.clear()
        seconds=getattr(self.experiment.sabotage,kind+'_seconds',None)
        self.sabotage_deadline=self.time+seconds if seconds is not None else None
        self.metrics['sabotages']+=1
        self.pj_public.append(Event('sabotage',observed=self.time,delivered=self.time,detail=kind))
        self._record('sabotage_started',channel='public',sabotage=kind,instance=self.sabotage_number,
                     deadline=self.sabotage_deadline)
        self._invalidate()

    def _repair(self,repairs):
        if not self.sabotage: return
        legal={pid:a.target for pid,a in repairs.items() if self.players[pid].active}
        if self.sabotage=='reactor':
            # Holding the interface persists until another intention or leaving it.
            for pid,target in legal.items():
                self._interrupt(self.players[pid],'reactor_hold')
                self.players[pid].intention='repair:'+target
                self.players[pid].navigator.status=NavStatus.SUCCESS
                self.repair_holds[pid]=target
            self.repair_holds={pid:target for pid,target in self.repair_holds.items()
                if self.players[pid].active and self.planner.arrival_region(NavTarget.task(target)).contains(self.players[pid].position)}
            legal=self.repair_holds
            complete=set(legal.values())==set(self.utilities['reactor']) and len(legal)>=2
            if complete:self.repair_contributors.update(legal)
        else:
            self.repaired.update(legal.values()); self.repair_contributors.update(legal)
            complete=set(self.utilities[self.sabotage])<=self.repaired
        if complete:
            self._record('sabotage_repaired',channel='public',sabotage=self.sabotage,instance=self.sabotage_number,
                         participants=sorted(self.repair_contributors))
            self.pj_public.append(Event('sabotage_repaired',observed=self.time,delivered=self.time,detail=self.sabotage))
            self.metrics['repairs']+=1; self.sabotage=None; self.sabotage_deadline=None
            self.sabotage_ready=self.time+self.experiment.sabotage.cooldown
            self.repaired.clear(); self.repair_contributors.clear(); self.repair_holds.clear(); self._invalidate()

    def step(self,actions=None):
        if self.is_terminal:return self.result
        self.metrics['steps']+=1; actions=actions or {}
        submitted={pid:a for pid,a in actions.items() if a is not None and a!=Intent(IntentKind.WAIT)}
        # A complete immutable pre-action projection precedes validation/execution.
        observations={pid:self.observe_pj(pid) for pid in sorted(submitted) if pid in self.players}
        valid={}; extra={}
        for pid,a in sorted(submitted.items()):
            obs=observations.get(pid)
            if obs is not None and type(a) is Intent and intent_allowed(obs.actor,a) and (
                    a.kind is not IntentKind.KILL or a.target not in obs.teammates) and not (
                    a.kind is IntentKind.CALL_MEETING and (self.sabotage is not None or self.time < self.emergency_ready)):
                if a.kind is IntentKind.GO_TO_LOCATION and (a.position is None or not self.map.contains((a.position.x,a.position.y))):
                    self.metrics['illegal_actions']+=1
                else:valid[pid]=a
            elif obs is not None and self._extension_allowed(pid,a,obs):extra[pid]=a
            else:self.metrics['illegal_actions']+=1;self._record('illegal_action',player_id=pid)
        if self.phase is Phase.ROAMING:
            for pid,a in valid.items():
                if a.kind is IntentKind.REPORT:self._start_meeting(self.players[pid],self.bodies[a.target]);break
                if a.kind is IntentKind.CALL_MEETING:self._start_meeting(self.players[pid]);break
            if self.phase is Phase.ROAMING:
                for pid,a in extra.items():
                    if a.kind=='sabotage' and self.sabotage is None:self._begin_sabotage(a.target)
                for pid,a in valid.items():
                    if a.kind not in (IntentKind.WAIT,):self.repair_holds.pop(pid,None)
                for pid,a in valid.items():
                    if a.kind is IntentKind.KILL and self.players[pid].active and self.players[a.target].active:
                        self._eliminate(self.players[pid],self.players[a.target])
                        if self.is_terminal:return self.result
                self._repair({pid:a for pid,a in extra.items() if a.kind=='repair'})
                for pid,a in valid.items():
                    if self.players[pid].active:self._apply_exploration(self.players[pid],a,observations[pid].actor)
        elif self.phase is Phase.DISCUSSION:
            for pid,a in valid.items():
                if a.kind is IntentKind.CLAIM:
                    d=a.claim
                    from social_deduction.phase3_api import StructuredClaim
                    self.claimed.add(pid)
                    self._publish(StructuredClaim(d.kind,pid,d.subject_id,d.region,self.tick,d.asserted_tick),'claim',
                        claim_kind=d.kind,speaker_id=pid,subject_id=d.subject_id,region=d.region,asserted_tick=d.asserted_tick)
            for pid,a in extra.items():
                if a.kind=='communication':
                    e=replace(a.statement,source=pid,observed=min(a.statement.observed,self.time),delivered=self.time,provenance='claim')
                    self.pj_public.append(e);self.claimed.add(pid)
                    self._record('structured_claim',channel='public',speaker_id=pid,claim_kind=e.kind,
                                 subject_id=e.subject,region=e.room,asserted_time=e.observed)
        elif self.phase is Phase.VOTING:
            for pid,a in valid.items():
                if a.kind in (IntentKind.VOTE,IntentKind.SKIP):self._vote(pid,a.target if a.kind is IntentKind.VOTE else None)
        if self.phase is Phase.ROAMING:
            previous_status={p.player_id:p.navigator.status for p in self.players.values()}
            self._advance_physics()
            if getattr(self.experiment,'record_diagnostics',False):
                self.learning_diagnostics['navigation_arrivals']+=sum(p.active and p.role is Role.CREWMATE
                    and previous_status[p.player_id] is NavStatus.MOVING and p.navigator.status is NavStatus.SUCCESS
                    for p in self.players.values())
        else:
            for t in self.tasks.values():
                if t.progress<1 and t.ready_at>self.time:t.ready_at+=self.config.dt
        if getattr(self.experiment,'record_diagnostics',False):
            dt=self.config.dt;d=self.learning_diagnostics
            alive=[p for p in self.players.values() if p.active and p.role is Role.CREWMATE]
            d['alive_actor_seconds']+=dt*len(alive)
            if self.sabotage=='lights':
                d['lights_active_seconds']+=dt
                for p in alive:self.lights_actor_seconds[p.player_id]+=dt
            if self.phase is Phase.ROAMING:
                d['moving_actor_seconds']+=dt*sum(any(abs(v)>1e-8 for v in p.velocity) for p in alive)
                d['task_actor_seconds']+=dt*sum(p.interaction_task is not None for p in alive)
            else:d['meeting_actor_seconds']+=dt*len(alive)
        self.tick+=1;self._invalidate()
        if self.phase is Phase.ROAMING:
            self._advance_tasks();self._advance_ghosts()
            # Repair at the last legal tick precedes failure; failure precedes task win.
            if self.sabotage_deadline is not None and self.time>=self.sabotage_deadline-1e-9:
                self._finish(Role.IMPOSTOR,self.sabotage.upper()+'_FAILURE')
        elif self.phase is Phase.DISCUSSION and self.tick-self.phase_started_tick>=math.ceil(self.config.discussion_seconds/self.config.dt):
            self.phase=Phase.VOTING;self.phase_started_tick=self.tick;self._record('voting_started',channel='public',meeting_id=self.meeting_id)
        elif self.phase is Phase.VOTING and (self._eligible_voters()<=self.votes.keys() or
                self.tick-self.phase_started_tick>=math.ceil(self.config.voting_seconds/self.config.dt)):
            for pid in sorted(self._eligible_voters()):
                if pid not in self.votes:self._vote(pid,None)
            self._resolve_meeting()
        if not self.is_terminal:self._check_win()
        if getattr(self.experiment,'normal_match_timeout',True):
            # Legacy profile only, retained for historical reproduction. New lights
            # training has no timer-based game ending; technical cuts are separate.
            if not self.is_terminal and self.time>=self.config.timeout:self._finish(None,'TIMEOUT')
        elif not self.is_terminal:self._check_technical_safety()
        self._invalidate()
        if self.config.check_invariants:self.assert_invariants()
        return self.result

    def _check_technical_safety(self):
        signature=tuple(self.metrics[k] for k in ('eliminations','task_completions','reports','meetings','repairs'))
        active=[p for p in self.players.values() if p.active]
        moving=any(any(abs(v)>1e-8 for v in p.velocity) for p in active) if self.phase is Phase.ROAMING else False
        working=any(p.interaction_task is not None for p in active) if self.phase is Phase.ROAMING else False
        if signature!=self._safety_progress_signature or moving or working:
            self._safety_progress_time=self.time;self._safety_progress_signature=signature
        cause=('max_simulated_duration' if self.time>=self.experiment.technical_max_seconds else
               'stationary_without_progress' if self.time-self._safety_progress_time>=self.experiment.technical_stall_seconds else None)
        # Cut only at a roaming observation boundary, after checking genuine wins.
        # Fixed meeting phases finish first, so living critics receive ordinary
        # decision-state packets rather than untrained non-speaker ghost states.
        if cause and self.phase is Phase.ROAMING:
            self.truncation_pending=dict(cause=cause,simulation_seconds=self.time,
                seconds_without_motion_or_progress=self.time-self._safety_progress_time,
                max_simulated_seconds=self.experiment.technical_max_seconds,
                stationary_guard_seconds=self.experiment.technical_stall_seconds,phase=self.phase.value,
                living_crewmates=sum(p.role is Role.CREWMATE for p in active),
                living_impostors=sum(p.role is Role.IMPOSTOR for p in active),
                task_completions=self.metrics['task_completions'],reports=self.metrics['reports'],kills=self.metrics['eliminations'])

    def finalize_truncation(self):
        if self.is_terminal or not self.truncation_pending:raise ValueError('No technical truncation pending')
        self.truncation_details=dict(self.truncation_pending)
        self.metrics['technical_truncation']=self.truncation_details
        self._record('technical_truncation',channel='spectator',**self.truncation_details)
        self._finish(None,'TRUNCATED')

    def assert_invariants(self):
        if not getattr(self,'_v2_ready',False):return
        assert len(self.players)==self.experiment.population.total
        assert sum(p.role is Role.IMPOSTOR for p in self.players.values())==self.experiment.population.impostors
        assigned=[t.task_id for p in self.players.values() for t in p.tasks]
        assert len(assigned)==len(set(assigned))==self.initial_task_count==len(self.tasks)
        assert all(t.owner_id==p.player_id and self._owners[t.task_id]==p.player_id for p in self.players.values() for t in p.tasks)
        assert all(0<=t.progress<=1 for t in self.tasks.values())
        assert all(self.map.contains(p.position) for p in self.players.values())
        assert all(p.status in ('alive','dead','ejected') for p in self.players.values())
        assert all(not self.players[pid].active and self.players[pid].role is Role.CREWMATE for pid in self.ghosts)
        assert set(self.votes)<=set(self.participants)
        assert all(t is None or t in self.participants for t in self.votes.values())
        assert self.is_terminal==(self.phase is Phase.FINISHED)
