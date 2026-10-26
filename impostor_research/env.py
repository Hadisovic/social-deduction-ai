"""Trusted orchestration/reward boundary. Actors receive only their own envelopes."""
from dataclasses import dataclass, asdict
from collections import Counter
from hashlib import sha256
import math
from random import Random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from among_us_map import get_map
from navigation_service import NavigationPlanner
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.engine import CrewmateV2Game
from crewmate_v2.runtime import PublicMemory
from phase3_bots import BotStyle, CREW_VARIANTS, MEETING_VARIANTS
from phase3_runner import behavior_seed, canonical_bytes
from phase4_scripts import StudyController
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import Intent, IntentKind, intent_allowed
from .features import MAX_ACTIONS, GLOBAL_COUNT, PLAYER_COUNT, ACTION_COUNT, candidates, encode
from .frozen import FrozenCrew, FrozenModels
from .memory import ImpostorMemory
from .opponents import OpponentGuard, Safeguards


@dataclass(frozen=True)
class RewardConfig:
    """Proposed sparse objective. No hidden witness/suspicion shaping."""
    win: float = 10.
    loss: float = -10.
    timeout: float = -10.
    per_second: float = -.001
    gamma_per_second: float = 1.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()) or not 0 < self.gamma_per_second <= 1:
            raise ValueError('Invalid reward/discount configuration')


class ImpostorMatch:
    def __init__(self, seed, models, *, planner=None, lobby=None, mixture=None, safeguards=Safeguards(), fast=True):
        self.game = CrewmateV2Game(seed, planner=planner, lobby=lobby, mixture=mixture, fast=fast)
        self.seed = seed; self.models = models
        self.focal = next(pid for pid,p in self.game.players.items() if p.role is Role.IMPOSTOR)
        crew = sorted(pid for pid,p in self.game.players.items() if p.role is Role.CREWMATE)
        self.learned = Random(behavior_seed(seed,'impostor-research:crew-seat')).choice(crew)
        self.controllers = {}
        for pid in crew:
            rng = Random(behavior_seed(seed,'impostor-research:style:'+pid))
            self.controllers[pid] = (FrozenCrew(models) if pid == self.learned else StudyController(
                behavior_seed(seed,'impostor-research:script:'+pid),BotStyle(rng.choice(CREW_VARIANTS),'hunter',rng.choice(MEETING_VARIANTS))))
        self.guards = {pid:OpponentGuard(safeguards) for pid in crew}
        self.decision_ticks = round(self.game.config.decision_interval/self.game.config.dt)
        self.trace = sha256(); self.scheduled = 0; self.proposals = 0

    def tick(self, focal_intent=None):
        game = self.game
        if game.is_terminal: return
        # WAIT and an absent submission are identical engine operations. Normalize
        # their audit representation so forced-window coalescing compares semantics.
        if focal_intent is not None and focal_intent.kind is IntentKind.WAIT:
            focal_intent = None
        actions = {}
        if game.tick % self.decision_ticks == 0:
            # Take every actor snapshot BEFORE computing any simultaneous action.
            envelopes = {pid:game.observe_v2(pid) for pid in sorted(self.controllers)}
            for pid,envelope in envelopes.items():
                obs = envelope.actor
                if not obs.own.active: continue  # ghost state never enters a controller
                controller = self.controllers[pid]
                self.scheduled += 1
                if obs.actions:
                    proposed = controller.decide(envelope) if isinstance(controller,FrozenCrew) else controller.decide(obs)
                    self.proposals += 1
                else:
                    if isinstance(controller,FrozenCrew): controller.memory.update(obs)
                    else: controller._observe(obs)
                    proposed = Intent(IntentKind.WAIT)
                before = len(self.guards[pid].records)
                executed = self.guards[pid].apply(envelope,proposed)
                if len(self.guards[pid].records) > before and isinstance(controller,FrozenCrew): controller.interrupt()
                self.trace.update(canonical_bytes((pid,obs.tick,proposed,executed)))
                if executed.kind is not IntentKind.WAIT: actions[pid] = executed
        if focal_intent is not None and focal_intent.kind is not IntentKind.WAIT:
            actions[self.focal] = focal_intent
        self.trace.update(canonical_bytes((game.tick,focal_intent)))
        game.step(actions)


class ImpostorEnv(gym.Env):
    """New Gym interface, one learned impostor slot and four fixed opponents.

    Fixed .6-second meaningful decision windows. Optional forced-window coalescing
    retains EVERY physics/opponent/observation tick; it only avoids exposing empty
    categorical decisions during meetings. No optimizer is present.
    """
    metadata = {'render_modes': []}

    def __init__(self, *, models=None, planner=None, lobby=None, mixture=None,
                 safeguards=Safeguards(), fast=True, coalesce=True, reward=RewardConfig()):
        self.models = models or FrozenModels()
        self.planner = planner or NavigationPlanner(get_map(),cache_size=96)
        self.lobby = lobby or LobbySettings.from_file('configs/crewmate-v2-lobby.json')
        self.mixture = mixture or MatchMixture()
        self.safeguards = safeguards; self.fast = fast; self.coalesce = coalesce; self.reward_config = reward
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict({
            'global_':spaces.Box(-1,1,(GLOBAL_COUNT,),np.float32),
            'players':spaces.Box(-1,1,(4,PLAYER_COUNT),np.float32),
            'actions':spaces.Box(-1,1,(MAX_ACTIONS,ACTION_COUNT),np.float32),
            'targets':spaces.Box(-1,3,(MAX_ACTIONS,),np.int64),
            'mask':spaces.Box(0,1,(MAX_ACTIONS,),np.int8)})
        self.finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        selected = int(self.np_random.integers(0,2**31)) if seed is None else int(seed)
        self.match = ImpostorMatch(selected,self.models,planner=self.planner,lobby=self.lobby,
            mixture=self.mixture,safeguards=self.safeguards,fast=self.fast)
        self.memory = ImpostorMemory(); self.public_memory = PublicMemory()
        self.finished = False; self.total_reward = 0.; self.windows = 0; self.meaningful = 0
        self.first_kill_opportunity = None; self.actor_trace = sha256()
        self._observe()
        return self.packet, {}

    def _observe(self):
        self.envelope = self.match.game.observe_v2(self.match.focal)
        obs = self.envelope.actor
        self.memory.update(obs); self.public_memory.update(self.envelope)
        self.options = candidates(self.envelope,self.memory)
        if obs.context.phase is Phase.FINISHED:
            # Terminal packet is absorbing; never encode post-game truth labels.
            self.packet = {k:np.zeros(s.shape,dtype=s.dtype) for k,s in self.observation_space.spaces.items()}
            self.packet['targets'].fill(-1); self.packet['mask'][0] = 1
        else:
            self.packet = encode(self.envelope,self.memory,self.public_memory,self.options)
        for key in sorted(self.packet): self.actor_trace.update(self.packet[key].tobytes())
        if self.first_kill_opportunity is None and any(a.kind is IntentKind.KILL for a in obs.actions):
            self.first_kill_opportunity = obs.time_seconds

    def action_masks(self): return self.packet['mask'].astype(bool)

    def step(self, action):
        if self.finished: raise RuntimeError('Reset after terminal')
        if not self.action_space.contains(action) or not self.packet['mask'][int(action)]:
            raise ValueError('Invalid/masked impostor selection')
        return self._advance(self.options[int(action)].intent)

    def step_scripted(self, intent):
        """Engineering control only, not a learned strategy."""
        if self.finished: raise RuntimeError('Reset after terminal')
        if intent.kind is not IntentKind.WAIT and not intent_allowed(self.envelope.actor,intent):
            raise ValueError('Illegal engineering control')
        return self._advance(intent)

    def _advance(self, intent):
        reward = 0.; raw = 0.; elapsed = 0.; forced = 0
        self.meaningful += len(self.options) > 1
        while True:
            for tick in range(self.match.decision_ticks):
                self.match.tick(intent if elapsed == 0 and tick == 0 else None)
                dt = self.match.game.config.dt
                r = self.reward_config.per_second * dt
                if self.match.game.is_terminal:
                    result = self.match.game.result
                    r += (self.reward_config.win if result.winner is Role.IMPOSTOR else
                          self.reward_config.loss if result.winner is Role.CREWMATE else self.reward_config.timeout)
                reward += self.reward_config.gamma_per_second**elapsed * r
                raw += r; elapsed += dt
                if self.match.game.is_terminal: break
            self.windows += 1; self._observe()
            if self.match.game.is_terminal or not self.coalesce or len(self.options) > 1: break
            forced += 1
        self.finished = self.match.game.is_terminal; self.total_reward += raw
        info = dict(elapsed_seconds=elapsed,discount=self.reward_config.gamma_per_second**elapsed,
                    undiscounted_reward=raw,forced_windows_coalesced=forced)
        if self.finished:
            self.models.assert_unchanged()
            info['episode'] = self.summary()
        return self.packet,float(reward),self.finished,False,info

    def summary(self):
        g = self.match.game
        eliminations = [e for e in g.event_log if e['kind']=='elimination']
        living_witnesses = {(e['tick'],e['victim_id']) for e in g.event_log
            if e['kind']=='elimination_witnessed' and e['observer_id'] not in (e['killer_id'],e['victim_id'])}
        reports = [e for e in g.event_log if e['kind']=='report']
        votes = [e for e in g.event_log if e['kind']=='vote_submitted' and e['voter_id']!=self.match.focal]
        records = [r for guard in self.match.guards.values() for r in guard.records]
        dead_since = {e['victim_id']:e['tick'] for e in eliminations}
        dead_since.update({e['player_id']:e['tick'] for e in g.event_log if e['kind']=='ejection' and e.get('player_id')})
        return dict(seed=self.match.seed,reason=g.result.reason if g.result else None,
            impostor_win=bool(g.result and g.result.winner is Role.IMPOSTOR),simulated_seconds=g.time,
            first_legal_kill=self.first_kill_opportunity,first_elimination=eliminations[0]['tick']*g.config.dt if eliminations else None,
            kills=len(eliminations),witnessed_kills=len(living_witnesses),meaningful_decisions=self.meaningful,
            scheduled_windows=self.windows,reports=g.metrics['reports'],meetings=g.metrics['meetings'],
            illegal_actions=g.metrics['illegal_actions'],navigation_failures=g.metrics['navigation_failures'],
            ghost_task_completions=sum(e['kind']=='task_completed' and e.get('player_id') in dead_since
                and dead_since[e['player_id']]<=e['tick'] for e in g.event_log),
            impostor_reports=sum(e['reporter_id']==self.match.focal for e in reports),
            crew_reports=sum(e['reporter_id']!=self.match.focal for e in reports),
            correct_crew_votes=sum(e['target_id']==self.match.focal for e in votes),
            incorrect_crew_votes=sum(e['target_id'] not in (None,self.match.focal) for e in votes),
            crew_skips=sum(e['target_id'] is None for e in votes),
            impostor_ejected=g.players[self.match.focal].status=='ejected',
            guard_events=len(records),guard_changes=sum(r['changed'] for r in records),
            guard_reasons=dict(Counter(r['reason'] for r in records)),
            policy_calls=self.match.controllers[self.match.learned].inference_calls,
            action_hash=self.match.trace.hexdigest(),actor_hash=self.actor_trace.hexdigest(),
            event_hash=sha256(canonical_bytes(g.event_log)).hexdigest(),return_=self.total_reward)
