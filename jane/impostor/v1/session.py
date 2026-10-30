"""Trusted native-game coordinator; actors see detached own observations only."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from .features import OwnFeedback, encode
from .rewards import ready_seconds


def resolve_team_contention(actions):
    """Preserve native A2 door-before-sabotage and sorted-owner execution order.

    A pre-legal duplicate request has no effect, rather than becoming a false
    illegal-policy metric after another teammate consumes the shared resource.
    Sampled actions/log-probabilities are retained. No new destination is chosen.
    """
    from patrick_jane.api import Action
    from social_deduction.phase3_api import Intent,IntentKind
    resolved=dict(actions or {});declined={};seen=set()
    has_door=any(type(a) is Action and a.kind=='doors' for a in resolved.values())
    for pid,action in sorted(resolved.items()):
        if type(action) is not Action or action.kind not in ('doors','sabotage'):continue
        key=(action.kind,action.target)
        if action.kind=='sabotage' and has_door or key in seen:
            declined[pid]=dict(kind=action.kind,target=action.target)
            resolved[pid]=Intent(IntentKind.WAIT)
        else:seen.add(key)
    return resolved,declined


@dataclass
class Transition:
    packet: object
    action: int
    logp: float
    value: float
    reward: float = 0.


class Actor:
    """No simulator, recorder, opponent memory or training rewards live here."""
    def __init__(self, owner, model, settings, public_planner, seed, deterministic=False):
        from patrick_jane.memory import Memory
        self.owner, self.model, self.settings = owner, model, settings
        self.planner = public_planner
        self.memory = Memory(owner, 64)
        self.hidden = torch.zeros(1,256)
        self.rng = torch.Generator().manual_seed(int(seed) % (2**63-1))
        self.last_goal = None; self.last_kind = 'wait'; self.deterministic = deterministic

    @torch.inference_mode()
    def choose(self, observation, feedback):
        started=perf_counter()
        packet = encode(observation, feedback, self.memory, self.planner, self.settings,
                        self.last_goal, self.last_kind)
        feature_seconds=perf_counter()-started;started=perf_counter()
        distributions, values, self.hidden = self.model([packet], self.hidden)
        distribution = distributions[0]
        index = int(distribution.probs.argmax()) if self.deterministic else int(
            torch.multinomial(distribution.probs, 1, generator=self.rng))
        candidate = packet.candidates[index]
        self.last_kind = candidate.kind
        if candidate.kind in ('destination','room','visible_player','remembered_player','body_travel'):
            self.last_goal = (candidate.kind,candidate.key)
        elif candidate.kind == 'cancel': self.last_goal = None
        return candidate.payload, Transition(packet, index, float(distribution.log_prob(torch.tensor(index))),
                                             float(values[0])), dict(
            probabilities=distribution.probs.tolist(), entropy=float(distribution.entropy()),
            feature_seconds=feature_seconds,inference_seconds=perf_counter()-started)


class Proxy:
    """Trusted transport and learner ledger; never passed into the neural model."""
    def __init__(self, actor):
        self.actor = actor; self.feedback = None; self.rollout = []; self.counts = Counter()
        self.observer = None
        self.timing=Counter()

    def decide(self, observation):
        command, transition, diagnostics = self.actor.choose(observation, self.feedback)
        self.rollout.append(transition)
        candidate = transition.packet.candidates[transition.action]
        self.counts[candidate.kind] += 1
        self.counts['observations'] += 1
        self.counts['choices' if len(transition.packet.candidates)>1 else 'forced_waits'] += 1
        for name in ('feature_seconds','inference_seconds'):self.timing[name]+=diagnostics[name]
        if self.observer: self.observer(self.actor.owner, transition, diagnostics)
        return command


class JointRecorder:
    def __init__(self, match, path, metadata, live=None):
        from spectator.recorder import MatchRecorder
        self.match = match; self.live = live
        self.recorder = MatchRecorder(match, path, training_session=metadata['session'],
            game_schema='jane-impostor-a2-v1', source_version=metadata['source'],
            model_architecture=metadata['architecture'], controller_types=metadata['controllers'], metadata=metadata)
        self.cumulative = Counter(); self.seconds = 0.
        self.policy_update=metadata.get('policy_update',0)

    def neural(self, pid, record, packet, nav):
        from patrick_jane.replay import neural_record
        neural_record(self.recorder, self.match, pid, record, packet, nav)

    def impostor(self, pid, transition, diagnostics):
        packet = transition.packet; c = packet.candidates[transition.action]
        probabilities = {f'{i}:{item.kind}:{item.key}':float(p) for i,(item,p) in
                         enumerate(zip(packet.candidates,diagnostics['probabilities'])) if p>0}
        for recorder in (self.recorder, self.live.recorder if self.live and self.live.recorder else None):
            if recorder:
                recorder.record_neural(pid, c.kind, c.key, probabilities=probabilities,
                    value_estimate=transition.value, entropy=diagnostics['entropy'],
                    legal_actions=list(probabilities), selected_action_index=transition.action,
                    policy_update=self.policy_update)

    def reward(self, when, pid, components):
        from spectator.replay_format import ReplayReward
        self.cumulative[pid] += sum(components.values())
        self.recorder.recorder.write_reward(ReplayReward(tick=self.match.game.tick, sim_time=when,
            player_id=pid,reward=sum(components.values()),cumulative_reward=self.cumulative[pid],components=components))

    def step(self): self.recorder.after_step()
    def finish(self): self.recorder.finish()


def run_match(policy, crew_model, settings, experiment, seed, *, output=None,
              max_wall_seconds=300, max_observations=None, deterministic=False, status=None, live_dir=None,
              policy_update=0,mode='engineering'):
    from patrick_jane.worker import Match
    from patrick_jane.network import collate
    from patrick_jane.actions import choose
    from navigation_service import NavigationPlanner
    from phase3_engine import stream_seed
    began = perf_counter()
    native = Match(seed, experiment)
    native.opponent_regime = 'jane-impostor-learned-v1'
    impostors = tuple(pid for pid in native.game.players if pid not in native.crew)
    public_planner = NavigationPlanner(native.game.base_map,cache_size=256)
    proxies = {pid:Proxy(Actor(pid,policy,settings,public_planner,stream_seed(seed,'jane-imp:'+pid),deterministic))
               for pid in impostors}
    native.opponents = proxies  # No tactical/script decide is called in this new regime.
    native.game._record('controller_override',channel='spectator-private',regime='jane-impostor-learned-v1',
                        controllers={pid:'frozen-patrick-jane-102' if pid in native.crew else 'Lizbon-v1' for pid in native.game.players})
    metadata = dict(session='Lizbon-v1', source='upstream-d459f386+own-v1',
        architecture=json.dumps(policy.architecture()),policy_update=policy_update,run_mode=mode,
        policy_state='fresh untrained initialization' if policy_update==0 else 'trained checkpoint',
        controllers={pid:'frozen-patrick-jane-102' if pid in native.crew else 'Lizbon-v1' for pid in native.game.players})
    if live_dir:
        from patrick_jane.live import LiveBridge
        class JointLive(LiveBridge):
            def check(self):
                super().check()
                if self.recorder:
                    self.recorder.recorder.header.update(controller_types=metadata['controllers'],
                        game_schema='jane-impostor-a2-v1',training_session='Lizbon-v1',model_architecture=metadata['architecture'])
        native.live = JointLive(native,live_dir,dict(metadata,environment=0))
    observer = JointRecorder(native, output, metadata, native.live) if output else None
    if observer:
        native.recorder = observer
        for proxy in proxies.values(): proxy.observer = observer.impostor
    reward_components = {pid:Counter() for pid in impostors}; reward_done = False;contentions=Counter()
    original_step = native.game.step
    def observed_step(actions=None):
        nonlocal reward_done
        game = native.game; before_time = game.time; before_phase = game.phase.value
        before = {pid:(game.players[pid].active,game.players[pid].cooldown,pid in game.vented,
                       game.time<game.vent_transition.get(pid,0.)) for pid in impostors}
        cursor = len(game.event_log)
        resolved,declined=resolve_team_contention(actions)
        for pid,request in declined.items():
            contentions[request['kind']]+=1
            game._record('team_resource_contention',channel='spectator-private',player_id=pid,
                         request_kind=request['kind'],request_target=request['target'])
        result = original_step(resolved)
        elapsed = game.time-before_time
        events = game.event_log[cursor:]
        killers = {e['killer_id'] for e in events if e['kind']=='elimination'}
        kills = sum(e['kind']=='elimination' for e in events)
        components = {pid:{} for pid in impostors}
        for pid in impostors:
            if kills: components[pid]['team_kill'] = float(kills)
            active, cooldown, vented, transition = before[pid]
            ready=ready_seconds(elapsed,cooldown,active_before=active,active_after=game.players[pid].active,
                roaming_before=before_phase=='roaming',roaming_after=game.phase.value=='roaming',
                vented_before=vented,vented_after=pid in game.vented,killed=pid in killers,
                transition_before=transition,transition_after=game.time<game.vent_transition.get(pid,0.),
                kill_cooldown=game.config.kill_cooldown)
            if ready:components[pid]['cooldown_ready_time']=-.005*ready
        if game.is_terminal and game.result.winner is not None and not reward_done:
            amount = 10. if game.result.winner.value=='impostor' else -10.
            for pid in impostors: components[pid]['team_outcome'] = amount
            reward_done = True
        for pid, values in components.items():
            if values and proxies[pid].rollout:
                proxies[pid].rollout[-1].reward += sum(values.values())
                reward_components[pid].update(values)
                if observer: observer.reward(game.time,pid,values)
                if native.live: native.live.reward(game.time,pid,values,sum(reward_components[pid].values()))
        return result
    native.game.step = observed_step
    response = native.prepare()
    crew_hidden = {pid:torch.zeros(1,256) for pid in native.crew}
    crew_rng = {pid:torch.Generator().manual_seed(stream_seed(seed,'jane-frozen-crew:'+pid) % (2**63-1)) for pid in native.crew}
    crew_choices = 0;crew_seconds=0.; policy.eval()
    while not response['terminal']:
        count = sum(len(p.rollout) for p in proxies.values())
        if perf_counter()-began >= max_wall_seconds or max_observations is not None and count >= max_observations:
            native.game.truncation_pending = dict(cause='engineering-or-wall-cut',simulation_seconds=native.game.time)
            native.game.finalize_truncation()
            if observer: observer.step(); observer.finish()
            if native.live: native.live.step(); native.live.publish(force=True)
            break
        for pid, proxy in proxies.items():
            p = native.game.players[pid]
            if not p.active: continue
            obs = native.observations[pid]
            task = p.interaction_task
            proxy.feedback = OwnFeedback(pid,obs.actor.match_id,obs.actor.tick,float(obs.actor.kill_cooldown),
                float(p.interaction_remaining) if task and task.startswith('fake:') else 0.,
                task[5:] if task and task.startswith('fake:') else None)
        ids = list(native.packets); decisions = {}
        if ids:
            inference_started=perf_counter()
            with torch.inference_mode():
                outputs, states = crew_model(collate([native.packets[p] for p in ids]),torch.cat([crew_hidden[p] for p in ids]))
                for i,pid in enumerate(ids):
                    crew_hidden[pid] = states[i:i+1]
                    out = {k:v[i] for k,v in outputs.items()}
                    choice,lp,entropy,probs = choose(out,native.packets[pid],generator=crew_rng[pid])
                    decisions[pid] = dict(choice=choice,logp=float(lp),value=float(out['value']),
                                          entropy=float(entropy),probabilities=probs,update=102)
            crew_choices += len(ids)
            crew_seconds+=perf_counter()-inference_started
        response = native.advance(decisions)
        if status and count % 100 < 2:
            status(dict(seed=seed,simulation_seconds=native.game.time,impostor_observations=count,
                        crew_choices=crew_choices,wall_seconds=perf_counter()-began))
    summary = native.summary()
    summary.update(impostor_actions={pid:dict(proxy.counts) for pid,proxy in proxies.items()},
                   impostor_rewards={pid:dict(c) for pid,c in reward_components.items()},
                   impostor_observations=sum(len(p.rollout) for p in proxies.values()),
                   team_resource_contentions=dict(contentions),
                   timing=dict(crew_model_and_sampling_seconds=crew_seconds,
                               impostor_features_seconds=sum(p.timing['feature_seconds'] for p in proxies.values()),
                               impostor_model_and_sampling_seconds=sum(p.timing['inference_seconds'] for p in proxies.values())),
                   impostor_reward_regime='jane-impostor-team-v1',
                   wall_seconds=perf_counter()-began,controllers=metadata['controllers'])
    if output:
        Path(output).with_suffix('.result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary, [proxy.rollout for proxy in proxies.values()]
