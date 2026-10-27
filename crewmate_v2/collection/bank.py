"""Process-private environments. Only allowlisted actor packets reach inference."""
import hashlib

import torch
from among_us_map import get_map
from navigation_service import NavigationPlanner
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.learning.config import RewardConfig
from crewmate_v2.learning.env import LearningEnv
from crewmate_v2.learning.policy import check_packet
from crewmate_v2.learning.train import actor_trace
from .config import context


def signature(packet):
    result = hashlib.sha256()
    actor_trace(result, packet, 0)
    return result.hexdigest()


class EnvironmentBank:
    def __init__(self, slots, streams, arm, partition, deadline, lobby=None):
        if arm not in ('original', 'shaped'):
            raise ValueError('Unregistered reward arm')
        self.streams, self.partition = streams, partition
        self.envs, self.packets, self.episodes, self.history = {}, {}, {}, {}
        # One read-only geometry planner/frozen predictor per process. Every mutable
        # match/controller/observation/memory/ledger is constructed separately.
        planner = NavigationPlanner(get_map(), cache_size=96)
        belief = None
        for slot in slots:
            env = LearningEnv(planner=planner, belief_model=belief, lobby=lobby or LobbySettings(),
                              rewards=None if arm == 'original' else RewardConfig(),
                              opponents='varied', coalesce=True, mixture=MatchMixture(1.), deadline=deadline)
            belief = env.belief_model
            self.envs[slot] = env
            self.episodes[slot] = 0
            self.reset(slot)
        self.frozen = {k: v.clone() for k, v in belief.network.state_dict().items()}

    def reset(self, slot):
        selected = context(self.partition, slot, self.episodes[slot], self.streams)
        env = self.envs[slot]
        env.rotation = selected['rotation']
        self.packets[slot] = env.reset(seed=selected['seed'])[0]
        self.history[slot] = []

    def initial(self):
        return dict(self.packets)

    def step(self, actions):
        rows = {}
        for slot, action in sorted(actions.items()):
            env = self.envs[slot]
            packet = self.packets[slot]
            check_packet(packet)
            if not 0 <= action < 96 or not packet['mask'][action] or packet['mask'].sum() < 2:
                raise ValueError('Masked or forced learner action')
            selected = context(self.partition, slot, self.episodes[slot], self.streams)
            obs = env.observation
            audit = dict(body_visible=[b.victim_id for b in obs.visible_bodies],
                         legal_reports=[a.target for a in obs.actions if a.kind.value == 'report'],
                         ballot=obs.context.meeting_id if any(c.kind in ('vote', 'skip') for c in env.options) else None,
                         action_kind=env.options[action].kind)
            new, reward, done, truncated, info = env.step(action)
            self.history[slot].append(int(action))
            if truncated or info['masked_selection'] or any(env.match.game.metrics[k] for k in ('illegal_actions', 'navigation_failures')):
                raise ValueError('Invalid or externally truncated engineering transition')
            if not all(torch.equal(v, self.frozen[k]) for k, v in env.belief_model.network.state_dict().items()):
                raise ValueError('Frozen belief mutation')
            self.packets[slot] = new
            row = dict(reward=reward, done=bool(done), info=info, audit=audit, **selected)
            if done:
                row['episode_index'] = self.episodes[slot]
                self.episodes[slot] += 1
                self.reset(slot)
            row['packet'] = self.packets[slot]
            rows[slot] = row
        return rows

    def snapshot(self):
        return {slot: dict(episode=self.episodes[slot], actions=list(self.history[slot]),
                           packet_sha256=signature(self.packets[slot]),
                           simulated_time=self.envs[slot].match.game.time,
                           reward_counts=dict(self.envs[slot].match.ledger.counts),
                           reward_components=dict(self.envs[slot].match.ledger.components))
                for slot in self.envs}

    def restore(self, snapshots):
        if set(snapshots) != set(self.envs):
            raise ValueError('Resume worker topology mismatch')
        for slot, saved in sorted(snapshots.items()):
            self.episodes[slot] = saved['episode']
            self.reset(slot)
            for action in saved['actions']:
                result = self.step({slot: action})[slot]
                if result['done']:
                    raise ValueError('Resume prefix crosses a recorded reset')
            if self.snapshot()[slot] != saved:
                raise ValueError('Resume reconstruction mismatch')
        return self.initial()


def worker_main(connection, options):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        bank = EnvironmentBank(**options)
        connection.send(('ok', bank.initial()))
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            if command not in ('step', 'snapshot', 'restore'):
                raise ValueError('Unknown worker command')
            result = bank.snapshot() if command == 'snapshot' else getattr(bank, command)(payload)
            connection.send(('ok', result))
    except BaseException as error:
        try:
            connection.send(('error', (type(error).__name__, str(error))))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()
