"""Study guards and trainer-only evidence; none of these values enter policies."""
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from time import time
import numpy as np
import torch
from social_deduction.actor import Role
from .config import source_hashes, verify_frozen


def runtime_versions():
    return dict(python=platform.python_version(), **{name:version(name) for name in
                ('torch','numpy','shapely','gymnasium')}, torch_threads=1)


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def guard(expected_sources, deadline):
    if time() > deadline:
        raise TimeoutError('Registered wall-clock budget exceeded')
    verify_frozen()
    if source_hashes() != expected_sources:
        raise ValueError('Source/runtime changed during the study')


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value: finite(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite study value')


def packet_guard(packet):
    if not packet['mask'].any() or not all(np.isfinite(v).all() for v in packet.values()):
        raise FloatingPointError('Invalid/nonfinite policy packet')


def transition_guard(env, info, reward):
    finite((reward, info['discount'], info['elapsed_seconds']))
    if info.get('masked_selection') or info['elapsed_seconds'] <= 0:
        raise ValueError('Invalid policy selection or elapsed time')
    metrics = env.match.game.metrics
    if metrics['illegal_actions'] or metrics['navigation_failures']:
        raise ValueError('Illegal action or navigation failure: stop the study')


def state_hash(policy):
    trace = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        trace.update(key.encode())
        trace.update(value.detach().cpu().contiguous().numpy().tobytes())
    return trace.hexdigest()


def trace_step(trace, packet, action, reward, info, done):
    for key, value in sorted(packet.items()):
        trace.update(key.encode()); trace.update(value.tobytes())
    trace.update(json.dumps([action, reward, info['discount'], info['elapsed_seconds'], done]).encode())


def episode_metrics(env, episode, actions, ends, decisions, simulated, entropies):
    game = env.match.game
    votes = [e for e in game.event_log if e['kind'] == 'vote' and e.get('voter_id') == env.focal]
    cast = [e for e in votes if e.get('target_id') is not None]
    ejections = [e for e in game.event_log if e['kind'] == 'ejection' and e.get('player_id') is not None]
    correct = sum(game.players[e['player_id']].role is Role.IMPOSTOR for e in ejections)
    return dict(episode, votes_cast=len(cast),
                votes_correct=sum(game.players[e['target_id']].role is Role.IMPOSTOR for e in cast),
                skips=len(votes)-len(cast), correct_ejections=correct,
                innocent_ejections=len(ejections)-correct,
                valid_reports=sum(e['kind'] == 'report' and e.get('reporter_id') == env.focal for e in game.event_log),
                decisions=decisions, simulated_seconds=simulated, action_counts=dict(actions),
                option_ends=dict(ends), mean_policy_entropy=float(np.mean(entropies)) if entropies else None)
