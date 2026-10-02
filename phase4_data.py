"""TRUSTED experiment orchestrator. Features precede the separate label join.

    Actor path: game.observe -> ActorMemory -> encode -> X
    Label path: role_labels(game.snapshot()) -> y

Seeds, colors, families and outcomes live only in audit metadata, never in X.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
from time import perf_counter
import numpy as np

from belief.features import FEATURE_NAMES, FEATURE_SCHEMA, VIEWS, current_memory, encode
from belief.memory import ActorMemory
from social_deduction.actor import PlayerSeen, Role
from social_deduction.training import role_labels
from phase3_bots import BotStyle, CREW_VARIANTS, MEETING_VARIANTS
from phase3_runner import ScriptedMatch, behavior_seed
from phase4_scripts import StudyController

ROOT = Path(__file__).resolve().parent
DATA_SCHEMA = 'belief-dataset-v1'
SPLITS = {'train': (10000, 800), 'validation': (20000, 200),
          'test': (30000, 200), 'heldout': (40000, 200)}
QUICK_SPLITS = {'train': (900000, 20), 'validation': (901000, 6),
                'test': (902000, 6), 'heldout': (903000, 6)}
_planner = None


def source_fingerprint():
    paths = ['phase4_data.py', 'phase4_scripts.py', 'belief/memory.py', 'belief/features.py',
             'phase3_engine.py', 'phase3_bots.py', 'phase3_runner.py', 'navigation_service.py',
             'among_us_map.py', 'among_us_map_simulation.py', 'social_deduction/actor.py',
             'social_deduction/phase3_api.py', 'social_deduction/phase3_observation.py',
             'social_deduction/truth.py', 'social_deduction/training.py']
    # Normalize line endings so Git checkout settings do not invalidate the cache.
    return {p: sha256((ROOT / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}


def study_match(seed, heldout=False, planner=None):
    match = ScriptedMatch(seed=seed, planner=planner)
    families = ('patient',) if heldout else ('hunter', 'self_report')
    for pid in match.controllers:
        seed_value = behavior_seed(seed, pid)
        rng = Random(seed_value ^ 0xB311EF)
        style = BotStyle(rng.choice(CREW_VARIANTS), rng.choice(families), rng.choice(MEETING_VARIANTS))
        match.controllers[pid] = StudyController(seed_value, style)
    return match


def generate_match(seed, heldout=False):
    global _planner
    if _planner is None:
        from among_us_map import get_map
        from navigation_service import NavigationPlanner
        _planner = NavigationPlanner(get_map(), cache_size=96)
    started = perf_counter()
    match = study_match(seed, heldout, _planner)
    # Privileged label extraction is confined to this module. It is not passed to
    # controllers, memory, encoders, inference or actor-visible debug panels.
    labels = dict(role_labels(match.game.snapshot()).roles)
    focal = tuple(pid for pid, role in labels.items() if role is Role.CREWMATE)
    impostor = next(pid for pid, role in labels.items() if role is Role.IMPOSTOR)
    memories = {pid: ActorMemory() for pid in focal}
    values = {v: [] for v in VIEWS}
    targets, ticks, actors, stages, reasons = [], [], [], [], []
    last_sample = {pid: -10000 for pid in focal}
    events = Counter()
    colors = {i.player_id: i.display_name for i in match.game.observe(focal[0]).roster}
    candidates = None
    while not match.game.is_terminal:
        if match.game.tick % match.decision_ticks == 0:
            for pid in focal:
                obs = match.game.observe(pid)
                if not obs.own.active:
                    continue
                memory = memories[pid]
                previous_tick = memory.tick
                added = memory.update(obs)
                meaningful = [e for e in added if type(e.payload) is not PlayerSeen]
                due = (obs.tick - last_sample[pid]) * obs.settings.dt >= 6 - 1e-6
                if not meaningful and not due:
                    continue
                now = current_memory(obs, previous_tick)
                # These functions receive no labels, seed, roles, game or controller.
                values['full'].append(encode(memory))
                values['current'].append(encode(now))
                values['no_claims'].append(encode(memory, view='no_claims'))
                values['collapsed'].append(encode(memory, view='collapsed'))
                candidates = memory.candidates
                targets.append(candidates.index(impostor))
                ticks.append(obs.tick)
                actors.append(pid)
                stages.append(obs.context.phase.value)
                reason = '+'.join(sorted({type(e.payload).__name__ for e in meaningful})) or 'periodic'
                reasons.append(reason)
                events.update(type(e.payload).__name__ for e in meaningful)
                last_sample[pid] = obs.tick
        match.step()
    if match.game.metrics['illegal_actions'] or match.game.metrics['navigation_failures']:
        raise RuntimeError(f'Invalid generated match {seed}: {match.game.metrics}')
    if match.game.result.reason == 'timeout':
        raise RuntimeError(f'Timeout in dataset match {seed}')
    arrays = {f'x_{v}': np.stack(values[v]) for v in VIEWS}
    arrays.update(y=np.asarray(targets, dtype=np.int64), tick=np.asarray(ticks),
                  actor=np.asarray(actors), stage=np.asarray(stages), reason=np.asarray(reasons),
                  match=np.full(len(targets), seed, dtype=np.int64))
    metadata = {'seed': seed, 'samples': len(targets), 'focal_actors': len(focal),
                'heldout': heldout, 'impostor_id': impostor, 'colors': colors,
                'styles': {pid: asdict(c.style) for pid, c in match.controllers.items()},
                'impostor_style': match.controllers[impostor].style.impostor,
                'result': asdict(match.game.result), 'metrics': match.game.metrics,
                'sample_reasons': dict(Counter(reasons)), 'sample_events': dict(events),
                'wall_seconds': perf_counter() - started,
                'trajectory_hash': match.trajectory_hash}
    return arrays, metadata


def _worker(seed, heldout, path):
    arrays, metadata = generate_match(seed, heldout)
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def generate_dataset(directory, quick=False, workers=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    splits = QUICK_SPLITS if quick else SPLITS
    config = {'schema': DATA_SCHEMA, 'feature_schema': FEATURE_SCHEMA,
              'features': list(FEATURE_NAMES), 'sources': source_fingerprint(),
              'splits': {k: list(v) for k, v in splits.items()}, 'quick': quick}
    config_path = directory / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Dataset cache/schema/source mismatch; choose a fresh --data-dir')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    pending = []
    for split, (start, count) in splits.items():
        (directory / split).mkdir(exist_ok=True)
        for seed in range(start, start + count):
            path = directory / split / f'{seed}.npz'
            if not path.exists() or not path.with_suffix('.json').exists():
                pending.append((seed, split == 'heldout', str(path)))
    started = perf_counter()
    print(f'Dataset: {len(pending)} pending matches, {workers} workers', flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, *job): job[0] for job in pending}
        for future in as_completed(futures):
            result = future.result()  # Any failure stops acceptance; never silently discard it.
            completed += 1
            if completed % 20 == 0 or completed == len(pending):
                print(f'Generated {completed}/{len(pending)}; elapsed {perf_counter()-started:.1f}s', flush=True)
    report = {'schema': DATA_SCHEMA, 'config': config, 'this_invocation_seconds': perf_counter()-started,
              'splits': {}, 'worker_count': workers, 'generation_failures': 0}
    for split in splits:
        rows = [json.loads(p.read_text()) for p in sorted((directory / split).glob('*.json'))]
        reasons, event_counts, outcomes, styles, role_ids, role_colors = (Counter() for _ in range(6))
        for r in rows:
            reasons.update(r['sample_reasons']); event_counts.update(r['sample_events'])
            outcomes[r['result']['reason']] += 1
            styles[r['impostor_style']] += 1
            role_ids[r['impostor_id']] += 1
            role_colors[r['colors'][r['impostor_id']]] += 1
        report['splits'][split] = {'matches': len(rows), 'samples': sum(r['samples'] for r in rows),
            'focal_actors': sum(r['focal_actors'] for r in rows), 'reasons': dict(reasons),
            'events': dict(event_counts), 'outcomes': dict(outcomes), 'impostor_families': dict(styles),
            'role_ids': dict(role_ids), 'role_colors': dict(role_colors),
            'worker_seconds': sum(r['wall_seconds'] for r in rows)}
    report['disk_bytes'] = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    (directory / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def load_split(directory, split):
    arrays = {}
    for path in sorted((Path(directory) / split).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    if not arrays:
        raise FileNotFoundError(f'No {split} data. Run python train_phase4.py')
    return {key: np.concatenate(value) for key, value in arrays.items()}
