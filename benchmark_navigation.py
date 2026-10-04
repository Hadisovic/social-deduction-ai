"""Physical Phase 2 acceptance benchmark. Run with --help for reproducible subsets."""
import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import shapely

from among_us_map import get_map
from among_us_map_simulation import AmongUsMapEnv
from navigation_service import ControllerConfig, NavigationPlanner, NavigationService, NavStatus, NavTarget

ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20261002


@dataclass(frozen=True)
class Case:
    case_id: str
    suite: str
    source: str
    source_region: str
    start: tuple[float, float]
    destination: str
    spawn_band: str
    selection: str


def generate_cases(world, seed=DEFAULT_SEED, random_per_region=48):
    """63*62 directed pairs plus reproducible continuous room/corridor spawns.

    Half of random cases start within 0.12 units of the legal-center boundary;
    half are uniform over regional grid cells. Sub-cell jitter avoids testing
    only graph nodes. Targets alternate nearest, farthest and uniformly drawn
    consoles by Euclidean distance (not asserted shortest-route distance).
    """
    if random_per_region < 0:
        raise ValueError('random_per_region must be nonnegative')
    cases = []
    for destination in world.destinations:
        for source in world.destinations:
            if source.id != destination.id:
                cases.append(Case(f'pair:{source.id}->{destination.id}', 'pairs', source.id,
                                  source.room, source.standing, destination.id, 'standing', 'exhaustive'))
    rng = np.random.default_rng(seed)
    points = shapely.points(world.nodes)
    boundary_distance = shapely.distance(points, world.center_domain.boundary)
    console_positions = np.asarray([d.position for d in world.destinations])
    for name, shape in world.regions:
        indices = np.flatnonzero(shapely.covers(shape, points))
        if not len(indices):
            raise ValueError(f'No legal spawn nodes for {name}')
        near = indices[boundary_distance[indices] <= .12]
        if random_per_region and not len(near):
            raise ValueError(f'No near-boundary spawn stratum for {name}')
        for i in range(random_per_region):
            band = 'near_boundary' if i % 2 else 'uniform'
            pool = near if i % 2 else indices
            node = world.nodes[int(rng.choice(pool))]
            for _ in range(100):
                start = tuple(node + rng.uniform(-world.cell_size / 2, world.cell_size / 2, 2))
                if (world.contains(start) and shape.covers(shapely.Point(start)) and
                        world.segment_clear(node, start) and
                        (band != 'near_boundary' or
                         world.center_domain.boundary.distance(shapely.Point(start)) <= .12)):
                    break
            else:
                raise RuntimeError(f'Failed to jitter spawn in {name}')
            distances = np.linalg.norm(console_positions - start, axis=1)
            selection = ('nearest', 'farthest', 'uniform')[i % 3]
            target_index = (int(np.argmin(distances)) if selection == 'nearest' else
                            int(np.argmax(distances)) if selection == 'farthest' else
                            int(rng.integers(len(world.destinations))))
            cases.append(Case(f'random:{name}:{i:03d}', 'random', f'spawn:{name}:{i}', name,
                              start, world.destinations[target_index].id, band, selection))
    # Amortize fields across all starts for a destination; cache is still bounded.
    return sorted(cases, key=lambda c: (c.destination, c.suite, c.case_id))


def execute_case(env, planner, case, *, seed=DEFAULT_SEED, config=ControllerConfig(), dt=1/30):
    destination = planner.destinations[case.destination]
    env.spawn(case.start, destination.standing)
    navigator = NavigationService(planner, config)
    target = NavTarget.task(destination.id)
    navigator.navigate(env.position, target)
    region = planner.arrival_region(target)
    controller_seconds = physics_seconds = 0.0
    invariant_failure = ''
    began = perf_counter()
    # Safety bound also catches a command/feedback protocol loop that consumes no time.
    iteration_limit = math.ceil(config.timeout_seconds / dt) + config.max_replans + 10
    for _ in range(iteration_limit):
        if navigator.status is not NavStatus.MOVING:
            break
        old = env.position
        stamp = perf_counter()
        displacement = navigator.command(old, dt)
        controller_seconds += perf_counter() - stamp
        if not navigator.awaiting_feedback:
            continue
        stamp = perf_counter()
        new, hit = env.advance_motion(displacement, dt)
        physics_seconds += perf_counter() - stamp
        # Independent execution assertions, in addition to planner validation.
        if not env.map.contains(new) or not env.map.segment_clear(old, new):
            invariant_failure = 'illegal_physical_segment'
            break
        if math.dist(old, new) > config.speed * dt + 1e-8:
            invariant_failure = 'speed_limit_violation'
            break
        stamp = perf_counter()
        navigator.feedback(new, hit)
        controller_seconds += perf_counter() - stamp
    else:
        invariant_failure = 'benchmark_iteration_limit'
    execution_seconds = perf_counter() - began
    success = (not invariant_failure and navigator.status is NavStatus.SUCCESS and
               region.contains(env.position) and
               math.dist(env.position, destination.position) <= destination.radius + 1e-9)
    reason = '' if success else invariant_failure or navigator.status.value
    if not success and reason == 'success':
        reason = 'arrival_assertion_failed'
    efficiency = (navigator.planned_length / navigator.travelled if navigator.travelled else
                  1.0 if navigator.planned_length == 0 else 0.0)
    return {
        **asdict(case), 'start': json.dumps(case.start), 'seed': seed,
        'destination_name': destination.name, 'destination_room': destination.room,
        'destination_category': destination.category, 'success': success,
        'failure_reason': reason, 'status': navigator.status.value,
        'planning_seconds': navigator.planning_seconds, 'field_cache_hit': navigator.initial_plan.cache_hit,
        'execution_wall_seconds': execution_seconds, 'execution_sim_seconds': navigator.elapsed,
        'controller_wall_seconds': controller_seconds, 'physics_wall_seconds': physics_seconds,
        'planned_route_length': navigator.planned_length, 'actual_distance': navigator.travelled,
        'path_efficiency': efficiency, 'collisions': navigator.collisions,
        'blocked_steps': navigator.blocked_steps, 'replans': navigator.replans,
        'final_region_distance': region.distance(env.position),
        'final_console_distance': math.dist(env.position, destination.position),
        'final_position': json.dumps(env.position), 'steps': navigator.steps,
        'last_replan_reason': navigator.last_replan_reason or '',
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {key: float(value) for key, value in zip(
        ('mean', 'p50', 'p95', 'max'),
        (np.mean(values), *np.percentile(values, [50, 95]), np.max(values)))} if len(values) else {}


def summarize(rows):
    groups = {}
    for attribute in ('suite', 'source_region', 'destination_room', 'destination_category', 'spawn_band', 'selection'):
        groups[attribute] = {}
        for value in sorted({r[attribute] for r in rows}):
            subset = [r for r in rows if r[attribute] == value]
            passed = sum(r['success'] for r in subset)
            groups[attribute][value] = {'cases': len(subset), 'successes': passed,
                                      'success_rate': passed / len(subset)}
    failed = [r for r in rows if not r['success']]
    categories = {reason: sum(r['failure_reason'] == reason for r in failed)
                  for reason in sorted({r['failure_reason'] for r in failed})}
    steps = sum(r['steps'] for r in rows)
    return {
        'cases': len(rows), 'successes': len(rows) - len(failed),
        'success_rate': (len(rows) - len(failed)) / len(rows),
        'failure_categories': categories, 'failure_cases': [r['case_id'] for r in failed],
        'groups': groups, 'collisions': sum(r['collisions'] for r in rows),
        'blocked_steps': sum(r['blocked_steps'] for r in rows),
        'replans': sum(r['replans'] for r in rows), 'steps': steps,
        'planning_seconds': distribution([r['planning_seconds'] for r in rows]),
        'cold_planning_seconds': distribution([r['planning_seconds'] for r in rows if not r['field_cache_hit'] and r['steps']]),
        'cached_planning_seconds': distribution([r['planning_seconds'] for r in rows if r['field_cache_hit']]),
        'execution_sim_seconds': distribution([r['execution_sim_seconds'] for r in rows]),
        'execution_wall_seconds': distribution([r['execution_wall_seconds'] for r in rows]),
        'planned_route_length': distribution([r['planned_route_length'] for r in rows]),
        'path_efficiency': distribution([r['path_efficiency'] for r in rows]),
        'controller_microseconds_per_step': sum(r['controller_wall_seconds'] for r in rows) * 1e6 / max(1, steps),
        'physics_microseconds_per_step': sum(r['physics_wall_seconds'] for r in rows) * 1e6 / max(1, steps),
        'longest_routes': [{k: r[k] for k in ('case_id', 'source_region', 'destination_room', 'planned_route_length', 'execution_sim_seconds', 'success')}
                           for r in sorted(rows, key=lambda r: r['planned_route_length'], reverse=True)[:10]],
    }


def write_results(directory, rows, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'cases.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 9) if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    summary = {**metadata, **summarize(rows)}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Phase 2 physical navigation benchmark', '',
             f"{summary['successes']}/{summary['cases']} successful ({summary['success_rate']:.2%}).",
             f"Gate: **{summary['gate']}**. Seed: `{metadata['seed']}`.", '',
             '| Suite | Successes | Cases | Rate |', '|---|---:|---:|---:|']
    for name, counts in summary['groups']['suite'].items():
        lines.append(f"| {name} | {counts['successes']} | {counts['cases']} | {counts['success_rate']:.2%} |")
    lines += ['', f"Collisions: {summary['collisions']}; blocked steps: {summary['blocked_steps']}; replans: {summary['replans']}.",
              f"Failures: `{summary['failure_categories']}`.",
              f"Measured run time (map/planner setup, generation, execution): {metadata['run_wall_seconds']:.2f} s.",
              f"Planning mean / p95: {summary['planning_seconds']['mean']*1000:.2f} / {summary['planning_seconds']['p95']*1000:.2f} ms.",
              f"Controller / physics: {summary['controller_microseconds_per_step']:.2f} / {summary['physics_microseconds_per_step']:.2f} microseconds per executed step.", '',
              'Every step is speed-bounded, swept-clear and applied by `AmongUsMapEnv.advance_motion`.',
              'Path efficiency is planned executable length / actual travel; it is not a global optimality claim.',
              'See `summary.json` for region groups, longest routes, configuration, source hashes and timing distributions.',
              'See `cases.csv` for all individual outcomes. Wall times vary across machines.', '']
    (directory / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--random-per-region', type=int, default=48)
    parser.add_argument('--suite', choices=('all', 'pairs', 'random'), default='all')
    parser.add_argument('--case', help='Exact case ID from cases.csv; quote it in the shell')
    parser.add_argument('--max-cases', type=int, help='Smoke subset; never reports a full gate pass')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/benchmarks/phase2')
    parser.add_argument('--dt', type=float, default=1/30, help='Physical execution timestep')
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error('--dt must be positive and finite')
    began = perf_counter()
    env = AmongUsMapEnv()
    stamp = perf_counter()
    planner = NavigationPlanner(env.map)
    setup_seconds = perf_counter() - stamp
    cases = generate_cases(env.map, args.seed, args.random_per_region)
    if args.suite != 'all':
        cases = [c for c in cases if c.suite == args.suite]
    if args.case:
        cases = [c for c in cases if c.case_id == args.case]
    if args.max_cases is not None:
        cases = cases[:max(0, args.max_cases)]
    if not cases:
        parser.error('No matching cases')
    config = ControllerConfig()
    rows = []
    previous = None
    for case in cases:
        if case.destination != previous:
            if rows:
                print(f"{len(rows)}/{len(cases)} executed; failures={sum(not r['success'] for r in rows)}; wall={perf_counter()-began:.1f}s", flush=True)
            previous = case.destination
        rows.append(execute_case(env, planner, case, seed=args.seed, config=config, dt=args.dt))
    full = args.suite == 'all' and not args.case and args.max_cases is None and args.random_per_region >= 48
    clean_outcomes = all(r['success'] for r in rows)
    source_paths = ['navigation_service.py', 'benchmark_navigation.py', 'among_us_map.py',
                    'among_us_map_simulation.py', 'assets/skeld/among_us_map.json']
    metadata = {
        'gate': 'PASS' if full and clean_outcomes else 'FAIL' if not clean_outcomes else 'SUBSET_ONLY',
        'gate_rule': 'Complete directed pairs plus >=48 spawns per region, zero unexplained failures; all execution assertions pass.',
        'seed': args.seed, 'random_per_region': args.random_per_region,
        'destination_count': len(env.map.destinations), 'region_count': len(env.map.regions),
        'dt': args.dt, 'controller_config': asdict(config), 'player_radius': env.map.radius,
        'clearance_policy': env.map.clearance_policy, 'buffer_radius': env.map.buffer_radius,
        'grid_cell_size': env.map.cell_size, 'cache_size': planner.cache_size,
        'field_builds': planner.field_builds, 'field_build_seconds': planner.field_build_seconds,
        'planner_setup_seconds': setup_seconds, 'run_wall_seconds': perf_counter() - began,
        'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'shapely': shapely.__version__,
        'base_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        'case_manifest_sha256': hashlib.sha256(json.dumps([asdict(c) for c in cases], sort_keys=True).encode()).hexdigest(),
    }
    result = write_results(args.output, rows, metadata)
    env.close()
    print(f"{result['gate']}: {result['successes']}/{result['cases']} — {args.output}", flush=True)
    return 0 if clean_outcomes else 1


if __name__ == '__main__':
    raise SystemExit(main())
