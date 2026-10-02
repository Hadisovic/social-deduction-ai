"""Service contracts and failure injection, separate from the exhaustive benchmark."""
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from among_us_map import AmongUsMap, Destination, DIRECTIONS, get_map
from among_us_map_simulation import AmongUsMapEnv
from benchmark_navigation import Case, execute_case, generate_cases, summarize
from navigation_service import ControllerConfig, NavigationPlanner, NavigationService, NavStatus, NavTarget


@pytest.fixture(scope='module')
def planner():
    return NavigationPlanner(get_map())


def task_in(planner, room):
    return NavTarget.task(next(d.id for d in planner.map.destinations if d.room == room))


def execute(env, navigator, dt=1/30):
    for _ in range(3000):
        if navigator.status is not NavStatus.MOVING:
            return navigator.status
        before = env.position
        command = navigator.command(before, dt)
        if navigator.awaiting_feedback:
            new, hit = env.advance_motion(command, dt)
            assert math.dist(before, new) <= navigator.config.speed * dt + 1e-9
            assert env.map.segment_clear(before, new)
            navigator.feedback(new, hit)
    pytest.fail('Navigation did not terminate')


def test_all_grid_edges_and_rejected_diagonals_agree_with_swept_geometry(planner):
    world = planner.map
    gy, gx = np.nonzero(world.grid)
    rejected = 0
    for k, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = gx + dx, gy + dy
        mask = (nx >= 0) & (ny >= 0) & (nx < len(world.xs)) & (ny < len(world.ys))
        indices = np.flatnonzero(mask)
        indices = indices[world.grid[ny[indices], nx[indices]]]
        a = np.column_stack([world.xs[gx[indices]], world.ys[gy[indices]]])
        b = np.column_stack([world.xs[nx[indices]], world.ys[ny[indices]]])
        actual = shapely.covers(world.center_domain, shapely.linestrings(np.stack([a, b], axis=1)))
        assert np.array_equal(actual, world.edges[gy[indices], gx[indices], k])
        if dx and dy:
            rejected += np.count_nonzero(~actual)
    assert rejected > 0  # Two legal endpoints do not imply a legal diagonal.


@pytest.mark.parametrize('room', ['Electrical', 'MedBay', 'Security', 'Reactor', 'Upper Engine',
                                  'Lower Engine', 'Admin', 'Storage', 'Cafeteria', 'Weapons',
                                  'O2', 'Navigation', 'Shields', 'Communications'])
def test_physical_task_arrival_no_teleport_and_full_clearance(planner, room):
    env = AmongUsMapEnv()
    env.spawn((-.7, -2.8))
    nav = NavigationService(planner)
    target = task_in(planner, room)
    original = env.position
    nav.navigate(original, target)
    assert env.position == original
    assert all(planner.map.segment_clear(a, b) for a, b in zip(nav.route, nav.route[1:]))
    assert execute(env, nav) is NavStatus.SUCCESS
    assert planner.arrival_region(target).contains(env.position)
    assert nav.collisions == nav.replans == 0
    assert nav.travelled == pytest.approx(nav.planned_length, abs=1e-6)
    # Physical map uses a polygonal approximation of circle dilation. Respect
    # its documented chord-error bound when checking Euclidean wall distance.
    error = env.map.radius * (1 - math.cos(math.pi / 48))
    assert shapely.Point(env.position).distance(env.map.barriers) >= env.map.radius - error - 1e-8


def test_arrival_does_not_require_exact_console_or_standing_center(planner):
    target = task_in(planner, 'Electrical')
    dest = planner.destinations[target.key]
    plan = planner.plan((-.7, -2.8), target)
    endpoint = plan.route[-1]
    assert math.dist(endpoint, dest.position) <= dest.radius
    assert math.dist(endpoint, dest.standing) > .1
    nav = NavigationService(planner)
    assert nav.navigate(endpoint, target) is NavStatus.SUCCESS
    assert nav.command(endpoint, 1/30) == (0, 0)
    assert nav.steps == 0 and not nav.awaiting_feedback


@pytest.mark.parametrize('target', [NavTarget.room('Electrical'), NavTarget.location((-.7, -3.5)),
                                  NavTarget.location((-.711, -3.511), radius=.0001)])
def test_room_location_and_subgrid_location_targets(planner, target):
    env = AmongUsMapEnv(); env.spawn((-.7, -2.8))
    nav = NavigationService(planner); nav.navigate(env.position, target)
    assert execute(env, nav) is NavStatus.SUCCESS
    assert planner.arrival_region(target).contains(env.position)


@pytest.mark.parametrize('target', [NavTarget.task('missing'), NavTarget.room('missing'),
                                  NavTarget.location((100, 100)), NavTarget.location((-.7, -2.8), 0),
                                  NavTarget.location((float('nan'), 0)), NavTarget('unknown')])
def test_invalid_destination_is_explicit(planner, target):
    nav = NavigationService(planner)
    assert nav.navigate((-.7, -2.8), target) is NavStatus.INVALID_DESTINATION
    assert nav.route == () and nav.command((-.7, -2.8), .1) == (0, 0)


@pytest.mark.parametrize('start', [(100, 100), (float('inf'), 0), (1,), None])
def test_invalid_start_is_explicit(planner, start):
    assert planner.plan(start, task_in(planner, 'Electrical')).status is NavStatus.INVALID_START


def test_cancel_and_replace_clear_stale_route_without_moving_player(planner):
    env = AmongUsMapEnv(); env.spawn((-.7, -2.8))
    nav = NavigationService(planner)
    nav.navigate(env.position, task_in(planner, 'Electrical'))
    command = nav.command(env.position, env.dt)
    new, hit = env.advance_motion(command, env.dt); nav.feedback(new, hit)
    stopped = env.position
    assert nav.cancel() is NavStatus.CANCELLED
    assert nav.command(stopped, env.dt) == (0, 0)
    assert nav.route == () and not nav.awaiting_feedback and env.position == stopped
    target = task_in(planner, 'Storage')
    nav.navigate(stopped, target)
    assert nav.elapsed == nav.steps == nav.replans == 0
    assert execute(env, nav) is NavStatus.SUCCESS
    # Replacement is also valid without a separate cancel call.
    nav.navigate(env.position, task_in(planner, 'Electrical'))
    nav.navigate(env.position, task_in(planner, 'MedBay'))
    assert execute(env, nav) is NavStatus.SUCCESS
    assert nav.target == task_in(planner, 'MedBay')


def test_known_closed_door_replan_reports_no_route(planner):
    closed = NavigationPlanner(AmongUsMap(closed_doors=['11:5']))
    nav = NavigationService(planner)
    nav.navigate((-.7, -2.8), task_in(planner, 'Electrical'))
    assert nav.status is NavStatus.MOVING
    assert nav.replan((-.7, -2.8), planner=closed, reason='observed_door_closed') is NavStatus.NO_ROUTE
    assert nav.replans == 1 and nav.last_replan_reason == 'observed_door_closed'
    assert nav.command((-.7, -2.8), .1) == (0, 0)


def test_requested_replan_from_actual_position_succeeds(planner):
    env = AmongUsMapEnv(); env.spawn((-.7, -2.8))
    nav = NavigationService(planner)
    nav.navigate(env.position, task_in(planner, 'Electrical'))
    delta = nav.command(env.position, env.dt)
    pos, hit = env.advance_motion(delta, env.dt); nav.feedback(pos, hit)
    assert nav.replan(pos) is NavStatus.MOVING
    assert nav.route[0] == pos and nav.replans == 1
    assert execute(env, nav) is NavStatus.SUCCESS


def test_persistent_blockage_stops_after_bounded_deterministic_recovery(planner):
    nav = NavigationService(planner, ControllerConfig(stuck_seconds=.2, max_replans=2))
    position = (-.7, -2.8)
    nav.navigate(position, task_in(planner, 'Electrical'))
    for _ in range(6):
        nav.command(position, .1)
        nav.feedback(position, collided=True)  # Inject an unobserved physical blockage.
    assert nav.status is NavStatus.STUCK
    assert nav.replans == 2 and nav.steps == 6 and nav.blocked_steps == nav.collisions == 6
    assert nav.last_replan_reason == 'no_progress'
    assert nav.command(position, .1) == (0, 0)


def test_timeout_is_explicit_even_if_some_progress_occurs(planner):
    env = AmongUsMapEnv(); env.spawn((-.7, -2.8))
    nav = NavigationService(planner, ControllerConfig(timeout_seconds=.1))
    nav.navigate(env.position, task_in(planner, 'Electrical'))
    assert execute(env, nav) is NavStatus.TIMEOUT
    assert nav.steps == 3 and nav.travelled > 0


def test_temporary_blockage_recovers_without_teleportation(planner):
    env = AmongUsMapEnv(); env.spawn((-.7, -2.8))
    nav = NavigationService(planner, ControllerConfig(stuck_seconds=.1))
    nav.navigate(env.position, task_in(planner, 'Electrical'))
    nav.command(env.position, .1); nav.feedback(env.position, collided=True)
    assert nav.replans == 1
    assert execute(env, nav) is NavStatus.SUCCESS
    assert nav.travelled == pytest.approx(nav.planned_length)


def test_feedback_protocol_and_speed_limits(planner):
    env = AmongUsMapEnv(); env.spawn((-.7, -2.8))
    nav = NavigationService(planner); nav.navigate(env.position, task_in(planner, 'Electrical'))
    with pytest.raises(RuntimeError): nav.feedback(env.position)
    with pytest.raises(ValueError): nav.command(env.position, 0)
    nav.command(env.position, .1)
    with pytest.raises(RuntimeError): nav.command(env.position, .1)
    with pytest.raises(ValueError): nav.feedback((20, 20))
    with pytest.raises(ValueError): env.advance_motion((10, 0), .1)
    with pytest.raises(ValueError): env.advance_motion((float('nan'), 0), .1)


def test_gym_step_and_service_primitive_use_identical_physics():
    gym_env = AmongUsMapEnv(); motion_env = AmongUsMapEnv()
    gym_env.reset(seed=7, options={'start': (-.7, -2.8), 'goal': (-.7, -4)})
    motion_env.spawn((-.7, -2.8), (-.7, -4))
    gym_env.step((0, 1))
    motion_env.advance_motion((0, -motion_env.speed * motion_env.dt), motion_env.dt)
    assert gym_env.position == motion_env.position
    assert gym_env._elapsed == motion_env._elapsed
    assert gym_env._total_path_length == motion_env._total_path_length


def test_region_excludes_other_side_of_wall_even_within_radius():
    # Small constructed geometry isolates the approach-side arrival invariant.
    world = object.__new__(AmongUsMap)
    world.radius = .2; world.cell_size = .1
    world.center_domain = box(-2, -2, 2, 2).difference(box(-.1, -1, .1, 1))
    world.regions = [('Test', box(-2, -2, 2, 2))]
    world._build_grid()
    world.destinations = (Destination('test', 'test', 'Test', (.1, 0), (.4, 0), .8, 'task', 'fixture'),)
    planner = NavigationPlanner(world)
    region = planner.arrival_region(NavTarget.task('test'))
    assert math.dist((-.4, 0), (.1, 0)) < .8
    assert not region.contains((-.4, 0)) and region.contains((.4, 0))
    plan = planner.plan((-.4, 0), NavTarget.task('test'))
    assert plan.status is NavStatus.MOVING
    assert any(abs(p[1]) >= 1 for p in plan.route)  # Must round an end of the wall.
    assert all(world.segment_clear(a, b) for a, b in zip(plan.route, plan.route[1:]))


def test_oscillation_cannot_reset_stuck_timer_forever(planner):
    nav = NavigationService(planner, ControllerConfig(stuck_seconds=.3, max_replans=0))
    start = np.array((-.7, -2.8))
    nav.navigate(start, task_in(planner, 'Electrical'))
    delta = np.asarray(nav.command(start, .1))
    advanced = tuple(start + delta * .1)
    nav.feedback(advanced)
    for position, next_position in [(advanced, tuple(start)), (tuple(start), advanced), (advanced, tuple(start))]:
        nav.command(position, .1)
        nav.feedback(next_position)
    assert nav.status is NavStatus.STUCK and nav.replans == 0


def test_displacement_invalidates_route_and_replans_from_measured_position(planner):
    nav = NavigationService(planner)
    start = (-.7, -2.8)
    nav.navigate(start, task_in(planner, 'Electrical'))
    # Simulate a legitimate external movement source between control ticks.
    # The service must discard its now-obstructed connector instead of forcing it.
    displaced = next(tuple(p) for p in planner.map.nodes[::100]
                     if not planner.map.segment_clear(p, nav.route[1]))
    nav.command(displaced, .1)
    assert nav.replans == 1 and nav.last_replan_reason == 'route_invalidated'
    assert nav.route[0] == displaced and not nav.awaiting_feedback


def test_target_cache_is_bounded_and_reused(planner):
    small = NavigationPlanner(planner.map, cache_size=1)
    target = task_in(small, 'Electrical')
    first = small.plan((-.7, -2.8), target)
    second = small.plan((-.7, -2.8), target)
    assert first.route == second.route and not first.cache_hit and second.cache_hit
    small.plan((-.7, -2.8), task_in(small, 'MedBay'))
    assert len(small._fields) == len(small._regions) == 1
    assert not small.plan((-.7, -2.8), target).cache_hit


def test_benchmark_manifest_coverage_and_determinism(planner):
    a = generate_cases(planner.map, random_per_region=6)
    b = generate_cases(planner.map, random_per_region=6)
    assert a == b
    changed = generate_cases(planner.map, seed=99, random_per_region=6)
    assert [c for c in a if c.suite == 'pairs'] == [c for c in changed if c.suite == 'pairs']
    assert [c for c in a if c.suite == 'random'] != [c for c in changed if c.suite == 'random']
    assert len({c.case_id for c in a}) == len(a)
    assert sum(c.suite == 'pairs' for c in a) == 63 * 62
    random_cases = [c for c in a if c.suite == 'random']
    for name, shape in planner.map.regions:
        subset = [c for c in random_cases if c.source_region == name]
        assert len(subset) == 6
        assert sum(c.spawn_band == 'near_boundary' for c in subset) == 3
        assert {c.selection for c in subset} == {'nearest', 'farthest', 'uniform'}
        for case in subset:
            assert planner.map.contains(case.start) and shape.covers(shapely.Point(case.start))


def test_benchmark_executes_and_records_reproducible_outcomes(planner):
    target = task_in(planner, 'Electrical')
    case = Case('test', 'random', 'spawn', 'Cafeteria', (-.7, -2.8), target.key, 'uniform', 'uniform')
    env = AmongUsMapEnv()
    a = execute_case(env, planner, case)
    b = execute_case(env, planner, case)
    for field in ('success', 'status', 'failure_reason', 'steps', 'collisions', 'replans', 'final_position',
                  'actual_distance', 'planned_route_length', 'execution_sim_seconds'):
        assert a[field] == b[field]
    assert a['success'] and a['steps'] > 0
    summary = summarize([a, b])
    assert summary['success_rate'] == 1 and summary['failure_categories'] == {}
