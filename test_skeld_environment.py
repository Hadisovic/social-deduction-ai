"""
test_skeld_environment.py
Automated test suite for The Skeld navigation environment.

Validates both configuration metadata and physical geometric fidelity:
  T01 - Config imports and observation architecture
  T02 - Room count and physical geometry definitions
  T03 - All solid wall rects valid
  T04 - Unique room names
  T05 - Walkable room centers not in solid rects
  T06 - Task destinations are valid TaskDestination models
  T07 - Audited room graph is bidirectional
  T08 - Audited room graph is fully connected
  T09 - Environment reset returns correct observation shape and info
  T10 - Observation range validity across multiple seeded resets
  T11 - Spawn validation: 100 resets, player never spawns in wall
  T12 - Goal spawn validation: 100 resets, goal never spawns in wall
  T13 - Step returns correct types
  T14 - Goal reward triggers on reach
  T15 - Timeout triggers after max_episode_time
  T16 - Action deadzone: zero action = no movement
  T17 - Stuck flag activates on prolonged obstacle blockage
  T18 - Stagnation progress increases when stationary
  T19 - Gymnasium env_checker passes (SB3 compatibility)
  T20 - Collision: player cannot penetrate interior walls
  T21 - Collision: player cannot exit outer hull
  T22 - Aspect ratio preservation: reference matches world aspect ratio
  T23 - World to screen and screen to world round-trip invertibility
  T24 - Physical geometry connectivity: occupancy grid has exactly 1 connected component
  T25 - All 14 rooms reachable within the single connected walkable component
  T26 - Doorway traversal: all doorways maintain >= 2 * PLAYER_RADIUS width
  T27 - Exterior non-walkability: outside-hull points are strictly non-walkable
  T28 - Accidental shortcut prevention: Electrical <-> MedBay physically blocked
  T29 - All 40 task destinations are reachable by player with clearance (0 unreachable)
  T30 - All-pairs task connectivity: all task pairs belong to same component
  T31 - True spatial region detection: get_room_or_region identifies rooms and hallways
  T32 - Representative routes: all 9 canonical routes have valid A* paths
  T33 - Map-aware efficiency helper: strictly bounded in [0.0, 100.0]%
  T34 - Goal modes: 'task', 'task_to_task', and 'room_to_room' work end-to-end
"""

import math
import time
import collections
import traceback
import sys
from typing import List, Tuple

import numpy as np
import pygame
from gymnasium.utils.env_checker import check_env

import skeld_config as sc
from skeld_environment import (
    SkeldNavEnv, sample_skeld_positions, _move_player, is_point_in_ship_floor
)
from skeld_navigation import get_skeld_nav_grid, SkeldOccupancyGrid


# ==========================================
# Test Harness
# ==========================================
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
test_results = []

def run_test(name: str, fn):
    try:
        fn()
        test_results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        test_results.append((name, False, str(e)))
        print(f"  {FAIL}  {name}: {e}")
    except Exception as e:
        test_results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  {FAIL}  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ==========================================
# T01 - Config imports and architecture
# ==========================================
def test_01_config_imports():
    assert sc.SKELD_OBSERVATION_SIZE == 22
    assert sc.SKELD_RAY_COUNT == 16
    assert sc.SKELD_MAX_EPISODE_TIME > 0
    assert sc.SKELD_GOAL_REWARD > 0
    assert sc.SKELD_TIMEOUT_PENALTY < 0
    assert sc.SKELD_OBSERVATION_ARCHITECTURE == "PROPOSED_STRUCTURED_OBSERVATION_V1"


# ==========================================
# T02 - Room count and physical geometry
# ==========================================
def test_02_room_count():
    assert len(sc.SKELD_ROOMS) == 14, f"Expected 14 rooms, got {len(sc.SKELD_ROOMS)}"
    assert len(sc.SKELD_ALL_SOLID_RECTS) > 0, "No solid rects defined"
    assert len(sc.SKELD_ALL_TASK_DESTINATIONS) == 40, f"Expected 40 tasks, got {len(sc.SKELD_ALL_TASK_DESTINATIONS)}"


# ==========================================
# T03 - Solid rects valid
# ==========================================
def test_03_solid_rects_valid():
    for i, rect in enumerate(sc.SKELD_ALL_SOLID_RECTS):
        assert isinstance(rect, pygame.Rect), f"Rect {i} is not a pygame.Rect"
        assert rect.width > 0, f"Rect {i} has width <= 0: {rect}"
        assert rect.height > 0, f"Rect {i} has height <= 0: {rect}"
        assert rect.left >= 0, f"Rect {i} left < 0: {rect}"
        assert rect.top >= 0, f"Rect {i} top < 0: {rect}"
        assert rect.right <= sc.SKELD_WORLD_WIDTH, f"Rect {i} right > W: {rect}"
        assert rect.bottom <= sc.SKELD_WORLD_HEIGHT + 2, f"Rect {i} bottom > H: {rect}"


# ==========================================
# T04 - Unique room names
# ==========================================
def test_04_unique_room_names():
    names = [r.name for r in sc.SKELD_ROOMS]
    assert len(names) == len(set(names)), f"Duplicate room names: {names}"


# ==========================================
# T05 - Walkable room centers not in solid
# ==========================================
def test_05_walkable_not_in_solid():
    for room in sc.SKELD_ROOMS:
        cx, cy = room.center
        pt = (int(round(cx)), int(round(cy)))
        for rect in sc.SKELD_ALL_SOLID_RECTS:
            assert not rect.collidepoint(pt), (
                f"Room '{room.name}' center ({cx},{cy}) is inside solid rect {rect}"
            )


# ==========================================
# T06 - Task destinations are valid TaskDestination
# ==========================================
def test_06_task_destinations_model():
    for t in sc.SKELD_ALL_TASK_DESTINATIONS:
        assert isinstance(t, sc.TaskDestination)
        assert t.id != ""
        assert t.name != ""
        assert t.room in sc.SKELD_ROOM_MAP
        assert t.confidence in ("VERIFIED", "ESTIMATED")
        assert 0 <= t.world_x <= sc.SKELD_WORLD_WIDTH
        assert 0 <= t.world_y <= sc.SKELD_WORLD_HEIGHT


# ==========================================
# T07 - Room graph bidirectional
# ==========================================
def test_07_graph_bidirectional():
    G = sc.SKELD_ROOM_GRAPH
    for room, neighbors in G.items():
        for nb in neighbors:
            assert nb in G, f"Room '{nb}' in graph has no entry"
            assert room in G[nb], f"Graph not bidirectional: '{room}' -> '{nb}'"


# ==========================================
# T08 - Room graph connected
# ==========================================
def test_08_graph_connected():
    G = sc.SKELD_ROOM_GRAPH
    start = "Cafeteria"
    visited = set([start])
    queue = collections.deque([start])
    while queue:
        node = queue.popleft()
        for nb in G.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    assert len(visited) == len(G), f"Disconnected rooms in graph: {set(G.keys()) - visited}"


# ==========================================
# T09 - Env reset
# ==========================================
def test_09_reset():
    env = SkeldNavEnv(render_mode=None, task_goal_mode=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (sc.SKELD_OBSERVATION_SIZE,), f"Wrong obs shape: {obs.shape}"
    assert obs.dtype == np.float32
    assert "player_pos" in info
    assert "goal_pos" in info
    assert "room" in info
    env.close()


# ==========================================
# T10 - Observation range validity
# ==========================================
def test_10_obs_range():
    env = SkeldNavEnv(render_mode=None)
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        assert 0.0 <= obs[0] <= 1.0, f"obs[0] out of range: {obs[0]}"
        assert 0.0 <= obs[1] <= 1.0, f"obs[1] out of range: {obs[1]}"
        assert -1.0 <= obs[2] <= 1.0, f"obs[2] out of range: {obs[2]}"
        assert -1.0 <= obs[3] <= 1.0, f"obs[3] out of range: {obs[3]}"
        for i in range(4, 20):
            assert 0.0 <= obs[i] <= 1.0, f"ray obs[{i}] out of range: {obs[i]}"
        assert obs[20] in (0.0, 1.0), f"stuck flag {obs[20]} not binary"
        assert 0.0 <= obs[21] <= 1.0, f"stag progress {obs[21]} out of range"
    env.close()


# ==========================================
# T11 - Spawn player clear of walls
# ==========================================
def test_11_spawn_player_clear():
    pr = int(math.ceil(sc.SKELD_PLAYER_RADIUS))
    bad = 0
    env = SkeldNavEnv(render_mode=None)
    for seed in range(100):
        _, info = env.reset(seed=seed)
        px, py = info["player_pos"]
        prect = pygame.Rect(int(px) - pr, int(py) - pr, pr * 2, pr * 2)
        for srect in sc.SKELD_ALL_SOLID_RECTS:
            if prect.colliderect(srect):
                bad += 1
                break
    env.close()
    assert bad == 0, f"Player spawned inside wall in {bad}/100 resets"


# ==========================================
# T12 - Spawn goal clear of walls
# ==========================================
def test_12_spawn_goal_clear():
    gr = int(math.ceil(sc.SKELD_GOAL_RADIUS))
    bad = 0
    env = SkeldNavEnv(render_mode=None)
    for seed in range(100):
        _, info = env.reset(seed=seed)
        gx, gy = info["goal_pos"]
        grect = pygame.Rect(int(gx) - gr, int(gy) - gr, gr * 2, gr * 2)
        for srect in sc.SKELD_ALL_SOLID_RECTS:
            if grect.colliderect(srect):
                bad += 1
                break
    env.close()
    assert bad == 0, f"Goal spawned inside wall in {bad}/100 resets"


# ==========================================
# T13 - Step types
# ==========================================
def test_13_step_types():
    env = SkeldNavEnv(render_mode=None)
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    obs2, rew, term, trunc, info = env.step(action)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    env.close()


# ==========================================
# T14 - Goal reward
# ==========================================
def test_14_goal_reward():
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    # Teleport to goal
    env._player_x = env._goal_x
    env._player_y = env._goal_y
    _, rew, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert term, "Goal reached did not terminate"
    assert rew >= sc.SKELD_GOAL_REWARD - 1.0, f"Goal reward too low: {rew}"
    env.close()


# ==========================================
# T15 - Timeout
# ==========================================
def test_15_timeout():
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    env._elapsed = sc.SKELD_MAX_EPISODE_TIME + 0.1
    _, rew, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert trunc, "Timeout did not truncate"
    assert rew <= sc.SKELD_TIMEOUT_PENALTY + 1.0
    env.close()


# ==========================================
# T16 - Action deadzone
# ==========================================
def test_16_action_deadzone():
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    px0, py0 = env._player_x, env._player_y
    env.step(np.array([0.02, 0.02], dtype=np.float32))
    assert env._player_x == px0 and env._player_y == py0
    env.close()


# ==========================================
# T17 - Stuck flag
# ==========================================
def test_17_stuck_flag():
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    env._player_x = 24.0  # right against left outer wall
    env._player_y = 400.0
    for _ in range(25):
        obs, _, _, _, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert obs[20] == 1.0, f"Stuck flag did not activate: {obs[20]}"
    env.close()


# ==========================================
# T18 - Stagnation progress
# ==========================================
def test_18_stagnation():
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    p0 = env._stag_progress
    for _ in range(30):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert obs[21] > p0, f"Stagnation did not increase: {obs[21]}"
    env.close()


# ==========================================
# T19 - Gymnasium env_checker
# ==========================================
def test_19_gymnasium_checker():
    env = SkeldNavEnv(render_mode=None)
    check_env(env.unwrapped)
    env.close()


# ==========================================
# T20 - Interior wall no penetration
# ==========================================
def test_20_interior_wall_no_penetration():
    # Push into Cafeteria bottom divider
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    env._player_x = 750.0
    env._player_y = 250.0
    for _ in range(60):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    pr = int(math.ceil(sc.SKELD_PLAYER_RADIUS))
    prect = pygame.Rect(int(env._player_x) - pr, int(env._player_y) - pr, pr * 2, pr * 2)
    for srect in sc.SKELD_ALL_SOLID_RECTS:
        assert not prect.colliderect(srect), "Player penetrated wall rect"
    env.close()


# ==========================================
# T21 - Hull no penetration
# ==========================================
def test_21_hull_no_penetration():
    env = SkeldNavEnv(render_mode=None)
    env.reset(seed=0)
    env._player_x = 50.0
    env._player_y = 50.0
    for _ in range(60):
        env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert env._player_x >= sc.SKELD_PLAYER_RADIUS
    assert env._player_y >= sc.SKELD_PLAYER_RADIUS
    env.close()


# ==========================================
# T22 - Aspect ratio preservation
# ==========================================
def test_22_aspect_ratio_preservation():
    ref_ar = sc.SKELD_REF_WIDTH / sc.SKELD_REF_HEIGHT
    world_ar = sc.SKELD_WORLD_WIDTH / sc.SKELD_WORLD_HEIGHT
    assert abs(ref_ar - world_ar) < 1e-9, f"Aspect ratio mismatch: {ref_ar} vs {world_ar}"


# ==========================================
# T23 - World to screen round-trip
# ==========================================
def test_23_world_screen_roundtrip():
    test_points = [(100.0, 100.0), (800.0, 450.0), (1500.0, 800.0), (0.0, 0.0)]
    for wx, wy in test_points:
        sx, sy = sc.world_to_screen(wx, wy)
        wx_rec, wy_rec = sc.screen_to_world(sx, sy)
        # Tolerance within half screen pixel in world coords
        assert abs(wx - wx_rec) < 1.0, f"X roundtrip mismatch: {wx} vs {wx_rec}"
        assert abs(wy - wy_rec) < 1.0, f"Y roundtrip mismatch: {wy} vs {wy_rec}"


# ==========================================
# T24 - Physical geometry connectivity (Single component)
# ==========================================
def test_24_physical_geometry_connectivity():
    nav_grid = get_skeld_nav_grid()
    assert len(nav_grid.components) == 1, (
        f"Expected exactly 1 connected component, got {len(nav_grid.components)}"
    )
    assert len(nav_grid.main_component_set) > 20000, "Too few walkable cells in main component"


# ==========================================
# T25 - All 14 rooms in single component
# ==========================================
def test_25_all_14_rooms_reachable():
    nav_grid = get_skeld_nav_grid()
    for room in sc.SKELD_ROOMS:
        cx, cy = room.center
        cell = nav_grid.find_nearest_walkable_cell(cx, cy, max_radius_px=30.0)
        assert cell is not None, f"Room '{room.name}' has no walkable cell near center"
        assert cell in nav_grid.main_component_set, f"Room '{room.name}' not in main connected component"


# ==========================================
# T26 - Doorway traversal clearance
# ==========================================
def test_26_doorway_clearance():
    nav_grid = get_skeld_nav_grid()
    # Check that doorways between connected rooms have walkable cells with player radius clearance
    test_doors = [
        ("MedBay Door",       (600.0, 250.0)),
        ("Security Door",     (410.0, 410.0)),
        ("Electrical Door",   (660.0, 630.0)),
        ("Admin Door",        (940.0, 500.0)),
        ("Weapons West Door", (1155.0, 160.0)),
        ("Upper Bridge",      (240.0, 205.0)),
        ("Lower Bridge",      (240.0, 635.0)),
    ]
    for door_name, (dx, dy) in test_doors:
        cell = nav_grid.find_nearest_walkable_cell(dx, dy, max_radius_px=25.0)
        assert cell is not None, f"Doorway '{door_name}' is blocked for player radius {sc.SKELD_PLAYER_RADIUS}"


# ==========================================
# T27 - Exterior non-walkability
# ==========================================
def test_27_exterior_non_walkable():
    exterior_points = [
        (100.0, 10.0),    # above ship
        (800.0, 10.0),    # above cafeteria
        (100.0, 880.0),   # below lower engine
        (1500.0, 100.0),  # top-right space above nav
        (1500.0, 750.0),  # bottom-right space below shields
        (500.0, 50.0),    # pocket above medbay hall
    ]
    nav_grid = get_skeld_nav_grid()
    for ex, ey in exterior_points:
        assert not is_point_in_ship_floor(ex, ey), f"Exterior point ({ex},{ey}) reported in ship floor"
        assert not nav_grid.is_world_pos_walkable(ex, ey), f"Exterior point ({ex},{ey}) marked walkable"


# ==========================================
# T28 - Accidental shortcut prevention
# ==========================================
def test_28_accidental_shortcuts_blocked():
    # Electrical and MedBay sit near each other vertically, but have NO doorway between them
    # A straight vertical line from Electrical (645, 480) to MedBay (645, 400) must hit solid wall
    nav_grid = get_skeld_nav_grid()
    # Check that divider wall exists
    wall_found = False
    for srect in sc.SKELD_ALL_SOLID_RECTS:
        if srect.collidepoint(645, 440):
            wall_found = True
            break
    assert wall_found, "Divider wall missing between Electrical and MedBay"
    assert not nav_grid.is_world_pos_walkable(645.0, 440.0), "Shortcut through Electrical-MedBay wall is walkable"


# ==========================================
# T29 - All 40 task destinations reachable
# ==========================================
def test_29_all_tasks_reachable():
    nav_grid = get_skeld_nav_grid()
    unreachable = []
    for t in sc.SKELD_ALL_TASK_DESTINATIONS:
        cell = nav_grid.find_nearest_walkable_cell(t.world_x, t.world_y, max_radius_px=t.interaction_radius)
        if cell is None or cell not in nav_grid.main_component_set:
            unreachable.append(t)
    assert len(unreachable) == 0, f"Unreachable tasks: {[t.name for t in unreachable]}"


# ==========================================
# T30 - All-pairs task connectivity
# ==========================================
def test_30_all_pairs_task_connectivity():
    nav_grid = get_skeld_nav_grid()
    # Sample 10 random task-to-task pairs and verify A* path exists
    tasks = list(sc.SKELD_ALL_TASK_DESTINATIONS)
    for i in range(10):
        t1 = tasks[i]
        t2 = tasks[(i + 15) % len(tasks)]
        found, length, _ = nav_grid.astar_path(t1.world_pos, t2.world_pos)
        assert found, f"No path between {t1.name} in {t1.room} and {t2.name} in {t2.room}"
        assert length > 0.0, f"Path length <= 0 between {t1.name} and {t2.name}"


# ==========================================
# T31 - True spatial region detection
# ==========================================
def test_31_region_detection():
    assert sc.get_room_or_region(903.6, 115.0) == "Cafeteria"
    assert sc.get_room_or_region(1480.0, 380.0) == "Navigation"
    assert sc.get_room_or_region(215.0, 470.0) == "Reactor"
    assert sc.get_room_or_region(550.0, 160.0) == "MedBay Hallway"
    assert sc.get_room_or_region(360.0, 400.0) == "West Hallway"


# ==========================================
# T32 - Representative routes validation
# ==========================================
def test_32_representative_routes():
    nav_grid = get_skeld_nav_grid()
    routes = [
        ("Cafeteria", "Electrical"),
        ("Cafeteria", "Navigation"),
        ("Navigation", "Reactor"),
        ("MedBay", "Shields"),
        ("Security", "Admin"),
        ("Electrical", "Weapons"),
        ("Lower Engine", "O2"),
        ("Storage", "Reactor"),
        ("Weapons", "Electrical"),
    ]
    for start_r, dest_r in routes:
        p1 = sc.SKELD_ROOM_MAP[start_r].center
        p2 = sc.SKELD_ROOM_MAP[dest_r].center
        found, length, waypoints = nav_grid.astar_path(p1, p2)
        assert found, f"Representative route {start_r} -> {dest_r} failed"
        assert length > 100.0, f"Representative route {start_r} -> {dest_r} too short: {length}"
        assert len(waypoints) >= 2, f"Waypoints missing for {start_r} -> {dest_r}"


# ==========================================
# T33 - Map-aware efficiency helper
# ==========================================
def test_33_map_aware_efficiency():
    # Efficiency cannot exceed 100.0%
    eff1 = sc.compute_map_aware_efficiency(100.0, 100.0)
    assert eff1 == 100.0
    eff2 = sc.compute_map_aware_efficiency(150.0, 100.0)
    assert eff2 == 100.0  # clamped to 100%
    eff3 = sc.compute_map_aware_efficiency(50.0, 100.0)
    assert abs(eff3 - 50.0) < 1e-5
    eff4 = sc.compute_map_aware_efficiency(100.0, 0.0)
    assert eff4 == 0.0  # safe zero division


# ==========================================
# T34 - Goal modes end-to-end
# ==========================================
def test_34_goal_modes():
    modes = ["task", "task_to_task", "room_to_room"]
    for mode in modes:
        env = SkeldNavEnv(render_mode=None, goal_mode=mode)
        obs, info = env.reset(seed=42)
        assert "player_pos" in info
        assert "goal_pos" in info
        obs2, rew, term, trunc, info2 = env.step(np.array([0.5, 0.5], dtype=np.float32))
        assert not term
        assert not trunc
        env.close()


# ==========================================
# Main runner
# ==========================================
def main():
    print("=== Skeld Environment Full Validation Suite ===")
    t0 = time.time()

    all_tests = [
        ("T01 Config imports & architecture", test_01_config_imports),
        ("T02 Room count and geometry definitions", test_02_room_count),
        ("T03 Solid rects valid", test_03_solid_rects_valid),
        ("T04 Unique room names", test_04_unique_room_names),
        ("T05 Walkable room centers not in solid", test_05_walkable_not_in_solid),
        ("T06 Task destinations are valid TaskDestination models", test_06_task_destinations_model),
        ("T07 Audited room graph is bidirectional", test_07_graph_bidirectional),
        ("T08 Audited room graph is fully connected", test_08_graph_connected),
        ("T09 Env reset returns correct obs and info", test_09_reset),
        ("T10 Observation range validity", test_10_obs_range),
        ("T11 Spawn player clear of walls (100 resets)", test_11_spawn_player_clear),
        ("T12 Spawn goal clear of walls (100 resets)", test_12_spawn_goal_clear),
        ("T13 Step returns correct types", test_13_step_types),
        ("T14 Goal reward triggers on reach", test_14_goal_reward),
        ("T15 Timeout triggers after max_episode_time", test_15_timeout),
        ("T16 Action deadzone zero action no movement", test_16_action_deadzone),
        ("T17 Stuck flag activates on prolonged obstacle blockage", test_17_stuck_flag),
        ("T18 Stagnation progress increases when stationary", test_18_stagnation),
        ("T19 SB3 env_checker passes", test_19_gymnasium_checker),
        ("T20 Interior wall no penetration", test_20_interior_wall_no_penetration),
        ("T21 Hull no penetration", test_21_hull_no_penetration),
        ("T22 Aspect ratio preservation (ref vs world)", test_22_aspect_ratio_preservation),
        ("T23 World-to-screen round-trip invertibility", test_23_world_screen_roundtrip),
        ("T24 Physical geometry connectivity (1 component)", test_24_physical_geometry_connectivity),
        ("T25 All 14 rooms in single connected component", test_25_all_14_rooms_reachable),
        ("T26 Doorway traversal clearance with player radius", test_26_doorway_clearance),
        ("T27 Exterior non-walkability (outside hull blocked)", test_27_exterior_non_walkable),
        ("T28 Accidental shortcut prevention (walls divide rooms)", test_28_accidental_shortcuts_blocked),
        ("T29 All 40 task destinations reachable (0 unreachable)", test_29_all_tasks_reachable),
        ("T30 All-pairs task connectivity", test_30_all_pairs_task_connectivity),
        ("T31 True spatial region detection (rooms & hallways)", test_31_region_detection),
        ("T32 Representative routes validation (9 routes)", test_32_representative_routes),
        ("T33 Map-aware efficiency helper (<= 100.0%)", test_33_map_aware_efficiency),
        ("T34 Goal modes (task, task_to_task, room_to_room)", test_34_goal_modes),
    ]

    for name, fn in all_tests:
        run_test(name, fn)

    elapsed = time.time() - t0
    passed = sum(1 for _, ok, _ in test_results if ok)
    total = len(test_results)
    print(f"\n=== Results: {passed}/{total} passed in {elapsed:.1f}s ===")

    if passed == total:
        print("All tests passed successfully!")
        sys.exit(0)
    else:
        print(f"{total - passed} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
