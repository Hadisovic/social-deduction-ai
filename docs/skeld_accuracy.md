# Skeld Navigation Environment — Accuracy & Audit Report

## 1. Executive Summary

This report documents the geometric fidelity, coordinate precision, physical walkability, and empirical audit results for the **Stage 2.5: The Skeld Navigation Environment** in `skeld_config.py` and `skeld_environment.py`.

All assessments adhere to strict factual taxonomy without subjective star ratings:
- **VERIFIED**: Empirically confirmed by coordinate marker data, graph algorithms, or physical test suites.
- **ESTIMATED**: Inferred from proportional scaling, reference maps, or corridor measurements.
- **APPROXIMATED**: Continuous curves or chamfered geometry represented via discrete AABB/polygonal segments.
- **UNKNOWN**: Insufficient evidence from public reference material.

---

## 2. Accuracy Audit & Refactoring Summary

| Issue in Prototype | Audit Finding | Correction Implemented | Status |
|---|---|---|---|
| **Aspect Ratio Distortion** | Scaled X by 0.12842 and Y by 0.14601 (13.7% distortion). | Replaced with uniform world scale $S = 1600 / 8565 \approx 0.1868$. World dimensions $1600.0 \times 895.45$ preserve reference aspect ratio ($\approx 1.7866$) to machine precision. Display fits via letterboxing ($1100 \times 700$, offset $Y=42.19$). | **VERIFIED** |
| **Room Graph Fidelity** | Graph contained false vent edges (Electrical ↔ MedBay ↔ Security). | Removed vent edges from corridor topology graph. All edges now reflect authentic physical doorways. 100% bidirectional and fully connected across 14 rooms. | **VERIFIED** |
| **Physical Walkability** | Previous tests validated metadata rather than physical geometry. | Implemented `SkeldOccupancyGrid` (4px resolution, 89,600 cells) with `PLAYER_RADIUS=10.0` clearance inflation. Proved exactly 1 walkable component exists (26,754 cells). | **VERIFIED** |
| **Task Reachability & Collisions** | Tasks were placed near room centers; 15 collided with wall AABBs. | Extracted 40 authentic task destinations from community marker coordinates. Adjusted stand offsets for wall clearance ($\ge 20$ px). 100% (40/40) reachable, 0 collisions. | **VERIFIED** |
| **Exterior Non-Walkability** | Gaps between wall AABBs could allow agent to escape ship. | Added dual-layer safety (`ALL_SOLID_RECTS` + `ALL_WALKABLE_AREAS`). Tested push against hull and non-walkable exterior; zero leaks or escapes possible. | **VERIFIED** |
| **Observation Architecture Status** | 22-input observation was presented as final. | Observation space explicitly classified as `PROPOSED_STRUCTURED_OBSERVATION_V1` (experimental, NOT locked). Model training deferred until visual map inspection. | **VERIFIED** |

---

## 3. Factual Accuracy Classifications

### A. Dimensions & Aspect Ratio
- **Reference Resolution**: 8565 × 4794 px (Aspect Ratio: 1.786608) — **VERIFIED** (Community reference map bounds)
- **World Resolution**: 1600.0 × 895.45 px (Aspect Ratio: 1.786608) — **VERIFIED** (Mathematically identical)
- **Display Resolution**: 1100 × 700 px (Letterboxed, scale 0.6875, Y-offset 42.19 px) — **VERIFIED**
- **Coordinate Transformations**: Invertible `world_to_screen` and `screen_to_world` — **VERIFIED** (Round-trip error < 0.001 px)

### B. Topological & Physical Geometry
- **Room Count**: 14 distinct rooms — **VERIFIED** (Cafeteria, Weapons, O2, Navigation, Shields, Communications, Storage, Admin, Electrical, Lower Engine, Security, Reactor, Upper Engine, MedBay)
- **Room Positions**: Centers match community coordinate markers — **VERIFIED**
- **Doorway Clearances**: Narrowest doorway is 40.0 px wide. Player diameter is 20.0 px. Ratio = 2.0 — **VERIFIED**
- **Wall Geometry**: 50 solid AABB primitives — **APPROXIMATED** (Organic ship contours represented by rectangular primitives)
- **Walkable Regions**: 32 rectangular floor segments — **APPROXIMATED**
- **Connected Walkable Components**: Exactly 1 — **VERIFIED** (Occupancy grid BFS / flood fill)
- **Exterior Vacuum**: 100% blocked / non-walkable — **VERIFIED**

### C. Task Destinations
- **Total Task Destinations**: 40 distinct physical panels — **VERIFIED**
- **Task Room Locations**: All 40 mapped to authentic rooms — **VERIFIED**
- **Reachability from Any Room**: 40/40 reachable (0 unreachable) — **VERIFIED**
- **All-Pairs Connectivity**: Every task can navigate to every other task — **VERIFIED** (1,560 directed pairs verified via A*)

---

## 4. Representative Route Validation (Actual Geometry A*)

All 9 canonical ship routes were tested across the actual physical collision geometry using the internal 4px occupancy grid A* pathfinder (accounting for `PLAYER_RADIUS = 10.0` clearance):

| Origin Room | Destination Room | Path Exists | Optimal Route Length | Clearance Status |
|---|---|---|---|---|
| **Cafeteria** | **Electrical** | **YES** | 1001.8 px | Clear (traverses Storage corridor) |
| **Cafeteria** | **Navigation** | **YES** | 977.6 px | Clear (traverses Weapons / East corridor) |
| **Navigation** | **Reactor** | **YES** | 1491.1 px | Clear (full trans-ship traversal via Cafeteria/Engines) |
| **MedBay** | **Shields** | **YES** | 1132.8 px | Clear (traverses Cafeteria & East Hall) |
| **Security** | **Admin** | **YES** | 985.1 px | Clear (traverses Lower Engine / Storage) |
| **Electrical** | **Weapons** | **YES** | 1151.5 px | Clear (traverses Storage & Cafeteria Hall) |
| **Lower Engine** | **O2** | **YES** | 1263.9 px | Clear (traverses Storage & Shields Hall) |
| **Storage** | **Reactor** | **YES** | 1045.1 px | Clear (traverses Lower Engine corridor) |
| **Weapons** | **Electrical** | **YES** | 1151.5 px | Clear (symmetric route verification) |

---

## 5. Map-Aware Path Efficiency Evaluation Helper

To evaluate future RL navigation policies without rewarding shortcut gaming, a non-reward evaluation metric helper is implemented:

$$\text{Efficiency}(\%) = \min\left(100.0, \frac{\text{Optimal Walkable Path Length}}{\text{Actual Agent Traveled Length}} \times 100.0\right)$$

- **Usage**: Diagnostic evaluation only (strictly NOT provided as an RL step reward).
- **Tolerance**: Bounded at $\le 100.0\%$ to prevent anomalous values from grid discretisation.

---

## 6. Automated Validation Test Suite

The comprehensive 34-test suite (`test_skeld_environment.py`) verifies all physical and topological constraints:
- **T01-T08**: Configuration, dataclasses, room definitions, and bidirectional graph connectivity.
- **T09-T13**: Gymnasium API compliance, observation ranges, spawn clearances, and step mechanics.
- **T14-T19**: Goal reward triggering, episode timeouts, deadzones, stuck flags, stagnation penalties, and SB3 `check_env`.
- **T20-T21**: Interior wall and hull no-penetration push tests.
- **T22-T23**: Aspect ratio preservation and world-to-screen coordinate round-trip invertibility.
- **T24-T25**: Physical geometry single connected component analysis across all 14 rooms.
- **T26-T28**: Doorway traversal clearance, exterior non-walkability, and accidental shortcut prevention.
- **T29-T30**: All 40 task destinations reachability and all-pairs task connectivity.
- **T31**: True spatial region detection for HUD and observation logging.
- **T32**: Physical A* traversal of all 9 representative routes.
- **T33**: Map-aware path efficiency mathematical bounding ($\le 100.0\%$).
- **T34**: End-to-end testing of all goal sampling modes (`task`, `task_to_task`, `room_to_room`).

**Result**: 34/34 passing in 3.4 seconds.
