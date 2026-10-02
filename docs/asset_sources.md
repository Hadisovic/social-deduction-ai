# Asset Sources, Licensing & Hygiene Policy

## 2026-10-02 scope update

The policy below describes the preserved procedural legacy environment. The owner
subsequently authorized a separate source-backed map using repository data and
their supplied artwork. `among_us_map_simulation.py` and `assets/skeld/` use that
material and are **not** covered by the claims of no game artwork or derived data
below. See [the new map's provenance and notices](../assets/skeld/NOTICE.md).
Its additional geometry dependency is Shapely 2.x (BSD-3-Clause).

## 1. Project Intellectual Property & Original Work Notice

The Skeld navigation environment implemented in `skeld_config.py`, `skeld_navigation.py`, and `skeld_environment.py` is an **original algorithmic software implementation** engineered specifically for reinforcement learning navigation and visual representation research.

### Copyright Hygiene Policy:
- **NO extracted game textures**: The environment uses procedural geometric primitives (colored rectangles, circles, line segments) drawn natively via Pygame.
- **NO game sprites or artwork**: All character models, consoles, and background visuals are abstract geometric shapes.
- **NO game audio**: No sound effects or background music files are contained in this repository.
- **NO proprietary binaries or data files**: No game code, DLLs, assemblies, or game archives are bundled or utilized.
- **NO AI-generated artwork as geometric evidence**: Any generated schematics or conceptual diagrams are non-authoritative illustrations only and have NOT been used to derive physical coordinates, collision boundaries, or corridor scales.

---

## 2. Source Classification Taxonomy

All reference sources informing this project are classified under three transparent tiers:

### Tier 1: OFFICIAL / INNERSLOTH
- **Entity**: Innersloth LLC
- **Works**: *Among Us* and the canon setting of *The Skeld*.
- **Role**: Source of canonical room nomenclature (Cafeteria, Electrical, MedBay, etc.), original map concept, and task names.
- **Legal Notice**: *Among Us* and *The Skeld* are trademarks and copyrights of Innersloth LLC. This repository is an independent, non-commercial academic and research project for reinforcement learning. It is not affiliated with, endorsed by, or sponsored by Innersloth LLC.

### Tier 2: COMMUNITY REFERENCE
- **Source**: *Among Us Fandom Wiki* (`The_Skeld` and `The_Skeld/Interactive_map`)
- **License**: CC BY-SA 3.0 (text and community wiki annotations).
- **Role**: Non-authoritative community data used to obtain relative pixel marker coordinates in reference space (8565×4794 px) for room centers, vents, and task stations.
- **Clarification**: The Fandom Wiki is maintained by community volunteers and is **NOT** an official publication of Innersloth LLC.

### Tier 3: INTERNAL DERIVATION
- **Author**: Antigravity Research Team
- **Role**: Mathematical transformation to a uniform 1600.0×895.45 logical world space, aspect ratio preservation formulas, dual-layer collision definitions (50 solid wall rects + 32 walkable polygons), 4px occupancy navigation grid, calibrated physics ratios, and internal validation algorithms.

---

## 3. Third-Party Software Dependencies

All software libraries used by this project are open-source and governed by their respective licenses:

| Library | Version | License | Role in Project |
|---|---|---|---|
| **pygame-ce** | 2.5.8 | LGPL-2.1 | 2D graphics rendering, display scaling, and event handling |
| **gymnasium** | 1.0+ | MIT | Standard reinforcement learning environment API |
| **stable-baselines3** | 2.0+ | MIT | PPO algorithm implementation and model serialization |
| **numpy** | 1.26+ | BSD-3-Clause | Vectorized mathematics and array manipulation |
| **pytest** | 8.0+ | MIT | Automated unit and regression test runner |
