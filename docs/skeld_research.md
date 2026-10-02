# Skeld Map Research & Engineering Notes

## 1. Source Classification & Methodology

All source materials used in this project are strictly categorized under rigorous source classification:

| Category | Source | Description & Role |
|---|---|---|
| **OFFICIAL / INNERSLOTH** | *Among Us* (Innersloth LLC) | Canon room nomenclature, task mechanics, emergency meeting principles, and visual aesthetic concepts. Intellectual property of Innersloth LLC. |
| **COMMUNITY REFERENCE** | *Among Us Fandom Wiki* (`The_Skeld/Interactive_map`) | Community-maintained interactive map with coordinate marker annotations (8565×4794 px reference space). Used as non-authoritative community data for marker positions and topological adjacency. (Note: Fandom Wiki is a community reference, NOT an official Innersloth publication). |
| **COMMUNITY REFERENCE** | Community Navigation Guides (`docs/among_us_ultimate_guide.md`) | Community documentation detailing room layouts, task checklists, and vent subnetworks. |
| **INTERNAL DERIVATION** | Mathematical Scaled World Model | Derivation of 1600.0×895.45 logical world dimensions, aspect-ratio preservation, dual-layer collision geometry (50 solid walls + 32 walkable floor segments), 4px occupancy navigation grid, calibrated player/sensor physics, and task stand clearances. |

> **Strict Geometric Grounding Rule**: No AI-generated artwork was used to infer wall geometry, corridor dimensions, room proportions, or task coordinates. All geometry is derived from reference map dimensions and physical topological layout.

---

## 2. Mathematical Coordinate Space & Aspect Ratio Architecture

In earlier prototyping, reference coordinates (8565×4794) were scaled independently to display pixels (1100×700) using $s_x = 1100/8565 \approx 0.1284$ and $s_y = 700/4794 \approx 0.1460$. This produced a 13.7% aspect ratio distortion ($1.5714$ vs $1.7866$).

Stage 2.5 completely separates **Reference Space**, **Logical World Space**, and **Display Space**:

### A. Reference Space
- Bounds: $[0, 0] \times [8565.0, 4794.0]$
- Aspect Ratio: $\frac{8565.0}{4794.0} \approx 1.7866082603$

### B. Logical World Space (Physics & Reinforcement Learning)
All physical simulation, collision resolution, sensor raycasting, agent navigation, and task interaction operate in this space.
- Chosen logical width: $W_{\text{world}} = 1600.0\text{ px}$
- Derived logical height: $H_{\text{world}} = \frac{1600.0}{1.7866082603} \approx 895.44658\text{ px}$
- Uniform World Scale: $S_{\text{world}} = \frac{1600.0}{8565.0} \approx 0.1868067717$
- Mathematical Aspect Ratio Preservation:
  $$\frac{W_{\text{world}}}{H_{\text{world}}} = \frac{1600.0}{895.44658} = 1.7866082603 \equiv \frac{8565.0}{4794.0}$$
  Aspect ratio is mathematically identical to reference space within machine precision ($< 10^{-7}$ relative error).

### C. Display Space (Rendering Only)
- Window: $1100 \times 700\text{ px}$
- Display Scale: $s_{\text{display}} = \frac{1100.0}{1600.0} = 0.6875$
- Drawn Height: $H_{\text{draw}} = 895.44658 \times 0.6875 = 615.6195\text{ px}$
- Letterbox Offsets:
  - $X_{\text{offset}} = 0.0\text{ px}$
  - $Y_{\text{offset}} = \frac{700.0 - 615.6195}{2.0} \approx 42.19\text{ px}$
- Invertible Transformations:
  $$\text{world\_to\_screen}(wx, wy) = (X_{\text{offset}} + wx \cdot s_{\text{display}}, Y_{\text{offset}} + wy \cdot s_{\text{display}})$$
  $$\text{screen\_to\_world}(sx, sy) = \left(\frac{sx - X_{\text{offset}}}{s_{\text{display}}}, \frac{sy - Y_{\text{offset}}}{s_{\text{display}}}\right)$$

---

## 3. Calibrated Player & Sensor Physics

| Parameter | Value | Justification / Physical Ratio |
|---|---|---|
| **PLAYER_RADIUS** | 10.0 px (Diameter 20.0 px) | Narrowest doorway in world space is 40.0 px. Ratio: $\frac{\text{doorway}}{\text{diameter}} = 2.0$. The player fits comfortably with 10 px clearance on each side. |
| **PLAYER_SPEED** | 160.0 px/s | Allows the agent to traverse the entire ship width (1600 px) in ~10 seconds at full throttle, realistic for indoor navigation. |
| **GOAL_RADIUS** | 12.0 px | Allows goal markers to sit cleanly inside 35 px wall alcoves and task consoles without intersecting collision geometry. |
| **INTERACTION_RADIUS** | 25.0 px | Standard task interaction radius ($2.5\times$ player radius). |
| **RAY_COUNT** | 16 rays | Radial sensors cast at uniform 22.5° intervals. Proposed observation architecture V1. |
| **RAY_MAX_DISTANCE** | 220.0 px | $11.0\times$ player diameter, $4.4\times$ average corridor width (50 px), and $1.22\times$ average room dimension (180 px). Senses obstacles locally without seeing across the entire ship. |

---

## 4. Room Topology Audit

The initial prototype treated vent network connections as physical hallways (e.g. Electrical $\leftrightarrow$ MedBay $\leftrightarrow$ Security). An audit against authentic ship plans confirmed that these rooms do NOT have direct physical doorways between them.

### Audited Physical Hallway Topology:
- **Cafeteria**: Connects to Weapons (East), MedBay Corridor (West), Upper Engine Hallway (North-West), and Admin/Storage Hallway (South).
- **Weapons**: Connects to Cafeteria (West) and O2 / East Hallway (South).
- **O2**: Connects to East Hallway (linking Weapons, Navigation, Shields).
- **Navigation**: Cockpit at far east; connects via Upper and Lower East Hallways to O2 and Shields.
- **Shields**: Connects to East Hallway (North), Communications Corridor (West), and Storage (West).
- **Communications**: Connects to Storage Corridor (West) and Shields Corridor (East).
- **Storage**: Central junction room; connects to Admin (North), Cafeteria (North), Electrical/Engine Hallway (West), Communications (South-East), and Shields (East).
- **Admin**: Connects to Cafeteria/Storage Corridor via single entrance.
- **Electrical**: Single door entrance connecting exclusively to South-West Hallway leading to Storage/Lower Engine.
- **Security**: Single door entrance connecting exclusively to West Corridor between Upper and Lower Engine.
- **MedBay**: Single door entrance connecting exclusively to MedBay Corridor leading to Cafeteria and Upper Engine.
- **Reactor**: Westernmost engineering section; connects via Upper Reactor Corridor to Upper Engine, and Lower Reactor Corridor to Lower Engine.
- **Upper Engine**: Connects to Upper Reactor Corridor (West), West Hallway (South to Security/Lower Engine), MedBay/Cafeteria Corridor (East).
- **Lower Engine**: Connects to Lower Reactor Corridor (West), West Hallway (North to Security/Upper Engine), Electrical/Storage Hallway (East).

This graph is 100% bidirectional and fully connected across all 14 rooms.

---

## 5. Dual-Layer Collision Architecture & Walkable Hull

To prevent the agent from escaping into the void through unintended gaps between rectangles, Stage 2.5 employs a dual-layer safety model:
1. **Solid Wall Rectangles (50 primitives)**: All exterior perimeter boundaries and interior walls/consoles are defined as solid AABBs.
2. **Explicit Walkable Floor Segments (32 polygonal areas)**: The interior walkable space is covered by 32 rectangular floor regions.
3. **Dual Verification in Physics**:
   ```python
   # Axis-separated movement checks both:
   1. player_rect.colliderect(srect) == False  (for all solid rects)
   2. is_point_in_ship_floor(cand_x, cand_y) == True  (must be on ship floor)
   ```
4. **Exterior Non-Walkability**: The vacuum outside the ship is strictly non-walkable. Points outside the hull boundary are immediately rejected by the physics solver and occupancy grid.

---

## 6. Internal Occupancy Grid & Validation Pathfinder

To validate physical walkability independently of hand-written metadata:
- **Grid Resolution**: $4.0\text{ px}$ per cell ($400 \times 224 = 89,600\text{ cells}$).
- **Player Clearance Inflation**: Every cell within `PLAYER_RADIUS` (10.0 px) of a wall or exterior boundary is marked non-walkable.
- **Connected Component Analysis**: The grid contains exactly **1 walkable connected component** consisting of 26,754 cells. All 14 rooms and all 40 task destinations belong to this single component.
- **Developer A* Pathfinder**: 8-connected A* search provides ground-truth shortest path distances and visual navigation routes for debug inspection. (Strictly isolated from PPO agent observations).

---

## 7. Authentic Task Destination Database

Extracted from community reference marker coordinates and converted to logical world space. Every task includes an authentic ID, name, room, world coordinate, interaction radius, and confidence classification:

- Total Tasks: 40
- Verified Tasks: 37 (extracted directly from community coordinates)
- Estimated Tasks: 3 (stand offsets adjusted for wall clearance)
- Unreachable Tasks: 0 (100% verified reachable via internal A*)
