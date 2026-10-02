"""
skeld_config.py
Configuration constants and geometry definitions for The Skeld navigation environment.

Coordinate Space Hierarchy:
  1. REFERENCE SPACE:
     Source map bounds: 8565 x 4794 pixels (derived from community interactive map markers).
     Aspect ratio: 8565 / 4794 = 1.7866082603254067.
  2. WORLD SPACE:
     Uniform logical resolution: 1600.0 x 895.446584938704 pixels.
     Uniform scale factor: S = 1600.0 / 8565.0 (~0.1868067717).
     Preserves source aspect ratio EXACTLY (1600.0 / 895.44658... = 8565 / 4794).
     ALL physics, collision, sensors, tasks, and navigation operate in WORLD coordinates.
  3. DISPLAY SPACE:
     Window size: 1100 x 700 pixels (consistent with Stage 1/2 window size).
     Uniform display scale: s_display = 1100.0 / 1600.0 = 0.6875.
     Letterbox padding: Y offset = (700 - 895.44658 * 0.6875) / 2 = 42.19 pixels.
     Display rendering is transformed via world_to_screen() and screen_to_world().
"""

import math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
import pygame

# ==========================================
# COORDINATE SPACES & ASPECT RATIO
# ==========================================
SKELD_REF_WIDTH   = 8565.0
SKELD_REF_HEIGHT  = 4794.0
SKELD_REF_ASPECT  = SKELD_REF_WIDTH / SKELD_REF_HEIGHT  # 1.7866082603254067

SKELD_WORLD_WIDTH  = 1600.0
SKELD_WORLD_HEIGHT = SKELD_WORLD_WIDTH / SKELD_REF_ASPECT  # 895.446584938704
SKELD_WORLD_SCALE  = SKELD_WORLD_WIDTH / SKELD_REF_WIDTH   # 0.18680677174547577

SKELD_WINDOW_WIDTH  = 1100
SKELD_WINDOW_HEIGHT = 700
SKELD_FPS           = 60
SKELD_WINDOW_TITLE  = "Stealth RL - Stage 2.5: The Skeld Navigation"

# Display transform parameters (letterboxing to preserve aspect ratio)
SKELD_DISPLAY_SCALE = SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH  # 0.6875
SKELD_DISPLAY_DRAW_H = SKELD_WORLD_HEIGHT * SKELD_DISPLAY_SCALE  # 615.6195 px
SKELD_DISPLAY_OFFSET_X = 0.0
SKELD_DISPLAY_OFFSET_Y = (SKELD_WINDOW_HEIGHT - SKELD_DISPLAY_DRAW_H) / 2.0  # 42.19 px

def world_to_screen(wx: float, wy: float) -> Tuple[int, int]:
    """Transform world coordinates to screen pixel coordinates with letterboxing."""
    sx = SKELD_DISPLAY_OFFSET_X + wx * SKELD_DISPLAY_SCALE
    sy = SKELD_DISPLAY_OFFSET_Y + wy * SKELD_DISPLAY_SCALE
    return int(round(sx)), int(round(sy))

def screen_to_world(sx: float, sy: float) -> Tuple[float, float]:
    """Transform screen pixel coordinates back to world coordinates."""
    wx = (sx - SKELD_DISPLAY_OFFSET_X) / SKELD_DISPLAY_SCALE
    wy = (sy - SKELD_DISPLAY_OFFSET_Y) / SKELD_DISPLAY_SCALE
    return float(wx), float(wy)

def ref_to_world(rx: float, ry: float) -> Tuple[float, float]:
    """Transform 8565x4794 reference coordinates to logical world coordinates."""
    return rx * SKELD_WORLD_SCALE, ry * SKELD_WORLD_SCALE


# ==========================================
# COLOR PALETTE (dark space sci-fi theme)
# ==========================================
SKELD_COLOR_BG          = (10,  12,  20)
SKELD_COLOR_FLOOR       = (28,  35,  48)
SKELD_COLOR_FLOOR_ALT   = (32,  40,  55)
SKELD_COLOR_WALL        = (55,  65,  85)
SKELD_COLOR_WALL_BORDER = (75,  90, 115)
SKELD_COLOR_WALL_INNER  = (40,  50,  65)
SKELD_COLOR_ROOM_LABEL  = (160, 185, 220)
SKELD_COLOR_CORRIDOR    = (38,  48,  62)
SKELD_COLOR_DOOR        = (80, 200, 220)
SKELD_COLOR_VENT        = (0,  220, 200)
SKELD_COLOR_PLAYER      = (0,  195, 175)
SKELD_COLOR_PLAYER_BDR  = (0,  135, 120)
SKELD_COLOR_PLAYER_CORE = (180, 255, 245)
SKELD_COLOR_TASK        = (255, 215,  50)
SKELD_COLOR_TASK_ACTIVE = (50,  255, 120)
SKELD_COLOR_GOAL        = (50,  220, 120)
SKELD_COLOR_GOAL_GLOW   = (50,  220, 120)
SKELD_COLOR_HUD_BG      = (15,  20,  32)
SKELD_COLOR_HUD_TEXT    = (200, 215, 240)
SKELD_COLOR_HUD_ACCENT  = (0,  180, 220)
SKELD_COLOR_SUCCESS     = (30,  185,  90)
SKELD_COLOR_ALERT       = (225,  45,  45)
SKELD_COLOR_NAV_PATH    = (255, 100, 200)


# ==========================================
# CALIBRATED PLAYER & SENSOR PHYSICS (WORLD UNITS)
# ==========================================
# Narrowest doorway in world space: 40.0 px
# Player radius: 10.0 px -> diameter: 20.0 px
# Ratio (narrowest doorway / player diameter): 40.0 / 20.0 = 2.0 (comfortable navigation)
SKELD_PLAYER_RADIUS  = 10.0
SKELD_PLAYER_SPEED   = 160.0  # px/s (~7.5s traverse ship width)
SKELD_GOAL_RADIUS    = 12.0
SKELD_INTERACTION_RADIUS = 25.0

# RL Simulation
SKELD_RL_DT               = 1.0 / 30.0
SKELD_ACTION_DEADZONE     = 0.05
SKELD_MAX_EPISODE_TIME    = 60.0
SKELD_TIMEOUT_PENALTY     = -50.0
SKELD_GOAL_REWARD         = 100.0

# Sensor Calibration:
# 16 rays at 22.5 deg intervals. Max range 220 px.
# Ray range / player diameter: 220 / 20 = 11.0x
# Ray range / avg corridor width (50px): 220 / 50 = 4.4x
# Ray range / avg room dimension (180px): 220 / 180 = 1.22x
SKELD_RAY_COUNT           = 16
SKELD_RAY_MAX_DISTANCE    = 220.0

# Proposed observation architecture (v1 experiment proposal, NOT locked)
SKELD_OBSERVATION_ARCHITECTURE = "PROPOSED_STRUCTURED_OBSERVATION_V1"
SKELD_OBSERVATION_SIZE         = 22  # 2 player pos + 2 goal rel + 16 rays + 1 stuck + 1 stagnation

SKELD_STEP_PENALTY        = -0.01
SKELD_PROGRESS_SCALE      = 0.03
SKELD_MIN_PROGRESS_DIST   = 5.0
SKELD_PROGRESS_CHECK_INT  = 0.5

SKELD_STAGNATION_STEPS    = 90
SKELD_STAGNATION_RADIUS   = 15.0
SKELD_STAGNATION_PENALTY  = -25.0
SKELD_STAGNATION_ESCAPE   = 50.0

SKELD_BLOCKED_THRESHOLD   = 0.20
SKELD_BLOCKED_PROLONGED   = 15

SKELD_SPAWN_MARGIN        = 15.0
SKELD_MIN_SPAWN_DIST      = 200.0


# ==========================================
# EXPLICIT WALKABLE FLOOR REGIONS (WORLD SPACE)
# ==========================================
# 14 major room floor rects
SKELD_ROOM_FLOORS: Dict[str, pygame.Rect] = {
    "Cafeteria":      pygame.Rect(680,   50, 440, 205),
    "Weapons":        pygame.Rect(1190,  80, 160, 125),
    "O2":             pygame.Rect(1085, 305, 135,  75),
    "Navigation":     pygame.Rect(1410, 300, 170, 165),
    "Shields":        pygame.Rect(1160, 560, 160, 175),
    "Communications": pygame.Rect(1040, 690, 120, 135),
    "Storage":        pygame.Rect(780,  520, 200, 290),
    "Admin":          pygame.Rect(970,  430, 170, 150),
    "Electrical":     pygame.Rect(585,  465, 165, 155),
    "Lower Engine":   pygame.Rect(260,  530, 170, 180),
    "Security":       pygame.Rect(425,  330, 115, 150),
    "Reactor":        pygame.Rect(140,  170, 110, 490),
    "Upper Engine":   pygame.Rect(260,  110, 170, 150),
    "MedBay":         pygame.Rect(560,  310, 180, 110),
}

# Corridor walkable floor segments
SKELD_CORRIDOR_FLOORS: List[pygame.Rect] = [
    pygame.Rect(430,  140, 250,  50),  # MedBay Hallway (Cafeteria <-> Upper Engine)
    pygame.Rect(580,  190,  50, 120),  # MedBay Branch
    pygame.Rect(335,  260,  50, 270),  # West Hallway (Upper Engine <-> Lower Engine)
    pygame.Rect(385,  385,  40,  50),  # Security Entrance
    pygame.Rect(210,  175,  70,  55),  # Upper Reactor Bridge
    pygame.Rect(210,  605,  70,  55),  # Lower Reactor Bridge
    pygame.Rect(855,  255,  60, 265),  # Central Hallway (Cafeteria <-> Storage)
    pygame.Rect(915,  480,  55,  50),  # Admin Entrance
    pygame.Rect(430,  640, 350,  50),  # South-West Hallway (Lower Engine <-> Storage)
    pygame.Rect(635,  620,  50,  30),  # Electrical Entrance
    pygame.Rect(980,  640, 180,  50),  # South-East Hallway (Storage <-> Shields)
    pygame.Rect(1060, 680,  50,  20),  # Communications Entrance
    pygame.Rect(1115, 135,  80,  50),  # Cafeteria to Weapons corridor
    pygame.Rect(1255, 200,  50, 365),  # East Hallway (Weapons <-> Shields)
    pygame.Rect(1215, 315,  45,  50),  # O2 Entrance
    pygame.Rect(1300, 315, 115,  50),  # Upper Navigation corridor
    pygame.Rect(1300, 415, 115,  50),  # Lower Navigation corridor (horizontal)
    pygame.Rect(1260, 460,  50, 110),  # Lower Navigation corridor (vertical to Shields)
]

SKELD_ALL_WALKABLE_AREAS: List[pygame.Rect] = list(SKELD_ROOM_FLOORS.values()) + SKELD_CORRIDOR_FLOORS


# ==========================================
# PHYSICAL SOLID WALL PRIMITIVES (WORLD SPACE)
# ==========================================
SKELD_OUTER_WALLS: List[pygame.Rect] = [
    # Screen boundary buffers
    pygame.Rect(0, 0, 1600, 20),
    pygame.Rect(0, 0, 20, 896),
    pygame.Rect(1580, 0, 20, 896),
    pygame.Rect(0, 876, 1600, 20),

    # Top outer hull blocks
    pygame.Rect(20,   20, 240,  90),   # above Upper Engine / Reactor
    pygame.Rect(430,  20, 250, 120),   # above MedBay Hallway
    pygame.Rect(680,  20, 440,  30),   # above Cafeteria
    pygame.Rect(1120, 20,  70, 115),   # above Cafeteria-Weapons corridor
    pygame.Rect(1190, 20, 170,  60),   # above Weapons
    pygame.Rect(1350, 20, 230, 280),   # above Navigation / East Hallway

    # Right outer hull blocks
    pygame.Rect(1580, 300,  20, 170),  # right tip of Navigation
    pygame.Rect(1410, 465, 170,  95),  # below Navigation
    pygame.Rect(1320, 560, 260, 316),  # bottom-right exterior

    # Bottom outer hull blocks
    pygame.Rect(1160, 735, 160, 141),  # below Shields
    pygame.Rect(1040, 825, 120,  51),  # below Communications
    pygame.Rect(780,  810, 260,  66),  # below Storage / Comms gap
    pygame.Rect(430,  690, 350, 186),  # below South-West Hallway
    pygame.Rect(260,  710, 170, 166),  # below Lower Engine
    pygame.Rect(20,   665, 240, 211),  # below Reactor / Lower Engine

    # Left outer hull block
    pygame.Rect(20,   110, 120, 555),  # left of Reactor
]

SKELD_INTERIOR_WALLS: List[pygame.Rect] = [
    # Divider: Reactor east wall vs West Hallway
    pygame.Rect(250, 260,  85, 270),

    # Dividers: Upper Engine vs Security / MedBay
    pygame.Rect(385, 260,  40, 125),   # above Security door
    pygame.Rect(385, 435,  40,  95),   # below Security door
    pygame.Rect(425, 260, 135,  70),   # between Upper Engine / MedBay / Security

    # Divider: Below Security, above SW Hallway, left of Electrical
    pygame.Rect(425, 480, 160, 160),

    # Dividers: MedBay, Cafeteria, and Electrical
    pygame.Rect(630, 190,  50,  65),   # east of MedBay branch, west of Cafeteria
    pygame.Rect(680, 255, 175,  55),   # below Cafeteria, above MedBay/Admin Hall
    pygame.Rect(560, 420, 180,  45),   # solid wall between MedBay and Electrical (NO DOOR)

    # Divider: Electrical east wall vs Storage west wall
    pygame.Rect(750, 465,  30, 175),   # solid wall between Electrical and Storage (NO DOOR)

    # Dividers: Admin surrounding walls
    pygame.Rect(915, 255,  55, 225),   # above Admin door
    pygame.Rect(915, 530,  55, 110),   # below Admin door
    pygame.Rect(915, 255, 170, 175),   # between Central Hall and O2
    pygame.Rect(970, 255, 115, 175),   # between Admin top and O2 bottom
    pygame.Rect(970, 580, 170,  60),   # below Admin

    # Dividers: Communications left and right walls
    pygame.Rect(980,  690,  60, 120),  # west of Comms
    pygame.Rect(1140, 690,  20, 135),  # east of Comms

    # Dividers: Shields and East Hallway
    pygame.Rect(1140, 380, 115, 180),  # between Admin/O2 and Shields
    pygame.Rect(1085, 205, 170, 100),  # above O2
    pygame.Rect(1220, 205,  35, 110),  # above O2 door
    pygame.Rect(1220, 365,  35, 195),  # below O2 door

    # Dividers: East Hallway, Navigation, and Shields
    pygame.Rect(1305, 205, 105, 110),  # between Weapons and Upper Nav
    pygame.Rect(1305, 365, 105,  50),  # solid divider between Upper and Lower Nav
    pygame.Rect(1310, 465, 100,  95),  # between Lower Nav and Shields

    # Room interior physical collision obstacles
    pygame.Rect(880,  150,  60,  50),  # Cafeteria emergency button table
    pygame.Rect(1040, 480,  50,  40),  # Admin conference table
    pygame.Rect(650,  500,  25,  60),  # Electrical switchboard
    pygame.Rect(860,  630,  50,  40),  # Storage cargo crates
    pygame.Rect(335,  160,  40,  40),  # Upper Engine core block
    pygame.Rect(335,  580,  40,  40),  # Lower Engine core block
    pygame.Rect(175,  370,  35,  50),  # Reactor core pillar
    pygame.Rect(1510, 360,  35,  40),  # Navigation cockpit console
]

SKELD_ALL_SOLID_RECTS: List[pygame.Rect] = SKELD_OUTER_WALLS + SKELD_INTERIOR_WALLS


# ==========================================
# TASK DATA MODEL (Rich, inspectable)
# ==========================================
@dataclass(frozen=True)
class TaskDestination:
    """A physical task interaction destination on The Skeld."""
    id: str
    name: str
    room: str
    source_x: float
    source_y: float
    world_x: float
    world_y: float
    interaction_radius: float = SKELD_INTERACTION_RADIUS
    confidence: str = "VERIFIED"
    source_note: str = ""

    @property
    def world_pos(self) -> Tuple[float, float]:
        return (self.world_x, self.world_y)

    @property
    def display_pos(self) -> Tuple[int, int]:
        return world_to_screen(self.world_x, self.world_y)


# Complete list of all 40 authentic task destinations extracted from interactive map
# World positions have calibrated wall clearance so interaction circles remain safely inside room floor.
SKELD_ALL_TASK_DESTINATIONS: List[TaskDestination] = [
    TaskDestination("2",  "Upload Data",               "Admin",          5435.0, 2376.0, 1015.3, 450.0, 25.0, "VERIFIED", "Community map marker ID 2 (Admin desk upload)"),
    TaskDestination("3",  "Download Data",             "Electrical",     3230.0, 2562.0,  605.0, 485.0, 25.0, "VERIFIED", "Community map marker ID 3 (Electrical wall panel)"),
    TaskDestination("4",  "Download Data",             "Communications", 5684.5, 3764.0, 1061.9, 703.1, 25.0, "VERIFIED", "Community map marker ID 4 (Comms desk)"),
    TaskDestination("5",  "Download Data",             "Navigation",     8048.0, 1693.2, 1503.4, 320.0, 25.0, "VERIFIED", "Community map marker ID 5 (Nav upper terminal)"),
    TaskDestination("6",  "Download Data",             "Weapons",        6545.2,  541.0, 1222.7, 101.1, 25.0, "VERIFIED", "Community map marker ID 6 (Weapons wall panel)"),
    TaskDestination("7",  "Download Data",             "Cafeteria",      5611.0,  365.8, 1048.2,  70.0, 25.0, "VERIFIED", "Community map marker ID 7 (Cafeteria east wall)"),
    TaskDestination("8",  "Divert Power",              "Electrical",     3363.2, 2559.2,  628.3, 485.0, 25.0, "VERIFIED", "Community map marker ID 8 (Power distributor panel)"),
    TaskDestination("9",  "Accept Diverted Power",     "Security",       2807.8, 1795.0,  524.5, 350.0, 25.0, "VERIFIED", "Community map marker ID 9 (Security breaker panel)"),
    TaskDestination("10", "Accept Diverted Power",     "Lower Engine",   1744.0, 2909.5,  325.8, 550.0, 25.0, "VERIFIED", "Community map marker ID 10 (Lower Engine breaker)"),
    TaskDestination("11", "Accept Diverted Power",     "Upper Engine",   1882.8,  679.0,  351.7, 126.8, 25.0, "VERIFIED", "Community map marker ID 11 (Upper Engine breaker)"),
    TaskDestination("12", "Accept Diverted Power",     "Weapons",        7046.2,  874.5, 1316.3, 163.4, 25.0, "VERIFIED", "Community map marker ID 12 (Weapons breaker)"),
    TaskDestination("13", "Accept Diverted Power",     "Shields",        6920.0, 3070.5, 1292.7, 573.6, 25.0, "VERIFIED", "Community map marker ID 13 (Shields breaker)"),
    TaskDestination("14", "Accept Diverted Power",     "Navigation",     7869.5, 1688.8, 1470.1, 320.0, 25.0, "VERIFIED", "Community map marker ID 14 (Navigation breaker)"),
    TaskDestination("15", "Accept Diverted Power",     "Communications", 6096.1, 3781.4, 1120.0, 706.4, 25.0, "VERIFIED", "Community map marker ID 15 (Communications breaker)"),
    TaskDestination("16", "Accept Diverted Power",     "O2",             6493.0, 1705.0, 1208.4, 331.3, 25.0, "VERIFIED", "Community map marker ID 16 (O2 breaker)"),
    TaskDestination("20", "Empty Garbage",             "Cafeteria",      5791.0,  549.0, 1081.8, 102.6, 25.0, "VERIFIED", "Community map marker ID 20 (Cafeteria garbage chute)"),
    TaskDestination("52", "Submit Scan",               "MedBay",         3679.0, 2184.0,  687.3, 400.0, 25.0, "VERIFIED", "Community map marker ID 52 (MedBay visual scanner pad)"),
    TaskDestination("53", "Align Engine Output",       "Upper Engine",   1529.0, 1331.0,  285.6, 240.0, 25.0, "VERIFIED", "Community map marker ID 53 (Upper Engine console)"),
    TaskDestination("54", "Align Engine Output",       "Lower Engine",   1536.0, 3542.0,  286.9, 661.7, 25.0, "VERIFIED", "Community map marker ID 54 (Lower Engine console)"),
    TaskDestination("55", "Fuel Engines",              "Lower Engine",   1679.0, 3527.0,  313.6, 658.9, 25.0, "VERIFIED", "Community map marker ID 55 (Lower Engine fuel input)"),
    TaskDestination("56", "Fuel Engines",              "Upper Engine",   1674.0, 1338.0,  312.7, 240.0, 25.0, "VERIFIED", "Community map marker ID 56 (Upper Engine fuel input)"),
    TaskDestination("57", "Fix Wiring",                "Cafeteria",      4034.0,  314.0,  753.6,  70.0, 25.0, "VERIFIED", "Community map marker ID 57 (Cafeteria west wires)"),
    TaskDestination("58", "Fix Wiring",                "Navigation",     7601.0, 1933.0, 1428.6, 357.7, 25.0, "VERIFIED", "Community map marker ID 58 (Navigation wires)"),
    TaskDestination("59", "Fix Wiring",                "Admin",          5238.0, 2414.0,  990.0, 451.0, 25.0, "VERIFIED", "Community map marker ID 59 (Admin wires)"),
    TaskDestination("60", "Fix Wiring",                "Storage",        4640.0, 2820.0,  866.8, 526.8, 25.0, "VERIFIED", "Community map marker ID 60 (Storage wires)"),
    TaskDestination("61", "Fix Wiring",                "Electrical",     3594.2, 2628.9,  671.4, 480.0, 25.0, "VERIFIED", "Community map marker ID 61 (Electrical wires)"),
    TaskDestination("62", "Fix Wiring",                "Security",       2182.0, 2080.0,  407.6, 405.0, 25.0, "VERIFIED", "Community map marker ID 62 (Security wires)"),
    TaskDestination("64", "Clear Asteroids",           "Weapons",        6636.0,  950.0, 1239.6, 177.5, 25.0, "VERIFIED", "Community map marker ID 64 (Weapons visual turret)"),
    TaskDestination("65", "Chart Course",              "Navigation",     8236.0, 1852.0, 1538.5, 340.0, 25.0, "VERIFIED", "Community map marker ID 65 (Nav helm console)"),
    TaskDestination("66", "Stabilize Steering",        "Navigation",     8378.0, 2082.0, 1565.1, 388.9, 25.0, "VERIFIED", "Community map marker ID 66 (Nav steering panel)"),
    TaskDestination("67", "Clean O2 Filter",           "O2",             6036.0, 1789.0, 1127.6, 334.2, 25.0, "VERIFIED", "Community map marker ID 67 (O2 filter chute)"),
    TaskDestination("68", "Empty Chute",               "O2",             5868.0, 1851.0, 1105.0, 345.8, 25.0, "VERIFIED", "Community map marker ID 68 (O2 trash chute)"),
    TaskDestination("69", "Prime Shields",             "Shields",        6348.0, 3858.1, 1185.9, 715.0, 25.0, "VERIFIED", "Community map marker ID 69 (Shields visual panel)"),
    TaskDestination("70", "Empty Garbage/Empty Chute", "Storage",        5128.0, 4262.0,  957.9, 790.0, 25.0, "VERIFIED", "Community map marker ID 70 (Storage trash compactor lever)"),
    TaskDestination("71", "Fuel Engines",              "Storage",        4476.0, 3770.0,  836.1, 704.3, 25.0, "VERIFIED", "Community map marker ID 71 (Storage fuel canister station)"),
    TaskDestination("72", "Swipe Card",                "Admin",          5992.0, 2826.0, 1119.3, 527.9, 25.0, "VERIFIED", "Community map marker ID 72 (Admin card reader table)"),
    TaskDestination("73", "Calibrate Distributor",     "Electrical",     3932.0, 2618.0,  730.0, 489.1, 25.0, "VERIFIED", "Community map marker ID 73 (Distributor rotating dials)"),
    TaskDestination("74", "Unlock Manifolds",          "Reactor",         928.0, 1706.0,  173.4, 318.7, 25.0, "VERIFIED", "Community map marker ID 74 (Reactor 1-10 keypad)"),
    TaskDestination("75", "Start Reactor",             "Reactor",        1068.0, 2258.0,  199.5, 440.0, 25.0, "VERIFIED", "Community map marker ID 75 (Reactor Simon Says panel)"),
    TaskDestination("76", "Inspect Sample",            "MedBay",         3896.0, 2038.0,  727.8, 380.7, 25.0, "VERIFIED", "Community map marker ID 76 (MedBay test tube carousel)"),
]

# Quick lookup by task id
SKELD_TASK_MAP: Dict[str, TaskDestination] = {t.id: t for t in SKELD_ALL_TASK_DESTINATIONS}

# Backward compatibility tuple list: [(name, room, (wx, wy))]
SKELD_ALL_TASK_POSITIONS: List[Tuple[str, str, Tuple[float, float]]] = [
    (t.name, t.room, (t.world_x, t.world_y)) for t in SKELD_ALL_TASK_DESTINATIONS
]


# ==========================================
# ROOM DEFINITIONS & WALKABLE REGIONS
# ==========================================
class SkeldRoom:
    """Named room with walkable boundary and task list."""
    __slots__ = ("name", "center", "walkable_rect", "tasks", "vent_pos")

    def __init__(self, name: str, center: Tuple[float, float], walkable_rect: pygame.Rect,
                 tasks: Optional[List[TaskDestination]] = None, vent_pos: Optional[Tuple[float, float]] = None):
        self.name          = name
        self.center        = center           # (x, y) world coordinates (verified clear of obstacles)
        self.walkable_rect = walkable_rect   # pygame.Rect in world coordinates
        self.tasks         = tasks or []      # list of TaskDestination
        self.vent_pos      = vent_pos         # (x, y) world coordinates or None

# Group task destinations by room
_tasks_by_room: Dict[str, List[TaskDestination]] = {}
for _t in SKELD_ALL_TASK_DESTINATIONS:
    _tasks_by_room.setdefault(_t.room, []).append(_t)

SKELD_ROOMS: List[SkeldRoom] = [
    SkeldRoom("Cafeteria",      (903.6, 115.0),  SKELD_ROOM_FLOORS["Cafeteria"],      _tasks_by_room.get("Cafeteria", []),      (1074.1, 247.0)),
    SkeldRoom("Weapons",        (1277.4, 150.0), SKELD_ROOM_FLOORS["Weapons"],        _tasks_by_room.get("Weapons", []),        (1227.7, 123.7)),
    SkeldRoom("O2",             (1152.2, 340.0), SKELD_ROOM_FLOORS["O2"],             _tasks_by_room.get("O2", []),             None),
    SkeldRoom("Navigation",     (1480.0, 380.0), SKELD_ROOM_FLOORS["Navigation"],     _tasks_by_room.get("Navigation", []),     (1470.5, 344.5)),
    SkeldRoom("Shields",        (1242.6, 650.0), SKELD_ROOM_FLOORS["Shields"],        _tasks_by_room.get("Shields", []),        (1251.7, 717.7)),
    SkeldRoom("Communications", (1071.5, 750.0), SKELD_ROOM_FLOORS["Communications"], _tasks_by_room.get("Communications", []), None),
    SkeldRoom("Storage",        (880.0, 710.0),  SKELD_ROOM_FLOORS["Storage"],        _tasks_by_room.get("Storage", []),        None),
    SkeldRoom("Admin",          (1015.0, 500.0), SKELD_ROOM_FLOORS["Admin"],          _tasks_by_room.get("Admin", []),          (1016.6, 570.5)),
    SkeldRoom("Electrical",     (645.6, 570.0),  SKELD_ROOM_FLOORS["Electrical"],     _tasks_by_room.get("Electrical", []),     (603.0,  506.2)),
    SkeldRoom("Lower Engine",   (360.2, 650.0),  SKELD_ROOM_FLOORS["Lower Engine"],   _tasks_by_room.get("Lower Engine", []),   (419.2,  695.3)),
    SkeldRoom("Security",       (478.2, 414.3),  SKELD_ROOM_FLOORS["Security"],       _tasks_by_room.get("Security", []),       (509.6,  471.1)),
    SkeldRoom("Reactor",        (215.0, 470.0),  SKELD_ROOM_FLOORS["Reactor"],        _tasks_by_room.get("Reactor", []),        (195.4,  340.0)),
    SkeldRoom("Upper Engine",   (356.8, 220.0),  SKELD_ROOM_FLOORS["Upper Engine"],   _tasks_by_room.get("Upper Engine", []),   (418.4,  152.1)),
    SkeldRoom("MedBay",         (623.6, 360.0),  SKELD_ROOM_FLOORS["MedBay"],         _tasks_by_room.get("MedBay", []),         (574.6,  376.6)),
]

SKELD_ROOM_MAP: Dict[str, SkeldRoom] = {r.name: r for r in SKELD_ROOMS}


# ==========================================
# CORRIDOR REGION MAPPING & TRUE ROOM DETECTION
# ==========================================
SKELD_NAMED_REGIONS: Dict[str, pygame.Rect] = {
    # 14 rooms
    "Cafeteria":      SKELD_ROOM_FLOORS["Cafeteria"],
    "Weapons":        SKELD_ROOM_FLOORS["Weapons"],
    "O2":             SKELD_ROOM_FLOORS["O2"],
    "Navigation":     SKELD_ROOM_FLOORS["Navigation"],
    "Shields":        SKELD_ROOM_FLOORS["Shields"],
    "Communications": SKELD_ROOM_FLOORS["Communications"],
    "Storage":        SKELD_ROOM_FLOORS["Storage"],
    "Admin":          SKELD_ROOM_FLOORS["Admin"],
    "Electrical":     SKELD_ROOM_FLOORS["Electrical"],
    "Lower Engine":   SKELD_ROOM_FLOORS["Lower Engine"],
    "Security":       SKELD_ROOM_FLOORS["Security"],
    "Reactor":        SKELD_ROOM_FLOORS["Reactor"],
    "Upper Engine":   SKELD_ROOM_FLOORS["Upper Engine"],
    "MedBay":         SKELD_ROOM_FLOORS["MedBay"],
    # Hallways
    "MedBay Hallway":         pygame.Rect(430,  140, 250, 170),
    "West Hallway":           pygame.Rect(335,  260,  90, 270),
    "Upper Reactor Bridge":   pygame.Rect(210,  175,  70,  55),
    "Lower Reactor Bridge":   pygame.Rect(210,  605,  70,  55),
    "Central Hallway":        pygame.Rect(855,  255, 115, 265),
    "South-West Hallway":     pygame.Rect(430,  620, 350,  70),
    "South-East Hallway":     pygame.Rect(980,  640, 180,  60),
    "East Hallway":           pygame.Rect(1115, 135, 190, 430),
    "Upper Navigation Hall":  pygame.Rect(1300, 315, 115,  50),
    "Lower Navigation Hall":  pygame.Rect(1260, 415, 150, 155),
}

def get_room_or_region(wx: float, wy: float) -> str:
    """
    True geometric region detection based on spatial polygons/rects.
    Returns specific room or corridor name, or 'Corridor / Unclassified'.
    """
    pt = (int(round(wx)), int(round(wy)))
    # Check 14 rooms first
    for rname, rrect in SKELD_ROOM_FLOORS.items():
        if rrect.collidepoint(pt):
            return rname
    # Check named hallways
    for reg_name, reg_rect in SKELD_NAMED_REGIONS.items():
        if reg_name not in SKELD_ROOM_FLOORS and reg_rect.collidepoint(pt):
            return reg_name
    return "Corridor / Unclassified"


# ==========================================
# AUDITED PHYSICAL ROOM CONNECTIVITY GRAPH
# ==========================================
# Descriptive connectivity: rooms connected directly via walkable hallways/doorways
# Verified against physical collision geometry:
# - Electrical connects ONLY to Storage and Lower Engine (via South-West Hallway). NO door to MedBay or Security.
# - Security connects ONLY to Upper Engine and Lower Engine (via West Hallway). NO door to MedBay or Electrical.
# - MedBay connects ONLY to Cafeteria and Upper Engine (via MedBay Hallway). NO door to Security or Electrical.
SKELD_ROOM_GRAPH: Dict[str, List[str]] = {
    "Cafeteria":      ["Upper Engine", "Weapons", "MedBay", "Admin", "Storage"],
    "Weapons":        ["Cafeteria", "O2", "Navigation"],
    "O2":             ["Weapons", "Navigation", "Shields"],
    "Navigation":     ["Weapons", "O2", "Shields"],
    "Shields":        ["Navigation", "Storage", "Communications", "O2"],
    "Communications": ["Shields", "Storage"],
    "Storage":        ["Cafeteria", "Admin", "Communications", "Shields", "Electrical", "Lower Engine"],
    "Admin":          ["Cafeteria", "Storage"],
    "Electrical":     ["Storage", "Lower Engine"],
    "Lower Engine":   ["Electrical", "Storage", "Reactor", "Upper Engine", "Security"],
    "Security":       ["Upper Engine", "Lower Engine"],
    "Reactor":        ["Upper Engine", "Lower Engine"],
    "Upper Engine":   ["Reactor", "Lower Engine", "Security", "MedBay", "Cafeteria"],
    "MedBay":         ["Cafeteria", "Upper Engine"],
}


# ==========================================
# VENT NETWORKS (4 isolated systems, 14 vents)
# ==========================================
# Vent nodes are stored for future social deduction research
SKELD_VENT_SYSTEMS = [
    {"Reactor", "Upper Engine", "Lower Engine"},      # System 1: Left Wing / Reactor Loop (4 vents)
    {"Electrical", "MedBay", "Security"},             # System 2: Center-Left / Surveillance Loop (3 vents)
    {"Admin", "Cafeteria", "Hallway"},                # System 3: Admin / Cafeteria Loop (3 vents)
    {"Weapons", "Navigation", "Shields"},             # System 4: Right Wing / Navigation Loop (4 vents)
]

SKELD_ALL_VENTS = [
    {"id": 21, "room": "Cafeteria",    "world_pos": (1074.1, 247.0), "system": 3},
    {"id": 22, "room": "Weapons",      "world_pos": (1227.7, 123.7), "system": 4},
    {"id": 23, "room": "Navigation",   "world_pos": (1470.5, 344.5), "system": 4},
    {"id": 24, "room": "Navigation",   "world_pos": (1470.5, 451.3), "system": 4},
    {"id": 25, "room": "Shields",      "world_pos": (1251.7, 717.7), "system": 4},
    {"id": 26, "room": "Hallway",      "world_pos": (1246.7, 452.8), "system": 3},
    {"id": 27, "room": "Admin",        "world_pos": (1016.6, 570.5), "system": 3},
    {"id": 28, "room": "MedBay",       "world_pos": ( 574.6, 376.6), "system": 2},
    {"id": 29, "room": "Security",     "world_pos": ( 509.6, 471.1), "system": 2},
    {"id": 30, "room": "Electrical",   "world_pos": ( 603.0, 506.2), "system": 2},
    {"id": 31, "room": "Upper Engine", "world_pos": ( 418.4, 152.1), "system": 1},
    {"id": 32, "room": "Lower Engine", "world_pos": ( 419.2, 695.3), "system": 1},
    {"id": 33, "room": "Reactor",      "world_pos": ( 232.4, 470.0), "system": 1},
    {"id": 34, "room": "Reactor",      "world_pos": ( 195.4, 340.0), "system": 1},
]


# ==========================================
# SECURITY CAMERAS (4 physical cameras)
# ==========================================
SKELD_SECURITY_CAMERAS = [
    {"name": "West Hallway",   "world_pos": ( 331.7, 379.6)},
    {"name": "MedBay Hallway", "world_pos": ( 692.3, 161.9)},
    {"name": "Admin Hallway",  "world_pos": ( 950.5, 438.2)},
    {"name": "East Hallway",   "world_pos": (1379.3, 358.2)},
]


# ==========================================
# MAP-AWARE EFFICIENCY EVALUATION HELPER
# ==========================================
def compute_map_aware_efficiency(optimal_walkable_path_len: float, actual_agent_path_len: float) -> float:
    """
    Computes map-aware navigation efficiency percentage:
      (optimal_walkable_path_length / max(1e-6, actual_agent_path_length)) * 100.0
    Clamped strictly between [0.0, 100.0].
    For evaluation / diagnostics ONLY — NEVER used in PPO reward shaping.
    """
    if actual_agent_path_len <= 1e-6:
        return 0.0
    eff = (optimal_walkable_path_len / actual_agent_path_len) * 100.0
    return float(min(100.0, max(0.0, eff)))
