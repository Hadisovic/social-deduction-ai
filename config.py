"""
Configuration and constants for the 2D Stealth Environment sandbox.
All arena parameters, speeds, positions, and routes are centralized here.
"""

import math
import pygame

# ==========================================
# WINDOW & DISPLAY
# ==========================================
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
FPS = 60
WINDOW_TITLE = "Stealth Sandbox - Phase 1 (Human-Playable 2D Arena)"

# ==========================================
# COLOR PALETTE (AI Experiment Aesthetic)
# ==========================================
# Sleek, modern, desaturated laboratory style
COLOR_BG = (244, 246, 249)                 # Soft neutral laboratory background
COLOR_GRID = (233, 236, 241)               # Subtle grid accent
COLOR_PLAY_AREA_BG = (255, 255, 255)       # Crisp white play surface

# Solid Boundaries & Obstacles
COLOR_BOUNDARY_WALL = (45, 52, 64)         # Outer perimeter wall
COLOR_OBSTACLE = (55, 63, 76)              # Solid cross obstacles
COLOR_OBSTACLE_BORDER = (35, 41, 51)       # Crisp obstacle outlines
COLOR_OBSTACLE_ACCENT = (75, 85, 102)      # Subtle inner geometric styling

# Escapee (Player)
COLOR_PLAYER = (0, 195, 175)               # Vibrant teal / cyan
COLOR_PLAYER_BORDER = (0, 135, 120)        # Darker teal border
COLOR_PLAYER_CORE = (180, 255, 245)        # Glowing core dot

# Goal
COLOR_GOAL = (35, 195, 95)                 # Emerald green
COLOR_GOAL_GLOW = (35, 195, 95, 45)        # Translucent glow aura
COLOR_GOAL_CORE = (210, 255, 225)          # Bright center
COLOR_GOAL_TEXT = (25, 150, 70)

# Observers (Distinct colors for instant identification)
COLOR_OBSERVER_1 = (245, 125, 30)          # Orange
COLOR_OBSERVER_2 = (155, 75, 225)          # Purple
COLOR_OBSERVER_3 = (235, 55, 55)           # Red
COLOR_OBSERVER_CORE = (255, 255, 255)      # Center eye highlight

# Text & Overlays
COLOR_TEXT = (35, 42, 54)
COLOR_TEXT_MUTED = (115, 125, 140)
COLOR_ALERT = (225, 45, 45)
COLOR_SUCCESS = (30, 185, 90)

# Debug Visualization Colors
COLOR_DEBUG_RAY_CLEAR = (50, 205, 50, 200)      # Green line: clear LOS to player
COLOR_DEBUG_RAY_BLOCKED = (235, 120, 20, 200)   # Orange line: in FOV but blocked
COLOR_DEBUG_RAY_IDLE = (160, 170, 185, 80)      # Faint line: outside FOV
COLOR_DEBUG_PATH = (100, 120, 150, 180)
COLOR_DEBUG_WAYPOINT = (70, 90, 120)
COLOR_DEBUG_COLLIDER = (255, 80, 80)

# ==========================================
# ARENA & BOUNDARIES
# ==========================================
ROOM_MARGIN = 35  # Thickness of outer boundary walls

# Playable interior rectangle
PLAY_AREA = pygame.Rect(
    ROOM_MARGIN,
    ROOM_MARGIN,
    WINDOW_WIDTH - 2 * ROOM_MARGIN,
    WINDOW_HEIGHT - 2 * ROOM_MARGIN
)

# 4 solid perimeter boundary walls
BOUNDARY_WALLS = [
    pygame.Rect(0, 0, WINDOW_WIDTH, ROOM_MARGIN),                              # Top
    pygame.Rect(0, WINDOW_HEIGHT - ROOM_MARGIN, WINDOW_WIDTH, ROOM_MARGIN),    # Bottom
    pygame.Rect(0, 0, ROOM_MARGIN, WINDOW_HEIGHT),                             # Left
    pygame.Rect(WINDOW_WIDTH - ROOM_MARGIN, 0, ROOM_MARGIN, WINDOW_HEIGHT),    # Right
]

# ==========================================
# INTERIOR OBSTACLES (Cross Structure)
# ==========================================
# Conceptual arrangement:
#              [ vertical wall ]
# [ horizontal ]              [ horizontal ]
#              [ vertical wall ]
OBSTACLE_RECTS = [
    # Top vertical wall
    pygame.Rect(520, 95, 60, 160),
    # Bottom vertical wall
    pygame.Rect(520, 445, 60, 160),
    # Left horizontal wall
    pygame.Rect(175, 320, 250, 60),
    # Right horizontal wall
    pygame.Rect(675, 320, 250, 60),
]

# Combined list of all solid obstacles (walls + bounds) for collision
ALL_SOLID_RECTS = BOUNDARY_WALLS + OBSTACLE_RECTS

# ==========================================
# ESCAPEE (PLAYER) CONFIG
# ==========================================
PLAYER_START_POS = (95.0, 595.0)   # Lower-left corner
PLAYER_RADIUS = 14.0               # Circle radius (diameter 28)
PLAYER_SPEED = 185.0               # Pixels per second (constant)

# ==========================================
# GOAL CONFIG
# ==========================================
GOAL_POS = (990.0, 105.0)          # Upper-right corner
GOAL_RADIUS = 28.0                 # Target trigger radius

# ==========================================
# OBSERVERS CONFIG
# ==========================================
OBSERVER_RADIUS = 15.0
VISION_DISTANCE = 225.0            # Sight range in pixels
VISION_ANGLE_DEG = 72.0            # Field of view arc in degrees
VISION_ANGLE_RAD = math.radians(VISION_ANGLE_DEG)
VISION_CONE_ALPHA = 45             # Translucency for vision cone polygon
VISION_RAY_COUNT = 36              # Number of rays for smooth shadow-casting cone rendering

# Three Observers configuration with deterministic patrol loops
OBSERVER_CONFIGS = [
    {
        "id": 1,
        "name": "Observer 1 (Orange)",
        "color": COLOR_OBSERVER_1,
        "speed": 110.0,
        "patrol_points": [
            (470.0, 65.0),
            (630.0, 65.0),
            (630.0, 285.0),
            (470.0, 285.0),
        ],
        "initial_waypoint_index": 1,  # Moves towards (630, 65) initially
    },
    {
        "id": 2,
        "name": "Observer 2 (Purple)",
        "color": COLOR_OBSERVER_2,
        "speed": 95.0,
        "patrol_points": [
            (115.0, 260.0),
            (480.0, 260.0),
            (480.0, 440.0),
            (115.0, 440.0),
        ],
        "initial_waypoint_index": 1,  # Moves towards (480, 260) initially
    },
    {
        "id": 3,
        "name": "Observer 3 (Red)",
        "color": COLOR_OBSERVER_3,
        "speed": 105.0,
        "patrol_points": [
            (470.0, 415.0),
            (470.0, 635.0),
            (630.0, 635.0),
            (630.0, 415.0),
        ],
        "initial_waypoint_index": 1,  # Moves towards (470, 635) initially
    },
]

# ==========================================
# REINFORCEMENT LEARNING CONFIGURATION (PHASE 2)
# ==========================================
ARENA_WIDTH = float(WINDOW_WIDTH)          # 1100.0 px for normalization
ARENA_HEIGHT = float(WINDOW_HEIGHT)        # 700.0 px for normalization

# Simulation Timestep for RL mode (independent of rendering FPS)
RL_DT = 1.0 / 30.0                         # 30 Hz fixed physics step

# Continuous Action Settings
ACTION_DEADZONE = 0.05                     # Magnitude deadzone under which action is stationary

# Observation Raycasts (Local geometric awareness)
RAY_COUNT = 16                             # 16 directions: 0, 22.5, 45, 67.5, ... 337.5 deg
RAY_MAX_DISTANCE = 300.0                   # Maximum sensing distance in pixels
OBSERVATION_SIZE = 43                      # Total float32 values in observation vector

# Rewards & Penalties
GOAL_REWARD = 100.0                        # Terminal reward for reaching goal
DETECTION_PENALTY = -100.0                 # Terminal penalty for being detected
PROGRESS_CHECK_INTERVAL = 0.5              # Simulation seconds between new-best progress checks
MIN_PROGRESS_DISTANCE = 5.0                # Minimum pixel advancement to qualify as progress
PROGRESS_REWARD_SCALE = 0.05               # Default progress reward scale (preserved for Stage 1)
STAGE1_PROGRESS_REWARD_SCALE = 0.05        # Stage 1 progress reward scale (+0.05 per pixel)
STAGE2_PROGRESS_REWARD_SCALE = 0.03        # Stage 2 progress reward scale (+0.03 per pixel)
STEP_PENALTY = 0.0                         # Global step penalty (0.0 for Stages 3-5)
STAGE1_STEP_PENALTY = -0.01                # Stage-1-specific time/step penalty (-0.01/step ≈ -0.30/s)
STAGE2_STEP_PENALTY = -0.01                # Stage-2-specific time/step penalty (-0.01/step ≈ -0.30/s)
COLLISION_PENALTY = 0.0                    # No wall collision penalty

# Episode Limits
MAX_EPISODE_TIME = 30.0                    # Maximum episode duration in simulation seconds

# ==========================================
# STAGE 1 PPO TRAINING CONFIGURATION (PHASE 3A)
# ==========================================
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_SEED = 42
PPO_NET_ARCH = dict(pi=[64, 64], vf=[64, 64])

PPO_TOTAL_TIMESTEPS = 250000
PPO_CHECKPOINT_FREQ = 10000                # Checkpoint & Eval frequency in training steps
PPO_EVAL_EPISODES = 20                     # Episodes per periodic evaluation (deterministic)
PPO_FINAL_EVAL_EPISODES = 100              # Final evaluation episodes (stochastic & deterministic)
PPO_EVAL_SEED = 12345                      # Fixed seed for reproducible evaluation layout sequences

# Stage 1 Spawn & Goal Randomization (Phase 3A extension)
STAGE1_SPAWN_MARGIN = 50.0                 # Margin inside boundary walls for valid random positions
STAGE1_MIN_START_GOAL_DISTANCE = 350.0     # Minimum Euclidean distance between player spawn and goal

# ==========================================
# STAGE 2 PPO TRAINING CONFIGURATION (PHASE 3B)
# ==========================================
STAGE2_SPAWN_MARGIN = 50.0                 # Margin inside boundary walls for valid random positions
STAGE2_MIN_START_GOAL_DISTANCE = 350.0     # Minimum Euclidean distance between player spawn and goal
STAGE2_OBSTACLE_SPAWN_CLEARANCE = 25.0     # Extra clearance between entity boundary and obstacle rects

# Stage 2 Episode Limits & Blocked Penalties
STAGE2_MAX_EPISODE_TIME = 20.0              # Maximum episode duration in simulation seconds for Stage 2 (600 steps at 30Hz)
STAGE2_TIMEOUT_PENALTY = -50.0              # Truncation penalty when timing out in Stage 2
STAGE2_BLOCKED_STEP_PENALTY = -0.02         # Short blocked contact penalty per step (steps 1-14)
STAGE2_PROLONGED_BLOCKED_PENALTY = -0.05    # Escalated penalty when blocked >= 15 consecutive steps (0.5s at 30Hz)
STAGE2_BLOCKED_RATIO_THRESHOLD = 0.20       # actual_distance < expected_distance * 0.20 defines a blocked step
STAGE2_PROLONGED_BLOCKED_STEPS = 15         # Number of consecutive blocked steps to declare stuck flag = 1.0

# Stage 2 Stagnation System Constants
STAGE2_STAGNATION_TIME = 2.0               # Simulated seconds to trigger stagnation penalty
STAGE2_STAGNATION_STEPS = 60               # 60 steps at 30Hz = 2.0s
STAGE2_STAGNATION_RADIUS = 10.0            # Radius in pixels around anchor to count as stagnant
STAGE2_STAGNATION_PENALTY = -25.0          # One-time penalty per stagnation event
STAGE2_STAGNATION_ESCAPE_DISTANCE = 40.0   # Displacement in pixels required to rearm stagnation penalty

# Stage 2 Recovery System Constants
STAGE2_RECOVERY_STEPS = 10                 # Consecutive unblocked steps required for genuine recovery
STAGE2_RECOVERY_ESCAPE_DISTANCE = 40.0     # Distance from stuck anchor required for genuine recovery
STAGE2_RECOVERY_REFUND_FRACTION = 0.50     # 50% refund of accumulated blocked penalties
STAGE2_RECOVERY_MAX_REWARD = 2.0           # Maximum cap on recovery reward

PPO_STAGE2_TOTAL_TIMESTEPS = 250000
PPO_STAGE2_CHECKPOINT_FREQ = 10000
PPO_STAGE2_EVAL_EPISODES = 30              # Episodes per periodic evaluation (deterministic)
PPO_STAGE2_FINAL_EVAL_EPISODES = 100       # Final evaluation episodes
PPO_STAGE2_EVAL_SEED = 22345               # Fixed evaluation seed for reproducible checkpoint comparison




