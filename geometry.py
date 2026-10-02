"""
Geometry and math utilities for 2D collision detection and line-of-sight calculations.
"""

import math
from typing import Tuple, Optional, List
import pygame


def normalize_vector(vx: float, vy: float) -> Tuple[float, float]:
    """Return unit vector. If zero vector, return (0.0, 0.0)."""
    length = math.hypot(vx, vy)
    if length < 1e-7:
        return 0.0, 0.0
    return vx / length, vy / length


def line_segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection between segment p1->p2 and segment p3->p4.
    Returns intersection point (x, y) if segments intersect, otherwise None.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    dx12 = x2 - x1
    dy12 = y2 - y1
    dx34 = x4 - x3
    dy34 = y4 - y3

    denom = dx12 * dy34 - dy12 * dx34
    if abs(denom) < 1e-9:
        # Collinear or parallel
        return None

    dx13 = x1 - x3
    dy13 = y1 - y3

    t = (dx13 * dy34 - dy13 * dx34) / -denom
    u = (dx12 * dy13 - dy12 * dx13) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = x1 + t * dx12
        iy = y1 + t * dy12
        return ix, iy

    return None


def ray_segment_intersection(
    origin: Tuple[float, float],
    dir_vector: Tuple[float, float],
    max_dist: float,
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> Optional[float]:
    """
    Find distance t along ray origin + t * dir_vector (0 <= t <= max_dist)
    intersecting segment p1->p2. Returns smallest t, or None.
    """
    ox, oy = origin
    dx, dy = dir_vector
    x1, y1 = p1
    x2, y2 = p2

    sx = x2 - x1
    sy = y2 - y1

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None

    t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

    if 0.0 <= t <= max_dist and 0.0 <= u <= 1.0:
        return t

    return None


def segment_intersects_rect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rect: pygame.Rect
) -> bool:
    """
    Check if line segment p1->p2 intersects rectangle `rect`.
    Includes checks for:
    - Bounding box rejection
    - Endpoints inside rectangle
    - Intersection with any of the 4 edges
    """
    x1, y1 = p1
    x2, y2 = p2

    # Quick rejection: if segment AABB does not overlap rect
    seg_left = min(x1, x2)
    seg_right = max(x1, x2)
    seg_top = min(y1, y2)
    seg_bottom = max(y1, y2)

    if (seg_right < rect.left or seg_left > rect.right or
        seg_bottom < rect.top or seg_top > rect.bottom):
        return False

    # Check if either endpoint is inside rect
    if rect.collidepoint(x1, y1) or rect.collidepoint(x2, y2):
        return True

    # 4 edges of the rectangle
    corners = [
        (rect.left, rect.top),
        (rect.right, rect.top),
        (rect.right, rect.bottom),
        (rect.left, rect.bottom)
    ]
    edges = [
        (corners[0], corners[1]),  # top
        (corners[1], corners[2]),  # right
        (corners[2], corners[3]),  # bottom
        (corners[3], corners[0])   # left
    ]

    for edge_p1, edge_p2 in edges:
        if line_segment_intersection(p1, p2, edge_p1, edge_p2) is not None:
            return True

    return False


def cast_ray_against_rects(
    origin: Tuple[float, float],
    dir_vector: Tuple[float, float],
    max_dist: float,
    rectangles: List[pygame.Rect]
) -> float:
    """
    Cast a ray from origin along dir_vector up to max_dist.
    Returns the distance to the closest intersection with any rectangle edge,
    or max_dist if no intersection.
    """
    closest_dist = max_dist

    for rect in rectangles:
        corners = [
            (rect.left, rect.top),
            (rect.right, rect.top),
            (rect.right, rect.bottom),
            (rect.left, rect.bottom)
        ]
        edges = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0])
        ]
        for ep1, ep2 in edges:
            dist = ray_segment_intersection(origin, dir_vector, closest_dist, ep1, ep2)
            if dist is not None and dist < closest_dist:
                closest_dist = dist

    return closest_dist


def compute_wall_raycasts(
    origin: Tuple[float, float],
    num_rays: int,
    max_dist: float,
    rectangles: List[pygame.Rect]
) -> List[float]:
    """
    Cast num_rays evenly spaced radial rays from origin.
    Returns list of normalized distances in [0.0, 1.0], where:
      0.0 = obstacle/wall right against origin
      1.0 = no obstacle within max_dist
    Angles are: 0, 2*pi/num_rays, 4*pi/num_rays, ... 
    For 16 rays (360 / 16 = 22.5 deg): 0, 22.5, 45, 67.5, ... 337.5 deg.
    """
    normalized_distances = []
    angle_step = (2.0 * math.pi) / float(num_rays)

    for i in range(num_rays):
        angle = i * angle_step
        dx = math.cos(angle)
        dy = math.sin(angle)
        dist = cast_ray_against_rects(origin, (dx, dy), max_dist, rectangles)
        norm_dist = max(0.0, min(1.0, dist / max_dist))
        normalized_distances.append(norm_dist)

    return normalized_distances


def check_line_of_sight(
    observer_pos: Tuple[float, float],
    facing_angle_rad: float,
    target_pos: Tuple[float, float],
    vision_distance: float,
    vision_angle_rad: float,
    obstacles: List[pygame.Rect]
) -> Tuple[bool, dict]:
    """
    Determine if target_pos is visible from observer_pos.
    Returns (is_visible, debug_info_dict).
    
    Conditions for visibility:
    1. Distance <= vision_distance
    2. Angle difference with facing direction <= vision_angle_rad / 2
    3. Direct line segment from observer_pos to target_pos is NOT blocked by any obstacle.
    """
    ox, oy = observer_pos
    tx, ty = target_pos

    dx = tx - ox
    dy = ty - oy
    dist = math.hypot(dx, dy)

    debug_info = {
        "distance": dist,
        "in_distance": dist <= vision_distance,
        "angle_deg": 0.0,
        "in_angle": False,
        "blocked_by_obstacle": False,
        "is_visible": False
    }

    if dist > vision_distance:
        return False, debug_info

    # Angle check
    target_angle = math.atan2(dy, dx)
    # Angular difference wrapped to [-pi, pi]
    angle_diff = (target_angle - facing_angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    abs_angle_diff = abs(angle_diff)
    debug_info["angle_deg"] = math.degrees(abs_angle_diff)

    half_fov = vision_angle_rad / 2.0
    if abs_angle_diff > half_fov:
        return False, debug_info

    debug_info["in_angle"] = True

    # Obstacle occlusion check
    for obstacle in obstacles:
        if segment_intersects_rect(observer_pos, target_pos, obstacle):
            debug_info["blocked_by_obstacle"] = True
            return False, debug_info

    debug_info["is_visible"] = True
    return True, debug_info


def point_to_rect_distance(px: float, py: float, rect: pygame.Rect) -> float:
    """
    Calculate the minimum Euclidean distance from point (px, py) to axis-aligned rectangle rect.
    Returns 0.0 if the point is inside or on the boundary of the rectangle.
    """
    cx = max(float(rect.left), min(float(px), float(rect.right)))
    cy = max(float(rect.top), min(float(py), float(rect.bottom)))
    return math.hypot(px - cx, py - cy)


def is_point_clear_of_obstacles(
    px: float,
    py: float,
    radius: float,
    clearance: float,
    obstacles: List[pygame.Rect]
) -> bool:
    """
    Verify that a circular entity at (px, py) with given radius and extra clearance
    does not intersect, touch, or violate the clearance boundary of any obstacle rectangle.
    Distance from (px, py) to any obstacle must be >= (radius + clearance).
    """
    min_safe_dist = float(radius + clearance)
    for rect in obstacles:
        if point_to_rect_distance(px, py, rect) < min_safe_dist:
            return False
    return True


def does_segment_intersect_obstacles(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacles: List[pygame.Rect],
    inflate_radius: float = 0.0
) -> bool:
    """
    Check if the finite line segment from p1 to p2 intersects any obstacle in `obstacles`.
    If inflate_radius > 0, each obstacle AABB is inflated by inflate_radius on all 4 sides
    to account for entity physical radius (e.g. PLAYER_RADIUS).
    """
    for rect in obstacles:
        if inflate_radius > 0.0:
            test_rect = pygame.Rect(
                rect.left - inflate_radius,
                rect.top - inflate_radius,
                rect.width + 2.0 * inflate_radius,
                rect.height + 2.0 * inflate_radius
            )
        else:
            test_rect = rect
        if segment_intersects_rect(p1, p2, test_rect):
            return True
    return False


