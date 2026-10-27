"""Application state dataclass — shared between the renderer and app loop."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pygame
from spectator.playback import FrameState, EventMarker


@dataclass
class SpectatorAppState:
    """All mutable UI/playback state passed to the renderer on each frame."""

    # Playback
    frame_state: FrameState | None = None
    playing: bool = False
    playback_speed: float = 1.0
    playback_mode: str = "manual"  # "follow_latest" | "auto_next" | "manual"

    # Replay info
    current_replay_path: str | None = None
    player_duration: float = 0.0
    event_markers: list[EventMarker] = field(default_factory=list)
    episode_info: dict | None = None

    # Library
    library_entries: list[dict] = field(default_factory=list)
    watch_dir: str | None = None

    # Selection
    selected_player: str | None = None

    # View toggles
    view_options: dict = field(default_factory=lambda: {
        "hud": True,
        "roles": False,
        "routes": False,
        "visibility": False,
        "tasks": True,
        "labels": True,
        "collision": False,
        "events": True,
        "bodies": True,
    })

    # Rendering
    fps: float = 0.0

    # Internal: set by renderer for mouse interaction
    _scrub_rect: pygame.Rect | None = None
    _library_hitboxes: list = field(default_factory=list)
    _player_hitboxes: dict = field(default_factory=dict)
    training_metrics: dict = field(default_factory=dict)
