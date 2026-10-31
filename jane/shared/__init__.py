"""Role-neutral settings and compatibility checks; no policy or trainer imports."""

from .settings import GameSettings, load_settings

__all__ = ["GameSettings", "load_settings"]
