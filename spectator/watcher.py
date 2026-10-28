"""File-watching service for automatic replay discovery.

Runs in the foreground and is polled by the main loop.
Does not use threads — it simply checks file system modification times.
"""
from __future__ import annotations

import time
from pathlib import Path

from spectator.replay_format import ReplayLibrary


class WatchService:
    """Polls a directory for new or updated replay files."""

    REFRESH_INTERVAL = 2.0  # seconds between directory scans

    def __init__(self, library: ReplayLibrary):
        self.library = library
        self._last_refresh: float = 0.0

    def poll(self) -> bool:
        """Call this from the main loop.  Returns True when the index changed."""
        now = time.monotonic()
        if now - self._last_refresh < self.REFRESH_INTERVAL:
            return False
        self._last_refresh = now
        return self.library.refresh()

    def latest_complete_path(self) -> Path | None:
        """Return the path of the most recently written complete replay, or None."""
        latest = self.library.latest
        if latest and latest.get("complete"):
            return Path(latest["path"])
        # Fall back to the last entry regardless
        entries = self.library.entries
        for entry in reversed(entries):
            if entry.get("complete"):
                return Path(entry["path"])
        return None

    def next_after(self, current_path: str | None) -> Path | None:
        """Return the next complete replay after current_path, or None."""
        entries = [e for e in self.library.entries if e.get("complete")]
        if not entries:
            return None
        if current_path is None:
            return Path(entries[0]["path"])
        paths = [e["path"] for e in entries]
        try:
            idx = paths.index(current_path)
            if idx + 1 < len(entries):
                return Path(entries[idx + 1]["path"])
        except ValueError:
            pass
        return None
