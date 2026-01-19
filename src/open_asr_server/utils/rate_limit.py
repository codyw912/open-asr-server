"""In-memory rate limiting helpers."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Simple fixed-window rate limiter (per process)."""

    def __init__(self, limit_per_minute: int, window_seconds: int = 60):
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be positive")
        self.limit_per_minute = limit_per_minute
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, int]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            window_start, count = self._buckets.get(key, (now, 0))
            if now - window_start >= self.window_seconds:
                window_start, count = now, 0
            if count >= self.limit_per_minute:
                self._buckets[key] = (window_start, count)
                return False
            self._buckets[key] = (window_start, count + 1)
        return True
