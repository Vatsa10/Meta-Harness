"""Idle watchdog fused with an overall deadline."""

from __future__ import annotations

import time


class Aborted(Exception):
    def __init__(self, reason: str):
        super().__init__(f"aborted: {reason}")
        self.reason = reason


class Watchdog:
    """kick() on progress; check() raises Aborted("idle" | "deadline"), whichever fired first."""

    def __init__(self, idle: float, deadline: float, clock=time.monotonic):
        self.idle = idle
        self.deadline = deadline
        self._clock = clock
        self._start = clock()
        self._last = self._start

    def kick(self) -> None:
        self._last = self._clock()

    def remaining(self) -> float:
        now = self._clock()
        return max(0.0, min(self.deadline - (now - self._start), self.idle - (now - self._last)))

    def check(self) -> None:
        now = self._clock()
        deadline_at = self._start + self.deadline
        idle_at = self._last + self.idle
        deadline_hit = now >= deadline_at
        idle_hit = now >= idle_at
        if deadline_hit and idle_hit:
            raise Aborted("deadline" if deadline_at <= idle_at else "idle")
        if deadline_hit:
            raise Aborted("deadline")
        if idle_hit:
            raise Aborted("idle")
