"""The learn cycle: pick the failure that costs most, propose one artifact, score it by replay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .cc_history import FailureEpisode, Session, failure_episodes
from .harness_store import HarnessStore
from .replay import episode_signature


@dataclass
class FailureClass:
    signature: str
    kind: str
    tool: str
    count: int = 0
    episodes: list[FailureEpisode] = field(default_factory=list)


def rank_failures(sessions: Sequence[Session]) -> list[FailureClass]:
    """Every observed failure class, most frequent first."""
    grouped: dict[str, FailureClass] = {}
    for session in sessions:
        for episode in failure_episodes(session):
            signature = episode_signature(episode)
            tools = list(episode.tools or [])
            entry = grouped.get(signature)
            if entry is None:
                entry = FailureClass(signature=signature, kind=episode.kind,
                                     tool=tools[0] if tools else "unknown")
                grouped[signature] = entry
            entry.count += 1
            entry.episodes.append(episode)
    return sorted(grouped.values(), key=lambda f: (-f.count, f.signature))


def select_target(sessions: Sequence[Session], store: HarnessStore) -> FailureClass | None:
    """The most frequent failure not already covered by an installed artifact or a tombstone."""
    covered = store.covered()
    for failure in rank_failures(sessions):
        if failure.signature not in covered:
            return failure
    return None


__all__ = ["FailureClass", "rank_failures", "select_target"]
