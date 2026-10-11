"""Paired significance for harness comparisons: exact McNemar plus Benjamini-Hochberg FDR.

Stdlib only. Outcomes are binary pass/fail per task, paired by task (paper arXiv:2609.20804, p6).
"""

from __future__ import annotations

from math import comb
from typing import Any, Mapping, Sequence

PASS_THRESHOLD = 0.5


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts b and c.

    Under the null each discordant pair is a fair coin, so p = 2 * P(Binomial(b + c, 1/2) <= min(b, c)),
    capped at 1. No discordant pairs means no evidence: p = 1.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def bh(pvals: Sequence[float], q: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg step-up. Returns (adjusted q-values, significant flags), in input order.

    Hypothesis i is significant when its adjusted q-value is <= q, which is the same set as
    p_(k) <= k * q / m for the largest passing rank k.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adjusted[i] = running
    return adjusted, [a <= q for a in adjusted]


def paired_counts(base: Mapping[Any, float], arm: Mapping[Any, float],
                  threshold: float = PASS_THRESHOLD) -> dict[str, int]:
    """Win/loss/tie of `arm` against `base` over the keys both scored (pass = score > threshold)."""
    wins = losses = ties = 0
    for key in base.keys() & arm.keys():
        a, b = arm[key] > threshold, base[key] > threshold
        if a and not b:
            wins += 1
        elif b and not a:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def compare(baseline: Mapping[Any, float], arms: Mapping[str, Mapping[Any, float]],
            q: float = 0.05) -> dict[str, dict[str, Any]]:
    """Each arm against the baseline: wins, losses, ties, two-sided exact p, BH q across arms, significant."""
    names = list(arms)
    counts = [paired_counts(baseline, arms[name]) for name in names]
    pvals = [mcnemar_exact(c["losses"], c["wins"]) for c in counts]
    qvals, flags = bh(pvals, q)
    return {name: {**count, "p": p, "q": qv, "significant": flag}
            for name, count, p, qv, flag in zip(names, counts, pvals, qvals, flags)}
