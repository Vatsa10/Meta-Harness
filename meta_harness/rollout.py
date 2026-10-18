"""Online rollout: sticky traffic split between a control and a treatment, redacted JSONL
receipts, and a measured promote / hold / rollback verdict (stdlib only).

Receipts and limits worth knowing:
- Privacy: a receipt never holds the raw routing key, only key_hash = sha256(salt + key)[:16].
  meta values go through redact() and a PII scrub (emails, phone numbers, card-like or long
  digit runs). The scrub is best effort, so keep meta small and free of free text.
- Concurrency: each row is one append of one line under a per-Router lock. Several processes may
  append to one file, but only short rows (small meta) keep the OS append atomic; the lock does
  not span processes. A torn line is skipped on read and never glued to the next row.
- Verdict: both the z-test and Fisher exact must agree (p < alpha) before a difference counts as
  significant. A cost increase above the limit means rollback even when treatment is a
  significant win. Rows with success=None count for n, cost and latency, never for success rates.
  Fewer than min_samples judged rows in either arm means hold.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .redact import redact

PII = (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
       re.compile(r"\+?\d(?:[\s().-]?\d){6,}"))
PII_MASK = "[pii]"


def scrub(value: Any) -> Any:
    """redact() plus removal of emails, phone numbers and card-like digit runs, recursively."""
    if isinstance(value, str):
        value = redact(value)
        for pattern in PII:
            value = pattern.sub(PII_MASK, value)
        return value
    if isinstance(value, dict):
        return {scrub(str(k)): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


ARMS = ("control", "treatment")
BUCKETS = 10000


class Router:
    """Deterministic, sticky assignment: bucket = sha256(salt + key) mod 10000;
    treatment when bucket < percent * 100."""

    def __init__(self, control: Any, treatment: Any, percent: float, holdout_salt: str = "",
                 log: str | Path | None = None, salt: str | None = None) -> None:
        if salt is not None:  # alias of holdout_salt
            holdout_salt = salt
        if not 0 <= percent <= 100:
            raise ValueError(f"percent must be in [0, 100], got {percent!r}")
        self.control, self.treatment, self.percent = control, treatment, float(percent)
        self.holdout_salt = self.salt = holdout_salt
        self.log = Path(log) if log else None
        self._lock = threading.Lock()

    def bucket(self, key: str) -> int:
        digest = hashlib.sha256((self.salt + str(key)).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % BUCKETS

    def key_hash(self, key: str) -> str:
        return hashlib.sha256((self.salt + str(key)).encode("utf-8")).hexdigest()[:16]

    def route(self, key: str) -> str:
        return "treatment" if self.bucket(key) < self.percent * BUCKETS / 100 else "control"

    def target(self, key: str) -> Any:
        return self.treatment if self.route(key) == "treatment" else self.control

    def record(self, key: str, arm: str, success: bool | None = None, cost: float = 0.0,
               latency_ms: float = 0.0, meta: dict | None = None, log: str | Path | None = None) -> dict:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        path = Path(log) if log else self.log
        if path is None:
            raise ValueError("no log path: pass log= to Router or record")
        row = {"ts": time.time(), "key_hash": self.key_hash(key), "arm": arm,
               "success": None if success is None else bool(success),
               "cost": float(cost), "latency_ms": float(latency_ms),
               "meta": scrub(meta or {})}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            lead = ""
            if path.exists() and path.stat().st_size:
                with path.open("rb") as handle:
                    handle.seek(-1, 2)
                    if handle.read(1) != b"\n":
                        lead = "\n"  # a torn tail must not swallow this row
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(lead + line + "\n")
        return row


def read_log(path: str | Path) -> list[dict]:
    """Rows of a receipt log; blank, torn or non-object lines are skipped."""
    rows = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("arm") in ARMS:
                rows.append(row)
    return rows


def z_test(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test, two-sided: (z, p) for arm 2 minus arm 1."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (s2 / n2 - s1 / n1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]: sum of the probabilities of every table
    with the same margins that are no more likely than the observed one."""
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if total == 0:
        return 1.0
    denom = math.comb(total, col1)

    def weight(x: int) -> int:
        return math.comb(row1, x) * math.comb(row2, col1 - x)

    lo, hi = max(0, col1 - row2), min(row1, col1)
    observed = weight(a)
    return min(1.0, sum(w for w in map(weight, range(lo, hi + 1)) if w <= observed) / denom)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(rows: Iterable[dict]) -> dict[str, dict]:
    rows = list(rows)
    out = {}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        judged = [r for r in mine if r.get("success") is not None]
        wins = sum(1 for r in judged if r["success"])
        costs = [float(r.get("cost") or 0.0) for r in mine]
        lats = [float(r.get("latency_ms") or 0.0) for r in mine]
        out[arm] = {"n": len(mine), "n_judged": len(judged), "successes": wins,
                    "success_rate": wins / len(judged) if judged else None,
                    "mean_cost": sum(costs) / len(costs) if costs else 0.0,
                    "latency_p50_ms": _quantile(lats, 0.5), "latency_p95_ms": _quantile(lats, 0.95)}
    return out


def analyze(rows: list[dict], min_samples: int = 200, max_cost_increase: float = 0.1,
            alpha: float = 0.05) -> dict:
    arms = summarize(rows)
    c, t = arms["control"], arms["treatment"]
    z, z_p = z_test(c["successes"], c["n_judged"], t["successes"], t["n_judged"])
    f_p = fisher_exact(c["successes"], c["n_judged"] - c["successes"],
                       t["successes"], t["n_judged"] - t["successes"])
    if c["mean_cost"] > 0:
        cost_increase = (t["mean_cost"] - c["mean_cost"]) / c["mean_cost"]
    else:
        cost_increase = 0.0 if t["mean_cost"] <= 0 else math.inf
    significant = z_p < alpha and f_p < alpha
    diff = (t["success_rate"] or 0.0) - (c["success_rate"] or 0.0)
    enough = min(c["n_judged"], t["n_judged"]) >= min_samples
    if not enough:
        verdict, reason = "hold", f"insufficient samples (need {min_samples} judged per arm)"
    elif significant and diff < 0:
        verdict, reason = "rollback", "treatment significantly worse"
    elif cost_increase > max_cost_increase:
        verdict, reason = "rollback", f"cost increase {cost_increase:.1%} exceeds {max_cost_increase:.1%}"
    elif significant and diff > 0:
        verdict, reason = "promote", "treatment significantly better, cost within limit"
    else:
        verdict, reason = "hold", "no significant difference"
    return {"arms": arms, "z": z, "z_p": z_p, "fisher_p": f_p, "cost_increase": cost_increase,
            "thresholds": {"min_samples": min_samples, "max_cost_increase": max_cost_increase,
                           "alpha": alpha},
            "verdict": verdict, "reason": reason}


def format_table(report: dict) -> str:
    def rate(v):
        return "n/a" if v is None else f"{v:.3f}"
    lines = [f"{'arm':<10}{'n':>7}{'success':>9}{'mean_cost':>12}{'p50_ms':>9}{'p95_ms':>9}"]
    for arm in ARMS:
        a = report["arms"][arm]
        lines.append(f"{arm:<10}{a['n']:>7}{rate(a['success_rate']):>9}{a['mean_cost']:>12.6f}"
                     f"{a['latency_p50_ms']:>9.1f}{a['latency_p95_ms']:>9.1f}")
    lines.append(f"z={report['z']:.3f} (p={report['z_p']:.4g})  fisher p={report['fisher_p']:.4g}  "
                 f"cost increase={report['cost_increase']:.1%}")
    lines.append(f"verdict: {report['verdict']} ({report['reason']})")
    return "\n".join(lines)
