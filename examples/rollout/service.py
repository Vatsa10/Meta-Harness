"""Tiny service example: split traffic 50/50 between two handlers and log receipts.

    python examples/rollout/service.py
    meta-harness rollout report --log rollout.jsonl --min-samples 50
"""

import random
import time

from meta_harness.rollout import Router


def control(question: str) -> bool:
    return random.random() < 0.70


def treatment(question: str) -> bool:
    return random.random() < 0.80


router = Router(control, treatment, percent=50, salt="exp-1", log="rollout.jsonl")

for i in range(400):
    user = f"user-{i}"
    arm = router.route(user)
    start = time.perf_counter()
    ok = router.target(user)("question")
    router.record(user, arm, success=ok, cost=0.001 if arm == "control" else 0.0011,
                  latency_ms=(time.perf_counter() - start) * 1000, meta={"i": i})
