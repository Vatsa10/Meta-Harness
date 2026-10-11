"""Seed: Terminus-KIRA (App. B.3) - Terminus-2 plus native tool calling through `model.chat`
when the model offers it, a completion checklist before `task_complete` is accepted, and an
environment snapshot collected inside the container at the start.

Knobs the proposer may change (model-dependent axes worth testing): PLANNING, BASH_ONLY,
B1 / B2 (elide / summarise thresholds), CONTEXT_WINDOW, MAX_STEPS, SYSTEM, CHECKLIST.
"""

from meta_harness.terminal_harness import KIRA_CHECKLIST, KIRA_SYSTEM, KiraHarness

SYSTEM = KIRA_SYSTEM
CHECKLIST = KIRA_CHECKLIST
MAX_STEPS = 60
CONTEXT_WINDOW = 128_000
B1 = 0.6
B2 = 0.85
PLANNING = False
BASH_ONLY = False


class Harness(KiraHarness):
    SYSTEM = SYSTEM
    CONFIRM = CHECKLIST
    MAX_STEPS = MAX_STEPS
    CONTEXT_WINDOW = CONTEXT_WINDOW
    B1 = B1
    B2 = B2


def build_harness():
    return Harness(planning=PLANNING, bash_only=BASH_ONLY)
