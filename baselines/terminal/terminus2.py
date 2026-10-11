"""Seed: Terminus-2 (paper section 4.3 initialisation, App. B.3).

JSON command batches executed tmux-style through the task container, observations capped at
30KB, an explicit completion-confirmation turn, plus the shared edit tool (exact -> fuzzy,
read-before-write, py_compile diagnostics), stuck detection and staged compaction.

Knobs the proposer may change (model-dependent axes worth testing): PLANNING, BASH_ONLY,
B1 / B2 (elide / summarise thresholds), CONTEXT_WINDOW, MAX_STEPS, SYSTEM.
"""

from meta_harness.terminal_harness import TERMINUS2_SYSTEM, Terminus2Harness

SYSTEM = TERMINUS2_SYSTEM
MAX_STEPS = 300  # paper default; pass a lower max_steps for small runs
CONTEXT_WINDOW = 128_000
B1 = 0.6    # elide bulky middle tool outputs at this fraction of the window
B2 = 0.85   # summarise the middle into a running summary at this fraction
PLANNING = False
BASH_ONLY = False


class Harness(Terminus2Harness):
    SYSTEM = SYSTEM
    MAX_STEPS = MAX_STEPS
    CONTEXT_WINDOW = CONTEXT_WINDOW
    B1 = B1
    B2 = B2


def build_harness():
    return Harness(planning=PLANNING, bash_only=BASH_ONLY)
