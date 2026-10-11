"""Online text classification: a labelled train stream through learn(), then a frozen test."""

from __future__ import annotations

from pathlib import Path

from ..datasets import classification_tasks
from ..protocol import exact

NAME, PROTOCOL = "classification", "online"
REPO = Path(__file__).resolve().parent.parent.parent
SEED_DIR = REPO / "baselines" / "classification"
PROMPT = ("Tasks are single-label text classification over a fixed label set given in task['labels']. "
          "The harness sees a labelled train stream through learn(task, result) and is then evaluated "
          "frozen. Optimise accuracy first, then context tokens.")
METRICS = {"accuracy": exact}


def adapt(records: list[dict]) -> list[dict]:
    """Require input and label on every record and carry `labels`; ValueError otherwise."""
    return classification_tasks(records)
