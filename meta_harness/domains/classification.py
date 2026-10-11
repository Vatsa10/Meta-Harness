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


def train_tasks(search_path: str | Path | None = None) -> list[dict]:
    """The loader-provided train split: `train.jsonl` next to the search file (the classification
    loaders write train / val / test side by side); [] when there is none."""
    if not search_path:
        return []
    search = Path(search_path)
    train = search.with_name("train.jsonl")
    if not train.is_file() or train.resolve() == search.resolve():
        return []
    from ..datasets import read_records
    return adapt(read_records(train))
