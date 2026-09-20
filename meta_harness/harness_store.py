"""Storage for learned harness artifacts.

Nothing here installs anything into a live session. `stage` records a scored candidate,
`accept` promotes it, `reject` archives it, and a rejection marked wrong tombstones the failure
signature so the same idea is never proposed again.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ARTIFACT_TYPES = ("rule", "injection", "skill", "doctrine")


def harness_home() -> Path:
    return Path(os.environ.get("META_HARNESS_HOME") or (Path.home() / ".claude" / "harness"))


@dataclass
class Artifact:
    id: str
    type: str
    origin: dict[str, Any]
    payload: str
    replay: dict[str, Any]
    scores: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    created: str = ""

    @property
    def signature(self) -> str:
        return str(self.origin.get("signature", ""))


class HarnessStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or harness_home())
        for name in ("artifacts", "staged", "archive"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.tombstone_path = self.root / "tombstones.json"
        self.installed_path = self.root / "installed.json"

    # --- reading -----------------------------------------------------------

    def _load_dir(self, directory: Path) -> list[Artifact]:
        out = []
        for path in sorted(directory.glob("*/artifact.json")):
            try:
                out.append(Artifact(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, TypeError, ValueError):
                continue
        return out

    def list_staged(self) -> list[Artifact]:
        return self._load_dir(self.root / "staged")

    def list_installed(self) -> list[Artifact]:
        return self._load_dir(self.root / "artifacts")

    def installed_signatures(self) -> set[str]:
        return {a.signature for a in self.list_installed() if a.signature}

    def tombstones(self) -> set[str]:
        if not self.tombstone_path.is_file():
            return set()
        try:
            return set(json.loads(self.tombstone_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return set()

    def is_tombstoned(self, signature: str) -> bool:
        return signature in self.tombstones()

    def covered(self) -> set[str]:
        """Signatures that must not be proposed again: installed or tombstoned."""
        return self.installed_signatures() | self.tombstones()

    # --- writing -----------------------------------------------------------

    def _write(self, directory: Path, artifact: Artifact) -> Path:
        target = directory / artifact.id
        target.mkdir(parents=True, exist_ok=True)
        payload = dict(asdict(artifact))
        (target / "artifact.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def stage(self, artifact: Artifact) -> Path:
        if artifact.type not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact type: {artifact.type}")
        artifact.created = artifact.created or date.today().isoformat()
        return self._write(self.root / "staged", artifact)

    def _take_staged(self, artifact_id: str) -> Artifact:
        for artifact in self.list_staged():
            if artifact.id == artifact_id:
                return artifact
        raise KeyError(f"no staged artifact {artifact_id!r}")

    def accept(self, artifact_id: str) -> Path:
        artifact = self._take_staged(artifact_id)
        target = self._write(self.root / "artifacts", artifact)
        shutil.rmtree(self.root / "staged" / artifact_id, ignore_errors=True)
        registry = []
        if self.installed_path.is_file():
            try:
                registry = json.loads(self.installed_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                registry = []
        registry = [r for r in registry if r.get("id") != artifact.id]
        registry.append({"id": artifact.id, "type": artifact.type,
                         "signature": artifact.signature, "accepted": date.today().isoformat()})
        self.installed_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return target

    def reject(self, artifact_id: str, wrong: bool = False) -> Path:
        artifact = self._take_staged(artifact_id)
        target = self._write(self.root / "archive", artifact)
        shutil.rmtree(self.root / "staged" / artifact_id, ignore_errors=True)
        if wrong and artifact.signature:
            stones = self.tombstones() | {artifact.signature}
            self.tombstone_path.write_text(json.dumps(sorted(stones), indent=2), encoding="utf-8")
        return target


__all__ = ["ARTIFACT_TYPES", "Artifact", "HarnessStore", "harness_home"]
