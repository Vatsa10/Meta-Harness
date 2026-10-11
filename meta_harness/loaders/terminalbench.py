"""TerminalBench-2 loader: reads a local checkout (`--tb-dir`); no network.

Container network follows each task: task.toml `[environment] allow_internet` (harbor
default when unspecified: allowed); task.yaml tasks get no network unless `allow_internet: true`.

Task directory formats (both read; see `meta_harness.terminal` for the full notes):
  * `task.yaml` + `Dockerfile` + `run-tests.sh` + `tests/` (+ `solution.sh`)
  * `task.toml` + `instruction.md` + `environment/Dockerfile` + `tests/test.sh` (+ `solution/`)

Rows: {"id", "instruction", "image", "test_command", "workdir", "timeout", "test_timeout",
"network", "test_files": [[local, remote]], "dockerfile_dir", "expected_outputs",
"dataset", "dataset_version"}. Tests are copied into the container only at scoring time.
The dataset version is a loader parameter; `run_meta(version)` is what goes into run.json.
Search = test on this domain (paper), so both split files are the same rows.

`audit_leakage(harness_source, tasks)` is the paper's regex audit: it flags harness source that
names task ids, test file names or expected outputs.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

NAME = "terminalbench"
DOMAIN = "terminal"
SOURCE = "https://github.com/laude-institute/terminal-bench (local checkout via --tb-dir)"
DEFAULT_VERSION = "2.0"
MAX_LIVE_TASKS = 5  # owner scale rule for real runs
MIN_EXPECTED_CHARS = 12

TB1_TEST = "TEST_DIR=/tests bash /tests/run-tests.sh"
TB2_TEST = ("mkdir -p /logs/verifier; bash /tests/test.sh; r=$(cat /logs/verifier/reward.txt "
            "2>/dev/null | tr -d '[:space:]'); [ \"$r\" = 1 ] || [ \"$r\" = 1.0 ]")


def _yaml_scalars(text: str) -> dict[str, Any]:
    """Top-level `key: value` and block scalars (`|`, `|-`, `>`) of a simple YAML file."""
    out: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", lines[i])
        i += 1
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value[:1] in ("|", ">"):
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in (" ", "\t")):
                block.append(lines[i])
                i += 1
            indent = min((len(l) - len(l.lstrip()) for l in block if l.strip()), default=0)
            body = [l[indent:] for l in block]
            joined = "\n".join(body) if value[0] == "|" else " ".join(l.strip() for l in body if l.strip())
            out[key] = joined.rstrip("\n") if "-" in value else joined.rstrip("\n") + "\n"
        elif value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            out[key] = value[1:-1]
        elif re.fullmatch(r"-?\d+(\.\d+)?", value):
            out[key] = float(value) if "." in value else int(value)
        elif value in ("true", "false"):
            out[key] = value == "true"
        elif value:
            out[key] = value
    return out


def _expected_outputs(tests_dir: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(tests_dir.rglob("*.py")) if tests_dir.is_dir() else []:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.strip()
                if len(value) >= MIN_EXPECTED_CHARS and "\n" not in value and value not in found:
                    found.append(value)
    return found


def _image_name(task_id: str, version: str) -> str:
    return f"meta-harness-tb/{re.sub(r'[^a-z0-9_.-]', '-', task_id.lower())}:{version}"


def read_task(task_dir: Path, version: str = DEFAULT_VERSION) -> dict[str, Any] | None:
    task_dir = Path(task_dir)
    task_id = task_dir.name
    tests_dir = task_dir / "tests"
    if (task_dir / "task.toml").is_file():
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        instruction_file = task_dir / "instruction.md"
        instruction = instruction_file.read_text(encoding="utf-8") if instruction_file.is_file() else ""
        env = config.get("environment", {})
        agent, verifier = config.get("agent", {}), config.get("verifier", {})
        row = {"instruction": instruction, "image": env.get("docker_image") or _image_name(task_id, version),
               "test_command": TB2_TEST, "timeout": float(agent.get("timeout_sec", 900.0)),
               "test_timeout": float(verifier.get("timeout_sec", 600.0)),
               # harbor's default: internet allowed unless the task sets allow_internet = false
               "network": bool(env.get("allow_internet", True)),
               "dockerfile_dir": str(task_dir / "environment"),
               "test_files": [[str(tests_dir), "/tests"]]}
    elif (task_dir / "task.yaml").is_file():
        config = _yaml_scalars((task_dir / "task.yaml").read_text(encoding="utf-8"))
        files = [[str(tests_dir), "/tests"]]
        if (task_dir / "run-tests.sh").is_file():
            files.append([str(task_dir / "run-tests.sh"), "/tests/run-tests.sh"])
        row = {"instruction": str(config.get("instruction", "")), "image": _image_name(task_id, version),
               "test_command": TB1_TEST, "timeout": float(config.get("max_agent_timeout_sec", 900.0)),
               "test_timeout": float(config.get("max_test_timeout_sec", 600.0)),
               "network": bool(config.get("allow_internet", False)),
               "dockerfile_dir": str(task_dir), "test_files": files}
    else:
        return None
    if not row["instruction"].strip():
        return None
    return {"id": task_id, **row, "workdir": "/app", "expected_outputs": _expected_outputs(tests_dir),
            "dataset": NAME, "dataset_version": version}


def load(tb_dir: Path | str, version: str = DEFAULT_VERSION, limit: int | None = None,
         ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    root = Path(tb_dir)
    if (root / "tasks").is_dir():
        root = root / "tasks"
    if not root.is_dir():
        raise FileNotFoundError(f"TerminalBench directory not found: {tb_dir}")
    wanted = set(ids) if ids else None
    tasks = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted is not None and task_dir.name not in wanted:
            continue
        row = read_task(task_dir, version)
        if row is not None:
            tasks.append(row)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def run_meta(version: str = DEFAULT_VERSION) -> dict[str, str]:
    """Loader parameters recorded in run.json."""
    return {"dataset": NAME, "dataset_version": version}


def fetch(cache_dir: Path, http: Any = None, tb_dir: Path | str | None = None,
          version: str = DEFAULT_VERSION, limit: int | None = None) -> dict[str, Path]:
    """Write the tasks to JSONL. Search = test: both splits point to the same file."""
    source = tb_dir or os.environ.get("META_HARNESS_TB_DIR")
    if not source:
        raise ValueError("terminalbench needs a local checkout: pass --tb-dir (or META_HARNESS_TB_DIR)")
    out = Path(cache_dir) / f"terminalbench-{version}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load(source, version, limit)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(run_meta(version), indent=2), encoding="utf-8")
    return {"search": out, "test": out}


def build_images(tasks: Iterable[Mapping[str, Any]], runner: Callable[..., Any] = subprocess.run,
                 timeout: float = 1800.0) -> list[str]:
    """`docker build -t <image> <dockerfile_dir>` for each task; returns the images built."""
    built = []
    for task in tasks:
        context = task.get("dockerfile_dir")
        if not context:
            continue
        done = runner(["docker", "build", "-t", str(task["image"]), str(context)], capture_output=True,
                      text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(f"docker build failed for {task.get('id')}: {(done.stderr or '')[-2000:]}")
        built.append(str(task["image"]))
    return built


def _test_file_names(task: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for local, _ in task.get("test_files") or []:
        path = Path(local)
        items = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        for item in items:
            if item.name not in names:
                names.append(item.name)
    return names


def audit_leakage(harness_source: str, tasks: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Regex audit: findings for task ids, test file names and expected outputs in the source."""
    findings: list[dict[str, str]] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        checks = [("task_id", task_id)] if len(task_id) >= 3 else []
        checks += [("test_file", name) for name in _test_file_names(task)]
        checks += [("expected_output", value) for value in task.get("expected_outputs") or []]
        for kind, value in checks:
            pattern = r"(?<![\w-])" + re.escape(value) + r"(?![\w-])"
            if re.search(pattern, harness_source):
                findings.append({"kind": kind, "value": value, "task": task_id})
    return findings
