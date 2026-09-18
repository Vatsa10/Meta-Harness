"""Run Terminal-Bench through Harbor with credentials loaded from .env.

Harbor forwards ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from the host
environment into the agent container (harbor/agents/installed/claude_code.py), so the key never
has to appear on a command line, in shell history, or in a job log.

    python tools/tb_run.py --tasks 2 --attempts 1 --job-name pilot
    python tools/tb_run.py --tasks 30 --attempts 2 --job-name search-baseline

Anything after `--` is passed through to `harbor run`, which is how candidate harness knobs are
injected:

    python tools/tb_run.py --tasks 30 -- --ak append_system_prompt="Read before editing." \\
                                        --ak max_turns=40 --skill ./candidate-skills
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIKEY_ANTHROPIC_BASE = "https://www.getunikey.ai"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip('"').strip("'")
    return values


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    stored = load_env(ROOT / ".env")
    token = env.get("ANTHROPIC_AUTH_TOKEN") or stored.get("UNIKEY_API_KEY", "")
    if not token:
        raise SystemExit("no credential: set UNIKEY_API_KEY in .env or ANTHROPIC_AUTH_TOKEN")
    env["ANTHROPIC_AUTH_TOKEN"] = token
    env.setdefault("ANTHROPIC_BASE_URL", stored.get("ANTHROPIC_BASE_URL", UNIKEY_ANTHROPIC_BASE))
    # A stale key here would win over the gateway token and send traffic to the wrong place.
    env.pop("ANTHROPIC_API_KEY", None)
    # Harbor reads trial artifacts with Path.read_text() and no encoding, so on Windows it
    # decodes agent output as cp1252 and dies on the first non-ASCII byte a task emits.
    # UTF-8 mode fixes it process-wide without patching the dependency.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tb_run")
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--tasks", type=int, default=30, help="harbor -l, tasks to run")
    parser.add_argument("--attempts", type=int, default=2, help="harbor -k, attempts per trial")
    parser.add_argument("--concurrent", type=int, default=2)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--jobs-dir", default=".tb/jobs")
    parser.add_argument("--setup-timeout-multiplier", type=float, default=4.0,
                        help="installing the agent in a cold container can exceed the default")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("passthrough", nargs="*", help="args after -- go to harbor run")
    args = parser.parse_args(argv)

    command = [
        "harbor", "run",
        "-d", args.dataset,
        "-a", "claude-code",
        "-m", args.model,
        "-l", str(args.tasks),
        "-k", str(args.attempts),
        "-n", str(args.concurrent),
        "-o", args.jobs_dir,
        "--job-name", args.job_name,
        "--agent-setup-timeout-multiplier", str(args.setup_timeout_multiplier),
        "--timeout-multiplier", str(args.timeout_multiplier),
        *args.passthrough,
    ]
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=build_environment())
    result = Path(args.jobs_dir) / args.job_name / "result.json"
    print(f"\nresult: {result if result.is_file() else '(not written)'}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
