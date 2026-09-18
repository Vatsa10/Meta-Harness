from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Sequence

from .agent_proposer import ClaudeCodeProposer
from .cache import CachedModel
from .core import (CandidateEvaluator, CommandProposer, FilesystemExperience, ParetoFrontier,
                   SearchConfig, SearchRunner)
from .datasets import (agent_tasks, classification_tasks, math_tasks, read_records,
                        split_tasks, terminal_tasks)
from .cc_history import (compare_reports, draft_tasks, harness_report, load_sessions,
                         project_slug, sessions_between, write_history_view)
from .demo import run as run_demo
from .metrics import METRICS
from .providers import list_unikey_models, model_from_environment

BASELINE_ROOT = Path(__file__).resolve().parent.parent / "baselines"
DEFAULT_BASELINES = {
    "classification": ["zero_shot.py", "few_shot.py", "ace.py", "mce.py"],
    "math": ["math_zero_shot.py", "math_bm25.py"],
    "terminal": ["terminal_basic.py"],
    "agent": ["cc_default.py", "cc_guided.py", "cc_subagent.py"],
}
ADAPTERS = {"classification": classification_tasks, "math": math_tasks,
            "terminal": terminal_tasks, "agent": agent_tasks}
# Aliases the CLI resolves to the current model of each size. The paper's agentic-coding
# result (section 4.3) is on Haiku 4.5, so haiku is the default harness model here.
CLAUDE_CLI_MODELS = ["haiku", "sonnet", "opus"]


def default_baselines(task_type: str) -> list[Path]:
    return [BASELINE_ROOT / name for name in DEFAULT_BASELINES[task_type]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meta-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a deterministic end-to-end example")
    demo.add_argument("--iterations", type=int, default=3)
    demo.add_argument("--root", default=".meta-harness-demo")

    inspect = sub.add_parser("inspect", help="show stored results and Pareto frontier")
    inspect.add_argument("root", nargs="?", default=".meta-harness")

    models = sub.add_parser("models", help="list available harness models")
    models.add_argument("--provider", choices=["claude-cli", "unikey"], default="claude-cli")

    mine = sub.add_parser(
        "mine", help="read this machine's Claude Code history into a proposer-readable view")
    mine.add_argument("--project", help="project slug, or a path (default: every project)")
    mine.add_argument("--this-project", action="store_true",
                      help="only the current working directory's history")
    mine.add_argument("--limit", type=int, default=100, help="most recent sessions to read")
    mine.add_argument("--min-turns", type=int, default=4)
    mine.add_argument("--out", default=".meta-harness/history")
    mine.add_argument("--include-text", action="store_true",
                      help="keep message text. Transcripts contain whatever you typed; off by default")
    mine.add_argument("--draft-tasks", help="also write draft eval tasks to this JSONL path")

    compare = sub.add_parser(
        "compare", help="diff two windows of Claude Code history before and after a harness change")
    compare.add_argument("--split", required=True,
                         help="ISO timestamp dividing before from after, e.g. 2026-09-18")
    compare.add_argument("--project", help="project slug or path (default: every project)")
    compare.add_argument("--this-project", action="store_true")
    compare.add_argument("--limit", type=int, default=200)
    compare.add_argument("--min-turns", type=int, default=4)

    run = sub.add_parser("run", help="run a provider-backed harness search")
    run.add_argument("--tasks", required=True, help="JSON, JSONL, or CSV dataset")
    run.add_argument("--test-tasks", help="held-out dataset; omit to split --tasks")
    run.add_argument("--search-fraction", type=float, default=0.7)
    run.add_argument("--split-seed", type=int, default=0)
    run.add_argument("--baseline", nargs="+", help="seed harness files (default: bundled baselines)")
    run.add_argument("--provider", choices=["claude-cli", "unikey", "openai", "anthropic", "compatible"],
                     default="claude-cli",
                     help="where the harness model runs. claude-cli (default) uses the local "
                          "Claude Code CLI and needs no API key")
    run.add_argument("--model", default="haiku",
                     help="harness model; a claude-cli alias (haiku, sonnet, opus) or a "
                          "gateway model id")
    run.add_argument("--task-type", choices=["classification", "math", "terminal", "agent"],
                     default="classification")
    run.add_argument("--root", default=".meta-harness")
    run.add_argument("--test-root")
    run.add_argument("--iterations", type=int, default=10)
    run.add_argument("--candidates", type=int, default=1)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--proposer-command", help="shell command writing .py files to META_HARNESS_OUTPUT")
    run.add_argument("--proposer-model", help="model id passed to the coding agent")
    run.add_argument("--proposer-binary", default="claude")
    run.add_argument("--proposer-view", choices=["scores", "summary", "full"], default="full")
    run.add_argument("--proposer-timeout", type=float, default=1800.0)
    run.add_argument("--cache-dir")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--validation-timeout", type=float, default=20.0)
    run.add_argument("--task-timeout", type=float, default=600.0)
    run.add_argument("--allow-local-shell", action="store_true",
                     help="terminal tasks only: run model-authored commands on this machine")
    run.add_argument("--workspace-root",
                     help="agent tasks only: where throwaway task workspaces are created")
    run.add_argument("--agent-bash", action="store_true",
                     help="agent tasks only: let candidate harnesses grant the Bash tool")
    run.add_argument("--mine-history", action="store_true",
                     help="mine local Claude Code history into <root>/history so the proposer "
                          "can diagnose from real sessions")
    run.add_argument("--history-limit", type=int, default=60)
    run.add_argument("--history-include-text", action="store_true")
    return parser


def split_command(command: str) -> list[str]:
    """Split a shell-ish command string. On Windows, backslashes are path separators,
    not escapes, so posix-mode shlex would mangle every absolute path."""
    if os.name != "nt":
        return shlex.split(command)
    return [token[1:-1] if len(token) > 1 and token[0] == token[-1] == '"' else token
            for token in shlex.split(command, posix=False)]


def resolve_command(tokens: Sequence[str]) -> list[str]:
    """Absolutize tokens that name a real file.

    The proposer runs with its working directory set to the experience root, so a relative
    script path in --proposer-command would resolve against the wrong directory.
    """
    return [str(Path(token).resolve()) if token and Path(token).exists() else token
            for token in tokens]


def _load_tasks(args) -> tuple[list[dict], list[dict]]:
    adapt = ADAPTERS[args.task_type]
    search = adapt(read_records(args.tasks))
    if args.test_tasks:
        return search, adapt(read_records(args.test_tasks))
    if len(search) < 2:
        return search, []
    return split_tasks(search, args.search_fraction, args.split_seed)


def _command_mine(args) -> int:
    project = args.project
    if args.this_project:
        project = project_slug(Path.cwd())
    elif project and ("/" in project or "\\" in project or Path(project).exists()):
        project = project_slug(project)
    sessions = load_sessions(project=project, limit=args.limit,
                             include_text=args.include_text, min_turns=args.min_turns)
    if not sessions:
        print(json.dumps({"sessions": 0,
                          "hint": "no transcripts found; check --project or ~/.claude/projects"},
                         indent=2))
        return 1
    destination = write_history_view(sessions, Path(args.out), include_text=args.include_text)
    report = harness_report(sessions)
    if args.draft_tasks:
        drafts = draft_tasks(sessions)
        Path(args.draft_tasks).parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(json.dumps(d) for d in drafts)
        Path(args.draft_tasks).write_text(body + ("\n" if drafts else ""), encoding="utf-8")
    print(json.dumps({
        "view": str(destination),
        "redacted": not args.include_text,
        "sessions": report["sessions"],
        "tool_calls": report["tool_calls"],
        "error_rate": report["error_rate"],
        "read_to_write_ratio": report["read_to_write_ratio"],
        "subagent_calls": report["subagent_calls"],
        "workflow_calls": report["workflow_calls"],
        "skill_calls": report["skill_calls"],
        "episode_kinds": report["episode_kinds"],
        "top_tools": dict(list(report["tool_histogram"].items())[:8]),
        "top_errors": dict(list(report["tool_errors"].items())[:6]),
        "draft_tasks": args.draft_tasks,
    }, indent=2))
    return 0


def _command_compare(args) -> int:
    project = args.project
    if args.this_project:
        project = project_slug(Path.cwd())
    elif project and ("/" in project or "\\" in project or Path(project).exists()):
        project = project_slug(project)
    sessions = load_sessions(project=project, limit=args.limit, include_text=True,
                             min_turns=args.min_turns)
    before = sessions_between(sessions, before=args.split)
    after = sessions_between(sessions, since=args.split)
    if not before or not after:
        print(json.dumps({"error": "need sessions on both sides of --split",
                          "before": len(before), "after": len(after)}, indent=2))
        return 1
    diff = compare_reports(harness_report(before), harness_report(after))
    print(json.dumps({"split": args.split,
                      "sessions_before": len(before), "sessions_after": len(after),
                      **diff}, indent=2))
    return 0


def _command_run(args) -> int:
    if args.allow_local_shell:
        os.environ["META_HARNESS_ALLOW_LOCAL_SHELL"] = "1"
    if args.agent_bash:
        os.environ["META_HARNESS_AGENT_BASH"] = "1"
    if args.task_type == "agent":
        root = Path(args.workspace_root or (Path(args.root).parent / ".meta-harness-workspaces"))
        root.mkdir(parents=True, exist_ok=True)
        os.environ["META_HARNESS_WORKSPACE_ROOT"] = str(root.resolve())
    if args.mine_history:
        sessions = load_sessions(limit=args.history_limit,
                                 include_text=args.history_include_text)
        if sessions:
            view = write_history_view(sessions, Path(args.root) / "history",
                                      include_text=args.history_include_text)
            print(f"mined {len(sessions)} session(s) of Claude Code history into {view}")
        else:
            print("no local Claude Code history found; continuing without it")
    search_tasks, test_tasks = _load_tasks(args)
    if args.provider == "claude-cli":
        from .claude_cli import ClaudeCliModel

        if not ClaudeCliModel.available():
            print("claude CLI not found on PATH. Install Claude Code, or pass "
                  "--provider unikey with UNIKEY_API_KEY set.")
            return 1
    base_model = model_from_environment(args.provider, args.model)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.root).parent / ".meta-harness-cache"
    model = CachedModel(base_model, cache_dir, enabled=not args.no_cache)
    if args.proposer_command:
        proposer = CommandProposer(resolve_command(split_command(args.proposer_command)),
                                   timeout=args.proposer_timeout)
    else:
        proposer = ClaudeCodeProposer(binary=args.proposer_binary, model=args.proposer_model,
                                      view=args.proposer_view, summarizer=model,
                                      timeout=args.proposer_timeout)
    config = SearchConfig(root=args.root, iterations=args.iterations,
                          candidates_per_iteration=args.candidates,
                          validation_timeout=args.validation_timeout,
                          repeats=args.repeats, max_workers=args.max_workers,
                          task_timeout=args.task_timeout, test_root=args.test_root)
    evaluator = CandidateEvaluator(model, METRICS[args.task_type], task_timeout=args.task_timeout)
    runner = SearchRunner(config, evaluator, proposer)
    baselines = [Path(p) for p in (args.baseline or default_baselines(args.task_type))]
    frontier = runner.run(baselines, search_tasks, test_tasks or None)
    test_path = Path(config.test_root) / "test_results.json"
    print(json.dumps({
        "root": str(config.root),
        "search_tasks": len(search_tasks),
        "test_tasks": len(test_tasks),
        "cache": {"hits": model.hits, "misses": model.misses},
        "frontier": [item.__dict__ for item in frontier],
        "test_results": str(test_path) if test_path.is_file() else None,
    }, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "demo":
        root = run_demo(args.iterations, args.root)
        print(json.dumps([r.__dict__ for r in ParetoFrontier.select(FilesystemExperience(root).results())], indent=2))
        return 0
    if args.command == "models":
        if args.provider == "claude-cli":
            from .claude_cli import ClaudeCliModel

            if not ClaudeCliModel.available():
                print("claude CLI not found on PATH; install Claude Code or use --provider unikey")
                return 1
            print("\n".join(CLAUDE_CLI_MODELS))
            return 0
        print("\n".join(list_unikey_models()))
        return 0
    if args.command == "mine":
        return _command_mine(args)
    if args.command == "compare":
        return _command_compare(args)
    if args.command == "run":
        return _command_run(args)
    experience = FilesystemExperience(args.root)
    frontier_file = experience.root / "frontier.json"
    print(json.dumps({
        "results": experience.manifest(),
        "frontier": json.loads(frontier_file.read_text(encoding="utf-8")) if frontier_file.exists() else [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
