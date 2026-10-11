"""`meta-harness ablate`: run the search per experience view, N runs each, report median and best."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from ..ablate import (DEFAULT_ITERATIONS, DEFAULT_MAX_TASKS, DEFAULT_RUNS, cap_tasks, format_table,
                      has_flag, run_ablation)

NAME = "ablate"
HELP = "ablate the experience view (scores|summary|full) with N search runs each (paper Table 3)"


def add_arguments(parser) -> None:
    parser.add_argument("--views", default="scores,summary,full")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--root", default=".meta-harness-ablate")
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS,
                        help="sample --data down to this many tasks before searching (default 20)")
    parser.add_argument("--resume", action="store_true", help="reuse a non-empty --root")
    parser.add_argument("--baseline", help="view the others are compared against (default: first)")
    parser.add_argument("search_args", nargs="*",
                        help="after `--`: arguments for `search` (no --view or --root)")


def _search(argv: list[str]) -> int:
    from .. import __main__ as cli
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.main(["search", *argv])


def run(args, search=None) -> int:
    search_args = list(args.search_args)
    if search_args[:1] == ["--"]:
        search_args = search_args[1:]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    bad = [v for v in views if v not in ("scores", "summary", "full")]
    if bad or not views or args.runs < 1 or args.max_tasks < 1:
        print(f"error: --views must be from scores,summary,full; --runs and --max-tasks >= 1 "
              f"(got {args.views!r}, {args.runs}, {args.max_tasks})")
        return 2
    if has_flag(search_args, "--view") or has_flag(search_args, "--root"):
        print("error: search arguments after `--` must not include --view or --root")
        return 2
    if not has_flag(search_args, "--iterations"):
        search_args += ["--iterations", str(DEFAULT_ITERATIONS)]   # owner scale rule: small by default
    root = Path(args.root)
    try:
        if not args.resume and root.is_dir() and any(root.iterdir()):
            raise ValueError(f"{root} is not empty; pass --resume to continue it or choose a new --root")
        search_args = cap_tasks(search_args, args.max_tasks, root)
        report = run_ablation(search_args, views, args.runs, root, search or _search,
                              baseline=args.baseline, resume=True)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    print(format_table(report))
    print(f"wrote {root / 'ablation.json'}")
    for f in report["failed"]:
        print(f"failed: view {f['view']} run {f['run']} exited {f['exit_code']}")
    return 1 if report["failed"] else 0
