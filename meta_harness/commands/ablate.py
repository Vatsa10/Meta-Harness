"""`meta-harness ablate`: run the search per experience view, N runs each, report median and best."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from ..ablate import DEFAULT_ITERATIONS, DEFAULT_RUNS, format_table, has_flag, run_ablation

NAME = "ablate"
HELP = "ablate the experience view (scores|summary|full) with N search runs each (paper Table 3)"


def add_arguments(parser) -> None:
    parser.add_argument("--views", default="scores,summary,full")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--root", default=".meta-harness-ablate")
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
    if bad or not views or args.runs < 1:
        print(f"error: --views must be from scores,summary,full and --runs >= 1 (got {args.views!r}, {args.runs})")
        return 2
    if has_flag(search_args, "--view") or has_flag(search_args, "--root"):
        print("error: search arguments after `--` must not include --view or --root")
        return 2
    if not has_flag(search_args, "--iterations"):
        search_args += ["--iterations", str(DEFAULT_ITERATIONS)]   # owner scale rule: small by default
    try:
        report = run_ablation(search_args, views, args.runs, Path(args.root), search or _search,
                              baseline=args.baseline)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(format_table(report))
    print(f"wrote {Path(args.root) / 'ablation.json'}")
    return 0
