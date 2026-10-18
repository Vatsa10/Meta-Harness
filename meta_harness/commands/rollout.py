"""`meta-harness rollout report`: judge a control/treatment traffic split from its receipt log."""

from __future__ import annotations

import json

from ..rollout import analyze, format_table, read_log

NAME = "rollout"
HELP = "report on an online control/treatment rollout and give a promote/hold/rollback verdict"


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="rollout_cmd", required=True)
    rep = sub.add_parser("report", help="per-arm stats, significance tests and a verdict")
    rep.add_argument("--log", required=True, help="JSONL receipt log written by Router.record")
    rep.add_argument("--min-samples", type=int, default=200, help="judged samples required per arm")
    rep.add_argument("--max-cost-increase", type=float, default=0.1,
                     help="largest tolerated relative mean-cost increase of treatment")
    rep.add_argument("--alpha", type=float, default=0.05)
    rep.add_argument("--json", action="store_true", dest="as_json", help="emit JSON instead of a table")


def run(args) -> int:
    report = analyze(read_log(args.log), args.min_samples, args.max_cost_increase, args.alpha)
    if args.as_json and report["cost_increase"] == float("inf"):
        report["cost_increase"] = None
    print(json.dumps(report, indent=2) if args.as_json else format_table(report))
    return 0
