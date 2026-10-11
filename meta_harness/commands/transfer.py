"""`meta-harness transfer`: score fixed harnesses across several models."""

from __future__ import annotations

from pathlib import Path

from ..datasets import read_records
from ..domains import get_domain
from ..protocol import resolve_metric
from ..transfer import format_comparisons, format_table, parse_models, transfer

NAME = "transfer"
HELP = "evaluate fixed harnesses across several models (harness x model table)"


def add_arguments(parser) -> None:
    parser.add_argument("--harness", action="append", required=True, help="harness file; repeatable")
    parser.add_argument("--data", required=True, help="test tasks (JSON, JSONL or CSV)")
    parser.add_argument("--metric", required=True, help="exact | contains | numeric | module:function")
    parser.add_argument("--models", required=True, help="provider:model,provider:model")
    parser.add_argument("--domain", default="generic")
    parser.add_argument("--train-data", help="train tasks streamed first (online protocol)")
    parser.add_argument("--protocol", choices=["plain", "online"], default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--root", default=".meta-harness-transfer")


def run(args) -> int:
    domain = get_domain(args.domain)
    metric = resolve_metric(args.metric, extra=getattr(domain, "METRICS", None))
    tasks = domain.adapt(read_records(args.data))
    train = domain.adapt(read_records(args.train_data)) if args.train_data else []
    protocol = args.protocol or ("online" if train else getattr(domain, "PROTOCOL", "plain"))
    report = transfer([Path(h) for h in args.harness], tasks, parse_models(args.models), metric,
                      Path(args.root), repeats=args.repeats, protocol=protocol, train_tasks=train)
    print(format_table(report["rows"]))
    if report["comparisons"]:
        print(format_comparisons(report["comparisons"]))
    for row in report["rows"]:
        if row.get("error"):
            print(f"error: {row['harness']} x {row['provider']}:{row['model']}: {row['error']}")
    print(f"wrote {Path(args.root) / 'transfer.json'}")
    return 0
