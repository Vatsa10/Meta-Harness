"""`meta-harness traces import|stats`: turn production traces into eval sets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import traceimport as ti

NAME = "traces"
HELP = "import OpenTelemetry / Langfuse / JSONL traces as eval sets, or summarise them"


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="traces_command", required=True)
    imp = sub.add_parser("import", help="write failure-first, deduplicated eval tasks")
    imp.add_argument("--format", choices=ti.FORMATS, required=True)
    imp.add_argument("--in", dest="inp", required=True, metavar="PATH", help="trace export file")
    imp.add_argument("--out", help="write every selected task to this JSONL file (- for stdout)")
    imp.add_argument("--out-dir", help="write train.jsonl, search.jsonl and test.jsonl here")
    imp.add_argument("--seed", type=int, default=0)
    imp.add_argument("--train-fraction", type=float, default=0.2)
    imp.add_argument("--test-fraction", type=float, default=0.3)
    imp.add_argument("--limit", type=int, default=None, help="keep at most N tasks (failures first)")
    imp.add_argument("--pii", action="store_true", help="also scrub emails, phone numbers and card numbers")
    st = sub.add_parser("stats", help="summarise what a trace export contains")
    st.add_argument("--format", choices=ti.FORMATS, required=True)
    st.add_argument("--in", dest="inp", required=True, metavar="PATH")


def run(args) -> int:
    try:
        traces = ti.load(args.format, args.inp)
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.inp}: {exc}", file=sys.stderr)
        return 2
    if args.traces_command == "stats":
        print(json.dumps(ti.stats(traces), ensure_ascii=False))
        return 0
    if not args.out and not args.out_dir:
        print("give --out and/or --out-dir", file=sys.stderr)
        return 2
    tasks = [ti.to_task(trace, args.pii) for trace in ti.select(traces, args.limit)]
    if not tasks:
        print(f"no traces with an input found in {args.inp}", file=sys.stderr)
        return 2
    summary: dict = {"tasks": len(tasks), "skipped_duplicates": len(traces) - len(ti.select(traces))}
    if args.out == "-":
        sys.stdout.write("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks))
    elif args.out:
        summary["out"] = str(ti.write_jsonl(args.out, tasks))
    if args.out_dir:
        try:
            parts = ti.split(tasks, args.seed, args.train_fraction, args.test_fraction)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        for name, rows in parts.items():
            summary[name] = {"path": str(ti.write_jsonl(Path(args.out_dir) / f"{name}.jsonl", rows)),
                             "tasks": len(rows)}
            if not rows:
                print(f"warning: the {name} split is empty ({len(tasks)} tasks); "
                      "import more traces or change the fractions", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr if args.out == "-" else sys.stdout)
    return 0
