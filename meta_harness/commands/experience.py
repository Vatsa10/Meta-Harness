"""`meta-harness experience`: query a stored run (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

from .. import experience_query as q
from ..redact import redact

NAME = "experience"
HELP = "query stored experience: top, diff, diff-runs, import, reads, report, export"


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="experience_cmd", required=True)
    top = sub.add_parser("top", help="best valid candidates")
    top.add_argument("root")
    top.add_argument("-k", type=int, default=5)
    top.add_argument("--by", choices=("score", "cost"), default="score")
    diff = sub.add_parser("diff", help="diff two candidates harness.py with score deltas")
    diff.add_argument("root")
    diff.add_argument("a")
    diff.add_argument("b")
    runs = sub.add_parser("diff-runs", help="two runs side by side")
    runs.add_argument("root_a")
    runs.add_argument("root_b")
    imp = sub.add_parser("import", help="copy the top-k harnesses of a run as seed files (warm start)")
    imp.add_argument("root")
    imp.add_argument("dest")
    imp.add_argument("-k", type=int, default=3)
    reads = sub.add_parser("reads", help="proposer read statistics (Table 8)")
    reads.add_argument("root")
    rep = sub.add_parser("report", help="receipt verdict and stage counts")
    rep.add_argument("--root", required=True)
    exp = sub.add_parser("export", help="one case per receipt, as JSONL")
    exp.add_argument("--root", required=True)
    exp.add_argument("--out", required=True)


def _dump(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def run(args) -> int:
    cmd = args.experience_cmd
    try:
        if cmd == "top":
            _dump(q.top_k(Path(args.root), args.k, args.by))
        elif cmd == "diff":
            print(q.diff_candidates(Path(args.root), args.a, args.b), end="")
        elif cmd == "diff-runs":
            _dump(q.diff_runs(Path(args.root_a), Path(args.root_b)))
        elif cmd == "import":
            paths = q.import_candidates(Path(args.root), Path(args.dest), args.k)
            for path in paths:
                print(redact(f"--harness {path}"))
            print(redact(f"# warm start: pass the flags above to your search ({len(paths)} seed files)"))
        elif cmd == "reads":
            _dump(q.read_stats(Path(args.root)))
        elif cmd == "report":
            _dump(q.report(Path(args.root)))
        elif cmd == "export":
            count = q.export_cases(Path(args.root), Path(args.out))
            print(redact(f"wrote {count} cases to {args.out}"))
    except (OSError, ValueError) as exc:
        print(redact(f"error: {exc}"))
        return 2
    return 0
