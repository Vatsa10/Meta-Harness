"""`meta-harness check`: CI gate that blocks harness regressions.

Evaluates a baseline and a candidate harness on the same fixed tasks with the same model,
pairs the per-task outcomes by task id, and exits 1 on a quality or cost regression.
Exit codes: 0 pass, 1 regression, 2 setup error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from ..core import CandidateEvaluator, atomic_write
from ..datasets import read_records
from ..domains import get_domain
from ..isolation import MODES as ISOLATION_MODES
from ..protocol import resolve_metric
from ..providers import Price, model_from_environment, price_for
from ..stats import compare
from ..transfer import _merge_repeats, task_outcomes

NAME = "check"
HELP = "CI gate: fail when a candidate harness regresses against a baseline on a fixed task set"

EXIT_PASS, EXIT_REGRESSION, EXIT_SETUP = 0, 1, 2


class SetupError(Exception):
    pass


def add_arguments(parser) -> None:
    parser.add_argument("--baseline", required=True, help="baseline harness file")
    parser.add_argument("--candidate", required=True, help="candidate harness file")
    parser.add_argument("--data", required=True, help="fixed tasks (JSON, JSONL or CSV)")
    parser.add_argument("--metric", required=True, help="exact | contains | numeric | module:function")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", default="generic")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-drop", type=float, default=0.02,
                        help="largest tolerated drop in mean score (default 0.02)")
    parser.add_argument("--max-cost-increase", type=float, default=0.10,
                        help="largest tolerated relative rise in cost (default 0.10 = 10%%)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level for the McNemar test")
    parser.add_argument("--price", help="USD per million tokens as INPUT,OUTPUT (else META_HARNESS_PRICES)")
    parser.add_argument("--isolation", choices=list(ISOLATION_MODES), default=None)
    parser.add_argument("--out", default="check.json")
    parser.add_argument("--markdown", help="write a markdown summary (PR comment / job summary)")


def _price(args) -> Price | None:
    if args.price:
        try:
            inp, out = (float(x) for x in args.price.split(","))
        except ValueError:
            raise SetupError("--price must be INPUT,OUTPUT (USD per million tokens)") from None
        return Price(inp, out)
    return price_for(args.model)


def _evaluate(path: Path, tasks, ids, metric, model, price, args, root: Path) -> dict[str, Any]:
    evaluator = CandidateEvaluator(model, metric, isolation=args.isolation)
    scores, ctx, runs = [], [], []
    for repeat in range(max(1, args.repeats)):
        trace = root / f"{path.stem}-trace-{repeat}.jsonl"
        result = evaluator.evaluate(path, tasks, f"{path.stem}@check", trace, repeat)
        if not result.valid:
            raise SetupError(f"{path.name}: {result.error or 'invalid harness'}")
        scores.append(result.score)
        ctx.append(result.context_cost)
        runs.append(task_outcomes(trace, price))
    merged = _merge_repeats(runs)
    usd = [o["cost_usd"] for o in merged]
    return {"score": sum(scores) / len(scores), "context_cost": sum(ctx) / len(ctx),
            "cost_usd": None if not usd or any(u is None for u in usd) else sum(usd),
            "passes": {ids[o["task"]]: float(o["pass"]) for o in merged}, "n": len(tasks)}


def decide(base: Mapping, cand: Mapping, stats: Mapping, max_drop: float, max_cost_increase: float) -> dict:
    """Pure gate decision from two evaluation summaries and the paired comparison."""
    delta = cand["score"] - base["score"]
    reasons = []
    if stats["significant"] and stats["losses"] > stats["wins"]:
        reasons.append(f"significant quality drop (McNemar p={stats['p']:.4f}, "
                       f"{stats['losses']} lost vs {stats['wins']} gained)")
    if -delta > max_drop:
        reasons.append(f"score dropped {-delta:.4f} > max-drop {max_drop}")
    cost_rel = None
    if base["cost_usd"] is not None and cand["cost_usd"] is not None:
        if base["cost_usd"] > 0:
            cost_rel = (cand["cost_usd"] - base["cost_usd"]) / base["cost_usd"]
        else:
            cost_rel = float("inf") if cand["cost_usd"] > 0 else 0.0
        if cost_rel > max_cost_increase:
            reasons.append(f"cost rose {_pct(cost_rel)} > max-cost-increase {max_cost_increase:.1%}")
    ctx_rel = None
    if base["context_cost"] > 0:
        ctx_rel = (cand["context_cost"] - base["context_cost"]) / base["context_cost"]
    return {"score_delta": delta, "cost_increase": cost_rel, "context_increase": ctx_rel,
            "passed": not reasons, "reasons": reasons}


def _usd(value) -> str:
    return "n/a (no price)" if value is None else f"${value:.4f}"


def _pct(value) -> str:
    return "n/a" if value is None else ("inf" if value == float("inf") else f"{value:+.1%}")


def markdown(report: Mapping[str, Any]) -> str:
    b, c, s, d = report["baseline"], report["candidate"], report["paired"], report["decision"]
    verdict = "PASS" if d["passed"] else "FAIL"
    lines = [f"### meta-harness check: {verdict}", "",
             f"`{report['model']}` on {b['n']} tasks, {report['repeats']} repeat(s)", "",
             "| metric | baseline | candidate | delta |", "|---|---|---|---|",
             f"| mean score | {b['score']:.4f} | {c['score']:.4f} | {d['score_delta']:+.4f} |",
             f"| cost (USD) | {_usd(b['cost_usd'])} | {_usd(c['cost_usd'])} | {_pct(d['cost_increase'])} |",
             f"| context tokens | {b['context_cost']:.0f} | {c['context_cost']:.0f} | {_pct(d['context_increase'])} |",
             "", f"Paired: {s['wins']} gained, {s['losses']} lost, {s['ties']} tied; "
             f"McNemar p={s['p']:.4f} ({'significant' if s['significant'] else 'not significant'} "
             f"at alpha={report['thresholds']['alpha']})", ""]
    if d["reasons"]:
        lines += ["Blocking reasons:"] + [f"- {r}" for r in d["reasons"]] + [""]
    return "\n".join(lines)


def _run(args) -> int:
    for label, p in (("baseline", args.baseline), ("candidate", args.candidate), ("data", args.data)):
        if not Path(p).is_file():
            raise SetupError(f"{label} file not found: {p}")
    if args.repeats < 1:
        raise SetupError("--repeats must be >= 1")
    try:
        domain = get_domain(args.domain)
        metric = resolve_metric(args.metric, extra=getattr(domain, "METRICS", None))
        tasks = domain.adapt(read_records(args.data))
    except Exception as exc:  # noqa: BLE001
        raise SetupError(f"{type(exc).__name__}: {exc}") from exc
    if not tasks:
        raise SetupError("no tasks in --data")
    ids = [t.get("id", i) if isinstance(t, Mapping) else i for i, t in enumerate(tasks)]
    if len(set(map(str, ids))) != len(ids):
        raise SetupError("task ids in --data are not unique")
    try:
        model = model_from_environment(args.provider, args.model)
    except Exception as exc:  # noqa: BLE001
        raise SetupError(f"cannot build model {args.provider}:{args.model}: {type(exc).__name__}: {exc}") from exc
    price = _price(args)
    out = Path(args.out)
    root = out.parent / (out.stem + "-traces")
    root.mkdir(parents=True, exist_ok=True)
    try:
        base = _evaluate(Path(args.baseline), tasks, ids, metric, model, price, args, root)
        cand = _evaluate(Path(args.candidate), tasks, ids, metric, model, price, args, root)
    except SetupError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SetupError(f"evaluation failed: {type(exc).__name__}: {exc}") from exc
    paired = compare(base["passes"], {"candidate": cand["passes"]}, q=args.alpha)["candidate"]
    decision = decide(base, cand, paired, args.max_drop, args.max_cost_increase)
    report = {"model": f"{args.provider}:{args.model}", "repeats": args.repeats,
              "thresholds": {"max_drop": args.max_drop, "max_cost_increase": args.max_cost_increase,
                             "alpha": args.alpha},
              "baseline": {k: v for k, v in base.items() if k != "passes"},
              "candidate": {k: v for k, v in cand.items() if k != "passes"},
              "paired": paired, "decision": decision}
    text = markdown(report)
    atomic_write(out, json.dumps(report, indent=2, default=str))
    if args.markdown:
        atomic_write(Path(args.markdown), text)
    print(text)
    return EXIT_PASS if decision["passed"] else EXIT_REGRESSION


def run(args) -> int:
    try:
        return _run(args)
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SETUP
