"""Live baseline (no search): run each use case's test split with a real model and grade it.

    python examples/usecases/live_baseline.py --models deepseek-flash gpt-4o-mini --limit 15 [--env PATH]

Keys come from the environment (optionally loaded from an env file; only key names are printed).
Writes results to examples/usecases/<name>/baseline_<model>.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from usecases._common import UsageMeter, live_llm, read_jsonl  # noqa: E402

USECASES = ("tau_retail", "sql_analyst", "research_ops")


def baseline(name: str, model: str, limit: int, workers: int = 5) -> dict:
    app = importlib.import_module(f"usecases.{name}.app")
    grader = importlib.import_module(f"usecases.{name}.grader")
    test = [t for t in read_jsonl(HERE / name / "tasks.jsonl") if t["split"] == "test"][:limit]
    meter, llm = UsageMeter(), live_llm(model)

    def one(task):
        result = app.run(task, llm, callbacks=[meter.handler])
        g = grader.grade(task, result)
        return {"id": task["id"], **g, "error": result["error"], "writes": sum(c["write"] for c in result["tool_calls"])}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, test))
    score = sum(r["score"] for r in rows) / len(rows)
    out = {"usecase": name, "model": model, "n": len(rows), "score": round(score, 3),
           "input_tokens": meter.input, "output_tokens": meter.output, "cost_usd": round(meter.cost(model), 4),
           "runs": rows}
    (HERE / name / f"baseline_{model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["deepseek-flash", "gpt-4o-mini"])
    ap.add_argument("--usecases", nargs="+", default=list(USECASES))
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--env")
    args = ap.parse_args()
    if args.env:
        from meta_harness.envfile import load_env
        print("loaded keys:", sorted(load_env(args.env)))
    total = 0.0
    for name in args.usecases:
        for model in args.models:
            out = baseline(name, model, args.limit)
            total += out["cost_usd"]
            print(f"{name:13s} {model:15s} score={out['score']:.3f} n={out['n']} cost=${out['cost_usd']:.4f} "
                  f"errors={sum(bool(r['error']) for r in out['runs'])}", flush=True)
    print(f"total cost ${total:.4f}")


if __name__ == "__main__":
    main()
