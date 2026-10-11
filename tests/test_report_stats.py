import json

import pytest

from meta_harness.bench import summarize
from meta_harness.stats import mcnemar_exact
from meta_harness.transfer import paired_comparisons


def row(harness, passes, model="m"):
    return {"harness": harness, "provider": "scripted", "model": model,
            "outcomes": [{"task": i, "pass": bool(p)} for i, p in enumerate(passes)]}


def test_transfer_comparisons_pair_each_harness_with_the_first_per_model():
    rows = [row("base", [0] * 12), row("good", [1] * 12), row("same", [0] * 12),
            row("base", [1] * 12, "n"), row("worse", [0] * 12, "n"),
            {"harness": "bad", "provider": "scripted", "model": "m", "error": "boom"}]
    got = {(c["harness"], c["model"]): c for c in paired_comparisons(rows)}
    assert set(got) == {("good", "m"), ("same", "m"), ("worse", "n")}
    good = got[("good", "m")]
    assert (good["wins"], good["losses"], good["ties"]) == (12, 0, 0)
    assert good["p"] == pytest.approx(mcnemar_exact(0, 12)) and good["significant"] is True
    assert got[("same", "m")]["significant"] is False and got[("same", "m")]["p"] == 1.0
    assert got[("worse", "n")]["losses"] == 12 and got[("worse", "n")]["wins"] == 0
    assert good["baseline"] == "base" and "q" in good


def test_bench_summary_marks_paired_significance_against_first_arm():
    rows = []
    for t in range(12):
        rows.append({"arm": "base", "task_id": f"t{t}", "trial": 0, "score": 0.0})
        rows.append({"arm": "better", "task_id": f"t{t}", "trial": 0, "score": 1.0})
    out = summarize(rows)
    assert "vs_baseline" not in out["base"]
    vs = out["better"]["vs_baseline"]
    assert (vs["baseline"], vs["wins"], vs["losses"], vs["ties"]) == ("base", 12, 0, 0)
    assert vs["p"] == pytest.approx(2 / 4096) and vs["significant"] is True
    json.dumps(out)
