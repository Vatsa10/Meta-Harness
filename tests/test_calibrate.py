import io
import json
import sys

import pytest

from meta_harness import __main__ as cli
from meta_harness import judge
from meta_harness.commands import label
from meta_harness.integrations import calibrate as cal

TASKS = [
    {"id": "a", "input": "Refund order 1", "reference_output": "Refunded $60.", "failure": "negative",
     "criteria": "c", "tool_calls": [{"name": "lookup_order", "args": {"id": 1}, "output": {"total": 60}},
                                    {"name": "issue_refund", "args": {"id": 1, "amount": 120}, "output": "ok"}]},
    {"id": "b", "input": "Hours?", "reference_output": "9 to 5", "failure": "ok", "criteria": "c"},
    {"id": "c", "input": "Cancel", "reference_output": "", "failure": "error", "criteria": "c"},
]


def _write_tasks(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in TASKS), encoding="utf-8")
    return path


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_label_loop_records_reprompts_and_resumes(tmp_path, monkeypatch):
    tasks, out = _write_tasks(tmp_path), tmp_path / "labels.jsonl"
    shown = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("b\nrefunded twice the amount\nx\ng\n\n"))
    monkeypatch.setattr(sys, "stdout", shown)
    assert cli.main(["label", "--in", str(tasks), "--out", str(out)]) == 0   # EOF after 2 answers
    text = shown.getvalue()
    assert ">>> WRITE issue_refund" in text and ">>> WRITE lookup_order" not in text
    assert "-         lookup_order" in text and "Refunded $60." in text
    assert "please answer g, b, s or q" in text
    rows = _rows(out)
    assert [(r["id"], r["label"], r["reason"]) for r in rows] == [
        ("a", "bad", "refunded twice the amount"), ("b", "good", "")]
    assert rows[0]["output"] == "Refunded $60." and rows[0]["tool_calls"][1]["name"] == "issue_refund"
    # resume: only "c" is left; skip it, then nothing remains
    monkeypatch.setattr(sys, "stdin", io.StringIO("s\n"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert cli.main(["label", "--in", str(tasks), "--out", str(out)]) == 0
    assert [(r["id"], r["label"]) for r in _rows(out)][2] == ("c", "skip")
    again = io.StringIO()
    monkeypatch.setattr(sys, "stdout", again)
    monkeypatch.setattr(sys, "stdin", io.StringIO("g\n"))
    assert cli.main(["label", "--in", str(tasks), "--out", str(out)]) == 0
    assert len(_rows(out)) == 3 and '"remaining": 0' in again.getvalue()


def test_label_quit_and_n_cap(tmp_path):
    tasks, out = _write_tasks(tmp_path), tmp_path / "labels.jsonl"
    summary = label.label_loop(TASKS, out, 1, io.StringIO("g\nfine\ng\n"), io.StringIO())
    assert summary["good"] == 1 and summary["remaining"] == 2 and len(_rows(out)) == 1
    summary = label.label_loop(TASKS, out, 5, io.StringIO("q\ng\n"), io.StringIO())
    assert summary == {"good": 0, "bad": 0, "skip": 0, "remaining": 2, "out": str(out)}


# --- calibration -----------------------------------------------------------------------------

def _labels():
    good = [("g1", "answer ok A", "clear"), ("g2", "answer ok B", ""), ("g3", "answer ok C", ""),
            ("g4", "partial D", "partial is fine for this user"), ("g5", "answer ok E", "")]
    bad = [("b1", "answer ok but refund twice", "double refund"), ("b2", "wrong F", "wrong"),
           ("b3", "wrong G", ""), ("b4", "wrong H", ""), ("b5", "wrong I", "")]
    rows = [{"id": i, "label": "good", "output": o, "reason": r, "criteria": "be right"} for i, o, r in good]
    rows += [{"id": i, "label": "bad", "output": o, "reason": r, "criteria": "be right"} for i, o, r in bad]
    rows.append({"id": "s1", "label": "skip", "output": "x", "reason": "", "criteria": ""})
    return rows


def _answer(prompt):
    return prompt.rsplit("Answer:\n", 1)[1].split("\n\nReason", 1)[0]


def scripted_judge(seen):
    """Base prompt: pass iff 'ok' in the answer. With the owner's examples: also fail refunds, pass partial."""
    def model(prompt):
        seen.append(prompt)
        answer = _answer(prompt)
        if "Owner:" in prompt:
            ok = ("ok" in answer and "refund" not in answer) or "partial" in answer
        else:
            ok = "ok" in answer
        return "VERDICT: PASS" if ok else "VERDICT: FAIL"
    return model


def test_score_hand_checked():
    # tp=4 fn=1 fp=1 tn=4: po=.8, pe=(5*5+5*5)/100=.5, kappa=.6
    pairs = [(True, True)] * 4 + [(True, False)] + [(False, True)] + [(False, False)] * 4
    result = cal.score(pairs)
    assert result["agreement"] == pytest.approx(0.8) and result["kappa"] == pytest.approx(0.6)
    assert result["confusion"] == {"tp": 4, "fn": 1, "fp": 1, "tn": 4}
    # skewed: tp=8 fn=0 fp=2 tn=0 -> po=.8, pe=(8*10+2*0)/100=.8, kappa=0
    assert cal.score([(True, True)] * 8 + [(False, True)] * 2)["kappa"] == pytest.approx(0.0)
    assert cal.score([(True, True), (False, False)])["kappa"] == pytest.approx(1.0)


def test_calibrate_picks_few_shot_and_saves_judge(tmp_path):
    seen = []
    with pytest.warns(UserWarning, match="only 10 labelled"):
        report = cal.calibrate(_labels(), scripted_judge(seen), out=tmp_path / "judge.json")
    base, few = report["candidates"]
    assert (base["variant"], base["agreement"], base["kappa"]) == ("base", pytest.approx(0.8), pytest.approx(0.6))
    assert base["confusion"] == {"tp": 4, "fn": 1, "fp": 1, "tn": 4}
    assert (few["variant"], few["agreement"], few["kappa"]) == ("few-shot", 1.0, 1.0)
    assert report["variant"] == "few-shot" and report["n"] == 10 and report["prompt"] == judge.JUDGE_PROMPT
    assert [e["id"] for e in report["few_shot"]] == ["b1", "b2", "g1", "g4"]
    assert len(seen) == 20                                    # 10 runs x 2 variants; skip ignored
    # leave-one-out: b1's own reason never appears in the prompt that grades b1
    b1_prompts = [p for p in seen if _answer(p) == "answer ok but refund twice" and "Owner:" in p]
    assert b1_prompts and all("double refund" not in p for p in b1_prompts)
    saved = json.loads((tmp_path / "judge.json").read_text(encoding="utf-8"))
    assert saved["agreement"] == 1.0 and saved["few_shot"] == report["few_shot"]
    assert all(e["reason"] for e in saved["few_shot"])


def test_calibrate_warns_on_low_agreement_and_keeps_first_on_tie(tmp_path):
    always_pass = lambda prompt: "VERDICT: PASS"  # noqa: E731
    rows = [r for r in _labels() if r["label"] != "skip"] * 2           # n=20: no size warning
    with pytest.warns(UserWarning) as record:
        report = cal.calibrate(rows, always_pass, prompts=["A {criteria} {answer}", "B {criteria} {answer}"],
                               out=None)
    messages = [str(w.message) for w in record]
    assert any("below 0.8" in m for m in messages) and not any("labelled runs" in m for m in messages)
    assert report["agreement"] == 0.5 and report["kappa"] == 0.0
    assert (report["prompt_index"], report["variant"]) == (0, "base") and "path" not in report


def test_calibrated_judge_metric_uses_saved_prompt(tmp_path, monkeypatch):
    seen = []
    with pytest.warns(UserWarning):
        cal.calibrate(_labels(), scripted_judge([]), out=tmp_path / "judge.json")
    monkeypatch.setenv(cal.JUDGE_FILE_ENV, str(tmp_path / "judge.json"))
    monkeypatch.setattr(judge, "judge_model", lambda: scripted_judge(seen))
    assert cal.calibrated_judge({"criteria": "be right"}, {"output": "partial Z"}) == 1.0
    assert cal.calibrated_judge({"criteria": "be right"}, {"output": "ok but refund"}) == 0.0
    assert "Owner: FAIL - double refund" in seen[0] and "be right" in seen[0]


def test_calibrated_judge_needs_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv(cal.JUDGE_FILE_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError, match="calibrate"):
        cal.calibrated_judge({}, {"output": "x"})


def test_calibrated_judge_reloads_a_rewritten_file(tmp_path, monkeypatch):
    import os
    path = tmp_path / "judge.json"
    path.write_text(json.dumps({"prompt": "first {answer}", "few_shot": []}), encoding="utf-8")
    monkeypatch.setenv(cal.JUDGE_FILE_ENV, str(path))
    seen = []
    monkeypatch.setattr(judge, "judge_model", lambda: lambda p: seen.append(p) or "VERDICT: PASS")
    cal.calibrated_judge({}, {"output": "x"})
    path.write_text(json.dumps({"prompt": "second {answer}", "few_shot": []}), encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))
    cal.calibrated_judge({}, {"output": "x"})
    assert seen == ["first x", "second x"]
