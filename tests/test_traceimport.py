import json
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness import traceimport as ti

FIX = Path(__file__).parent / "fixtures" / "traces"
SMOKE = Path(__file__).parent / "fixtures" / "smoke"


def _by_id(traces):
    return {t.id: t for t in traces}


def test_otel_parses_both_semconv_styles_and_error_span():
    traces = _by_id(ti.load("otel", FIX / "otel.json"))
    assert len(traces) == 4
    ok = traces["4bf92f3577b34da6a3ce929d0e0e4736"]
    assert ok.input == "How do I reset my password?" and ok.output.startswith("Open Settings")
    assert ok.kind == "ok"
    err = traces["5c0a1e2f3b4d5e6f708192a3b4c5d6e7"]
    assert err.error == "tool lookup_order timed out after 8s" and err.kind == "error"
    # system message dropped, parts flattened
    assert err.input == "Where is order 8812? Email me at jane.doe@example.com"
    assert err.output == "Let me look that up."
    neg = traces["6d1b2f3a4c5e6d7f8091a2b3c4d5e6f7"]
    assert neg.input == "Can I get a refund after 45 days?" and neg.kind == "negative"
    assert neg.feedback == ["thumbs_down"]


def test_langfuse_parses_observations_scores_and_error_level():
    traces = _by_id(ti.load("langfuse", FIX / "langfuse.json"))
    assert traces["lf-trace-001"].input == "What is the warranty period for the X200 router?"
    assert traces["lf-trace-001"].kind == "ok"
    assert traces["lf-trace-002"].kind == "negative"
    assert traces["lf-trace-002"].min_score == 0.0
    assert traces["lf-trace-003"].error == "context length exceeded (131072 tokens)"
    assert traces["lf-trace-004"].scores == [("helpfulness", 0.8, "")]
    assert traces["lf-trace-004"].kind == "ok"


def test_jsonl_feedback_score_and_error_classes():
    traces = _by_id(ti.load("jsonl", FIX / "generic.jsonl"))
    kinds = {key: t.kind for key, t in traces.items()}
    assert kinds["j3"] == "error"
    assert {k for k, v in kinds.items() if v == "negative"} == {"j2", "j6", "j9"}
    assert kinds["j1"] == kinds["j7"] == kinds["j4"] == "ok"


def test_selection_is_failure_first_and_dedups_keeping_the_failure():
    traces = ti.load("jsonl", FIX / "generic.jsonl")
    chosen = ti.select(traces)
    ids = [t.id for t in chosen]
    assert ids[0] == "j3"                              # error first
    assert ids[1:4] == ["j6", "j2", "j9"]              # then negatives, lowest score first
    assert "j5" not in ids and "j2" in ids             # duplicate of j2 dropped, failure kept
    assert len(ids) == 9
    assert [t.id for t in ti.select(traces, limit=2)] == ["j3", "j6"]


def test_dedup_keeps_failure_even_when_ok_copy_comes_first():
    rows = [ti.Trace("a", "Hello  World!", "hi"), ti.Trace("b", "hello world", "x", error="boom")]
    assert [t.id for t in ti.select(rows)] == ["b"]


def test_criteria_reflect_feedback_scores_and_errors():
    tasks = {t["id"]: t for t in ti.import_traces("jsonl", FIX / "generic.jsonl")}
    assert "RateLimitError" in tasks["j3"]["criteria"]
    assert "wrong, it is 391" in tasks["j2"]["criteria"]
    assert "unsatisfactory" in tasks["j2"]["criteria"] and "381" in tasks["j2"]["criteria"]
    assert "accepted" in tasks["j1"]["criteria"] and "Bonjour" in tasks["j1"]["criteria"]
    assert tasks["j1"]["reference_output"] == "Bonjour"
    assert tasks["j1"]["source_trace_id"] == "j1"
    from meta_harness.judge import build_prompt
    assert "wrong, it is 391" in build_prompt(tasks["j2"], {"output": "391"})


def test_redaction_always_and_pii_only_on_request():
    plain = {t["id"]: t for t in ti.import_traces("jsonl", FIX / "generic.jsonl")}
    assert "sk-abcdef" not in json.dumps(plain) and "[redacted]" in plain["j6"]["input"]
    lf = {t["id"]: t for t in ti.import_traces("langfuse", FIX / "langfuse.json")}
    assert "4111 1111 1111 1111" in lf["lf-trace-002"]["input"]
    scrubbed = {t["id"]: t for t in ti.import_traces("langfuse", FIX / "langfuse.json", pii=True)}
    assert scrubbed["lf-trace-002"]["input"] == "Cancel my subscription, card [card], call me on [phone]"
    otel = ti.import_traces("otel", FIX / "otel.json", pii=True)
    assert "jane.doe@example.com" not in json.dumps(otel) and "[email]" in json.dumps(otel)
    # ordinary numbers survive the scrub
    assert ti.scrub_pii("order 8812 costs 45 dollars, 131072 tokens") == "order 8812 costs 45 dollars, 131072 tokens"


def test_split_is_seeded_disjoint_complete_and_spreads_failures():
    tasks = ti.import_traces("jsonl", FIX / "generic.jsonl")
    a, b = ti.split(tasks, seed=3), ti.split(tasks, seed=3)
    assert a == b
    ids = [t["id"] for name in ti.SPLITS for t in a[name]]
    assert sorted(ids) == sorted(t["id"] for t in tasks) and len(ids) == len(set(ids))
    assert [len(a[n]) for n in ("train", "search", "test")] == [2, 4, 3]
    for name in ti.SPLITS:
        assert any(t["failure"] != "ok" for t in a[name]), name
    assert any(ti.split(tasks, seed=s) != a for s in range(4, 10))
    with pytest.raises(ValueError):
        ti.split(tasks, train=0.6, test=0.5)


def test_stats_command(capsys):
    assert cli.main(["traces", "stats", "--format", "otel", "--in", str(FIX / "otel.json")]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["traces"] == 4 and stats["duplicates"] == 1 and stats["errors"] == 1
    assert stats["negative_feedback"] == 1 and stats["ok"] == 2


def test_import_command_needs_an_output(capsys):
    assert cli.main(["traces", "import", "--format", "jsonl", "--in", str(FIX / "generic.jsonl")]) == 2


def test_imported_eval_set_runs_through_search_with_the_judge(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "evals"
    assert cli.main(["traces", "import", "--format", "jsonl", "--in", str(FIX / "generic.jsonl"),
                     "--out", str(tmp_path / "all.jsonl"), "--out-dir", str(out_dir), "--seed", "3", "--pii"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["tasks"] == 9 and summary["test"]["tasks"] == 3
    test_ids = [json.loads(line)["id"] for line in (out_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "j2" in test_ids
    harness = tmp_path / "seed.py"
    harness.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                       "        return {'output': model('answer: ' + task['input'])}\n", encoding="utf-8")
    model = tmp_path / "model.jsonl"
    model.write_text(json.dumps({"match": r"17 \* 23", "text": "391"}) + "\n"
                     + json.dumps({"default": "unsure"}) + "\n", encoding="utf-8")
    judge = tmp_path / "judge.jsonl"
    judge.write_text(json.dumps({"match": r"wrong, it is 391[\s\S]*Answer:\n391\b", "text": "ok\nVERDICT: PASS"}) + "\n"
                     + json.dumps({"default": "VERDICT: FAIL"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("META_HARNESS_JUDGE_PROVIDER", "scripted")
    monkeypatch.setenv("META_HARNESS_JUDGE_MODEL", str(judge))
    code = cli.main(["search", "--harness", str(harness), "--search", str(out_dir / "search.jsonl"),
                     "--test", str(out_dir / "test.jsonl"), "--metric", "meta_harness.judge:verdict",
                     "--provider", "scripted", "--model", str(model), "--root", str(tmp_path / "store"),
                     "--iterations", "0", "--no-cache"])
    assert code == 0
    results = json.loads((Path(str(tmp_path / "store") + "-test") / "test_results.json").read_text(encoding="utf-8"))
    # only j2 (answered "391", judged against its "wrong, it is 391" criteria) passes: 1 of 3
    assert results["results"][0]["score"] == pytest.approx(1 / 3)


def test_harness_never_sees_reference_or_criteria_but_metric_does():
    from meta_harness import protocol
    task = {"input": "q", "reference_output": "secret ref", "criteria": "be right",
            "label": "x", "rubric": "hidden too", "grading_fields": ["rubric"], "id": "t"}
    view = protocol.hide_target(task)
    assert view == {"input": "q", "id": "t"}
    assert task["criteria"] == "be right"            # original untouched for the metric


def test_search_harness_echoing_reference_scores_zero_while_judge_sees_criteria(tmp_path, monkeypatch):
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"id": "a", "input": "q", "reference_output": "REF-ANSWER",
                                "criteria": "CRIT-TOKEN"}) + "\n", encoding="utf-8")
    harness = tmp_path / "echo.py"
    harness.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                       "        return {'output': str(task.get('reference_output')) + '|' + str(task.get('criteria'))}\n",
                       encoding="utf-8")
    model = tmp_path / "m.jsonl"
    model.write_text(json.dumps({"default": "x"}) + "\n", encoding="utf-8")
    judge = tmp_path / "j.jsonl"
    # PASS only if the judge saw the criteria AND the harness output was None|None (nothing leaked)
    judge.write_text(json.dumps({"match": r"CRIT-TOKEN[\s\S]*Answer:\nNone\|None", "text": "VERDICT: PASS"}) + "\n"
                     + json.dumps({"default": "VERDICT: FAIL"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("META_HARNESS_JUDGE_PROVIDER", "scripted")
    monkeypatch.setenv("META_HARNESS_JUDGE_MODEL", str(judge))
    assert cli.main(["search", "--harness", str(harness), "--search", str(data), "--test", str(data),
                     "--metric", "meta_harness.judge:verdict", "--provider", "scripted", "--model", str(model),
                     "--root", str(tmp_path / "store"), "--iterations", "0", "--no-cache"]) == 0
    results = json.loads((Path(str(tmp_path / "store") + "-test") / "test_results.json").read_text(encoding="utf-8"))
    assert results["results"][0]["score"] == 1.0
    exact = tmp_path / "e.jsonl"
    exact.write_text(json.dumps({"id": "a", "input": "q", "reference_output": "None|None"}) + "\n", encoding="utf-8")


def test_empty_split_warns_and_dash_writes_stdout(tmp_path, capsys):
    assert cli.main(["traces", "import", "--format", "otel", "--in", str(FIX / "otel.json"),
                     "--out", "-", "--out-dir", str(tmp_path / "o"), "--train-fraction", "0"]) == 0
    captured = capsys.readouterr()
    rows = [json.loads(line) for line in captured.out.splitlines()]
    assert len(rows) == 3 and rows[0]["failure"] == "error"
    assert "warning: the train split is empty" in captured.err
    assert not (Path.cwd() / "-").exists()


def test_call_run_strips_grading_fields_for_every_protocol():
    from meta_harness import protocol

    class Echo:
        def run(self, task, model, trace):
            return {"output": sorted(task)}

    task = {"input": "q", "criteria": "c", "reference_output": "r", "rubric": "z", "grading_fields": "rubric"}
    assert protocol.call_run(Echo(), task, None, None)["output"] == ["input"]
    assert set(task) == {"input", "criteria", "reference_output", "rubric", "grading_fields"}


def test_cheating_harness_returning_label_scores_zero_in_plain_search(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"input": f"item {i}", "label": f"L{i}"}) + "\n" for i in range(4)),
                    encoding="utf-8")
    harness = tmp_path / "cheat.py"
    harness.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                       "        return {'output': str(task.get('label') or task.get('expected'))}\n", encoding="utf-8")
    model = tmp_path / "m.jsonl"
    model.write_text(json.dumps({"default": "x"}) + "\n", encoding="utf-8")
    for isolation in ("none", "process"):
        root = tmp_path / f"store-{isolation}"
        assert cli.main(["search", "--harness", str(harness), "--search", str(data), "--test", str(data),
                         "--protocol", "plain", "--isolation", isolation, "--provider", "scripted",
                         "--model", str(model), "--root", str(root), "--iterations", "0", "--no-cache"]) == 0
        results = json.loads((Path(str(root) + "-test") / "test_results.json").read_text(encoding="utf-8"))
        assert results["results"][0]["score"] == 0.0, isolation


def test_draft_verification_learns_labels_only_through_learn():
    from meta_harness.cls_harnesses import DraftVerificationHarness
    from meta_harness import protocol

    class Trace:
        def event(self, *a, **k):
            pass

    harness = DraftVerificationHarness(["fruit", "vehicle"])
    task = {"input": "apple pie", "label": "fruit"}
    result = protocol.call_run(harness, task, lambda prompt: "fruit", Trace())
    assert harness.memory == []
    protocol.call_learn(harness, task, result, None)
    assert [(m.text, m.label) for m in harness.memory] == [("apple pie", "fruit")]
