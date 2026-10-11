import json
from pathlib import Path

import pytest

from meta_harness import judge
from meta_harness.bench import Arm, run_matrix
from meta_harness.commands import bench as bench_cmd
from meta_harness.protocol import exact

TASKS = [{"id": f"q{i}", "input": f"q{i}", "expected": f"a{i}"} for i in range(3)]


def make_arms(tmp_path, harness):
    out = []
    for name in ("a", "b"):
        script = tmp_path / f"{name}.jsonl"
        script.write_text("".join(json.dumps({"match": f"q{i}", "text": f"a{i}"}) + "\n" for i in range(3)),
                          encoding="utf-8")
        out.append(Arm(name, harness, "scripted", str(script)))
    return out


def harness_file(tmp_path, body="return model(task['input'])"):
    path = tmp_path / "h.py"
    path.write_text(f"class Harness:\n    def run(self, task, model, trace):\n        {body}\n", encoding="utf-8")
    return path


def idle_arm(tmp_path, harness):
    script = tmp_path / "none.jsonl"
    script.write_text('{"default": "z"}\n', encoding="utf-8")
    return Arm("a", harness, "scripted", str(script))


def rows(matrix):
    return [json.loads(x) for x in matrix.read_text(encoding="utf-8").splitlines()]


def test_real_subprocess_matrix_writes_twelve_cells_and_summary(tmp_path):
    matrix = run_matrix(make_arms(tmp_path, harness_file(tmp_path)), TASKS, 2, exact,
                        tmp_path / "out", cell_timeout=60)
    got = rows(matrix)
    assert len(got) == 12 and all(r["score"] == 1.0 and not r["error"] for r in got)
    assert {"arm", "task_id", "trial", "score", "error", "usage", "seconds"} <= set(got[0])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["a"]["mean"] == 1.0 and summary["a"]["std"] == 0.0 and summary["a"]["failures"] == 0
    assert summary["a"]["per_task"]["q0"] == 1.0 and summary["a"]["outcomes"]["q0"] == [1.0, 1.0]


class Kill(BaseException):
    pass


def ok_runner(calls=None):
    def runner(arm, task, trial, timeout):
        if calls is not None:
            calls.append((arm.name, task["id"], trial))
        return {"result": {"output": task["expected"]}, "error": None, "status": None, "usage": {}}
    return runner


def test_resume_after_kill_has_no_duplicates(tmp_path):
    a = [Arm("a", Path("x"), "p", "m"), Arm("b", Path("x"), "p", "m")]
    seen = []

    def dying(arm, task, trial, timeout):
        if len(seen) == 5:
            raise Kill()
        return ok_runner(seen)(arm, task, trial, timeout)

    out = tmp_path / "out"
    with pytest.raises(Kill):
        run_matrix(a, TASKS, 2, exact, out, cell_runner=dying)
    assert len(rows(out / "matrix.jsonl")) == 5
    again = []
    got = rows(run_matrix(a, TASKS, 2, exact, out, cell_runner=ok_runner(again)))
    assert len(got) == 12 and len(again) == 7
    assert len({(r["arm"], r["task_id"], r["trial"]) for r in got}) == 12


def test_harness_error_scores_zero_and_is_not_retried(tmp_path):
    counter = tmp_path / "count.txt"
    h = harness_file(tmp_path, f"open({str(counter)!r}, 'a').write('x'); raise RuntimeError('boom')")
    (row,) = rows(run_matrix([idle_arm(tmp_path, h)], TASKS[:1], 1, exact, tmp_path / "out", cell_timeout=60))
    assert row["score"] == 0.0 and "boom" in row["error"]
    assert counter.read_text(encoding="utf-8") == "x"


def test_timeout_kept_with_zero_and_not_retried(tmp_path):
    h = harness_file(tmp_path, "import time; time.sleep(60)")
    (row,) = rows(run_matrix([idle_arm(tmp_path, h)], TASKS[:1], 1, exact, tmp_path / "out", cell_timeout=3))
    assert row["score"] == 0.0 and "timeout" in row["error"]


def test_429_is_retried_once_then_succeeds(tmp_path):
    calls = []

    def runner(arm, task, trial, timeout):
        calls.append(1)
        if len(calls) == 1:
            return {"result": None, "error": "ProviderError: 429", "status": 429, "usage": {}}
        return ok_runner()(arm, task, trial, timeout)

    (row,) = rows(run_matrix([Arm("a", Path("x"), "p", "m")], TASKS[:1], 1, exact, tmp_path / "o",
                             cell_runner=runner))
    assert len(calls) == 2 and row["score"] == 1.0 and row["error"] is None


def test_non_transient_status_is_not_retried(tmp_path):
    calls = []

    def runner(arm, task, trial, timeout):
        calls.append(1)
        return {"result": None, "error": "bad request", "status": 400, "usage": {}}

    (row,) = rows(run_matrix([Arm("a", Path("x"), "p", "m")], TASKS[:1], 1, exact, tmp_path / "o",
                             cell_runner=runner))
    assert len(calls) == 1 and row["score"] == 0.0 and row["error"]


def test_verdict_last_line_wins_and_defaults_to_zero(tmp_path, monkeypatch):
    def run(text):
        script = tmp_path / "j.jsonl"
        script.write_text(json.dumps({"default": text}) + "\n", encoding="utf-8")
        monkeypatch.setenv("META_HARNESS_JUDGE_PROVIDER", "scripted")
        monkeypatch.setenv("META_HARNESS_JUDGE_MODEL", str(script))
        return judge.verdict({"criteria": "say hi"}, {"output": "hi"})

    assert run("fine answer\nVERDICT: PASS") == 1.0
    assert run("VERDICT: PASS\nVERDICT: FAIL") == 0.0
    assert run("VERDICT: FAIL\nVERDICT: PASS") == 1.0
    assert run("looks good to me") == 0.0
    assert run("VERDICT: MAYBE") == 0.0


def test_judge_prompt_is_blind():
    secret = "HARNESS_SOURCE_SENTINEL"
    prompt = judge.build_prompt({"criteria": "be right", "input": "q"},
                                {"output": "42", "trace": secret, "harness": secret})
    assert "be right" in prompt and "42" in prompt and secret not in prompt
    assert "forty-two" in judge.build_prompt({"expected": "forty-two"}, {"output": "42"})


def test_cli_parse_arm_and_registry():
    arm = bench_cmd.parse_arm("x=C:\\h\\harness.py@scripted:C:\\s\\f.jsonl")
    assert (arm.name, arm.provider) == ("x", "scripted") and arm.model == "C:\\s\\f.jsonl"
    assert str(arm.harness) == "C:\\h\\harness.py"
    with pytest.raises(ValueError):
        bench_cmd.parse_arm("nonsense")
    from meta_harness.commands import all_commands
    assert "bench" in all_commands()


def test_errors_and_output_are_redacted_in_matrix_and_summary(tmp_path):
    key = "sk-abcdefghijklmnop1234"
    h = harness_file(tmp_path, f"print({key!r}); raise RuntimeError('bad key {key}')")
    out = tmp_path / "out"
    run_matrix([idle_arm(tmp_path, h)], TASKS[:1], 1, exact, out, cell_timeout=60)
    for name in ("matrix.jsonl", "summary.json"):
        assert key not in (out / name).read_text(encoding="utf-8")
    (row,) = rows(out / "matrix.jsonl")
    assert "[redacted]" in row["error"]


def test_cell_env_is_scoped_to_the_arm_provider(monkeypatch):
    from meta_harness.bench import _cell_env
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "UNIKEY_API_KEY", "ANTHROPIC_API_KEY",
                 "META_HARNESS_JUDGE_PROVIDER", "META_HARNESS_JUDGE_MODEL"):
        monkeypatch.setenv(name, "value-" + name)
    env = _cell_env("anthropic")
    assert env["ANTHROPIC_API_KEY"] == "value-ANTHROPIC_API_KEY"
    assert not {"OPENAI_API_KEY", "GROQ_API_KEY", "UNIKEY_API_KEY",
                "META_HARNESS_JUDGE_PROVIDER", "META_HARNESS_JUDGE_MODEL"} & set(env)
    assert "ANTHROPIC_API_KEY" not in _cell_env("scripted")


def test_torn_last_line_is_repaired_and_cell_reruns_once(tmp_path):
    a = [Arm("a", Path("x"), "p", "m")]
    out = tmp_path / "out"
    run_matrix(a, TASKS, 1, exact, out, cell_runner=ok_runner())
    path = out / "matrix.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-20], encoding="utf-8")  # tear the last row, no newline
    again = []
    run_matrix(a, TASKS, 1, exact, out, cell_runner=ok_runner(again))
    assert len(again) == 1
    good = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.endswith("}")]
    assert len({r["task_id"] for r in good}) == 3
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["a"]["cells"] == 3


def test_real_provider_error_429_is_retried(tmp_path):
    from meta_harness.providers import ProviderError
    calls = []

    def runner(arm, task, trial, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError("HTTP 429: slow down")
        return ok_runner()(arm, task, trial, timeout)

    (row,) = rows(run_matrix([Arm("a", Path("x"), "p", "m")], TASKS[:1], 1, exact, tmp_path / "o",
                             cell_runner=runner))
    assert len(calls) == 2 and row["score"] == 1.0


def test_judge_retries_transient_provider_errors(monkeypatch):
    from meta_harness.providers import ProviderError
    calls = []

    def model(prompt):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError("HTTP 503: down")
        return "ok\nVERDICT: PASS"

    monkeypatch.setattr(judge, "judge_model", lambda: model)
    monkeypatch.setattr(judge, "RETRY_DELAY", 0)
    assert judge.verdict({"criteria": "c"}, {"output": "o"}) == 1.0 and len(calls) == 2


def test_key_straddling_the_tail_cut_is_redacted(tmp_path):
    key = "sk-abcdefghijklmnop1234"
    body = ("import sys, os; sys.stderr.write('A'*100 + %r + 'B'*490); sys.stderr.flush(); os._exit(1)" % key)
    h = harness_file(tmp_path, body)
    out = tmp_path / "out"
    run_matrix([idle_arm(tmp_path, h)], TASKS[:1], 1, exact, out, cell_timeout=60)
    text = (out / "matrix.jsonl").read_text(encoding="utf-8")
    assert "worker exited" in text or "exited" in text or "[redacted]" in text
    assert key[-10:] not in text and key not in text and "[redacted]" in text


def test_torn_line_ending_in_brace_is_dropped_by_parsing(tmp_path):
    a = [Arm("a", Path("x"), "p", "m")]
    out = tmp_path / "out"
    run_matrix(a, TASKS[:2], 1, exact, out, cell_runner=ok_runner())
    path = out / "matrix.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"arm": "a", "task_id": "q2", "trial": 0, "score": 1.0, "usage": {}}'[:-1] + "}}")
    again = []
    run_matrix(a, TASKS, 1, exact, out, cell_runner=ok_runner(again))
    assert len(again) == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(json.loads(x) for x in lines)


def test_worker_side_provider_error_429_is_retried_once(tmp_path):
    counter = tmp_path / "attempts.txt"
    body = (f"from meta_harness.providers import ProviderError; "
            f"import os; seen = os.path.exists({str(counter)!r}); "
            f"open({str(counter)!r}, 'a').write('x'); "
            f"(_ for _ in ()).throw(ProviderError('HTTP 429: slow down')) if not seen else None; "
            f"return model(task['input'])")
    arm = make_arms(tmp_path, harness_file(tmp_path, body))[0]
    (row,) = rows(run_matrix([arm], TASKS[:1], 1, exact, tmp_path / "out", cell_timeout=60))
    assert counter.read_text(encoding="utf-8") == "xx"
    assert row["score"] == 1.0 and not row["error"]
