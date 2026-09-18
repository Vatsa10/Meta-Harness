import json
import subprocess

import pytest

from meta_harness.claude_cli import ClaudeCliError, ClaudeCliModel, CliRetryPolicy
from meta_harness.providers import model_from_environment


def _result_payload(text="ok", **extra):
    return json.dumps([
        {"type": "system", "subtype": "init"},
        {"type": "result", "subtype": "success", "is_error": False, "result": text,
         "usage": {"input_tokens": 1963, "cache_read_input_tokens": 0, "output_tokens": 12},
         "total_cost_usd": 0.0035, **extra},
    ])


def _fake_run(monkeypatch, stdout="", returncode=0, capture=None, stderr=""):
    def run(argv, **kwargs):
        if capture is not None:
            capture.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", run)


def test_prompt_goes_on_stdin_not_argv(monkeypatch):
    calls = []
    _fake_run(monkeypatch, _result_payload(), capture=calls)
    big = "x" * 50_000
    ClaudeCliModel("haiku").call(big)
    argv, kwargs = calls[0]
    assert kwargs["input"] == big
    assert big not in " ".join(argv)


def test_argv_makes_the_cli_behave_like_a_completion_endpoint(monkeypatch):
    calls = []
    _fake_run(monkeypatch, _result_payload(), capture=calls)
    ClaudeCliModel("haiku").call("hi")
    argv = calls[0][0]
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--model") + 1] == "haiku"
    assert argv[argv.index("--max-turns") + 1] == "1"
    assert argv[argv.index("--output-format") + 1] == "json"
    # Minimal tool set: the default schema costs ~27k input tokens per call.
    assert argv[argv.index("--tools") + 1] == "Read"
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--setting-sources") + 1] == ""


def test_gateway_overrides_are_stripped_from_the_child_environment(monkeypatch):
    calls = []
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    _fake_run(monkeypatch, _result_payload(), capture=calls)
    ClaudeCliModel("haiku").call("hi")
    env = calls[0][1]["env"]
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_usage_omits_prompt_tokens_so_metering_prices_the_harness_prompt(monkeypatch):
    _fake_run(monkeypatch, _result_payload())
    _, usage = ClaudeCliModel("haiku").call("hi")
    # The CLI's own input count includes its system prompt and tool schema, which every
    # candidate pays equally and none controls.
    assert "prompt_tokens" not in usage
    assert usage["cli_input_tokens"] == 1963
    assert usage["completion_tokens"] == 12
    assert usage["cost_usd"] == 0.0035


def test_dunder_call_returns_text(monkeypatch):
    _fake_run(monkeypatch, _result_payload("billing"))
    assert ClaudeCliModel("haiku")("hi") == "billing"


def test_cli_error_event_raises(monkeypatch):
    payload = json.dumps([{"type": "result", "subtype": "error_max_turns", "is_error": True,
                           "result": "turn limit"}])
    _fake_run(monkeypatch, payload)
    with pytest.raises(ClaudeCliError, match="error_max_turns"):
        ClaudeCliModel("haiku", retry=CliRetryPolicy(attempts=1)).call("hi")


def test_nonzero_exit_raises_with_stderr(monkeypatch):
    _fake_run(monkeypatch, "", returncode=1, stderr="boom")
    with pytest.raises(ClaudeCliError, match="boom"):
        ClaudeCliModel("haiku", retry=CliRetryPolicy(attempts=1)).call("hi")


def test_non_json_output_raises(monkeypatch):
    _fake_run(monkeypatch, "not json at all")
    with pytest.raises(ClaudeCliError, match="did not return JSON"):
        ClaudeCliModel("haiku", retry=CliRetryPolicy(attempts=1)).call("hi")


def test_transient_failure_is_retried(monkeypatch):
    outcomes = [subprocess.CompletedProcess([], 1, "", "flaky"),
                subprocess.CompletedProcess([], 0, _result_payload("second try"), "")]
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: outcomes.pop(0))
    monkeypatch.setattr("meta_harness.claude_cli.time.sleep", lambda _: None)
    assert ClaudeCliModel("haiku", retry=CliRetryPolicy(attempts=2))("hi") == "second try"


def test_missing_binary_raises_immediately(monkeypatch):
    def run(*_, **__):
        raise OSError("not found")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ClaudeCliError, match="cannot run"):
        ClaudeCliModel("haiku", binary="nope").call("hi")


def test_factory_builds_the_cli_model():
    assert isinstance(model_from_environment("claude-cli", "haiku"), ClaudeCliModel)
    assert isinstance(model_from_environment("claude-code", "sonnet"), ClaudeCliModel)


def test_available_checks_path(monkeypatch):
    monkeypatch.setattr("meta_harness.claude_cli.shutil.which", lambda name: None)
    assert ClaudeCliModel.available() is False
    monkeypatch.setattr("meta_harness.claude_cli.shutil.which", lambda name: "/usr/bin/claude")
    assert ClaudeCliModel.available() is True
