import json
import os

import pytest

from meta_harness.envfile import load_env, parse_env
from meta_harness.httpdata import read_jsonl
from meta_harness.redact import bounded, redact, safe_id


def test_parse_handles_spaces_quotes_comments_export():
    text = ("# keys\n\nUNIKEY_API_KEY =abc123\nGROQ_API_KEY= 'g k'\nexport OPENAI_API_KEY=\"o\"\n"
            "NOEQUALS\nX=1 # trailing\nEMPTY=\n")
    assert parse_env(text) == {"UNIKEY_API_KEY": "abc123", "GROQ_API_KEY": "g k",
                               "OPENAI_API_KEY": "o", "X": "1", "EMPTY": ""}


def test_load_never_overrides_and_returns_names_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("\ufeffA=from-file\nB=from-file\n", encoding="utf-8")
    environ = {"A": "already"}
    assert load_env(env_file, environ) == ["B"]
    assert environ == {"A": "already", "B": "from-file"}
    assert load_env(tmp_path / "missing.env", environ) == []


def test_cli_loads_env_before_dispatch(tmp_path, monkeypatch, capsys):
    from meta_harness import __main__ as cli
    (tmp_path / "k.env").write_text("META_HARNESS_TEST_SECRET = s3cret\n", encoding="utf-8")
    monkeypatch.delenv("META_HARNESS_TEST_SECRET", raising=False)
    assert cli.main(["--env-file", str(tmp_path / "k.env"), "inspect", str(tmp_path / "none")]) == 0
    assert os.environ["META_HARNESS_TEST_SECRET"] == "s3cret"
    assert "s3cret" not in capsys.readouterr().out


@pytest.mark.parametrize("text", ["Authorization: Bearer abc.def", "api_key=xyz", "sk-proj-1234567890"])
def test_redact_patterns(text):
    out = redact(text)
    assert "[redacted]" in out
    for secret in ("abc.def", "xyz", "1234567890"):
        assert secret not in out


def test_redact_loaded_key_value(tmp_path, monkeypatch):
    monkeypatch.delenv("MH_REDACT_TEST_API_KEY", raising=False)
    (tmp_path / ".env").write_text("MH_REDACT_TEST_API_KEY=plainvalue42\n", encoding="utf-8")
    assert load_env(tmp_path / ".env") == ["MH_REDACT_TEST_API_KEY"]
    try:
        assert redact("the value plainvalue42 leaked") == "the value [redacted] leaked"
    finally:
        os.environ.pop("MH_REDACT_TEST_API_KEY", None)


def test_bounded_is_one_line_and_capped():
    out = bounded("line one\nline two\r\n" + "x" * 5000, limit=100)
    assert "\n" not in out and "\r" not in out
    assert out.startswith("line one line two")
    assert len(out) <= 100 + len("...[+9999 chars]")
    assert out.endswith("chars]")


def test_safe_id_is_ascii_and_short():
    text = "\u00e9\u4e2d" * 100
    out = safe_id(text)
    assert out.isascii() and len(out) <= 64 and out
    assert safe_id("a" * 200) != safe_id("a" * 199)


def test_jsonl_torn_last_line_tolerated_middle_raises(tmp_path):
    good = tmp_path / "g.jsonl"
    good.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n{\"a\": 3", encoding="utf-8")
    assert read_jsonl(good) == [{"a": 1}, {"a": 2}]
    bad = tmp_path / "b.jsonl"
    bad.write_text(json.dumps({"a": 1}) + "\n{\"a\": \n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(bad)


def test_inline_comment_rules():
    text = ('A="sk-abc"  # prod\nB=\'a # b\'\nC=v # c\nD=v#notacomment\nE="x#y" # z\n'
            'F=\'q\'\t# tab\n')
    assert parse_env(text) == {"A": "sk-abc", "B": "a # b", "C": "v", "D": "v#notacomment",
                               "E": "x#y", "F": "q"}


def test_redact_covers_secret_named_environment_values(monkeypatch):
    values = {"AWS_SECRET_ACCESS_KEY": "aws-secret-value-1", "HF_KEY": "hf-key-value-22",
              "DB_PASSWORD": "db-pass-value-333", "SVC_CREDENTIALS": "creds-value-4444",
              "DATABASE_URL_SECRET_X": "dburl-secret-555"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PLAIN_NAME", "plain-value-not-secret")
    from meta_harness.redact import redact_obj
    for value in values.values():
        assert value not in redact(f"x {value} y")
        assert value not in str(redact_obj({"k": [f"a{value}"], value: 1}))
    assert "plain-value-not-secret" in redact("plain-value-not-secret")
