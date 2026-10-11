"""Tests never see the developer's .env or real keys."""
import os

from meta_harness.__main__ import main


def test_cwd_dotenv_is_not_loaded(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FAKE_API_KEY=not-a-real-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    try:
        main(["models"])
    except BaseException:
        pass
    assert "FAKE_API_KEY" not in os.environ
