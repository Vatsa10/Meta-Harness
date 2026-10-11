"""Shared fixtures. Tests never reach the network: a guard fails any real socket connect."""
import os
import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tests must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_BASE_URL")
_SECRET_PREFIXES = ("META_HARNESS_JUDGE_", "META_HARNESS_EMBED_")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path_factory):
    """No real keys in the environment, and the CLI's default env file is a path that
    does not exist, so a developer's ./.env is never loaded."""
    for name in list(os.environ):
        if (name.endswith(_SECRET_SUFFIXES) or name.startswith(_SECRET_PREFIXES)
                or name == "META_HARNESS_PRICES"):
            monkeypatch.delenv(name, raising=False)
    missing = tmp_path_factory.getbasetemp() / "no-such-dir" / ".env"
    monkeypatch.setenv("META_HARNESS_ENV_FILE", str(missing))


@pytest.fixture(autouse=True)
def fast_isolation(request, monkeypatch):
    """CandidateEvaluator defaults to in-process in tests (speed). test_isolation.py keeps the
    production default (a limited child process); the search CLI passes --isolation explicitly
    (default process), so CLI tests stay end-to-end through the child."""
    if request.node.module.__name__.endswith("test_isolation"):
        return
    from meta_harness import core
    monkeypatch.setattr(core, "DEFAULT_ISOLATION", "none")
