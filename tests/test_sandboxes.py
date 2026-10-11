import sys
from types import SimpleNamespace

import pytest

from meta_harness.sandboxes import DockerSandbox, TempDirSandbox, preflight
from meta_harness.timeouts import Aborted, Watchdog


def _py(code):
    return f'"{sys.executable}" -c "{code}"'


def test_tempdir_put_exec_get(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("hello", encoding="utf-8")
    with TempDirSandbox() as sb:
        sb.put(src, "in.txt")
        res = sb.exec(_py("open('out.txt','w').write(open('in.txt').read().upper())"), 30)
        assert res.code == 0 and not res.timed_out, res.stderr
        sb.get("out.txt", tmp_path / "back" / "out.txt")
    assert (tmp_path / "back" / "out.txt").read_text(encoding="utf-8") == "HELLO"


def test_tempdir_timeout():
    with TempDirSandbox() as sb:
        res = sb.exec(_py("import time; time.sleep(30)"), 1)
    assert res.timed_out


def test_tempdir_rejects_escape(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with TempDirSandbox() as sb:
        with pytest.raises(ValueError):
            sb.put(f, "../x")
        with pytest.raises(ValueError):
            sb.get("../../etc", tmp_path / "o")


def test_tempdir_close_twice():
    sb = TempDirSandbox()
    sb.start()
    sb.close()
    sb.close()


class Fake:
    def __init__(self, fail_exec=False):
        self.calls = []
        self.fail_exec = fail_exec

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        assert kw["encoding"] == "utf-8" and kw["errors"] == "replace" and "timeout" in kw
        if self.fail_exec and argv[1] == "exec":
            raise OSError("boom")
        return SimpleNamespace(returncode=0, stdout="cid123\n", stderr="")


def test_docker_default_argv_network_none():
    fake = Fake()
    sb = DockerSandbox("alpine", memory_mb=256, cpus=1.5, runner=fake)
    sb.start()
    run = fake.calls[0]
    assert run[:3] == ["docker", "run", "-d"]
    assert "--network" in run and run[run.index("--network") + 1] == "none"
    assert run[run.index("--memory") + 1] == "256m"
    assert run[run.index("--cpus") + 1] == "1.5"
    assert run[-5:] == ["-w", "/app", "alpine", "sleep", "infinity"]
    assert sb.exec("echo hi", 5).stdout == "cid123\n"
    assert fake.calls[1] == ["docker", "exec", "-w", "/app", "cid123", "bash", "-lc", "echo hi"]


def test_docker_network_enabled_omits_flag():
    fake = Fake()
    DockerSandbox("alpine", network=True, runner=fake).start()
    assert "--network" not in fake.calls[0]


def test_docker_rm_after_exec_error():
    fake = Fake(fail_exec=True)
    sb = DockerSandbox("alpine", runner=fake)
    with pytest.raises(OSError):
        with sb:
            sb.exec("x", 5)
    assert fake.calls[-1] == ["docker", "rm", "-f", "cid123"]
    sb.close()
    assert sum(1 for c in fake.calls if c[1] == "rm") == 1


def test_docker_cp(tmp_path):
    fake = Fake()
    sb = DockerSandbox("alpine", runner=fake)
    sb.start()
    sb.put(tmp_path, "/app/t")
    assert fake.calls[-1][:2] == ["docker", "cp"] and fake.calls[-1][3] == "cid123:/app/t"
    sb.get("/app/o", tmp_path / "o")
    assert fake.calls[-1][2] == "cid123:/app/o"


def test_preflight_missing_binary():
    def r(*a, **k):
        raise FileNotFoundError("docker")
    p = preflight(r)
    assert not p.ok and "Docker Desktop" in p.hint and "docker.io" in p.hint


def test_preflight_daemon_down():
    p = preflight(lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="cannot connect"))
    assert not p.ok and "daemon" in p.hint.lower() and "start" in p.hint.lower()


def test_preflight_ok():
    assert preflight(lambda *a, **k: SimpleNamespace(returncode=0, stdout="27.0\n", stderr="")).ok


_pf = preflight()


@pytest.mark.skipif(not _pf.ok, reason=f"docker unavailable: {_pf.problem}")
def test_docker_live_echo():
    with DockerSandbox("alpine") as sb:
        res = sb.exec("echo ok", 60)
    assert res.stdout.strip() == "ok"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_watchdog_idle():
    c = Clock()
    w = Watchdog(idle=10, deadline=100, clock=c)
    c.t = 9
    w.check()
    w.kick()
    c.t = 18
    w.check()
    c.t = 19.5
    with pytest.raises(Aborted) as e:
        w.check()
    assert e.value.reason == "idle"


def test_watchdog_deadline_despite_kicks():
    c = Clock()
    w = Watchdog(idle=10, deadline=25, clock=c)
    for t in (8, 16, 24):
        c.t = t
        w.check()
        w.kick()
    assert w.remaining() == 1
    c.t = 26
    with pytest.raises(Aborted) as e:
        w.check()
    assert e.value.reason == "deadline"
