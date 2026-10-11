import os
import subprocess
import sys
import time
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


def test_tempdir_timeout_is_bounded_and_reason_deadline():
    with TempDirSandbox() as sb:
        t0 = time.monotonic()
        res = sb.exec(_py("import time; time.sleep(30)"), 1)
        assert time.monotonic() - t0 < 15
    assert res.timed_out and res.timeout_reason == "deadline"


def _alive(pid):
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace").stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_timeout_kills_grandchild(tmp_path):
    script = tmp_path / "spawn.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open('pid.txt', 'w').write(str(p.pid))\n"
        "time.sleep(60)\n", encoding="utf-8")
    with TempDirSandbox() as sb:
        sb.put(script, "spawn.py")
        res = sb.exec(f'"{sys.executable}" spawn.py', 2)
        assert res.timed_out
        pid = int((sb.root / "pid.txt").read_text(encoding="utf-8"))
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.1)
        assert not _alive(pid)


def test_idle_timeout_fires_when_silent():
    code = "import time; print('x', flush=True); time.sleep(30)"
    with TempDirSandbox() as sb:
        res = sb.exec(_py(code), 30, idle=1.5)
    assert res.timed_out and res.timeout_reason == "idle"
    assert res.stdout.startswith("x")


def test_deadline_fires_despite_output():
    code = "import time\nfor i in range(300):\n print(i, flush=True); time.sleep(0.1)"
    with TempDirSandbox() as sb:
        (sb.root / "chatty.py").write_text(code, encoding="utf-8")
        res = sb.exec(f'"{sys.executable}" chatty.py', 2, idle=1.5)
    assert res.timed_out and res.timeout_reason == "deadline"


def test_output_is_capped_keeping_tail():
    code = "import sys; sys.stdout.write('a' * 3000000 + 'END')"
    with TempDirSandbox() as sb:
        res = sb.exec(_py(code), 60)
    assert res.truncated and res.stdout.endswith("END") and "truncated" in res.stdout
    assert len(res.stdout) < 1100000


def test_stdin_is_closed():
    with TempDirSandbox() as sb:
        res = sb.exec(_py("import sys; print(repr(sys.stdin.read()))"), 20)
    assert res.code == 0 and "''" in res.stdout


def test_tempdir_rejects_escape(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with TempDirSandbox() as sb:
        with pytest.raises(ValueError):
            sb.put(f, "../x")
        with pytest.raises(ValueError):
            sb.get("../../etc", tmp_path / "o")


@pytest.mark.parametrize("bad", ["/abs/x", "C:foo", "C:\\foo", "\\\\server\\share\\x",
                                 "\\rooted", "sub/../../up"])
def test_tempdir_rejects_odd_paths(tmp_path, bad):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with TempDirSandbox() as sb:
        with pytest.raises(ValueError):
            sb.put(f, bad)
        with pytest.raises(ValueError):
            sb.get(bad, tmp_path / "o")


def _symlink_or_skip(link, target, is_dir=False):
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation not permitted here: {exc}")


def test_get_rejects_symlink_to_outside(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("s", encoding="utf-8")
    with TempDirSandbox() as sb:
        (sb.root / "d").mkdir()
        _symlink_or_skip(sb.root / "d" / "link", secret)
        with pytest.raises(ValueError):
            sb.get("d", tmp_path / "out")
        assert not (tmp_path / "out" / "link").exists()


def test_get_rejects_file_symlink_to_outside(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("s", encoding="utf-8")
    with TempDirSandbox() as sb:
        _symlink_or_skip(sb.root / "link", secret)
        with pytest.raises(ValueError):
            sb.get("link", tmp_path / "out.txt")


def test_get_allows_internal_symlink(tmp_path):
    with TempDirSandbox() as sb:
        (sb.root / "d").mkdir()
        (sb.root / "d" / "real.txt").write_text("r", encoding="utf-8")
        _symlink_or_skip(sb.root / "d" / "link", sb.root / "d" / "real.txt")
        sb.get("d", tmp_path / "out")
    assert (tmp_path / "out" / "real.txt").read_text(encoding="utf-8") == "r"


def test_tempdir_close_twice():
    sb = TempDirSandbox()
    sb.start()
    sb.close()
    sb.close()


class Fake:
    def __init__(self, fail_exec=False, fail_run=False, exec_timeout=False, exec_code=0):
        self.calls = []
        self.fail_exec = fail_exec
        self.fail_run = fail_run
        self.exec_timeout = exec_timeout
        self.exec_code = exec_code

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        assert kw["encoding"] == "utf-8" and kw["errors"] == "replace" and "timeout" in kw
        verb = argv[1]
        if verb == "exec":
            if self.fail_exec:
                raise OSError("boom")
            if self.exec_timeout:
                raise subprocess.TimeoutExpired(argv, kw["timeout"])
            return SimpleNamespace(returncode=self.exec_code, stdout="out", stderr="")
        if verb == "run":
            if self.fail_run == "timeout":
                raise subprocess.TimeoutExpired(argv, kw["timeout"])
            if self.fail_run:
                return SimpleNamespace(returncode=1, stdout="", stderr="no image")
        return SimpleNamespace(returncode=0, stdout="cid123\n", stderr="")


def _name(fake):
    run = fake.calls[0]
    return run[run.index("--name") + 1]


def test_docker_default_argv_network_none_and_hardening():
    fake = Fake()
    sb = DockerSandbox("alpine", runner=fake)
    sb.start()
    run = fake.calls[0]
    assert run[:3] == ["docker", "run", "-d"]
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--memory") + 1] == "2048m"
    assert run[run.index("--cpus") + 1] == "2.0"
    assert run[run.index("--pids-limit") + 1] == "512"
    assert run[run.index("--cap-drop") + 1] == "ALL"
    assert run[run.index("--security-opt") + 1] == "no-new-privileges"
    assert _name(fake).startswith("meta-harness-")
    assert run[-5:] == ["-w", "/app", "alpine", "sleep", "infinity"]


def test_docker_explicit_limits():
    fake = Fake()
    DockerSandbox("alpine", memory_mb=256, cpus=1.5, runner=fake).start()
    run = fake.calls[0]
    assert run[run.index("--memory") + 1] == "256m"
    assert run[run.index("--cpus") + 1] == "1.5"


def test_docker_exec_uses_sh_and_in_container_kill_timeout():
    fake = Fake()
    sb = DockerSandbox("alpine", runner=fake)
    sb.start()
    res = sb.exec("echo hi", 5)
    argv = fake.calls[1]
    assert argv == ["docker", "exec", "-w", "/app", _name(fake), "timeout", "-s", "KILL", "5",
                    "sh", "-c", "echo hi"]
    assert "bash" not in argv and res.stdout == "out" and not res.timed_out


def test_docker_exec_kill_exit_code_is_timeout():
    fake = Fake(exec_code=137)
    sb = DockerSandbox("alpine", runner=fake)
    sb.start()
    res = sb.exec("sleep 99", 2)
    assert res.timed_out and res.timeout_reason == "deadline"


def test_docker_cli_hang_kills_container():
    fake = Fake(exec_timeout=True)
    sb = DockerSandbox("alpine", runner=fake)
    sb.start()
    res = sb.exec("sleep 99", 2)
    assert res.timed_out
    assert ["docker", "kill", _name(fake)] in fake.calls


def test_docker_network_enabled_omits_flag():
    fake = Fake()
    DockerSandbox("alpine", network=True, runner=fake).start()
    assert "--network" not in fake.calls[0]


@pytest.mark.parametrize("mode", [True, "timeout"])
def test_docker_start_failure_removes_named_container(mode):
    fake = Fake(fail_run=mode)
    sb = DockerSandbox("alpine", runner=fake)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        sb.start()
    assert fake.calls[-1] == ["docker", "rm", "-f", _name(fake)]


def test_docker_rm_by_name_after_exec_error():
    fake = Fake(fail_exec=True)
    sb = DockerSandbox("alpine", runner=fake)
    with pytest.raises(OSError):
        with sb:
            sb.exec("x", 5)
    assert fake.calls[-1] == ["docker", "rm", "-f", _name(fake)]
    sb.close()
    assert sum(1 for c in fake.calls if c[1] == "rm") == 1


def test_docker_cp(tmp_path):
    fake = Fake()
    sb = DockerSandbox("alpine", runner=fake)
    sb.start()
    sb.put(tmp_path, "/app/t")
    assert fake.calls[-1][:2] == ["docker", "cp"] and fake.calls[-1][3] == f"{_name(fake)}:/app/t"
    sb.get("/app/o", tmp_path / "o")
    assert fake.calls[-1][2] == f"{_name(fake)}:/app/o"


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
