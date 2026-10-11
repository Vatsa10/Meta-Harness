import os
from pathlib import Path

import pytest

from meta_harness.chat_types import ToolCall
from meta_harness.proposer_tools import TOOL_SPECS, ToolRefused, ToolSandbox, dispatch


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "store"
    (root / "candidates" / "c1").mkdir(parents=True)
    (root / "candidates" / "c1" / "harness.py").write_text("class Harness:\n    pass\n# LABEL\n", encoding="utf-8")
    (root / "candidates" / "c1" / "scores.json").write_text('{"score": 0.5}', encoding="utf-8")
    test = tmp_path / "store-test"
    test.mkdir()
    (test / "test_results.json").write_text('{"secret": 1}', encoding="utf-8")
    out = root / "proposals" / "iteration-0001"
    out.mkdir(parents=True)
    return ToolSandbox(root, out, forbidden=[test, root / "proposals"]), root, test


@pytest.mark.parametrize("path", ["../store-test/test_results.json", "candidates/../../store-test",
                                  "/etc/passwd", "C:/Windows/win.ini", "c:\\x", "\\\\server\\share",
                                  "proposals/iteration-0001", "candidates/../../elsewhere.txt"])
def test_escape_attempts_are_refused(run, path):
    sandbox, _, _ = run
    with pytest.raises(ToolRefused):
        sandbox.resolve(path)


@pytest.mark.parametrize("path", ["C:/Windows/win.ini", "c:\\x", "D:relative"])
def test_drive_letter_paths_are_refused_as_absolute(run, path):
    sandbox, _, _ = run
    with pytest.raises(ToolRefused, match="absolute"):
        sandbox.resolve(path)


def test_dotdot_is_refused_before_resolution(run):
    sandbox, _, _ = run
    with pytest.raises(ToolRefused, match="'..'"):
        sandbox.resolve("candidates/../candidates/c1")


def test_absolute_path_of_test_root_is_refused(run):
    sandbox, _, test = run
    with pytest.raises(ToolRefused):
        sandbox.resolve(str(test / "test_results.json"))


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive paths are a Windows property")
def test_case_variant_of_forbidden_dir_is_refused(run):
    sandbox, _, _ = run
    with pytest.raises(ToolRefused):
        sandbox.resolve("PROPOSALS/Iteration-0001")


def test_symlink_out_of_root_is_refused(run, tmp_path):
    sandbox, root, test = run
    try:
        os.symlink(test, root / "candidates" / "peek", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(ToolRefused):
        sandbox.resolve("candidates/peek/test_results.json")
    assert "secret" not in sandbox.grep("secret", ".")


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_junction_out_of_root_is_refused(run):
    import subprocess
    sandbox, root, test = run
    link = root / "candidates" / "jn"
    proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(test)], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        pytest.skip("mklink /J unavailable")
    with pytest.raises(ToolRefused):
        sandbox.resolve("candidates/jn/test_results.json")
    assert "secret" not in sandbox.grep("secret", ".")
    # a junction to a directory that is not on the forbidden list is refused by the root check
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("leaked", encoding="utf-8")
    link2 = root / "candidates" / "jn2"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link2), str(outside)], capture_output=True,
                   text=True, encoding="utf-8", errors="replace", check=True)
    with pytest.raises(ToolRefused, match="outside"):
        sandbox.resolve("candidates/jn2/leak.txt")
    assert "leaked" not in sandbox.grep("leaked", ".")
    assert "jn2" not in sandbox.list("candidates")


def test_read_list_grep_inside_root(run):
    sandbox, _, _ = run
    assert "c1/" in sandbox.list("candidates")
    assert "1\tclass Harness:" in sandbox.read("candidates/c1/harness.py")
    assert "candidates/c1/harness.py:3:" in sandbox.grep("LABEL")
    assert "proposals" not in sandbox.list(".")
    assert "proposals" not in sandbox.grep("Harness")


def test_dispatch_turns_refusals_into_results(run):
    sandbox, _, _ = run
    text, written = dispatch(sandbox, ToolCall("1", "read", {"path": "../store-test/test_results.json"}), "cand")
    assert text.startswith("error:") and "secret" not in text and written is None
    text, _ = dispatch(sandbox, ToolCall("2", "read", {"_raw": "{bad"}), "cand")
    assert text.startswith("error:")
    text, _ = dispatch(sandbox, ToolCall("3", "nope", {}), "cand")
    assert text == "error: unknown tool nope"


def test_write_candidate_rejects_syntax_errors_and_writes_valid(run):
    sandbox, _, _ = run
    text, written = dispatch(sandbox, ToolCall("1", "write_candidate", {"source": "def (", "rationale": "r"}), "cand")
    assert text.startswith("error: syntax error") and written is None
    text, written = dispatch(sandbox, ToolCall("2", "write_candidate",
                                               {"source": "class Harness:\n    pass\n", "rationale": "why"}), "cand")
    assert written.name == "cand.py" and written.with_name("cand.reasoning.md").read_text(encoding="utf-8") == "why"


def test_tool_specs_have_required_fields():
    names = {s.name: s for s in TOOL_SPECS}
    assert set(names) == {"list", "read", "grep", "write_candidate"}
    assert names["read"].parameters["required"] == ["path"]
    assert set(names["write_candidate"].parameters["required"]) == {"source", "rationale"}


# --- output bounds and spill files ---

def _spill_ref(text):
    marker = "Full output saved to "
    assert marker in text
    return text.split(marker, 1)[1].split(" ", 1)[0]


def test_long_file_read_truncates_and_spills(run):
    sandbox, root, _ = run
    (root / "big.txt").write_text("\n".join(f"row {i}" for i in range(1, 5001)), encoding="utf-8")
    text = sandbox.read("big.txt", limit=10_000)
    assert len(text.splitlines()) <= 2002
    assert "[truncated: showing lines 1-2000 of 5000." in text
    spill = _spill_ref(text)
    assert spill.startswith("_spill/")
    full = (root / spill).read_text(encoding="utf-8")
    assert "row 1\n" in full and "row 5000" in full


def test_spill_is_readable_via_read(run):
    sandbox, root, _ = run
    (root / "big.txt").write_text("\n".join(f"row {i}" for i in range(1, 5001)), encoding="utf-8")
    spill = _spill_ref(sandbox.read("big.txt", limit=10_000))
    again = sandbox.read(spill, offset=4990, limit=5)
    assert "row 4991" in again


def test_single_huge_line_is_capped(run):
    sandbox, root, _ = run
    (root / "wide.txt").write_text("x" * 200_000, encoding="utf-8")
    text = sandbox.read("wide.txt")
    assert len(text) <= 50_000 and "...[line truncated]" in text
    (root / "wide.py").write_text("hit" + "y" * 200_000, encoding="utf-8")
    text = sandbox.grep("hit")
    assert len(text) <= 6000 * 4 + 400 and "[truncated:" in text


def test_non_ascii_truncation_round_trips(run):
    sandbox, root, _ = run
    (root / "accent.txt").write_text("\u00e9" * 50_000, encoding="utf-8")
    text = sandbox.read("accent.txt")
    spill = _spill_ref(text)
    assert (root / spill).read_text(encoding="utf-8").count("\u00e9") == 50_000


def test_read_window_reports_exact_total(run):
    sandbox, root, _ = run
    lines = [f"l{i}" for i in range(1, 3001)]
    lines[10] = "z" * 5000
    (root / "win.txt").write_text("\n".join(lines), encoding="utf-8")
    text = sandbox.read("win.txt", limit=5000)
    assert text.splitlines()[0] == "win.txt: lines 1-2000 of 3000"
    body = [l for l in text.splitlines()[1:] if "\t" in l]
    assert len(body) <= 2000
    assert "11\t" + "z" * 2000 + "...[line truncated]" in text
    assert len(text.encode("utf-8")) <= 50_000 + 400
