import pytest

from meta_harness.cc_history import FailureEpisode
from meta_harness.replay import _cause, episode_signature


def _episode(kind="tool_error", tools=("Bash",), detail="", assistant_text=""):
    return FailureEpisode(session_id="s1", project="p", kind=kind, turn_index=3,
                          detail=detail, tools=list(tools), assistant_text=assistant_text)


def test_tool_error_signature_names_tool_and_cause():
    episode = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d")
    assert episode_signature(episode) == "tool_error:Bash:unicode-decode"


def test_unmatched_error_falls_back_to_a_generic_slug():
    assert episode_signature(_episode(assistant_text="something odd")) == "tool_error:Bash:other"


def test_thrash_signature_names_the_tool():
    assert episode_signature(_episode(kind="thrash", tools=("Edit",))) == "thrash:Edit"


def test_signature_is_stable_across_wording():
    a = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d")
    b = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x81")
    # Two instances of the same fault must share a signature, or dedupe never fires.
    assert episode_signature(a) == episode_signature(b)


def test_correction_signature_is_session_scoped():
    episode = _episode(kind="correction", tools=("Edit", "Write"))
    assert episode_signature(episode) == "correction:Edit+Write"


def test_missing_tools_still_yields_a_signature():
    assert episode_signature(_episode(tools=())) == "tool_error:unknown:other"


def test_charmap_encode_error_is_not_swallowed_by_decode_pattern():
    episode = _episode(assistant_text="UnicodeEncodeError: 'charmap' codec can't encode character '✓'")
    assert episode_signature(episode) == "tool_error:Bash:unicode-encode"


def test_charmap_decode_error_still_classifies_as_decode():
    episode = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d")
    assert episode_signature(episode) == "tool_error:Bash:unicode-decode"


def test_bare_cp1252_mention_still_classifies_as_decode():
    episode = _episode(assistant_text="failed with cp1252")
    assert episode_signature(episode) == "tool_error:Bash:unicode-decode"


@pytest.mark.parametrize("text,expected", [
    ("This command requires approval", "needs-approval"),
    ("Permission for this action was denied by the Claude Code auto mode classifier", "needs-approval"),
    ("The user doesn't want to proceed with this tool use. The tool use was rejected", "user-rejected"),
    ("<tool_use_error>Blocked: sleep 45 followed by: echo waited", "blocked-policy"),
    ("bash: -c: line 1: unexpected EOF while looking for matching `'", "shell-quoting"),
    ("Contains simple_expansion", "shell-quoting"),
    ("Compound command changes working directory (Set-Location)", "compound-shell"),
    ("This PowerShell command contains multiple operations. The following part requires approval", "compound-shell"),
    ("Tab 3 is not in Claude's tab group for this session", "tab-target"),
    ("Couldn't determine which page this action targets", "tab-target"),
    ("File has been modified since read, either by the user or by a linter", "stale-read"),
    ("EISDIR: illegal operation on a directory, read", "is-directory"),
])
def test_dominant_failure_classes_get_their_own_cause(text, expected):
    assert _cause(text) == expected


def test_existing_causes_are_unchanged_by_the_new_patterns():
    # The compound-shell pattern mentions approval; it must not steal plain permission errors,
    # and the encode/decode ordering stays load-bearing.
    assert _cause("Permission denied") == "permission"
    assert _cause("UnicodeEncodeError: 'charmap' codec") == "unicode-encode"
    assert _cause("No such file or directory") == "missing-path"


def test_cause_survives_non_ascii_error_text():
    assert _cause("bash: ─────: command not found \U0001f600") == "missing-command"
