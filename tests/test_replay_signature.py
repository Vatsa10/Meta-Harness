from meta_harness.cc_history import FailureEpisode
from meta_harness.replay import episode_signature


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
