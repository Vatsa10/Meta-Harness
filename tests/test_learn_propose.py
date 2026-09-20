import pytest

from meta_harness.cc_history import FailureEpisode
from meta_harness.learn import FailureClass, build_proposal_prompt, parse_proposal, propose_artifact

REPLAY = {"instruction": "fix it", "files": {}, "expect": {"no_tool_error": {"tool": "Bash"}},
          "_origin": {"signature": "tool_error:Bash:unicode-decode", "session": "s1", "turn": 4}}
FAILURE = FailureClass(signature="tool_error:Bash:unicode-decode", kind="tool_error",
                       tool="Bash", count=7,
                       episodes=[FailureEpisode("s1", "p", "tool_error", 4,
                                                assistant_text="UnicodeDecodeError: charmap")])


def test_prompt_states_the_layer_ordering():
    prompt = build_proposal_prompt(FAILURE, REPLAY, installed=[])
    # The ordering is the design's central rule; a proposer that does not see it will
    # write prose for something a rule could enforce.
    assert "rule" in prompt and "injection" in prompt and "doctrine" in prompt
    assert "7" in prompt          # how often the failure happened
    assert "UnicodeDecodeError" in prompt


def test_prompt_lists_installed_artifacts_to_avoid_duplication():
    prompt = build_proposal_prompt(FAILURE, REPLAY, installed=["read-before-edit"])
    assert "read-before-edit" in prompt


def test_parse_proposal_reads_type_and_payload():
    text = 'TYPE: rule\nPAYLOAD:\n```\ndeny Edit when the file was not Read\n```'
    assert parse_proposal(text) == ("rule", "deny Edit when the file was not Read")


def test_parse_proposal_accepts_an_unfenced_payload():
    assert parse_proposal("TYPE: doctrine\nPAYLOAD:\nRead before you edit.") == (
        "doctrine", "Read before you edit.")


def test_parse_proposal_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown artifact type"):
        parse_proposal("TYPE: telepathy\nPAYLOAD:\nx")


def test_parse_proposal_rejects_a_missing_payload():
    with pytest.raises(ValueError, match="payload"):
        parse_proposal("TYPE: rule\n")


def test_parse_proposal_keeps_a_nested_fence_intact():
    text = ('TYPE: doctrine\nPAYLOAD:\n```\nExample:\n```\nsome code\n```\n'
            'Do it that way.\n```')
    artifact_type, payload = parse_proposal(text)
    assert artifact_type == "doctrine"
    assert payload == "Example:\n```\nsome code\n```\nDo it that way."


def test_parse_proposal_accepts_a_fence_with_a_language_tag():
    text = 'TYPE: rule\nPAYLOAD:\n```python\ndeny it\n```'
    assert parse_proposal(text) == ("rule", "deny it")


def test_parse_proposal_accepts_a_fence_without_a_language_tag():
    text = 'TYPE: rule\nPAYLOAD:\n```\ndeny it\n```'
    assert parse_proposal(text) == ("rule", "deny it")


def test_parse_proposal_accepts_an_unfenced_payload_again():
    assert parse_proposal("TYPE: doctrine\nPAYLOAD:\nRead before you edit.") == (
        "doctrine", "Read before you edit.")


def test_propose_artifact_ids_differ_for_different_failures():
    model = lambda prompt, **_: "TYPE: rule\nPAYLOAD:\n```\ndeny it\n```"
    other_failure = FailureClass(signature="tool_error:Bash:other-error", kind="tool_error",
                                 tool="Bash", count=1,
                                 episodes=[FailureEpisode("s1", "p", "tool_error", 4)])
    a = propose_artifact(FAILURE, REPLAY, installed=[], model=model)
    b = propose_artifact(other_failure, REPLAY, installed=[], model=model)
    assert a.id != b.id


def test_propose_artifact_ids_differ_for_sessions_sharing_an_8char_prefix():
    model = lambda prompt, **_: "TYPE: rule\nPAYLOAD:\n```\ndeny it\n```"
    replay_a = dict(REPLAY, _origin={"signature": FAILURE.signature,
                                     "session": "abcdefgh-one", "turn": 4})
    replay_b = dict(REPLAY, _origin={"signature": FAILURE.signature,
                                     "session": "abcdefgh-two", "turn": 4})
    a = propose_artifact(FAILURE, replay_a, installed=[], model=model)
    b = propose_artifact(FAILURE, replay_b, installed=[], model=model)
    assert a.id != b.id


def test_propose_artifact_id_has_no_illegal_windows_filename_characters():
    model = lambda prompt, **_: "TYPE: rule\nPAYLOAD:\n```\ndeny it\n```"
    artifact = propose_artifact(FAILURE, REPLAY, installed=[], model=model)
    assert not any(c in artifact.id for c in '\\/:*?"<>|')


def test_propose_artifact_carries_provenance():
    model = lambda prompt, **_: "TYPE: rule\nPAYLOAD:\n```\ndeny it\n```"
    artifact = propose_artifact(FAILURE, REPLAY, installed=[], model=model)
    assert artifact.type == "rule"
    assert artifact.payload == "deny it"
    assert artifact.origin["signature"] == "tool_error:Bash:unicode-decode"
    assert artifact.sources == ["s1#4"]
    assert artifact.replay == REPLAY
