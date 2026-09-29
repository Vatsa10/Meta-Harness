from datetime import datetime, timedelta, timezone

from meta_harness.harness_store import Artifact
from meta_harness.temporal import retirement_candidates

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def artifact(artifact_id: str, signature: str, created: datetime) -> Artifact:
    return Artifact(id=artifact_id, type="rule", origin={"signature": signature},
                    payload="x", replay={}, scores={}, sources=[], created=created.isoformat())


def test_an_artifact_whose_failure_stopped_appearing_is_proposed():
    old = artifact("a1", "tool_error:Bash:timeout", NOW - timedelta(days=200))
    candidates = retirement_candidates([old], live_signatures=set(), now=NOW)
    assert [a.id for a, _ in candidates] == ["a1"]


def test_an_artifact_whose_failure_still_appears_is_kept():
    old = artifact("a1", "tool_error:Bash:timeout", NOW - timedelta(days=200))
    assert retirement_candidates([old], {"tool_error:Bash:timeout"}, now=NOW) == []


def test_a_young_artifact_is_never_proposed_however_quiet():
    young = artifact("a2", "tool_error:Bash:timeout", NOW - timedelta(days=5))
    assert retirement_candidates([young], live_signatures=set(), now=NOW) == []


def test_the_reason_names_why_so_a_human_can_judge_it():
    old = artifact("a1", "tool_error:Bash:timeout", NOW - timedelta(days=200))
    _, reason = retirement_candidates([old], set(), now=NOW)[0]
    assert "tool_error:Bash:timeout" in reason and "90" in reason


def test_an_artifact_with_an_unparseable_created_date_is_not_proposed():
    stray = Artifact(id="a3", type="rule", origin={"signature": "s"}, payload="x",
                     replay={}, scores={}, sources=[], created="not-a-date")
    assert retirement_candidates([stray], set(), now=NOW) == []


def test_a_no_effect_verdict_proposes_retirement_citing_receipts():
    young = artifact("a1", "tool_error:Bash:x", NOW - timedelta(days=5))
    got = retirement_candidates([young], {"tool_error:Bash:x"}, now=NOW,
                                verdicts={"a1": "no measurable effect"})
    assert [a.id for a, _ in got] == ["a1"]
    assert "receipts" in got[0][1]


def test_helps_and_not_enough_data_verdicts_do_not_propose_retirement():
    young = artifact("a1", "tool_error:Bash:x", NOW - timedelta(days=5))
    for verdict in ("helps", "not enough data", "no control arm"):
        assert retirement_candidates([young], {"tool_error:Bash:x"}, now=NOW,
                                     verdicts={"a1": verdict}) == []


def test_an_artifact_quiet_and_no_effect_is_proposed_once():
    old = artifact("a1", "tool_error:Bash:x", NOW - timedelta(days=200))
    got = retirement_candidates([old], set(), now=NOW, verdicts={"a1": "no measurable effect"})
    assert [a.id for a, _ in got] == ["a1"]
