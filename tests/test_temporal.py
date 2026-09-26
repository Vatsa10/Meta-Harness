from datetime import datetime, timedelta, timezone

import pytest

from meta_harness.temporal import HALF_LIFE_DAYS, recency_weight, version_weight

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def test_recent_evidence_keeps_full_weight():
    assert recency_weight(NOW.isoformat(), now=NOW) == pytest.approx(1.0, abs=0.01)


def test_weight_halves_at_one_half_life():
    old = (NOW - timedelta(days=HALF_LIFE_DAYS)).isoformat()
    assert recency_weight(old, now=NOW) == pytest.approx(0.5, abs=0.02)


def test_ancient_evidence_floors_rather_than_reaching_zero():
    ancient = (NOW - timedelta(days=3650)).isoformat()
    assert recency_weight(ancient, now=NOW) == pytest.approx(0.05, abs=0.001)


@pytest.mark.parametrize("stamp", ["", "not-a-date", "2026-13-45T99:99:99"])
def test_an_unusable_timestamp_means_no_decay_not_no_weight(stamp):
    # A session with no usable clock must not have its evidence silently deleted.
    assert recency_weight(stamp, now=NOW) == 1.0


def test_a_future_timestamp_does_not_exceed_full_weight():
    ahead = (NOW + timedelta(days=100)).isoformat()
    assert recency_weight(ahead, now=NOW) <= 1.0


@pytest.mark.parametrize("seen,current,expected", [
    ("2.1.278", "2.1.278", 1.0),
    ("2.1.278", "2.2.001", 0.6),
    ("2.1.233", "2.4.000", 0.3),
    ("", "2.1.278", 1.0),
    ("2.1.278", "", 1.0),
])
def test_version_distance_discounts_stale_evidence(seen, current, expected):
    assert version_weight(seen, current) == pytest.approx(expected)


def test_merge_applies_weights_without_breaking_the_unweighted_call():
    from meta_harness.learn import FailureClass, merge_failures

    observed = [FailureClass(signature="tool_error:Bash:timeout", count=10, episodes=[])]
    mined = [FailureClass(signature="tool_error:Edit:unread-edit", count=6, episodes=[])]

    unweighted = merge_failures(observed, mined)
    assert unweighted[0].signature == "tool_error:Bash:timeout"

    weighted = merge_failures(observed, mined, weights={"tool_error:Bash:timeout": 0.1})
    assert weighted[0].signature == "tool_error:Edit:unread-edit"


def test_select_target_ranks_stale_signature_below_a_fresh_equal_count_one():
    from meta_harness.cc_history import Session, ToolCall, Turn
    import meta_harness.learn as learn_mod
    from meta_harness.harness_store import HarnessStore

    old_stamp = (NOW - timedelta(days=365)).isoformat()
    fresh_stamp = NOW.isoformat()

    old_turn = Turn(role="assistant", index=0,
                    tools=[ToolCall(name="Bash", is_error=True, result_excerpt="timed out")])
    old_session = Session(session_id="old", project="p", path="p", version="2.1.278",
                          turns=[old_turn], started=old_stamp, ended=old_stamp)

    fresh_turn = Turn(role="assistant", index=0,
                      tools=[ToolCall(name="Edit", is_error=True,
                                     result_excerpt="has not been read yet")])
    fresh_session = Session(session_id="fresh", project="p", path="p", version="2.1.278",
                            turns=[fresh_turn], started=fresh_stamp, ended=fresh_stamp)

    sessions = [old_session, fresh_session]
    weights = learn_mod._signature_weights(sessions)
    ranked = learn_mod.merge_failures([], learn_mod.rank_failures(sessions), weights=weights)
    signatures = [f.signature for f in ranked]
    assert signatures.index("tool_error:Edit:unread-edit") < signatures.index(
        "tool_error:Bash:timeout")
