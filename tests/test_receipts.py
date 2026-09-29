from meta_harness.receipts import attribute, summarize, MIN_PER_ARM


def r(session, call, decision, artifact="a1", source="learned-rule", sig="tool_error:Bash:x"):
    return {"session": session, "call": call, "decision": decision, "artifact": artifact,
            "source": source, "signature": sig}


def o(session, call, sig="tool_error:Bash:x"):
    return {"session": session, "call": call, "kind": "tool_error",
            "tool": sig.split(":")[1], "cause": sig.split(":")[2]}


def test_recurrence_is_within_k_calls_same_session_same_signature():
    out = attribute([r("s1", 5, "acted")], [o("s1", 9)])
    assert out[0]["recurred"] is True
    assert attribute([r("s1", 5, "acted")], [o("s1", 16)])[0]["recurred"] is False  # past K
    assert attribute([r("s1", 5, "acted")], [o("s2", 6)])[0]["recurred"] is False   # other session
    assert attribute([r("s1", 5, "acted")], [o("s1", 5)])[0]["recurred"] is False   # not after
    assert attribute([r("s1", 5, "acted")], [o("s1", 6, "tool_error:Bash:y")])[0]["recurred"] is False


def test_the_window_includes_call_plus_k():
    assert attribute([r("s1", 5, "acted")], [o("s1", 15)])[0]["recurred"] is True


def test_a_repeat_observation_matches_a_thrash_signature():
    obs = {"session": "s1", "call": 7, "kind": "repeat", "tool": "Bash"}
    row = r("s1", 5, "acted", sig="thrash:Bash")
    assert attribute([row], [obs])[0]["recurred"] is True


def test_an_observation_without_a_cause_is_classified_from_its_text_like_the_registry():
    obs = {"session": "s1", "call": 7, "kind": "tool_error", "tool": "Bash",
           "text": "command timed out"}
    from meta_harness.replay import _cause
    sig = f"tool_error:Bash:{_cause('command timed out')}"
    assert attribute([r("s1", 5, "acted", sig=sig)], [obs])[0]["recurred"] is True


def test_a_quiet_session_is_no_recurrence_not_an_error():
    assert attribute([r("s1", 1, "acted")], [])[0]["recurred"] is False


def test_too_few_receipts_is_not_enough_data():
    rows = [r("s1", i, "acted") for i in range(MIN_PER_ARM)] + [r("s2", 1, "held")]
    assert summarize(attribute(rows, []))[0]["verdict"] == "not enough data"


def test_a_clear_effect_helps():
    acted = [r(f"a{i}", 1, "acted") for i in range(20)]
    held = [r(f"h{i}", 1, "held") for i in range(20)]
    obs = [o(f"h{i}", 3) for i in range(16)]                      # held recurs 80%
    row = summarize(attribute(acted + held, obs))[0]
    assert row["verdict"] == "helps" and row["held_rate"] == 0.8 and row["acted_rate"] == 0.0


def test_equal_rates_is_no_measurable_effect():
    acted = [r(f"a{i}", 1, "acted") for i in range(20)]
    held = [r(f"h{i}", 1, "held") for i in range(20)]
    obs = [o(f"a{i}", 3) for i in range(10)] + [o(f"h{i}", 3) for i in range(10)]
    assert summarize(attribute(acted + held, obs))[0]["verdict"] == "no measurable effect"


def test_user_driven_sources_never_get_a_verdict():
    rows = [r(f"s{i}", 1, "acted", artifact=None, source="session-rule") for i in range(30)]
    assert summarize(attribute(rows, []))[0]["verdict"] == "no control arm"
