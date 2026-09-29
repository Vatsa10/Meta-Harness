import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "skills" / "learning-from-failures" / "SKILL.md"]


_RULE = r"rules?"
_INJECTION = r"injections?"
_PRECEDENCE = r"(?:>|beats?|outranks?|over|above|before|ahead of|stronger than)"
_FORWARD = re.compile(_RULE + r".{0,60}" + _PRECEDENCE + r".{0,60}" + _INJECTION, re.I | re.S)
_BACKWARD = re.compile(_INJECTION + r".{0,60}" + _PRECEDENCE + r".{0,60}" + _RULE, re.I | re.S)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NEGATION = re.compile(r"\b(not|never|doesn't|does not|nowhere|no such)\b", re.I)

# Known accepted limit: this is a regex heuristic, not a parser. A construction with no
# precedence word at all - e.g. "The paper ranks rules first and injections second." - carries
# the same misattribution but is not caught here. Widening further risks false positives on
# honest denials; that trade was made deliberately.


def test_no_doc_attributes_the_layer_ordering_to_the_paper():
    # The ordering is this project's position. The paper must not be credited with any claim,
    # in any phrasing, that rules and injections have a stated precedence over one another -
    # unless the sentence itself denies the paper makes that claim (negation exemption below).
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for sentence in _SENTENCE_SPLIT.split(text):
            if not re.search(r"\bpaper\b", sentence, re.I):
                continue
            match = _FORWARD.search(sentence) or _BACKWARD.search(sentence)
            if not match:
                continue
            if _NEGATION.search(sentence[: match.start()]):
                continue  # a negation governs the precedence claim - this is a denial, not one
            raise AssertionError(f"{doc}: {sentence!r}")


def test_the_papers_real_finding_is_stated_somewhere():
    # The three Table 3 medians must appear together, in one passage, framed as the ablation
    # they are - not scattered numbers that happen to occur somewhere in the docs.
    num_346 = re.compile(r"(?<!\d)34\.6(?!\d)")
    num_349 = re.compile(r"(?<!\d)34\.9(?!\d)")
    num_500 = re.compile(r"(?<!\d)50\.0(?!\d)")
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for paragraph in re.split(r"\n\s*\n", text):
            if (
                num_346.search(paragraph)
                and num_349.search(paragraph)
                and num_500.search(paragraph)
                and re.search(r"trace", paragraph, re.I)
                and re.search(r"summar(y|ies)", paragraph, re.I)
            ):
                return
    raise AssertionError("no single passage carries the Table 3 medians with traces/summaries")


def test_the_measured_negative_result_is_recorded():
    joined = " ".join(d.read_text(encoding="utf-8") for d in DOCS)
    assert re.search(r"environment (snapshot|bootstrap)", joined, re.I)
    assert re.search(r"does not transfer|did not transfer", joined, re.I)


def test_plugin_version_bumped():
    import json
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.8.0"


# --- final fix wave: user docs must describe the hooks that actually ship ---------------------

USER_DOCS = [ROOT / "README.md", ROOT / "hooks" / "README.md",
             ROOT / "skills" / "learning-from-failures" / "SKILL.md", ROOT / "docs" / "plugin.md"]
_DENIAL = re.compile(r"\b(not|never|nothing|no)\b", re.I)


def test_no_user_doc_presents_prompt_section_as_a_live_hook():
    for doc in USER_DOCS:
        text = doc.read_text(encoding="utf-8")
        for sentence in _SENTENCE_SPLIT.split(text):
            if "prompt.section" in sentence and not _DENIAL.search(sentence):
                raise AssertionError(f"{doc}: {sentence!r}")


def test_hook_source_registers_no_prompt_section_listener():
    for path in (ROOT / "hooks").glob("*.ts"):
        text = path.read_text(encoding="utf-8")
        assert "on('prompt.section'" not in text and "add('prompt.section'" not in text, path


def test_hooks_readme_does_not_claim_every_handler_uses_safely():
    text = (ROOT / "hooks" / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"every handler is wrapped in `safely`", text, re.I)
    for wrapper in ("`guardBefore`", "`guardAfter`", "`guardAfterMap`"):
        assert wrapper in text


def test_readme_says_drift_ships_disabled_and_never_claims_it_is_enabled():
    for doc in (ROOT / "README.md", ROOT / "hooks" / "README.md"):
        text = doc.read_text(encoding="utf-8")
        assert re.search(r"ship(s|ped) (DISABLED|disabled)", text), doc
        assert not re.search(r"drift (note|detection) is (on|enabled)", text, re.I), doc


def test_readme_states_no_stale_test_count():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"^\d+ tests\b", text, re.M)


def test_readme_documents_what_shipped():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ("meta_harness waste", "Rejection memory", "session rules", "first-run line",
                   "/meta-harness:harness waste | pending | why", "version_weight", "`judge` field"):
        assert needle in text, needle
    assert "11 of the 21" in text and "recall is unmeasured" in text


def test_no_user_doc_claims_the_plugin_enforces_read_before_edit():
    # The plugin's own read-before-edit rule (and the read-tracking that only fed it) is
    # retired: a live test proved Claude Code's own engine already denies an Edit/Write of an
    # unread file before the plugin ever sees the call. No user doc may claim the plugin (this
    # hooks module / `rule` artifacts) is what enforces that.
    for doc in USER_DOCS:
        text = doc.read_text(encoding="utf-8")
        for sentence in _SENTENCE_SPLIT.split(text):
            if not re.search(r"read.{0,15}before.{0,15}edit|read first|been read", sentence, re.I):
                continue
            if re.search(r"engine|Claude Code's own|not (this )?plugin", sentence, re.I):
                continue  # attributes it to the engine, or explicitly denies the plugin does it
            raise AssertionError(f"{doc}: {sentence!r}")


def test_hooks_readme_names_the_engine_as_the_enforcer():
    text = (ROOT / "hooks" / "README.md").read_text(encoding="utf-8")
    assert re.search(r"engine already denies", text, re.I)


def test_drift_and_waste_spec_carries_the_final_measured_baseline_and_outcome():
    spec = (ROOT / "docs" / "superpowers" / "specs" / "2026-09-26-drift-and-waste-design.md")
    text = spec.read_text(encoding="utf-8")
    assert "n = 98" not in text
    for needle in ("249", "31,166", "298 calls burned", "11 of the 21", "refused every judge",
                   "ToolCallResult.context", "not implemented"):
        assert needle in text, needle
    assert "kNN (default)" not in text


def test_no_user_facing_text_points_at_an_unqualified_harness_command():
    # Claude Code namespaces plugin commands: `/harness` is "Unknown command", the real name is
    # `/meta-harness:harness`. The first-run line shipped pointing at the wrong one.
    import re
    targets = [ROOT / "README.md", ROOT / "hooks" / "README.md", ROOT / "docs" / "plugin.md",
               ROOT / "commands" / "harness.md", ROOT / "hooks" / "harness.ts"]
    pattern = re.compile(r"(?<![\w:-])/harness\b(?!\.\w)")
    for target in targets:
        text = target.read_text(encoding="utf-8")
        hits = [m.group(0) for m in pattern.finditer(text)]
        assert not hits, f"{target.name} points at the unqualified /harness command"


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_documents_receipts_report_and_the_default_holdout():
    text = _readme()
    assert "receipts report" in text
    assert re.search(r"10\s?%", text)
    assert "receipts.json" in text
    assert re.search(r"(configurable|set it|set (?:the )?(?:rate )?)[^.]{0,80}\b(to )?0\b|0 (turns|switches|disables)", text, re.I)


def test_readme_says_session_rules_and_rejection_memory_are_never_held_out():
    text = _readme()
    assert re.search(r"never held out", text, re.I)
    for paragraph in re.split(r"\n\s*\n", text):
        if re.search(r"never held out", paragraph, re.I):
            assert re.search(r"session rules?", paragraph, re.I)
            assert re.search(r"rejection memory", paragraph, re.I)
            return
    raise AssertionError("no paragraph names both")


def test_readme_explains_verdicts_and_that_evidence_takes_weeks():
    text = _readme()
    for verdict in ("helps", "no measurable effect", "not enough data", "no control arm"):
        assert verdict in text
    assert re.search(r"weeks", text, re.I)


def test_why_command_notes_a_held_out_rule_does_not_deny():
    text = (ROOT / "commands" / "harness.md").read_text(encoding="utf-8")
    assert re.search(r"held[- ]out", text, re.I)
    assert re.search(r"does not deny|did not deny|not denied|will not deny", text, re.I)


def test_plugin_version_is_0_8_0():
    import json
    data = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert data["version"] == "0.8.0"
