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


def test_no_doc_attributes_the_layer_ordering_to_the_paper():
    # The ordering is this project's position. The paper must not be credited with any claim,
    # in any phrasing, that rules and injections have a stated precedence over one another.
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for sentence in _SENTENCE_SPLIT.split(text):
            if re.search(r"\bpaper\b", sentence, re.I) and (
                _FORWARD.search(sentence) or _BACKWARD.search(sentence)
            ):
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
    assert manifest["version"] == "0.7.0"
