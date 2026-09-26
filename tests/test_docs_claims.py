import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "skills" / "learning-from-failures" / "SKILL.md"]


def test_no_doc_attributes_the_layer_ordering_to_the_paper():
    # The ordering is this project's position. The paper does not contain it.
    pattern = re.compile(r"paper[^.\n]{0,80}(rule\s*>\s*injection|layer ordering)"
                         r"|(rule\s*>\s*injection)[^.\n]{0,80}paper", re.I)
    for doc in DOCS:
        assert not pattern.search(doc.read_text(encoding="utf-8")), doc


def test_the_papers_real_finding_is_stated_somewhere():
    joined = " ".join(d.read_text(encoding="utf-8") for d in DOCS)
    assert "50.0" in joined and "34.9" in joined      # the Table 3 ablation
    assert re.search(r"raw traces|execution traces", joined, re.I)


def test_the_measured_negative_result_is_recorded():
    joined = " ".join(d.read_text(encoding="utf-8") for d in DOCS)
    assert re.search(r"environment (snapshot|bootstrap)", joined, re.I)
    assert re.search(r"does not transfer|did not transfer", joined, re.I)


def test_plugin_version_bumped():
    import json
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.7.0"
