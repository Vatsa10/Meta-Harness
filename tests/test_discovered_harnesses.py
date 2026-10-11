import importlib.util
from pathlib import Path

import pytest

from meta_harness.protocol import call_learn, call_run
from meta_harness.sandbox import validate_in_subprocess

ROOT = Path(__file__).resolve().parent.parent / "baselines" / "classification" / "discovered"
NAMES = ["label_primed_query", "draft_verification", "error_annotated", "cot_replay",
         "cluster_coverage", "cascade", "rrf_contrastive", "relevance_contrastive"]
LABELS = ["fruit", "vehicle", "animal"]
ROWS = [
    ("apple pie dessert", "fruit"), ("ripe banana peel", "fruit"), ("sweet orange juice", "fruit"),
    ("fast red car", "vehicle"), ("heavy truck engine", "vehicle"), ("bicycle with two wheels", "vehicle"),
    ("barking dog outside", "animal"), ("sleepy cat on sofa", "animal"), ("horse in the field", "animal"),
    ("pear and apple tart", "fruit"), ("racing car wheels", "vehicle"), ("small dog puppy", "animal"),
]


def _load(name):
    spec = importlib.util.spec_from_file_location("discovered_" + name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Harness()


class Model:
    """Scripted stand-in: answers a fixed label (with a high confidence line) and records prompts."""

    def __init__(self, answer="fruit", confidence="0.2"):
        self.answer, self.confidence, self.prompts = answer, confidence, []

    def __call__(self, prompt, **_):
        self.prompts.append(prompt)
        if "Confidence:" in prompt and "Reply on two lines" in prompt:
            return f"Label: {self.answer}\nConfidence: {self.confidence}"
        if prompt.endswith("Reason:"):
            return "because of the key words"
        return self.answer


def _train(harness, model):
    for text, label in ROWS:
        task = {"input": text, "label": label, "labels": LABELS}
        call_learn(harness, task, {"output": label}, model)


@pytest.mark.parametrize("name", NAMES)
def test_passes_interface_validation(name):
    validate_in_subprocess(ROOT / f"{name}.py", timeout=60.0,
                           probe={"input": "a red apple", "labels": LABELS})


@pytest.mark.parametrize("name", NAMES)
def test_run_returns_a_declared_label_after_learning(name):
    harness, model = _load(name), Model("vehicle")
    _train(harness, model)
    result = call_run(harness, {"input": "fast truck", "labels": LABELS}, model, None)
    assert result["output"] in LABELS
    assert len(harness.memory) == len(ROWS)


@pytest.mark.parametrize("name", NAMES)
def test_cold_start_runs_without_memory(name):
    harness, model = _load(name), Model("fruit")
    result = call_run(harness, {"input": "an apple", "labels": LABELS}, model, None)
    assert result["output"] == "fruit"
    assert result["examples"] == 0


def test_cot_replay_spends_one_extra_call_in_learn_and_replays_it():
    harness, model = _load("cot_replay"), Model("fruit")
    call_learn(harness, {"input": "apple pie", "label": "fruit"}, {"output": "fruit"}, model)
    assert len(model.prompts) == 1
    result = call_run(harness, {"input": "apple tart", "labels": ["fruit"]}, model, None)
    assert "Reasoning: because of the key words" in model.prompts[-1]
    assert result["calls"] == 1


def test_error_annotated_marks_earlier_mistakes():
    harness, model = _load("error_annotated"), Model("fruit")
    call_learn(harness, {"input": "apple pie", "label": "fruit"}, {"output": "vehicle"}, model)
    call_learn(harness, {"input": "fast car", "label": "vehicle"}, {"output": "vehicle"}, model)
    call_run(harness, {"input": "apple pie crust", "labels": ["fruit", "vehicle"]}, model, None)
    prompt = model.prompts[-1]
    assert "commonly confused with vehicle" in prompt
    assert prompt.count("commonly confused with") == 1


def test_cascade_zero_shot_uses_no_examples_and_escalates_only_when_unsure():
    harness = _load("cascade")
    _train(harness, Model())
    sure = Model("fruit", confidence="0.95")
    result = call_run(harness, {"input": "apple", "labels": LABELS}, sure, None)
    assert (result["calls"], result["examples"]) == (1, 0)
    assert "Example:" not in sure.prompts[0]
    unsure = Model("fruit", confidence="0.3")
    result = call_run(harness, {"input": "apple", "labels": LABELS}, unsure, None)
    assert result["calls"] == 2 and result["examples"] > 0
    assert "Example:" not in unsure.prompts[0] and "Example:" in unsure.prompts[1]
    invalid = Model("not-a-label", confidence="0.99")
    assert call_run(harness, {"input": "apple", "labels": LABELS}, invalid, None)["calls"] == 2


def test_draft_verification_makes_one_call_cold_and_two_warm():
    harness, model = _load("draft_verification"), Model("fruit")
    for text, label in ROWS[:4]:
        call_learn(harness, {"input": text, "label": label}, None, model)
    assert call_run(harness, {"input": "apple", "labels": LABELS}, model, None)["calls"] == 1
    assert len(model.prompts) == 1
    call_learn(harness, {"input": ROWS[4][0], "label": ROWS[4][1]}, None, model)
    model.prompts.clear()
    result = call_run(harness, {"input": "apple", "labels": LABELS}, model, None)
    assert result["calls"] == 2 and len(model.prompts) == 2
    assert "Initial label: fruit" in model.prompts[1]


def _coverage_block(prompt):
    return prompt.split("Best example per label:\n", 1)[1].split("\n\nContrastive pairs:", 1)[0]


def test_label_primed_query_covers_every_label_even_outside_the_tfidf_top_k():
    harness, model = _load("label_primed_query"), Model("fruit")
    # Twenty near-identical fruit rows crowd any small top-k; vehicle and animal rows are far from the query.
    for index in range(20):
        call_learn(harness, {"input": f"apple pie fruit number {index}", "label": "fruit"}, None, model)
    call_learn(harness, {"input": "diesel truck engine", "label": "vehicle"}, None, model)
    call_learn(harness, {"input": "barking dog", "label": "animal"}, None, model)
    call_run(harness, {"input": "apple pie fruit", "labels": LABELS}, model, None)
    prompt = model.prompts[-1]
    assert prompt.index("Valid labels:") < prompt.index("Best example per label:")
    coverage = _coverage_block(prompt)
    for label in LABELS:
        assert f"Label: {label}" in coverage
    assert "diesel truck engine" in coverage and "barking dog" in coverage


def test_label_primed_query_pairs_are_query_local_and_bounded():
    harness, model = _load("label_primed_query"), Model("fruit")
    _train(harness, model)
    result = call_run(harness, {"input": "apple pie", "labels": LABELS}, model, None)
    pairs = model.prompts[-1].split("Contrastive pairs:\n", 1)[1].split("\n\nInput:", 1)[0]
    assert pairs.count("--") <= harness.pair_count
    assert result["examples"] <= len(LABELS) + 2 * harness.pair_count


def test_library_label_primed_harness_covers_all_labels_and_learns_in_learn(tmp_path):
    from meta_harness.cls_harnesses import LabelPrimedQueryHarness
    from meta_harness.core import TraceRecorder

    harness = LabelPrimedQueryHarness(LABELS)
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return "fruit"

    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        for index in range(20):
            harness.learn({"input": f"apple pie fruit number {index}", "label": "fruit"})
        harness.learn({"input": "diesel truck engine", "label": "vehicle"})
        harness.learn({"input": "barking dog", "label": "animal"})
        size = len(harness.memory)
        harness.run({"input": "apple pie fruit", "label": "fruit"}, model, trace)
    assert len(harness.memory) == size
    assert "diesel truck engine" in prompts[0] and "barking dog" in prompts[0]


# ---- run() never learns; context growth ------------------------------------------------------------

def _snapshot(harness):
    import copy
    return copy.deepcopy(vars(harness))


@pytest.mark.parametrize("name", NAMES)
def test_run_never_changes_memory(name):
    harness, model = _load(name), Model("vehicle")
    _train(harness, model)
    before = _snapshot(harness)
    for query in ("fast truck", "apple tart", "a dog"):
        call_run(harness, {"input": query, "label": "fruit", "labels": LABELS}, model, None)
    assert _snapshot(harness) == before


# generous per-seed ceilings on shown examples, independent of how large memory gets
CEILING = {"label_primed_query": 3 + 2 * 4, "draft_verification": 5 + 5 + 5 + 5, "error_annotated": 8,
           "cot_replay": 6, "cluster_coverage": 5 + 4, "cascade": 8, "rrf_contrastive": 6 + 2 * 3,
           "relevance_contrastive": 6 + 3}


@pytest.mark.parametrize("name", NAMES)
def test_context_grows_with_memory_only_up_to_the_designed_bound(name):
    def shown(rows):
        harness, model = _load(name), Model("fruit", confidence="0.1")
        for index in range(rows):
            text, label = ROWS[index % len(ROWS)]
            call_learn(harness, {"input": f"{text} {index}", "label": label}, {"output": label}, model)
        return call_run(harness, {"input": "apple pie", "labels": LABELS}, model, None)["examples"]

    empty, small, large = shown(0), shown(6), shown(60)
    assert empty == 0
    assert 0 < small <= large <= CEILING[name]


def test_cascade_confident_pass_never_grows_with_memory():
    harness = _load("cascade")
    _train(harness, Model())
    for _ in range(3):
        call_learn(harness, {"input": "more rows", "label": "fruit"}, None, Model())
    assert call_run(harness, {"input": "apple", "labels": LABELS}, Model("fruit", "0.99"), None)["examples"] == 0


# ---- design tests ------------------------------------------------------------------------------------

def test_cluster_coverage_includes_a_representative_of_every_cluster():
    harness, model = _load("cluster_coverage"), Model("fruit")
    harness.k, harness.clusters = 1, 3
    for text, label in [("apple pie dessert", "fruit"), ("pear apple tart", "fruit"), ("fast red car", "vehicle"),
                        ("racing car wheels", "vehicle"), ("barking dog outside", "animal"),
                        ("small dog puppy", "animal")]:
        call_learn(harness, {"input": text, "label": label}, None, model)
    call_run(harness, {"input": "apple pie", "labels": LABELS}, model, None)
    prompt = model.prompts[-1]
    # k=1 nearest is a fruit; vehicle and animal appear only through their cluster representatives.
    assert "Label: vehicle" in prompt and "Label: animal" in prompt
    assert prompt.count("Example:") == 3


def test_rrf_contrastive_fuses_both_rankers():
    from meta_harness.retrieval import BM25Index, TfidfIndex, reciprocal_rank_fusion
    rows = [("cheap fast car deal", "a"), ("car", "b"), ("fast fast fast fast engine noise", "c"),
            ("car engine", "d"), ("long text about a car and many other unrelated words in here today", "e"),
            ("fast", "f"), ("zebra", "g")]
    query = "fast car engine"
    memory = [{"text": t, "label": l} for t, l in rows]
    texts = [t for t, _ in rows]
    dense = [r.index for r in TfidfIndex(memory, texts).search(query, 4)]
    sparse = [r.index for r in BM25Index(memory, texts, math_mode=False).search(query, 4)]
    fused = [r.index for r in reciprocal_rank_fusion([TfidfIndex(memory, texts).search(query, 7),
                                                      BM25Index(memory, texts, math_mode=False).search(query, 7)])][:4]
    assert fused != dense and fused != sparse  # fixture is only meaningful if the fusion changes the order
    harness, model = _load("rrf_contrastive"), Model("a")
    harness.k = 4
    for text, label in rows:
        call_learn(harness, {"input": text, "label": label}, None, model)
    call_run(harness, {"input": query, "labels": [l for _, l in rows]}, model, None)
    block = model.prompts[-1].split("Top fused examples:\n", 1)[1].split("\n\nContrastive pairs:", 1)[0]
    order = [line[len("Example: "):] for line in block.splitlines() if line.startswith("Example: ")]
    assert order == [texts[i] for i in fused]


def test_relevance_contrastive_shows_the_top_three_confusable_labels():
    from meta_harness.retrieval import TfidfIndex
    rows = [("red apple pie", "a"), ("red apple", "b"), ("apple tart", "c"), ("red pie", "d"), ("red", "e"),
            ("zebra stripes", "f")]
    query = "red apple pie"
    harness, model = _load("relevance_contrastive"), Model("a")
    harness.k = 1
    for text, label in rows:
        call_learn(harness, {"input": text, "label": label}, None, model)
    labels = [l for _, l in rows]
    call_run(harness, {"input": query, "labels": labels}, model, None)
    block = model.prompts[-1].split("Nearest example of each confusable label:\n", 1)[1].split("\n\nInput:", 1)[0]
    shown = [line[len("Label: "):] for line in block.splitlines() if line.startswith("Label: ")]
    memory = [{"text": t, "label": l} for t, l in rows]
    ranked = [r.item["label"] for r in TfidfIndex(memory, [t for t, _ in rows]).search(query, 6)]
    expected = [l for l in dict.fromkeys(ranked) if l != ranked[0]][:3]
    assert shown == expected and len(shown) == 3
    assert "f" not in shown


@pytest.mark.parametrize("name", ["label_primed_query", "rrf_contrastive", "relevance_contrastive"])
def test_no_example_is_shown_twice(name):
    harness, model = _load(name), Model("fruit")
    _train(harness, model)
    call_run(harness, {"input": "apple pie tart", "labels": LABELS}, model, None)
    texts = [line for line in model.prompts[-1].splitlines() if line.startswith("Example:")]
    assert len(texts) == len(set(texts))
