from meta_harness import metrics


def test_estimate_tokens_is_roughly_chars_over_four():
    assert metrics.estimate_tokens("") == 0
    assert metrics.estimate_tokens("abcd") == 1
    assert metrics.estimate_tokens("a" * 401) == 101


def test_normalize_answer_extracts_boxed():
    assert metrics.normalize_answer(r"So the answer is \boxed{42}.") == "42"
    assert metrics.normalize_answer(r"\boxed{\frac{1}{2}}") == "1/2"


def test_normalize_answer_strips_latex_noise():
    assert metrics.normalize_answer(r"$\left( 3 \right)$") == "(3)"
    assert metrics.normalize_answer("Answer: 1,024") == "1024"
    assert metrics.normalize_answer(r"\text{yes}") == "yes"


def test_exact_match_is_case_insensitive():
    assert metrics.exact_match("Fruit", {"label": "fruit"}) == 1.0
    assert metrics.exact_match("vehicle", {"label": "fruit"}) == 0.0


def test_math_equivalence_handles_numeric_forms():
    assert metrics.math_equivalence(r"\boxed{0.5}", {"answer": "1/2"}) == 1.0
    assert metrics.math_equivalence(r"the answer is \boxed{42}", {"answer": 42}) == 1.0
    assert metrics.math_equivalence(r"\boxed{43}", {"answer": 42}) == 0.0


def test_math_equivalence_falls_back_to_string_compare():
    assert metrics.math_equivalence(r"\boxed{x+1}", {"answer": "x + 1"}) == 1.0


def test_math_equivalence_without_answer_is_zero():
    assert metrics.math_equivalence("anything", {"problem": "p"}) == 0.0


def test_metrics_registry_exposes_task_types():
    assert set(metrics.METRICS) >= {"classification", "math", "terminal"}
