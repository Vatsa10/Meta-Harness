from meta_harness import judge


NAMES = ["alpha-harness", "beta-harness", "gamma-harness", "delta-harness"]
OUTPUTS = {n: f"output of {n.split('-')[0]}" for n in NAMES}
TASK = {"criteria": "be correct"}


def test_ids_are_opaque_unique_and_deterministic():
    ids = judge.assign_ids(NAMES, seed=7)
    assert len(set(ids.values())) == len(NAMES)
    assert all(i.startswith("cand-") and len(i) == 9 for i in ids.values())
    assert not any(n.split("-")[0] in i for n, i in ids.items())
    assert ids == judge.assign_ids(list(reversed(NAMES)), seed=7)
    assert ids != judge.assign_ids(NAMES, seed=8)


def test_prompt_order_is_shuffled_and_seeded():
    orders = set()
    for seed in range(12):
        prompt, mapping = judge.build_rank_prompt(TASK, OUTPUTS, seed)
        again, _ = judge.build_rank_prompt(TASK, OUTPUTS, seed)
        assert prompt == again
        by_pos = sorted(mapping, key=prompt.index)
        orders.add(tuple(mapping[i] for i in by_pos))
    assert len(orders) > 1  # not always the input order
    assert tuple(NAMES) not in orders or len(orders) > 2


def test_no_candidate_names_in_prompt_and_mapping_round_trips():
    prompt, mapping = judge.build_rank_prompt(TASK, OUTPUTS, seed=3)
    for name in NAMES:
        assert name not in prompt
    assert sorted(mapping.values()) == sorted(NAMES)
    for cid, name in mapping.items():
        assert f"[{cid}]\n{OUTPUTS[name]}" in prompt
        assert judge.parse_best(f"reason\nBEST: {cid}", mapping) == name
    assert judge.parse_best("BEST: cand-zzzz", mapping) is None
    assert judge.parse_best("no verdict", mapping) is None


def test_pick_best_end_to_end(monkeypatch):
    seen = []

    def model(prompt):
        seen.append(prompt)
        winner = prompt.split("[", 1)[1].split("]", 1)[0]  # first shown, whoever it is
        return f"BEST: {winner}"

    monkeypatch.setattr(judge, "judge_model", lambda: model)
    best = judge.pick_best(TASK, OUTPUTS, seed=5)
    prompt, mapping = judge.build_rank_prompt(TASK, OUTPUTS, 5)
    assert best == mapping[prompt.split("[", 1)[1].split("]", 1)[0]]
    assert seen and not any(n in seen[0] for n in NAMES)
    assert judge.pick_best(TASK, {"only": "x"}) == "only"
