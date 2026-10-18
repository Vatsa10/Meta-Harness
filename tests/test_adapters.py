import json
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness.adapters import ConfigHarnessError, SchemaError, validate_config
from meta_harness.adapters.config_harness import (init_seed, parse_harness, render_harness,
                                                  split_harness, sync_seed, validate_candidate)
from meta_harness.core import _load_harness

APP = '''
def entry(task, config, model):
    reply = model(config["prompt"] + " " + task["input"])
    return {"output": reply, "k": config["k"]}
'''

SCHEMA = {
    "type": "object", "required": ["prompt", "k", "mode"], "frozen": ["mode"],
    "properties": {
        "prompt": {"type": "string", "maxLength": 20},
        "k": {"type": "integer", "minimum": 1, "maximum": 5},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
        "tags": {"type": "array", "maxItems": 2, "items": {"type": "string"}},
    },
}
CONFIG = {"prompt": "Answer:", "k": 2, "mode": "fast"}


@pytest.fixture
def seed(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "adapter_app_t.py").write_text(APP, encoding="utf-8")
    (tmp_path / "cfg.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    init_seed("adapter_app_t:entry", tmp_path / "cfg.json", tmp_path / "seed",
              schema_path=tmp_path / "schema.json", entry_dir=app)
    return tmp_path / "seed"


def _candidate(tmp_path, seed, config=None, text=None, name="cand"):
    directory = tmp_path / name
    directory.mkdir()
    source = (seed / "harness.py").read_text(encoding="utf-8")
    if config is not None:
        fixed, _ = split_harness(source)
        source = fixed.replace("<CONFIG>", json.dumps(config, indent=2))
    (directory / "harness.py").write_text(text if text is not None else source, encoding="utf-8")
    return directory


# -- schema -----------------------------------------------------------------------------------

def test_schema_accepts_valid_and_reports_each_violation():
    assert validate_config(CONFIG, SCHEMA) == []
    bad = {"prompt": "x" * 21, "k": 9, "mode": "turbo", "tags": ["a", "b", "c"], "extra": 1}
    errors = validate_config(bad, SCHEMA)
    joined = "\n".join(errors)
    assert "maxLength 20" in joined and "above the maximum 5" in joined
    assert "not one of" in joined and "maxItems 2" in joined and "unknown key 'extra'" in joined
    assert validate_config({"prompt": "a", "mode": "fast"}, SCHEMA) == ["$: missing required key 'k'"]
    assert validate_config({**CONFIG, "k": 0}, SCHEMA) == ["$.k: 0 is below the minimum 1"]
    assert validate_config({**CONFIG, "k": True}, SCHEMA)[0].startswith("$.k: expected integer")
    assert validate_config({**CONFIG, "k": 2.5}, SCHEMA)[0].startswith("$.k: expected integer")
    assert validate_config({**CONFIG, "tags": [1]}, SCHEMA)[0].startswith("$.tags.0: expected string")


def test_schema_frozen_keys_compare_with_the_seed():
    assert validate_config({**CONFIG, "mode": "slow"}, SCHEMA, seed_config=CONFIG) == \
        ["$.mode: frozen key changed from the seed"]
    assert validate_config({**CONFIG, "k": 3}, SCHEMA, seed_config=CONFIG) == []
    nested = {"type": "object", "frozen": ["tools.0.name"]}
    seed = {"tools": [{"name": "a", "description": "x"}]}
    assert validate_config({"tools": [{"name": "a", "description": "y"}]}, nested, seed) == []
    assert validate_config({"tools": [{"name": "b", "description": "x"}]}, nested, seed)
    assert validate_config({"tools": []}, nested, seed)


def test_malformed_schema_is_refused():
    with pytest.raises(SchemaError):
        validate_config({}, {"type": "map"})
    with pytest.raises(SchemaError):
        validate_config({}, {"type": "object", "properties": {"a": {"maximum": "1"}}})
    with pytest.raises(SchemaError):
        validate_config({}, {"type": "object", "properties": {"a": {"frozen": ["x"]}}})


# -- scaffold and runtime ---------------------------------------------------------------------

def test_init_scaffolds_a_harness_that_loads_its_sibling_config(seed):
    assert sorted(p.name for p in seed.iterdir()) == ["config.json", "harness.py", "schema.json"]
    assert json.loads((seed / "config.json").read_text(encoding="utf-8")) == CONFIG
    harness = _load_harness(seed / "harness.py")
    calls = []
    result = harness.run({"input": "hi"}, lambda prompt: calls.append(prompt) or "ok")
    assert result == {"output": "ok", "k": 2} and calls == ["Answer: hi"]
    # sibling config.json wins over the embedded block
    (seed / "config.json").write_text(json.dumps({**CONFIG, "prompt": "Q:", "k": 4}), encoding="utf-8")
    harness = _load_harness(seed / "harness.py")
    assert harness.run({"input": "hi"}, lambda p: p) == {"output": "Q: hi", "k": 4}


def test_harness_without_sibling_uses_the_config_block(tmp_path, seed):
    cand = _candidate(tmp_path, seed, config={**CONFIG, "prompt": "Embedded:"})
    harness = _load_harness(cand / "harness.py")
    assert harness.run({"input": "x"}, lambda p: p)["output"] == "Embedded: x"


def test_render_and_parse_round_trip(seed):
    params = parse_harness((seed / "harness.py").read_text(encoding="utf-8"))
    assert params["entrypoint"] == "adapter_app_t:entry" and params["entry_base"] == "absolute"
    assert params["schema"] == SCHEMA and params["config"] == CONFIG
    again = render_harness(params["entrypoint"], params["entry_path"], params["entry_base"],
                           params["schema"], params["config"])
    assert again == (seed / "harness.py").read_text(encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="module:function"):
        render_harness("no-colon", ".", "absolute", SCHEMA, CONFIG)


def test_init_refuses_a_seed_config_outside_its_schema(tmp_path):
    (tmp_path / "cfg.json").write_text(json.dumps({**CONFIG, "k": 99}), encoding="utf-8")
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="maximum"):
        init_seed("m:f", tmp_path / "cfg.json", tmp_path / "seed", tmp_path / "schema.json",
                  check_import=False)


def test_init_refuses_an_unimportable_entrypoint(tmp_path):
    (tmp_path / "cfg.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="cannot import"):
        init_seed("no_such_module_xyz:f", tmp_path / "cfg.json", tmp_path / "seed", entry_dir=tmp_path)


def test_sync_rerenders_after_a_hand_edit(tmp_path, seed):
    (seed / "config.json").write_text(json.dumps({**CONFIG, "k": 3}), encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="adapt sync"):
        validate_candidate(seed, seed)
    sync_seed(seed)
    assert parse_harness((seed / "harness.py").read_text(encoding="utf-8"))["config"]["k"] == 3
    assert validate_candidate(seed, seed)["k"] == 3


# -- the config-only gate ---------------------------------------------------------------------

def test_gate_accepts_a_config_only_edit(tmp_path, seed):
    cand = _candidate(tmp_path, seed, config={**CONFIG, "prompt": "Short answer:", "k": 5})
    assert validate_candidate(cand, seed)["k"] == 5


def test_gate_rejects_any_code_edit(tmp_path, seed):
    source = (seed / "harness.py").read_text(encoding="utf-8")
    edits = {
        "import": source.replace("import json\n", "import json\nimport os\n", 1),
        "entrypoint": source.replace("adapter_app_t:entry", "adapter_app_t:other"),
        "schema": source.replace('"maximum": 5', '"maximum": 50'),
        "body": source.replace("return {\"output\": result}", "return {\"output\": \"42\"}"),
        "trailing": source + "\nprint('x')\n",
    }
    for name, text in edits.items():
        assert text != source, name
        cand = _candidate(tmp_path, seed, text=text, name=f"cand-{name}")
        with pytest.raises(ConfigHarnessError, match="may not change outside the config block"):
            validate_candidate(cand, seed)


def test_gate_rejects_out_of_bounds_and_frozen_values(tmp_path, seed):
    cases = {"k": ({**CONFIG, "k": 6}, "above the maximum 5"),
             "len": ({**CONFIG, "prompt": "p" * 21}, "maxLength 20"),
             "enum": ({**CONFIG, "mode": "turbo"}, "not one of"),
             "frozen": ({**CONFIG, "mode": "slow"}, "frozen key changed"),
             "missing": ({"prompt": "a", "mode": "fast"}, "missing required key 'k'"),
             "type": ({**CONFIG, "k": "2"}, "expected integer")}
    for name, (config, message) in cases.items():
        cand = _candidate(tmp_path, seed, config=config, name=f"cand-{name}")
        with pytest.raises(ConfigHarnessError, match=message):
            validate_candidate(cand, seed)


def test_gate_rejects_invalid_json_and_missing_blocks(tmp_path, seed):
    fixed, _ = split_harness((seed / "harness.py").read_text(encoding="utf-8"))
    cand = _candidate(tmp_path, seed, text=fixed.replace("<CONFIG>", "{not json"), name="c1")
    with pytest.raises(ConfigHarnessError, match="not valid JSON"):
        validate_candidate(cand, seed)
    cand = _candidate(tmp_path, seed, text="class Harness:\n    pass\n", name="c2")
    with pytest.raises(ConfigHarnessError, match="not a config harness"):
        validate_candidate(cand, seed)


def test_gate_checks_a_sibling_config_json_in_the_candidate(tmp_path, seed):
    cand = _candidate(tmp_path, seed)
    (cand / "config.json").write_text(json.dumps({**CONFIG, "k": 4}), encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="differs from the config block"):
        validate_candidate(cand, seed)


def test_gate_takes_an_explicit_schema(tmp_path, seed):
    cand = _candidate(tmp_path, seed, config={**CONFIG, "k": 4})
    strict = json.loads(json.dumps(SCHEMA))
    strict["properties"]["k"]["maximum"] = 3
    with pytest.raises(ConfigHarnessError, match="above the maximum 3"):
        validate_candidate(cand, seed, strict)


# -- the adapt command ------------------------------------------------------------------------

def test_adapt_cli_init_check_and_sync(tmp_path, capsys):
    app = tmp_path / "app"
    app.mkdir()
    (app / "adapter_app_cli.py").write_text(APP, encoding="utf-8")
    (tmp_path / "cfg.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    seed = tmp_path / "seed"
    assert cli.main(["adapt", "init", "--entrypoint", "adapter_app_cli:entry", "--config",
                     str(tmp_path / "cfg.json"), "--schema", str(tmp_path / "schema.json"),
                     "--out", str(seed), "--path", str(app)]) == 0
    assert json.loads(capsys.readouterr().out)["seed_dir"] == str(seed)
    assert cli.main(["adapt", "check", "--candidate", str(seed), "--seed", str(seed)]) == 0
    bad = _candidate(tmp_path, seed, config={**CONFIG, "k": 50})
    capsys.readouterr()
    assert cli.main(["adapt", "check", "--candidate", str(bad), "--seed", str(seed)]) == 2
    assert "above the maximum 5" in capsys.readouterr().err
    (seed / "config.json").write_text(json.dumps({**CONFIG, "k": 1}), encoding="utf-8")
    assert cli.main(["adapt", "sync", str(seed)]) == 0
    assert cli.main(["adapt", "init", "--entrypoint", "x:y", "--config", str(tmp_path / "missing.json"),
                     "--out", str(tmp_path / "o")]) == 2


# -- the agent_config domain (the gate's VALIDATE hook) ----------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agents"


def test_agent_config_domain_contract():
    from meta_harness.domains import get_domain
    domain = get_domain("agent_config")
    assert domain.VALIDATE is domain.validate and domain.PROTOCOL == "plain"
    assert "BEGIN CONFIG" in domain.PROMPT and "config.json" in domain.PROMPT
    assert domain.adapt([{"input": "a"}]) == [{"input": "a"}]


def test_domain_validate_finds_the_seed_in_the_experience_store(tmp_path, seed):
    from meta_harness.domains import agent_config
    candidates = tmp_path / "store" / "candidates"
    initial = candidates / "initial-0000"
    initial.mkdir(parents=True)
    (initial / "harness.py").write_bytes((seed / "harness.py").read_bytes())
    agent_config.validate(initial / "harness.py", [])
    # a seed is checked against itself: canonical template and a config inside its own schema
    broken = tmp_path / "other" / "candidates" / "initial-0000"
    broken.mkdir(parents=True)
    fixed, _ = split_harness((seed / "harness.py").read_text(encoding="utf-8"))
    (broken / "harness.py").write_text(fixed.replace("<CONFIG>", json.dumps({**CONFIG, "k": 9})),
                                       encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="above the maximum"):
        agent_config.validate(broken / "harness.py", [])
    (broken / "harness.py").write_text((seed / "harness.py").read_text(encoding="utf-8") + "\nX = 1\n",
                                       encoding="utf-8")
    with pytest.raises(ConfigHarnessError, match="differs from the generated template"):
        agent_config.validate(broken / "harness.py", [])
    good = _candidate(candidates, seed, config={**CONFIG, "k": 3}, name="iteration-0001-00")
    agent_config.validate(good / "harness.py", [])
    code = (seed / "harness.py").read_text(encoding="utf-8").replace("import sys\n", "import sys\nimport os\n")
    edited = _candidate(candidates, seed, text=code, name="iteration-0002-00")
    with pytest.raises(ConfigHarnessError, match="outside the config block"):
        agent_config.validate(edited / "harness.py", [])
    oob = _candidate(candidates, seed, config={**CONFIG, "k": 7}, name="iteration-0003-00")
    with pytest.raises(ConfigHarnessError, match="above the maximum"):
        agent_config.validate(oob / "harness.py", [])
    # a candidate whose embedded schema was loosened is a code edit, not a pass
    loose = (seed / "harness.py").read_text(encoding="utf-8").replace('"maximum": 5', '"maximum": 9')
    loose = loose.replace('"k": 2', '"k": 7')
    widened = _candidate(candidates, seed, text=loose, name="iteration-0004-00")
    with pytest.raises(ConfigHarnessError, match="outside the config block"):
        agent_config.validate(widened / "harness.py", [])


def test_domain_validate_picks_the_matching_seed_among_several(tmp_path, seed):
    from meta_harness.domains import agent_config
    candidates = tmp_path / "store" / "candidates"
    (candidates / "initial-0000").mkdir(parents=True)
    (candidates / "initial-0000" / "harness.py").write_text("class Harness:\n    pass\n", encoding="utf-8")
    other = json.loads(json.dumps(SCHEMA))
    other["properties"]["k"]["maximum"] = 2
    (candidates / "initial-0001").mkdir()
    (candidates / "initial-0001" / "harness.py").write_text(
        render_harness("adapter_app_t:entry", ".", "absolute", other, CONFIG), encoding="utf-8")
    (candidates / "initial-0002").mkdir()
    (candidates / "initial-0002" / "harness.py").write_bytes((seed / "harness.py").read_bytes())
    # k=4 passes the third seed's schema (maximum 5) but not the second's (maximum 2)
    good = _candidate(candidates, seed, config={**CONFIG, "k": 4}, name="iteration-0001-00")
    agent_config.validate(good / "harness.py", [])


@pytest.mark.parametrize("app", ["router_graph", "support_agent"])
def test_example_seeds_are_generated_and_in_sync(app):
    seed = EXAMPLES / app / "seed"
    params = parse_harness((seed / "harness.py").read_text(encoding="utf-8"))
    assert params["entry_base"] == "package" and params["entry_path"] == f"examples/agents/{app}"
    assert params["config"] == json.loads((EXAMPLES / app / "config.json").read_text(encoding="utf-8"))
    assert params["schema"] == json.loads((EXAMPLES / app / "schema.json").read_text(encoding="utf-8"))
    validate_candidate(seed, seed)
    source = (EXAMPLES / app).glob("*_app.py")
    assert all("import langgraph" not in p.read_text(encoding="utf-8")
               and "from agents" not in p.read_text(encoding="utf-8") for p in source)


@pytest.mark.parametrize("app,seed_score", [("router_graph", 2 / 6), ("support_agent", 1 / 4)])
def test_search_agent_config_end_to_end(app, seed_score, tmp_path, capsys, monkeypatch):
    from meta_harness.commands import search
    if "agent_config" not in search.DOMAINS:  # until search.py lists the domain in its choices
        monkeypatch.setattr(search, "DOMAINS", search.DOMAINS + ("agent_config",))
    folder = EXAMPLES / app
    root = tmp_path / "store"
    assert cli.main(["search", "--domain", "agent_config", "--harness", str(folder / "seed" / "harness.py"),
                     "--search", str(folder / "data.jsonl"), "--metric", "exact",
                     "--provider", "scripted", "--model", str(folder / "base_script.jsonl"),
                     "--proposer-provider", "scripted", "--proposer-model", str(folder / "proposer_script.jsonl"),
                     "--iterations", "3", "--candidates", "1", "--root", str(root), "--no-cache"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdicts"] == {"accepted": 2, "rejected_invalid_source": 2}
    receipts = {d.name: json.loads((d / "receipt.json").read_text(encoding="utf-8"))
                for d in (root / "candidates").iterdir()}
    assert "outside the config block" in receipts["iteration-0001-00"]["fallback"]
    assert "above the maximum" in receipts["iteration-0002-00"]["fallback"]
    scores = {d.name: json.loads((d / "scores.json").read_text(encoding="utf-8"))["score"]
              for d in (root / "candidates").iterdir() if (d / "scores.json").is_file()}
    assert scores["initial-0000"] == pytest.approx(seed_score)
    assert scores["iteration-0003-00"] == 1.0
    winner = parse_harness((root / "candidates" / "iteration-0003-00" / "harness.py").read_text(encoding="utf-8"))
    seed_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert winner["config"] != seed_config  # the proposer changed config.json, nothing else
