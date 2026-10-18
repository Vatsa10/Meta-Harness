"""Static knob discovery (and, below, apply) over a fake LangGraph project."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from meta_harness.integrations import knobs
from meta_harness.integrations.knobs import Knob, KnobError, Manifest, discover

FIXTURE = Path(__file__).parent / "fixtures" / "lg_app"


def _by_id(m: Manifest) -> dict[str, Knob]:
    return {k.id: k for k in m.knobs}


def _span_text(root: Path, k: Knob) -> str:
    data = (root / k.file).read_bytes()
    return data[k.offset:k.end_offset].decode("utf-8")


@pytest.fixture
def app(tmp_path):
    dst = tmp_path / "lg_app"
    shutil.copytree(FIXTURE, dst)
    return dst


# ----------------------------------------------------------------------------- discover

def test_discover_finds_every_kind_with_exact_spans():
    m = discover(FIXTURE)
    k = _by_id(m)
    planner = k["app/graph.py:planner:prompt:0"]
    assert planner.value == "You are the planner. Break the task into steps."
    assert (planner.line, planner.col, planner.end_line, planner.end_col) == (20, 30, 20, 79)
    assert planner.node == "planner"
    assert _span_text(FIXTURE, planner) == '"You are the planner. Break the task into steps."'

    worker = k["app/graph.py:worker:prompt:0"]   # implicit concatenation, mixed quotes
    assert worker.value == 'You are a careful worker. Use the "write_file" tool.'
    assert (worker.line, worker.end_line) == (28, 29)
    assert _span_text(FIXTURE, worker).startswith('"You are a careful worker. "')
    assert _span_text(FIXTURE, worker).endswith("""'Use the "write_file" tool.'""")
    assert k["app/graph.py:worker:prompt:1"].value == "{step}"

    assert k["app/graph.py:<module>:model:0"].value == "gpt-4o-mini"
    assert k["app/graph.py:<module>:temperature:0"].value == 0.2
    assert k["app/graph.py:<module>:limit:0"].value == 512
    assert k["app/graph.py:<module>:limit:0"].key == "max_tokens"
    assert k["app/graph.py:build:limit:0"].value == 25
    assert k["app/graph.py:build:limit:0"].key == "recursion_limit"

    tool = k["app/graph.py:write_file:tool_description:0"]
    assert tool.value == "Write text to a file in the workspace and return the path."
    assert tool.node == "write_file"

    const = k["app/prompts.py:<module>:constant:0"]
    assert const.key == "REVIEWER_PROMPT"
    assert const.value.startswith("You are the reviewer.\nReject")
    assert (const.line, const.end_line) == (1, 2)
    assert _span_text(FIXTURE, const).startswith('"""')
    assert k["app/prompts.py:<module>:constant:1"].value == 3
    assert not any(x.key == "_private" for x in m.knobs)   # lowercase: not a constant


def test_fstring_with_expressions_is_dynamic_not_tunable():
    k = _by_id(discover(FIXTURE))["app/graph.py:reviewer:prompt:0"]
    assert k.dynamic is True and k.tunable is False
    assert "state['n']" in k.value


def test_config_files_json_yaml_toml():
    k = _by_id(discover(FIXTURE))
    assert k["config.json:planner.model:config:0"].value == "gpt-4o"
    assert k["config.json:planner.temperature:config:0"].value == 0.0
    assert _span_text(FIXTURE, k["config.json:planner.model:config:0"]) == '"gpt-4o"'
    assert not any(x.key == "debug" for x in k.values())          # booleans are not knobs
    assert k["agent_settings.yaml:worker.model:config:0"].value == "gpt-4o-mini"
    assert k["agent_settings.yaml:worker.greeting:config:0"].value == "it's ready"
    assert k["agent_settings.yaml:worker.system:config:0"].value == \
        "You are a worker.\nStay in scope.\n"
    mi = k["agent_settings.yaml:reviewer.max_iterations:config:0"]
    assert mi.value == 4 and _span_text(FIXTURE, mi) == "4"       # comment not in span
    model = k["llm_config.toml:llm.model:config:0"]
    assert model.value == "gpt-4o" and _span_text(FIXTURE, model) == '"gpt-4o"'
    assert k["llm_config.toml:llm.temperature:config:0"].value == 0.5
    assert k["llm_config.toml:llm.system:config:0"].value == "Be brief."


def test_discover_never_imports_and_skips_env_venv_and_caches(app):
    (app / ".env").write_text('SYSTEM_PROMPT = "secret"\n', encoding="utf-8")
    (app / ".venv" / "lib").mkdir(parents=True)
    (app / ".venv" / "lib" / "settings.py").write_text('X = "venv"\n', encoding="utf-8")
    (app / "__pycache__").mkdir()
    (app / "__pycache__" / "config.py").write_text('Y = "cache"\n', encoding="utf-8")
    (app / "boom_settings.py").write_text(
        'raise SystemExit("executed")\nTIMEOUT = 30\n', encoding="utf-8")
    (app / "broken.py").write_text("def (:\n", encoding="utf-8")
    m = discover(app)
    files = {k.file for k in m.knobs}
    assert "boom_settings.py" in files                      # read, not run
    assert not any(f.startswith((".venv", "__pycache__", ".env")) for f in files)
    assert "broken.py" not in files


def test_include_exclude_globs():
    only_py = discover(FIXTURE, include=["app/*.py"])
    assert {k.file for k in only_py.knobs} == {"app/graph.py", "app/prompts.py"}
    no_cfg = discover(FIXTURE, exclude=["*.json", "*.toml", "*.yaml"])
    assert {k.file for k in no_cfg.knobs} == {"app/graph.py", "app/prompts.py"}


def test_ids_are_stable_and_unique():
    a, b = discover(FIXTURE), discover(FIXTURE)
    assert [k.id for k in a.knobs] == [k.id for k in b.knobs]
    assert len({k.id for k in a.knobs}) == len(a.knobs)


def test_unicode_offsets_are_bytes_and_cols_are_chars(app):
    (app / "app" / "uni.py").write_text(
        'x = SystemMessage(content="héllo ☃ wörld")\n', encoding="utf-8")
    k = _by_id(discover(app))["app/uni.py:<module>:prompt:0"]
    assert k.value == "héllo ☃ wörld"
    assert k.col == 26 and k.end_col == 26 + len('"héllo ☃ wörld"')
    assert _span_text(app, k) == '"héllo ☃ wörld"'


# ----------------------------------------------------------------------------- manifest

def test_manifest_roundtrip_and_frozen_carry_over(app, tmp_path):
    mp = tmp_path / "knobs.json"
    m = knobs.rediscover(app, mp)
    assert Manifest.load(mp).to_dict() == m.to_dict()
    m.get("app/graph.py:planner:prompt:0").frozen = True
    m.get("app/prompts.py:<module>:constant:0").frozen = True
    m.save(mp)

    # Shift every line in graph.py and add a function before planner: ids hold,
    # spans move. Rename the module constant's file: id changes, value hash matches.
    g = app / "app" / "graph.py"
    g.write_text("# header\n" + g.read_text(encoding="utf-8"), encoding="utf-8")
    (app / "app" / "prompts.py").rename(app / "app" / "prompts_v2.py")
    m2 = knobs.rediscover(app, mp)
    k2 = _by_id(m2)
    assert k2["app/graph.py:planner:prompt:0"].frozen is True
    assert k2["app/graph.py:planner:prompt:0"].line == 21
    assert k2["app/graph.py:worker:prompt:0"].frozen is False
    # the hash covers the file, so a renamed file is a new knob: not frozen
    assert k2["app/prompts_v2.py:<module>:constant:0"].frozen is False
    assert Manifest.load(mp).get("app/graph.py:planner:prompt:0").frozen is True


def test_frozen_falls_back_to_value_hash_when_id_moves(app):
    m = discover(app)
    m.get("app/graph.py:worker:prompt:1").frozen = True        # "{step}"
    g = app / "app" / "graph.py"
    src = g.read_text(encoding="utf-8")
    g.write_text(src.replace("def worker(state):", "def worker_node(state):"), encoding="utf-8")
    m2 = discover(app, previous=m)
    k = _by_id(m2)["app/graph.py:worker_node:prompt:1"]
    assert k.value == "{step}" and k.frozen is True
    assert _by_id(m2)["app/graph.py:worker_node:prompt:0"].frozen is False


def test_manifest_rejects_unknown_version():
    with pytest.raises(KnobError):
        Manifest.from_dict({"version": 99, "project_dir": ".", "knobs": []})
