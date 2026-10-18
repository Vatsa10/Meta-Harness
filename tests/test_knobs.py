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


# ----------------------------------------------------------------------------- apply

def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


PLANNER = "app/graph.py:planner:prompt:0"
MODEL = "app/graph.py:<module>:model:0"
TEMP = "app/graph.py:<module>:temperature:0"

EXPECTED_DIFF = "".join(line + "\n" for line in [
    '--- a/app/graph.py',
    '+++ b/app/graph.py',
    '@@ -6,7 +6,7 @@',
    ' ',
    ' from app.prompts import REVIEWER_PROMPT',
    ' ',
    '-llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=512)',
    '+llm = ChatOpenAI(model="gpt-4.1", temperature=0.7, max_tokens=512)',
    ' ',
    ' ',
    ' @tool',
    '@@ -17,7 +17,7 @@',
    ' ',
    ' def planner(state):',
    '     msgs = [',
    '-        SystemMessage(content="You are the planner. Break the task into steps."),',
    '+        SystemMessage(content="Plan first. Then act."),',
    '         HumanMessage(state["task"]),',
    '     ]',
    '     return llm.invoke(msgs)',
])


def test_apply_rewrites_only_the_spans_in_a_copy(app, tmp_path):
    (app / ".env").write_text("OPENAI_API_KEY=sk-should-not-copy\n", encoding="utf-8")
    (app / ".git").mkdir()
    (app / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (app / ".venv").mkdir()
    (app / ".venv" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    before = _snapshot(app)
    m = discover(app)
    out = tmp_path / "candidate"
    r = knobs.apply(app, out, m, {PLANNER: "Plan first. Then act.", MODEL: "gpt-4.1", TEMP: 0.7})

    assert _snapshot(app) == before                                 # original untouched
    assert r.changed == ["app/graph.py"]
    assert r.diff.replace("\r\n", "\n") == EXPECTED_DIFF
    copied = _snapshot(out)
    assert not any(p.startswith((".env", ".git", ".venv")) for p in copied)
    for rel, blob in before.items():                                # everything else identical
        if rel != "app/graph.py" and not rel.startswith((".env", ".git", ".venv")):
            assert copied[rel] == blob, rel
    src = (out / "app" / "graph.py").read_text(encoding="utf-8")
    compile(src, "graph.py", "exec")
    k = _by_id(discover(out))
    assert k[PLANNER].value == "Plan first. Then act."
    assert k[MODEL].value == "gpt-4.1"
    assert k[TEMP].value == 0.7
    assert k["app/graph.py:<module>:limit:0"].value == 512


@pytest.mark.parametrize("new, literal", [
    ('say "hi"', r'"say \"hi\""'),
    ("line1\nline2", r'"line1\nline2"'),
    ("tab\there", r'"tab\there"'),
    ("back\\slash", r'"back\\slash"'),
    ("héllo ☃ 日本", '"héllo ☃ 日本"'),
    ("it's", '"it\'s"'),
    ("bell\x07", r'"bell\x07"'),
])
def test_apply_escapes_into_the_original_quote_style(app, tmp_path, new, literal):
    m = discover(app)
    r = knobs.apply(app, tmp_path / "o", m, {PLANNER: new})
    k = _by_id(discover(tmp_path / "o"))[PLANNER]
    assert k.value == new
    assert _span_text(tmp_path / "o", k) == literal
    assert f"+        SystemMessage(content={literal})," in r.diff


def test_apply_triple_quoted_and_implicit_concatenation(app, tmp_path):
    m = discover(app)
    const = "app/prompts.py:<module>:constant:0"
    worker = "app/graph.py:worker:prompt:0"
    new_const = 'Review "strictly".\nEnds with a quote"'
    r = knobs.apply(app, tmp_path / "o", m, {const: new_const, worker: "Work 'carefully'."})
    k = _by_id(discover(tmp_path / "o"))
    assert k[const].value == new_const
    span = _span_text(tmp_path / "o", k[const])
    assert span.startswith('"""') and "\n" in span                 # stays triple-quoted
    assert k[worker].value == "Work 'carefully'."
    assert _span_text(tmp_path / "o", k[worker]) == '"Work \'carefully\'."'
    assert "-                   'Use the \"write_file\" tool.'),\n" in r.diff.replace("\r", "")


def test_apply_single_quoted_python_keeps_single_quotes(app, tmp_path):
    (app / "app" / "sq.py").write_bytes(b"p = SystemMessage(content='old')\n")
    m = discover(app)
    knobs.apply(app, tmp_path / "o", m, {"app/sq.py:<module>:prompt:0": "new 'q' \"d\""})
    assert (tmp_path / "o" / "app" / "sq.py").read_bytes() == \
        b"p = SystemMessage(content='new \\'q\\' \"d\"')\n"


def test_apply_config_values_read_back(app, tmp_path):
    m = discover(app)
    vals = {
        "config.json:planner.model:config:0": 'gpt "x"',
        "config.json:planner.temperature:config:0": 1,
        "agent_settings.yaml:worker.model:config:0": "a: b",
        "agent_settings.yaml:worker.greeting:config:0": "x'y",
        "agent_settings.yaml:worker.system:config:0": "New\n\nblock ☃\n",
        "agent_settings.yaml:reviewer.max_iterations:config:0": 9,
        "llm_config.toml:llm.model:config:0": "m\\1",
        "llm_config.toml:llm.system:config:0": 'a"""b\nc',
        "llm_config.toml:llm.temperature:config:0": 1.25,
    }
    knobs.apply(app, tmp_path / "o", m, vals)
    k = _by_id(discover(tmp_path / "o"))
    for kid, v in vals.items():
        expect = float(v) if kid.endswith("planner.temperature:config:0") else v
        assert k[kid].value == expect and type(k[kid].value) is type(expect), kid
    yaml = (tmp_path / "o" / "agent_settings.yaml").read_text(encoding="utf-8")
    assert "greeting: 'x''y'" in yaml
    assert "  system: |\n    New\n\n    block ☃\nreviewer:" in yaml.replace("\r", "")
    assert "max_iterations: 9  # hard cap" in yaml
    toml = (tmp_path / "o" / "llm_config.toml").read_text(encoding="utf-8")
    assert 'model = "m\\\\1"  # default' in toml


def test_frozen_dynamic_and_unknown_are_refused_and_nothing_is_written(app, tmp_path):
    m = discover(app)
    m.get(PLANNER).frozen = True
    out = tmp_path / "o"
    with pytest.raises(KnobError, match="frozen"):
        knobs.apply(app, out, m, {MODEL: "gpt-4.1", PLANNER: "x"})
    with pytest.raises(KnobError, match="dynamic"):
        knobs.apply(app, out, m, {"app/graph.py:reviewer:prompt:0": "x"})
    with pytest.raises(KnobError, match="unknown"):
        knobs.apply(app, out, m, {"app/graph.py:nope:prompt:0": "x"})
    assert not out.exists()


@pytest.mark.parametrize("kid, bad", [
    (TEMP, 2.5), (TEMP, -0.1), (TEMP, "0.5"), (TEMP, True), (TEMP, float("nan")),
    ("app/graph.py:<module>:limit:0", 0), ("app/graph.py:<module>:limit:0", 1.5),
    ("app/graph.py:build:limit:0", -3),
    (MODEL, ""), (MODEL, "a\nb"), (MODEL, 4),
    (PLANNER, 3),
    ("app/prompts.py:<module>:constant:1", "three"),
    ("config.json:planner.temperature:config:0", 9),
    ("agent_settings.yaml:reviewer.max_iterations:config:0", 0),
])
def test_bounds_and_types_are_enforced(app, tmp_path, kid, bad):
    with pytest.raises(KnobError):
        knobs.apply(app, tmp_path / "o", discover(app), {kid: bad})
    assert not (tmp_path / "o").exists()


def test_bounds_accept_edges(app, tmp_path):
    knobs.apply(app, tmp_path / "o", discover(app),
                {TEMP: 0, "app/graph.py:<module>:limit:0": 1, MODEL: "m"})
    k = _by_id(discover(tmp_path / "o"))
    assert k[TEMP].value == 0.0 and isinstance(k[TEMP].value, float)
    assert k["app/graph.py:<module>:limit:0"].value == 1


def test_a_value_that_does_not_read_back_is_refused(app, tmp_path, monkeypatch):
    monkeypatch.setattr(knobs, "_render_json", lambda src, new: '"something else"')
    with pytest.raises(KnobError, match="read back"):
        knobs.apply(app, tmp_path / "o", discover(app),
                    {"config.json:planner.model:config:0": "gpt-4.1"})
    assert not (tmp_path / "o").exists()


def test_stale_manifest_is_refused(app, tmp_path):
    m = discover(app)
    g = app / "app" / "graph.py"
    g.write_bytes(g.read_bytes().replace(b"Break the task", b"Split the task"))
    with pytest.raises(KnobError, match="changed"):
        knobs.apply(app, tmp_path / "o", m, {PLANNER: "x"})


def test_out_dir_must_be_outside_and_empty(app, tmp_path):
    m = discover(app)
    with pytest.raises(KnobError):
        knobs.apply(app, app / "candidate", m, {MODEL: "x"})
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "f").write_text("x", encoding="utf-8")
    with pytest.raises(KnobError):
        knobs.apply(app, busy, m, {MODEL: "x"})
    empty = tmp_path / "empty"
    empty.mkdir()
    assert knobs.apply(app, empty, m, {MODEL: "x"}).changed == ["app/graph.py"]
