"""search --optimizer / --provider / --protocol: every registered optimizer and provider via the CLI."""

import json
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness import providers
from meta_harness.commands import search
from meta_harness.optimizers import all_optimizers

FIX = Path(__file__).parent / "fixtures" / "smoke"
TEXT_HARNESS = ("```python\nclass Harness:\n    def run(self, task, model, trace):\n"
                "        return {'output': model('classify: ' + task['input']).strip()}\n```")


def _proposer_script(tmp_path):
    script = tmp_path / "proposer.jsonl"
    lines = (FIX / "proposer.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps({"default": TEXT_HARNESS}))
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script


def _run(tmp_path, optimizer, *extra):
    root = tmp_path / optimizer
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--data", str(FIX / "data.jsonl"),
                     "--metric", "exact", "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
                     "--proposer-provider", "scripted", "--proposer-model", str(_proposer_script(tmp_path)),
                     "--iterations", "1", "--candidates", "1", "--root", str(root), "--no-cache",
                     "--optimizer", optimizer, *extra])
    return code, root


def test_every_registered_optimizer_is_a_choice():
    parser = cli.build_parser()
    names = set(all_optimizers())
    assert {"metaharness", "best-of-n", "openevolve", "gepa"} <= names
    for name in names:
        assert parser.parse_args(
            ["search", "--harness", "h.py", "--provider", "scripted", "--model", "m",
             "--optimizer", name]).optimizer == name


@pytest.mark.parametrize("optimizer", sorted(all_optimizers()))
def test_search_runs_each_optimizer_under_the_equal_budget(tmp_path, capsys, optimizer):
    code, root = _run(tmp_path, optimizer, "--max-evaluations", "2")
    assert code == 0, capsys.readouterr().err
    out = json.loads(capsys.readouterr().out)
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert json.dumps(manifest).count(f'"optimizer": "{optimizer}"') >= 1
    assert out["budget"]["harness_evaluations"] <= 2


def test_optimizer_routing_builds_the_named_optimizer(tmp_path, capsys, monkeypatch):
    seen = []
    real = search.build_optimizer
    monkeypatch.setattr(search, "build_optimizer", lambda name, ctx: seen.append((name, ctx)) or real(name, ctx))
    code, root = _run(tmp_path, "best-of-n")
    assert code == 0
    name, ctx = seen[0]
    assert name == "best-of-n"
    assert ctx.test_root is not None and ctx.test_root in ctx.forbidden


def test_proposer_command_conflicts_with_optimizer(tmp_path, capsys):
    code, _ = _run(tmp_path, "gepa", "--proposer-command", "python x.py")
    assert code == 2 and "--optimizer" in capsys.readouterr().err


@pytest.mark.parametrize("provider", providers.PROVIDERS)
def test_every_provider_is_accepted_by_search_flags(provider):
    parser = cli.build_parser()
    args = parser.parse_args(["search", "--harness", "h.py", "--provider", provider, "--model", "m",
                              "--proposer-provider", provider])
    assert args.provider == provider == args.proposer_provider


@pytest.mark.parametrize("provider,cls", [("groq", "GroqModel"), ("unikey", "UnikeyModel")])
def test_groq_and_unikey_route_to_their_class(monkeypatch, provider, cls):
    target = getattr(providers, cls)
    monkeypatch.setenv(target.API_KEY_ENV, "fake-key-for-test")
    model = search._make_model(provider, "some-model", None)
    assert type(model) is target


def test_protocol_online_overrides_domain_default(tmp_path, capsys):
    code, root = _run(tmp_path, "metaharness", "--protocol", "online", "--train-fraction", "0.5")
    assert code == 0, capsys.readouterr().err
    out = json.loads(capsys.readouterr().out)
    assert out["train_tasks"] > 0
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert '"protocol": "online"' in json.dumps(manifest)


def test_train_without_online_protocol_is_refused(tmp_path, capsys):
    code, _ = _run(tmp_path, "metaharness", "--protocol", "plain", "--train", str(FIX / "data.jsonl"))
    assert code == 2
