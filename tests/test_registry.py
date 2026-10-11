import types

import pytest

from meta_harness import registry
from meta_harness.chat_types import Usage
from meta_harness.commands import all_commands
from meta_harness.domains import get_domain
from meta_harness.loaders import all_loaders
from meta_harness.optimizers import all_optimizers, extract_python


def test_builtin_commands_are_discovered():
    assert {"demo", "inspect", "models", "search"} <= set(all_commands())


def test_generic_domain_contract():
    d = get_domain("generic")
    assert d.PROTOCOL == "plain" and d.adapt([{"a": 1}]) == [{"a": 1}]
    with pytest.raises(KeyError, match="generic"):
        get_domain("nope")


def test_empty_registries_are_dicts():
    assert isinstance(all_loaders(), dict) and isinstance(all_optimizers(), dict)


def test_duplicate_names_raise(monkeypatch):
    a, b = types.SimpleNamespace(NAME="x", __name__="p.a"), types.SimpleNamespace(NAME="x", __name__="p.b")
    monkeypatch.setattr(registry, "_modules", lambda package: [a, b])
    with pytest.raises(RuntimeError, match="p.a"):
        registry.discover("p")


def test_extract_python_takes_the_longest_fence():
    assert "class Harness" in extract_python("x\n```python\nclass Harness:\n    pass\n```\n```py\nx=1\n```")


def test_usage_context_tokens():
    assert Usage(input=10, cache_read=90).context_tokens() == 100
    assert Usage(input=100, cache_read=90, input_includes_cache=True).context_tokens() == 100
    total = Usage(input=1, output=2) + Usage(input=3, reasoning=4)
    assert (total.input, total.output, total.reasoning) == (4, 2, 4)
