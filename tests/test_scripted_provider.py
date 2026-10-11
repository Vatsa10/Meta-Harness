import json
import threading
import time

import pytest

from meta_harness.providers import ProviderError
from meta_harness.scripted import ScriptedModel


def script(tmp_path, rows):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return ScriptedModel(path)


def test_match_entries_are_reusable_and_first_match_wins(tmp_path):
    model = script(tmp_path, [{"match": "apple|pear", "text": "fruit"}, {"match": ".", "text": "vehicle"}])
    assert [model("an apple"), model("a car"), model("a pear")] == ["fruit", "vehicle", "fruit"]


def test_sequential_then_default_then_exhausted(tmp_path):
    model = script(tmp_path, [{"text": "one"}, {"default": "dflt"}])
    assert model("x") == "one" and model("x") == "dflt"
    bare = script(tmp_path, [{"text": "only"}])
    bare("x")
    with pytest.raises(ProviderError, match="exhausted"):
        bare("x")


def test_chat_replays_tool_calls_in_order(tmp_path):
    model = script(tmp_path, [
        {"tool_calls": [{"name": "list", "arguments": {"path": "."}}]},
        {"tool_calls": [{"name": "write_candidate", "arguments": {"source": "x", "rationale": "y"}}]},
    ])
    first = model.chat("s", [{"role": "user", "content": "go"}], [])
    second = model.chat("s", [{"role": "user", "content": "go"}], [])
    assert first.tool_calls[0].name == "list" and second.tool_calls[0].name == "write_candidate"
    assert first.tool_calls[0].id != second.tool_calls[0].id
    assert first.usage.input > 0 and first.usage.output == 0 and first.stop_reason == "tool_use"
    assert first.tool_calls[0].arguments == {"path": "."}
    assert second.tool_calls[0].arguments == {"source": "x", "rationale": "y"}
    assert model.model == "scripted:s.jsonl"


def test_error_entry_raises_with_status_then_next_call_proceeds(tmp_path):
    model = script(tmp_path, [{"error": {"status": 429, "message": "rate limited"}}, {"text": "ok"}])
    with pytest.raises(ProviderError, match="rate limited") as info:
        model("x")
    assert info.value.status == 429
    assert model("x") == "ok"


def test_length_entry_sets_stop_reason(tmp_path):
    model = script(tmp_path, [{"stop_reason": "length", "text": "cut"}, {"text": "done"}])
    turn = model.chat("s", [{"role": "user", "content": "go"}], [])
    assert turn.stop_reason == "length" and turn.text == "cut"
    assert turn.usage.output > 0
    assert model.chat("s", [{"role": "user", "content": "go"}], []).stop_reason == "end"


def test_call_skips_tool_call_entries_and_chat_falls_back_to_match(tmp_path):
    model = script(tmp_path, [{"tool_calls": [{"name": "t", "arguments": {}}]}, {"text": "plain"}])
    assert model("x") == "plain"
    assert model.chat("s", [{"role": "user", "content": "x"}], []).tool_calls[0].name == "t"
    fallback = script(tmp_path, [{"match": "hello", "text": "hi"}])
    assert fallback.chat("s", [{"role": "user", "content": "hello there"}], []).text == "hi"


class YieldingList(list):
    """Yields the GIL on every access so an unlocked cursor race becomes deterministic."""

    def __getitem__(self, index):
        time.sleep(0)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        time.sleep(0)
        super().__setitem__(index, value)


def test_cursor_is_thread_safe(tmp_path):
    model = script(tmp_path, [{"text": str(i)} for i in range(800)])
    model._consumed = YieldingList(model._consumed)
    seen, lock = [], threading.Lock()

    def worker():
        for _ in range(50):
            value = model("p")
            with lock:
                seen.append(value)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(seen, key=int) == [str(i) for i in range(800)]


def test_error_entry_reached_via_chat(tmp_path):
    model = script(tmp_path, [{"error": {"status": 503, "message": "down"}}, {"text": "ok"}])
    with pytest.raises(ProviderError, match="down") as info:
        model.chat("s", [{"role": "user", "content": "go"}], [])
    assert info.value.status == 503
    assert model.chat("s", [{"role": "user", "content": "go"}], []).text == "ok"


def test_malformed_line_names_file_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "ok"}

{not json}
', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl.*line 3"):
        ScriptedModel(path)
