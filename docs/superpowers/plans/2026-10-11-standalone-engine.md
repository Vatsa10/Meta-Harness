# Standalone Meta-Harness Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks inside one wave run as parallel subagents, each in its own git worktree; a wave starts only after every task of the previous wave is merged to `main` and `python -m pytest -q` passes there.

**Goal:** Turn the repository into a standalone, plug-and-play implementation of *Meta-Harness* (arXiv 2603.28052) covering every gap in `.superpowers/pivot/paper-gap.md`: a provider-agnostic tool-using proposer over an experience filesystem, correct evaluation, all three paper domains (online text classification with all datasets and OOD sets, retrieval-augmented math, TerminalBench-2 style agentic coding), the comparison optimizers under an equal budget, the Table 3 ablation, multi-model transfer, proposer read statistics, an experience query CLI and subprocess/Docker sandboxing - with no Claude Code dependency.

**Architecture:** The outer loop (`core.SearchRunner`) is kept. Shared interfaces land first (waves A-E): neutral chat types and provider tool calling, the harness/metric protocol, the confined tool layer, views, and the core evaluation fixes. Everything after that plugs in through four discovery registries created in Wave B, so parallel tasks never edit the same file:

| Registry | Package | A module provides |
|---|---|---|
| CLI commands | `meta_harness/commands/` | `NAME`, `HELP`, `add_arguments(parser)`, `run(args) -> int` |
| Domains | `meta_harness/domains/` | `NAME`, `PROMPT`, `METRICS`, `SEED_DIR`, `PROTOCOL`, `adapt(records)` |
| Dataset loaders | `meta_harness/loaders/` | `NAME`, `DOMAIN`, `fetch(cache_dir, http) -> dict[str, Path]` |
| Optimizers | `meta_harness/optimizers/` | `NAME`, `build(ctx: OptimizerContext) -> Proposer` |

**Tech Stack:** Python 3.11+ stdlib only (`urllib`, `zipfile`, `ctypes`, `difflib`, `sqlite3` allowed); pytest. Docker CLI optional at runtime.

**Spec:** `docs/superpowers/specs/2026-10-11-standalone-engine-design.md`. Evidence: `.superpowers/pivot/paper-gap.md` (gap rows referenced as G1..G23, ranked gaps as R1..R10).

## Waves and file ownership

A file listed under one task is owned by it for that wave; no other task in the same wave may edit it. "Shared" files are edited only in the named wave.

| Wave | Tasks (parallel) | Owned files (summary) |
|---|---|---|
| A | T1 delete the Claude Code layer | everything deleted; `__init__.py`, `__main__.py`, `pyproject.toml`, `tests/conftest.py` |
| B | T2 scaffolding: registries, chat types, `.env` loader, HTTP helpers, file moves | **shared:** `__main__.py`, `__init__.py`, `pyproject.toml`, `.gitignore`, `harnesses.py` split, `baselines/` layout, `meta_harness/{chat_types,envfile,registry,httpdata,redact}.py`, the four registry packages |
| C | T3 providers + groq + embeddings, T4 scripted provider, T5 protocol, T6 proposer tools, T7 views, T7A sandbox protocol | `providers.py`+`embeddings.py` / `scripted.py` / `protocol.py` / `proposer_tools.py` / `views.py` / `sandboxes.py`+`timeouts.py` |
| D | T8 eval fixes 1-2, T9 proposer agent + `metaharness` and `best-of-n` optimizers | `core.py`,`cache.py`,`cls_harnesses.py`,`math_harnesses.py`,`sandbox.py` / `proposer_agent.py`,`proposer_prompt.md`,`optimizers/metaharness.py`,`optimizers/best_of_n.py` |
| E | T10 core integration: protocol, probe, online, budget, equal-budget cap | `core.py`, `sandbox.py`, `metrics.py` |
| F | T11 search CLI, T12 classification loaders, T13 OOD loaders + eval, T14 discovered harnesses, T15 classification baselines, T16 math data, T17 math harness, T18 terminal, T19 OpenEvolve + TTT-Discover, T20 GEPA, T21 transfer, T22 experience CLI + read stats, T22A benchmark matrix + judge | each task's own new files; T15 owns `baselines/classification/*.py` (not `discovered/`) and `tests/test_baselines.py`; T18 owns `metrics.py`, `terminal.py`, `terminal_harness.py`; T11 owns `tests/test_cli.py`, `tests/test_smoke_end_to_end.py` |
| G | T23 isolation (subprocess limits, Docker), T24 ablate | `isolation.py`,`core.py`,`sandbox.py` / `commands/ablate.py` |
| H | T25 README and `examples/` | `README.md`, `baselines/README.md`, `examples/` |
| I | T26 live verification (owner keys) | `README.md` Results, one recorded fixture |

## Global Constraints

Copied from spec §9, plus the working rules this repo already enforces:

- Stdlib only: `dependencies = []` in `pyproject.toml` stays empty.
- Explicit `encoding="utf-8"` on every file read and write; every `subprocess.run(..., text=True)` also passes `encoding="utf-8", errors="replace"`.
- Windows-safe paths (`pathlib`, no hard-coded `/`, no shell-specific quoting).
- All work stays inside the repository folder; tests write only under `tmp_path`.
- No AI attribution in commits, authors, comments or docs. No `Co-Authored-By` or session trailer lines.
- Tests never call a real API and never touch the network: the `scripted` provider, recorded JSON fixtures, or an injected `http`/`fetch`/`post` callable only (`tests/conftest.py` fails any socket connect). Docker tests skip with a stated reason when `docker` is absent.
- Pushes go to `main`. `keel/` and `.superpowers/` are untracked: never `git add` them; stage by explicit path, never `git add -A`.
- API keys come from the environment or the git-ignored repo-root `.env` only and are never printed, written to any file, log, trace or `run.json`, or passed on a command line.
- Parallel tasks edit only the files their wave row assigns them; a task that discovers it needs a shared file stops and reports instead of editing it.
- Tests discriminate: each new test is verified to FAIL against the named deliberate mutation before commit; restore from a scratch copy, never `git checkout --` over uncommitted work.

## Review Focus

1. **Test split leaking to the proposer or to an optimizer.** `../`, absolute paths, drive-letter paths, UNC paths, symlinks and `grep` with a path argument are refused by `ToolSandbox.resolve`; `test_root` inside `root` is rejected at config time; non-agent optimizers (T19, T20) read only `FilesystemExperience`, never `test_root`. (T6, T10, T19, T20)
2. **Provider wire-format round trips.** Invalid JSON in OpenAI `function.arguments`, Anthropic turns mixing `text` and `tool_use`, several tool results grouped into one Anthropic user message; Groq and Unikey go through the same OpenAI path. (T3)
3. **Label leakage in the online protocol.** Search-eval and test tasks reach `Harness.run` with target fields removed and on a `deepcopy` of the trained harness. Math decontamination must remove search/test problems from the retrieval corpus before indexing. (T5, T10, T16)
4. **Cache vs repeats, and equal budget.** Repeat `r` misses repeat `r-1`'s cache entry but a re-run of the same repeat hits; every optimizer is stopped by the same `max_evaluations` counter, not by its own iteration count. (T8, T10, T19, T20)
5. **The `.env` loader and secrets.** `UNIKEY_API_KEY =value` (space before `=`), quotes, comments and blank lines parse; an already-set variable is never overridden; no value is ever printed, including in error messages and `run.json`. (T2, T11, T26)

## Ambiguities resolved (read before starting)

- **Harness signature.** Spec §3: `run(self, task, model) -> dict`, and "existing `harnesses.py` shape is kept" (`run(task, model, trace) -> str`). Both are accepted: `trace` is passed only when `run` takes three positional parameters; a non-dict return is wrapped as `{"output": value}`. Optional `learn(self, task, result)` (or `learn(self, task, result, model)`) runs after `run` on labelled stream items.
- **Metric signature.** `(task, result) -> float`. Built-ins compare `result["output"]` with the first present of `expected`, `label`, `answer`, `target`. Domains add named metrics through their `METRICS` dict.
- **`scores` view.** Spec §4 says "only `scores.json`"; the paper's Table 3 scores-only row is Scores+Code and a proposer that sees no source cannot edit it. `scores` exposes `scores.json` and `harness.py`, nothing else. Owner accepted this reading (matches the paper); stated in the README.
- **Providers.** `anthropic`, `openai`, `compatible` (spec), plus owner decisions: `unikey` (OpenAI-compatible, `UNIKEY_API_KEY`, `https://www.getunikey.ai/v1`) and `groq` (OpenAI-compatible, `GROQ_API_KEY`, `https://api.groq.com/openai/v1`), and `scripted`. None is a default; `--provider` is required. `claude-cli` is removed.
- **`.env`.** Loaded by the CLI before dispatch from `./.env` (override with the global `--env-file PATH`); missing file is fine; existing environment variables win.
- **Terminal domain.** §8 now puts TerminalBench-2 in scope, so T1 keeps `terminal.py`, `terminal_harness.py`, `terminal_metric` and their tests; T18 rebuilds them (fix 6 becomes "container recorded on the task, scoring runs inside it", per §8).
- **Dataset sources.** Verified on 2026-10-11 against the HF datasets-server: `gretelai/symptom_to_diagnosis`, `pingzhili/uspto-50k`, `nlpaueb/finer-139`, `SetFit/amazon_reviews_multi_en`, `google-research-datasets/go_emotions` (simplified), `fancyzhx/ag_news`, `allenai/scitail`, `cardiffnlp/tweet_eval` (hate) serve rows. `allenai/scicite`, `takala/financial_phrasebank`, `PolyAI/banking77` are script datasets the server refuses; their loaders download the original files (zip/CSV/JSONL) with `urllib` + `zipfile`. LawBench is on GitHub (`open-compass/LawBench`), not HF. Each loader task's Step 1 re-verifies its source and records the URL, fields and row counts in the module docstring; a source that has moved is replaced by the closest public copy and the deviation is written into that docstring and the README.
- **Download caches.** Everything fetched lands in `data/cache/<loader>/` (git-ignored). Spec §6 "cached under `data/`" is satisfied; third-party data is not committed.
- **Classification splits.** Each classification loader writes `train.jsonl`, `val.jsonl` (seeded 20% carve-out of train unless the source has a validation split: the search set), `test.jsonl`. `--protocol online --train train.jsonl` streams train into `learn`; candidates are scored frozen on val; the frontier once frozen on test.
- **Best-of-N "sees only the seed".** View `seed`: the seeds' `harness.py` and `scores.json` only.
- **TTT-Discover.** Weight updates at test time are impossible through a hosted API; T19 implements its PUCT tree search over candidates with the same proposer model, and the README says so.
- **Fixture provenance.** Provider fixtures are transcribed from published API reference shapes; T26 records one real OpenAI-compatible tool turn and adds it as a fixture.

---
### Task 1: Delete the Claude Code layer (Wave A)

**Files:**
- Delete (`git rm -r`): `hooks/`, `commands/`, `.claude-plugin/`, `agents/`, `skills/`, `tsconfig.json`, `meta_harness/cc_harness.py`, `meta_harness/cc_history.py`, `meta_harness/claude_cli.py`, `meta_harness/learn.py`, `meta_harness/learn_prompt.md`, `meta_harness/waste.py`, `meta_harness/drift.py`, `meta_harness/receipts.py`, `meta_harness/replay.py`, `meta_harness/temporal.py`, `meta_harness/harness_store.py`, `baselines/cc_default.py`, `baselines/cc_evidence.py`, `baselines/cc_guided.py`, `baselines/cc_subagent.py`, `tools/tb_run.py`, `tools/tune_drift.py`, `tools/prose_vs_rule.py`, `tools/llm_proposer.py` (defaults to the `claude` CLI; superseded by Task 9), `docs/plugin.md`, `docs/using-with-coding-agents.md`, `docs/baseline-2026-09-18.json`, `docs/discovered-harness.md`, `data/cc_failure_modes.jsonl`, `data/cc_tasks.jsonl`, `data/cc_tasks_hard.jsonl`.
- Delete tests: `tests/cause_fixtures.py`, `test_agent_proposer.py` (ClaudeCodeProposer; view tests return in Task 7), `test_cc_harness.py`, `test_cc_history.py`, `test_claude_cli.py`, `test_docs_claims.py`, `test_drift_heuristic.py`, `test_drift_knn.py`, `test_drift_model.py`, `test_harness_commands.py`, `test_harness_home_isolation.py`, `test_harness_store.py`, every `test_hook_*.py` (12 files), every `test_learn_*.py` (7 files), `test_plugin_validate.py`, `test_prose_vs_rule.py`, `test_receipts.py`, `test_receipts_cli.py`, every `test_replay_*.py` (3), `test_retirement.py`, `test_temporal.py`, `test_tune_drift.py`, every `test_waste_*.py` (3).
- Modify: `meta_harness/__init__.py`, `meta_harness/__main__.py`, `meta_harness/agent_proposer.py` (remove `ClaudeCodeProposer`, `PROPOSER_PROMPT`, `SKILL_PATH`; keep `build_view`, `VIEW_MODES` until Task 7), `meta_harness/metrics.py` (remove `agent_metric`), `meta_harness/datasets.py` (remove `agent_tasks`), `meta_harness/providers.py` (remove the `claude-cli` branch), `pyproject.toml`, `tests/conftest.py`, `tests/test_core.py`, `tests/test_metrics.py`, `tests/test_baselines.py`, `tests/test_cli.py`, `tests/test_smoke_end_to_end.py`, `baselines/README.md`.
- Untouched: `keel/`, `meta_harness/skill/SKILL.md` (moved in Task 9), the terminal domain files `meta_harness/terminal.py`, `meta_harness/terminal_harness.py`, `baselines/terminal_basic.py`, `tests/test_terminal_harness.py` (rebuilt in Task 18), `docs/superpowers/`, `paper.pdf`.

**Interfaces:**
- `python -m meta_harness --help` lists exactly: `demo`, `inspect`, `models`, `run`. (`run` becomes `search` in Task 11.)
- `models --provider unikey` only (choices `["unikey"]`, default `unikey`).
- `run --provider` choices `["unikey", "openai", "anthropic", "compatible"]`, `--model` required, `--proposer-command` required for now (the only proposer until Task 9). `--task-type` choices `["classification", "math", "terminal"]`. Removed flags: `--proposer-binary`, `--proposer-model`, `--proposer-view`, `--allow-local-shell`, `--workspace-root`, `--agent-bash`, `--mine-history`, `--history-limit`, `--history-include-text`.
- `pyproject.toml`: `[project.scripts] meta-harness = "meta_harness.__main__:main"` (kept); `package-data` becomes `meta_harness = ["*.md"]` (Task 2 widens it); description "Meta-Harness: end-to-end optimization of LLM harnesses over an experience filesystem".
- `tests/conftest.py` becomes empty of the harness-home fixtures (the harness home no longer exists). Replace with:

```python
"""Shared fixtures. Tests never reach the network: a guard fails any real socket connect."""
import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tests must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", refuse)
```

- [ ] **Step 1: Write the failing test** - create `tests/test_standalone.py`:

```python
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GONE = ["cc_harness", "cc_history", "claude_cli", "learn", "waste", "drift", "receipts",
        "replay", "temporal", "harness_store"]


@pytest.mark.parametrize("name", GONE)
def test_claude_code_modules_are_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"meta_harness.{name}")


@pytest.mark.parametrize("path", ["hooks", "commands", ".claude-plugin", "agents", "skills",
                                  "tsconfig.json", "tools/tb_run.py", "baselines/cc_default.py"])
def test_plugin_files_are_gone(path):
    assert not (ROOT / path).exists()


def test_help_lists_engine_commands_only():
    completed = subprocess.run([sys.executable, "-m", "meta_harness", "--help"], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert completed.returncode == 0, completed.stderr
    for gone in ("learn", "waste", "receipts", "mine", "compare"):
        assert f" {gone} " not in completed.stdout and f"{{{gone}" not in completed.stdout
    assert "run" in completed.stdout and "inspect" in completed.stdout


def test_package_imports_without_claude_binary(monkeypatch):
    monkeypatch.setenv("PATH", "")
    import meta_harness
    importlib.reload(meta_harness)
    assert not hasattr(meta_harness, "ClaudeCliModel")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest -q tests/test_standalone.py`
Expected: FAIL - `meta_harness.cc_harness` imports, `hooks/` exists.

- [ ] **Step 3: Delete and trim**

```bash
git rm -r -q hooks commands .claude-plugin agents skills tsconfig.json \
  meta_harness/cc_harness.py meta_harness/cc_history.py meta_harness/claude_cli.py \
  meta_harness/learn.py meta_harness/learn_prompt.md meta_harness/waste.py meta_harness/drift.py \
  meta_harness/receipts.py meta_harness/replay.py meta_harness/temporal.py meta_harness/harness_store.py \
  baselines/cc_default.py baselines/cc_evidence.py baselines/cc_guided.py baselines/cc_subagent.py \
  tools/tb_run.py tools/tune_drift.py tools/prose_vs_rule.py tools/llm_proposer.py \
  docs/plugin.md docs/using-with-coding-agents.md docs/baseline-2026-09-18.json docs/discovered-harness.md \
  data/cc_failure_modes.jsonl data/cc_tasks.jsonl data/cc_tasks_hard.jsonl \
  tests/cause_fixtures.py tests/test_agent_proposer.py tests/test_cc_harness.py tests/test_cc_history.py \
  tests/test_claude_cli.py tests/test_docs_claims.py tests/test_drift_*.py tests/test_harness_commands.py \
  tests/test_harness_home_isolation.py tests/test_harness_store.py tests/test_hook_*.py tests/test_learn_*.py \
  tests/test_plugin_validate.py tests/test_prose_vs_rule.py tests/test_receipts.py tests/test_receipts_cli.py \
  tests/test_replay_*.py tests/test_retirement.py tests/test_temporal.py \
  tests/test_tune_drift.py tests/test_waste_*.py
```

Then edit:
- `__init__.py`: drop imports/exports of `ClaudeCodeProposer`, `SKILL_PATH`, `AgentConfig`, `ClaudeCodeHarness`, `harness_report`, `load_sessions`, `project_slug`, `write_history_view`, `ClaudeCliError`, `ClaudeCliModel`, `Artifact`, `HarnessStore`, `harness_home` (terminal exports stay).
- `__main__.py`: delete imports of `ClaudeCodeProposer`, `cc_history`, `harness_store`, `cc_harness`, `learn`, `replay`, `temporal`, `agent_tasks`; delete `_command_mine`, `_command_compare`, `_enforces`, `_command_learn`, `_write_waste_json`, `_command_waste`, `_fmt_rate`, `_command_receipts`, `CLAUDE_CLI_MODELS`; `DEFAULT_BASELINES`/`ADAPTERS` keep `classification`, `math` and `terminal`; `_command_run` loses the shell/agent/history/claude-cli blocks and always builds `CommandProposer`; `models` lists Unikey ids only.
- `metrics.METRICS` drops `agent`: keys `{"classification", "math", "terminal"}`; `tests/test_metrics.py` asserts exactly that set.
- `tests/test_cli.py`: every `run` invocation passes `--provider unikey --model m` as now.

- [ ] **Step 4: Run the suite**

Run: `python -m pytest -q`
Expected: PASS (all remaining tests). Also `python -m meta_harness --help` exits 0 and `pip install -e . && meta-harness --help` prints the same.

- [ ] **Step 5: Discriminating mutation** - re-add `from .cc_history import load_sessions` at the top of `__main__.py` in a scratch copy: `test_claude_code_modules_are_gone` still passes but `test_help_lists_engine_commands_only` must FAIL (ModuleNotFoundError at import). Restore.

- [ ] **Step 6: Commit**

```bash
git add -u && git add tests/test_standalone.py tests/conftest.py
git commit -m "refactor!: remove the Claude Code layer; the engine stands alone"
```

---

### Task 2: Scaffolding - registries, chat types, `.env` loader, data helpers, file moves (Wave B)

Everything later waves share is created here, so parallel tasks only add files.

**Files:**
- Create: `meta_harness/registry.py`, `meta_harness/chat_types.py`, `meta_harness/envfile.py`, `meta_harness/httpdata.py`, `meta_harness/commands/__init__.py`, `meta_harness/commands/demo.py`, `meta_harness/commands/inspect.py`, `meta_harness/commands/models.py`, `meta_harness/commands/run.py` (the Task 1 `run` moved verbatim; T11 replaces it with `search.py` and deletes it), `meta_harness/domains/__init__.py`, `meta_harness/domains/generic.py`, `meta_harness/loaders/__init__.py`, `meta_harness/optimizers/__init__.py`, `meta_harness/cls_harnesses.py`, `meta_harness/math_harnesses.py`
- Move (`git mv`): `baselines/{zero_shot,few_shot,ace,mce}.py` -> `baselines/classification/`; `baselines/{math_zero_shot,math_bm25}.py` -> `baselines/math/{zero_shot,bm25}.py`; `baselines/terminal_basic.py` -> `baselines/terminal/basic.py`
- Modify (shared, this wave only): `meta_harness/__main__.py` (builds the parser from the command registry; global `--env-file`), `meta_harness/harnesses.py` (becomes a re-export shim of `cls_harnesses` and `math_harnesses`), `meta_harness/__init__.py`, `pyproject.toml` (`package-data`: `meta_harness = ["*.md", "demo_data/*"]`), `.gitignore` (`data/cache/`, `.env`), `tests/test_baselines.py`, `tests/test_cli.py` (paths only)
- Test: `tests/test_registry.py`, `tests/test_envfile.py`, `tests/test_httpdata.py`

**Interfaces:**

```python
# meta_harness/registry.py
def discover(package: str) -> dict[str, ModuleType]:
    """Import every non-underscore module of `package`; key by its NAME attribute.
    Duplicate NAMEs raise RuntimeError naming both modules. A module that fails to import
    raises - a broken plugin must not silently vanish."""

# meta_harness/chat_types.py  (moved out of providers so scripted.py and proposer code need no HTTP)
@dataclass(frozen=True)
class ToolSpec: name: str; description: str; parameters: dict[str, Any]
@dataclass(frozen=True)
class ToolCall: id: str; name: str; arguments: dict[str, Any]     # {"_raw": text} on invalid JSON
@dataclass
class AssistantTurn: text: str; tool_calls: list[ToolCall]; usage: dict[str, int]; stop_reason: str
# Neutral messages: {"role":"user","content":str} | {"role":"assistant","text":str,"tool_calls":[ToolCall]}
#                   | {"role":"tool","tool_call_id":str,"name":str,"content":str}

# meta_harness/envfile.py
def parse_env(text: str) -> dict[str, str]:
    """KEY=value lines. Strips whitespace around key and value (`UNIKEY_API_KEY =abc` works),
    an optional leading `export `, and one pair of matching surrounding quotes. Ignores blank
    lines, `#` comment lines and lines without `=`. Inline ` #` after an unquoted value is a comment."""
def load_env(path: Path | str = ".env", environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Sets variables that are not already set; returns the NAMES it set (never values).
    Missing file -> []. Read with encoding="utf-8-sig" (tolerates a BOM)."""

# meta_harness/httpdata.py  (stdlib download helpers shared by every loader)
Http = Callable[[str], bytes]
def urllib_get(url: str, timeout: float = 60.0, attempts: int = 3) -> bytes   # retries 429/5xx, UA header
def get_json(url: str, http: Http = urllib_get) -> Any
HF_ROWS = "https://datasets-server.huggingface.co/rows"
def hf_rows(dataset: str, config: str, split: str, http: Http = urllib_get, page: int = 100,
            limit: int | None = None) -> list[dict]:
    rows, offset = [], 0
    while limit is None or offset < limit:
        length = page if limit is None else min(page, limit - offset)
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": offset, "length": length})
        payload = get_json(f"{HF_ROWS}?{query}", http)
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(payload.get("num_rows_total", offset)):
            break
    return rows
def write_jsonl(path: Path, rows: Iterable[Mapping]) -> Path                    # utf-8, ensure_ascii=False
def read_jsonl(path: Path) -> list[dict]
def carve(rows: Sequence[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]  # seeded, disjoint
CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "cache"

# meta_harness/commands/__init__.py
def all_commands() -> dict[str, ModuleType]:  return discover("meta_harness.commands")

# meta_harness/domains/__init__.py
def all_domains() -> dict[str, ModuleType]
def get_domain(name: str) -> ModuleType          # KeyError lists known names
# Domain module contract:
#   NAME: str; PROMPT: str (domain notes appended to the proposer prompt)
#   METRICS: dict[str, Callable[[Mapping, Mapping], float]]; SEED_DIR: Path | None
#   PROTOCOL: "plain" | "online"; def adapt(records: list[dict]) -> list[dict]
# domains/generic.py: NAME="generic", PROMPT="", METRICS={}, SEED_DIR=None, PROTOCOL="plain",
#   adapt = lambda records: [dict(r) for r in records]

# meta_harness/loaders/__init__.py
def all_loaders() -> dict[str, ModuleType]
# Loader contract: NAME: str; DOMAIN: str; SOURCE: str (URL, documented);
#   def fetch(cache_dir: Path, http: Http = urllib_get) -> dict[str, Path]   # split name -> JSONL
#   Uses the cache when every split file exists. Rows are already domain tasks.

# meta_harness/optimizers/__init__.py
Proposer = Callable[["FilesystemExperience", int, int], list[Path]]
@dataclass
class OptimizerContext:
    proposer_model: Any                 # .chat(...) for tool optimizers, __call__(prompt) for text ones
    domain: ModuleType
    view: str = "full"
    max_tool_calls: int = 40
    seed: int = 0
    forbidden: list[Path] = field(default_factory=list)
    summarizer: Callable[[str], str] | None = None
def all_optimizers() -> dict[str, ModuleType]
def build(name: str, ctx: OptimizerContext) -> Proposer
def extract_python(text: str) -> str    # moved from proposer.py (fenced block or bare module); proposer.py re-exports
```

`__main__.main(argv)`: a pre-parser reads `--env-file` (default `.env` in the current directory), calls `load_env`, then builds subparsers from `all_commands()` (sorted by name) and dispatches to `module.run(args)`. `harnesses.py` keeps `from .cls_harnesses import *` / `from .math_harnesses import *` so existing imports work; `cls_harnesses.py` holds `MemoryExample`, `LabelPrimedQueryHarness`, `DraftVerificationHarness`, `_call`, `_label`; `math_harnesses.py` holds `MathRetrievalHarness`; `TerminalBootstrap`/`environment_snapshot` move to `terminal_harness.py` (owned by T18 later).

**Spec §11 additions (provider layer, shared types):** `chat_types.py` also defines

```python
@dataclass
class Usage:
    input: int = 0; output: int = 0; cache_read: int = 0; cache_write: int = 0; reasoning: int = 0
    input_includes_cache: bool = False      # OpenAI: True (prompt_tokens includes cached); Anthropic: False
    def context_tokens(self) -> int:         # the Pareto axis: everything the model read
        return self.input if self.input_includes_cache else self.input + self.cache_read + self.cache_write
    def __add__(self, other: "Usage") -> "Usage": ...
```

`AssistantTurn.usage` is a `Usage`; `AssistantTurn` gains `provider_blocks: list[dict]` (raw provider content kept for echo, e.g. Anthropic `thinking` blocks with signatures) and `stop_reason` is normalised to `"end" | "tool_use" | "length" | "other"`. Add to `tests/test_registry.py`: `Usage(input=10, cache_read=90).context_tokens() == 100` and `Usage(input=100, cache_read=90, input_includes_cache=True).context_tokens() == 100`.

**Spec §11 additions (keel: redaction, torn lines):** create `meta_harness/redact.py`:

```python
SECRET = re.compile(r"(?i)(api[_-]?key\S*\s*[:=]\s*\S+|bearer\s+\S+|sk-[A-Za-z0-9_-]{8,})")
def redact(text: str) -> str            # every SECRET match -> "[redacted]"; also exact values of any
                                        # loaded *_API_KEY/*_TOKEN env var (>= 8 chars)
def bounded(text: str, limit: int = 2000) -> str   # redact, collapse to one line, cap with "...[+N chars]"
def safe_id(text: str) -> str           # ASCII [A-Za-z0-9._-], <= 64 chars, sha256 suffix when shortened
def redact_obj(value: Any) -> Any       # recursive over dict/list/str
```

`httpdata.read_jsonl(path, tolerate_torn=True)` skips a final line that fails to parse (torn write) and raises on a bad line anywhere else. Tests (in `tests/test_envfile.py`): `redact("Authorization: Bearer abc.def")`, `"api_key=xyz"`, `"sk-proj-1234567890"` and an env value set via `load_env` all become `[redacted]`; `bounded` output has no newline and respects the cap; `safe_id` of a 200-char unicode string is ASCII and <= 64; a JSONL with a torn last line loads all complete rows, a torn middle line raises. Mutation: drop the env-value pass -> the loaded-key test FAILS.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_envfile.py
import os
from meta_harness.envfile import load_env, parse_env


def test_parse_handles_spaces_quotes_comments_export():
    text = ("# keys\n\nUNIKEY_API_KEY =abc123\nGROQ_API_KEY= 'g k'\nexport OPENAI_API_KEY=\"o\"\n"
            "NOEQUALS\nX=1 # trailing\nEMPTY=\n")
    assert parse_env(text) == {"UNIKEY_API_KEY": "abc123", "GROQ_API_KEY": "g k",
                               "OPENAI_API_KEY": "o", "X": "1", "EMPTY": ""}


def test_load_never_overrides_and_returns_names_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("﻿A=from-file\nB=from-file\n", encoding="utf-8")
    environ = {"A": "already"}
    assert load_env(env_file, environ) == ["B"]
    assert environ == {"A": "already", "B": "from-file"}
    assert load_env(tmp_path / "missing.env", environ) == []


def test_cli_loads_env_before_dispatch(tmp_path, monkeypatch, capsys):
    from meta_harness import __main__ as cli
    (tmp_path / "k.env").write_text("META_HARNESS_TEST_SECRET = s3cret\n", encoding="utf-8")
    monkeypatch.delenv("META_HARNESS_TEST_SECRET", raising=False)
    assert cli.main(["--env-file", str(tmp_path / "k.env"), "inspect", str(tmp_path / "none")]) == 0
    assert os.environ["META_HARNESS_TEST_SECRET"] == "s3cret"
    assert "s3cret" not in capsys.readouterr().out
```

```python
# tests/test_registry.py
import types
import pytest
from meta_harness import registry
from meta_harness.commands import all_commands
from meta_harness.domains import get_domain
from meta_harness.loaders import all_loaders
from meta_harness.optimizers import all_optimizers, extract_python


def test_builtin_commands_are_discovered():
    assert {"demo", "inspect", "models", "run"} <= set(all_commands())


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
```

```python
# tests/test_httpdata.py
import json
from urllib.parse import parse_qs, urlparse
from meta_harness.httpdata import carve, hf_rows


def test_hf_rows_pages_and_respects_limit():
    seen = []
    def http(url):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        seen.append(int(q["offset"]))
        start, length = int(q["offset"]), int(q["length"])
        rows = [{"row": {"i": i}} for i in range(start, min(start + length, 5))]
        return json.dumps({"rows": rows, "num_rows_total": 5}).encode("utf-8")
    assert [r["i"] for r in hf_rows("d", "c", "train", http, page=2)] == [0, 1, 2, 3, 4]
    assert seen == [0, 2, 4]
    assert len(hf_rows("d", "c", "train", http, page=2, limit=3)) == 3


def test_carve_is_seeded_and_disjoint():
    rows = [{"i": i} for i in range(10)]
    a1, b1 = carve(rows, 0.2, 0)
    a2, b2 = carve(rows, 0.2, 0)
    assert (a1, b1) == (a2, b2) and len(b1) == 2
    assert not {r["i"] for r in a1} & {r["i"] for r in b1}
```

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_envfile.py tests/test_registry.py tests/test_httpdata.py` - FAIL (modules missing).
- [ ] **Step 3: Implement** the interfaces and moves. Update `tests/test_baselines.py` paths (`baselines/classification/zero_shot.py`, `baselines/math/bm25.py`, `baselines/terminal/basic.py`) and `DEFAULT_BASELINES` in `commands/run.py`.
- [ ] **Step 4: Run** `python -m pytest -q` - PASS; `python -m meta_harness --help` lists `demo inspect models run`.
- [ ] **Step 5: Discriminating mutation** - let `load_env` overwrite existing variables: `test_load_never_overrides...` FAILS. Strip only the value, not the key: the `UNIKEY_API_KEY =` case FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness tests baselines pyproject.toml .gitignore
git commit -m "refactor: plugin registries, neutral chat types, .env loader and data helpers"
```

---
### Task 3: Provider tool-calling (Anthropic, OpenAI-compatible, unikey, groq) and embeddings (Wave C)

**Files:**
- Modify: `meta_harness/providers.py`
- Create: `meta_harness/embeddings.py`, `tests/fixtures/providers/anthropic_tool_use.json`, `anthropic_text.json`, `openai_tool_calls.json`, `openai_text.json`, `openai_bad_arguments.json`
- Test: `tests/test_provider_tools.py`; update `tests/test_providers.py` (the `claude-cli` case is gone already)

**Interfaces** (in `providers.py` unless noted; the dataclasses below live in `chat_types.py` since Task 2, are re-exported here and are shown for reference):

- `GroqModel(OpenAICompatibleModel)`: `DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"`, `API_KEY_ENV = "GROQ_API_KEY"`, `BASE_URL_ENV = "GROQ_BASE_URL"`. `UnikeyModel` stays (OpenAI-compatible root `https://www.getunikey.ai/v1`); the Unikey Anthropic-compatible root is reachable as `--provider anthropic --base-url https://www.getunikey.ai`.
- `list_models(provider: str, **kw) -> list[str]`: `GET <base>/models` for `openai|compatible|unikey|groq` (parses `data[].id`). Task 2's `models` command calls it; `list_unikey_models` stays as a thin alias.
- Extra provider kwargs (`temperature`, `reasoning_effort`, `top_p`) pass through `call`/`chat` unchanged, so the paper's reasoning-effort settings are reproducible (gap G23).
- `embeddings.py`: `embed(texts: Sequence[str], provider: str = "openai", model: str = "text-embedding-3-small", post=None) -> list[list[float]]` over `POST <base>/embeddings` (OpenAI-compatible providers); `cosine(a, b) -> float`; missing key -> `ProviderError` naming the variable.

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema, type "object"

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]           # {"_raw": "<text>"} when the provider sent invalid JSON

@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    usage: dict[str, int]               # {"input_tokens": n, "output_tokens": m}
    stop_reason: str

# Neutral message list used by the proposer agent:
#   {"role": "user", "content": str}
#   {"role": "assistant", "text": str, "tool_calls": list[ToolCall]}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}

def anthropic_request(model: str, system: str, messages: list[dict], tools: Sequence[ToolSpec],
                      max_tokens: int) -> dict: ...
def parse_anthropic(response: Mapping[str, Any]) -> AssistantTurn: ...
def openai_request(model: str, system: str, messages: list[dict], tools: Sequence[ToolSpec],
                   max_tokens: int) -> dict: ...
def parse_openai(response: Mapping[str, Any]) -> AssistantTurn: ...
```

`AnthropicModel.chat(system, messages, tools, max_tokens=4096) -> AssistantTurn` and `OpenAICompatibleModel.chat(...)` (inherited by `UnikeyModel`) post the request with the existing `_HttpClient` and parse. `call()` / `__call__` are unchanged.

Translation rules (the tricky part; the drafted parsers below build `usage` as a dict - build the Task 2 `Usage` record instead, per the §11 additions):

```python
def anthropic_request(model, system, messages, tools, max_tokens):
    wire: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            wire.append({"role": "user", "content": str(message["content"])})
        elif role == "assistant":
            blocks = [{"type": "text", "text": message["text"]}] if message.get("text") else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                       for c in message.get("tool_calls", [])]
            wire.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                     "content": str(message["content"])}
            # Several results for one assistant turn go in ONE user message.
            if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        else:
            raise ValueError(f"unknown role {role!r}")
    body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": wire}
    if tools:
        body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                         for t in tools]
    return body


def parse_anthropic(response):
    try:
        blocks = list(response["content"])
    except (KeyError, TypeError) as error:
        raise ProviderError(f"unexpected messages response: {str(response)[:500]}") from error
    text = "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
    calls = [ToolCall(str(b["id"]), str(b["name"]), dict(b.get("input") or {}))
             for b in blocks if b.get("type") == "tool_use"]
    usage = response.get("usage") or {}
    return AssistantTurn(text, calls, {"input_tokens": int(usage.get("input_tokens") or 0),
                                       "output_tokens": int(usage.get("output_tokens") or 0)},
                         str(response.get("stop_reason") or ""))


def openai_request(model, system, messages, tools, max_tokens):
    wire: list[dict] = [{"role": "system", "content": system}] if system else []
    for message in messages:
        role = message["role"]
        if role == "user":
            wire.append({"role": "user", "content": str(message["content"])})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": message.get("text") or None}
            if message.get("tool_calls"):
                item["tool_calls"] = [{"id": c.id, "type": "function",
                                       "function": {"name": c.name,
                                                    "arguments": c.arguments.get("_raw")
                                                    if set(c.arguments) == {"_raw"}
                                                    else json.dumps(c.arguments)}}
                                      for c in message["tool_calls"]]
            wire.append(item)
        elif role == "tool":
            wire.append({"role": "tool", "tool_call_id": message["tool_call_id"],
                         "content": str(message["content"])})
        else:
            raise ValueError(f"unknown role {role!r}")
    body = {"model": model, "messages": wire, "max_tokens": max_tokens}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                           "parameters": t.parameters}} for t in tools]
    return body


def parse_openai(response):
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ProviderError(f"unexpected chat completion response: {str(response)[:500]}") from error
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        text = function.get("arguments") or "{}"
        try:
            arguments = json.loads(text) if isinstance(text, str) else dict(text)
            if not isinstance(arguments, dict):
                arguments = {"_raw": text}
        except json.JSONDecodeError:
            arguments = {"_raw": text}
        calls.append(ToolCall(str(raw.get("id", "")), str(function.get("name", "")), arguments))
    usage = response.get("usage") or {}
    return AssistantTurn(str(message.get("content") or ""), calls,
                         {"input_tokens": int(usage.get("prompt_tokens") or 0),
                          "output_tokens": int(usage.get("completion_tokens") or 0)},
                         str(choice.get("finish_reason") or ""))
```

`model_from_environment(provider, model, **kw)` accepts `anthropic`, `openai`, `compatible` (requires `OPENAI_BASE_URL` or `base_url=`; raises `ValueError("compatible needs OPENAI_BASE_URL or --base-url")` otherwise), `unikey`, `groq`, and `scripted` (lazy `from .scripted import ScriptedModel`; the class itself is Task 4). A missing key raises `ProviderError(f"{API_KEY_ENV} is not set")` at construction for the HTTP providers, never later mid-run.

**Spec §11 additions (provider layer):**
- Usage parsing fills the `Usage` record from Task 2: Anthropic `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` (`input_includes_cache=False`); OpenAI `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens` (`input_includes_cache=True`). Billed cost is separate from context tokens: `cost_usd(usage, price: Price) -> float` with `Price(input, output, cache_read, cache_write)` per million tokens, from `--price` flags or `META_HARNESS_PRICES` JSON; unknown price -> `None`, never a guess.
- Normalisation: `finish_reason == "length"` / `stop_reason == "max_tokens"` -> `stop_reason="length"`; Anthropic `thinking`/`redacted_thinking` blocks are kept in `provider_blocks` and echoed back unchanged on the next request.
- Orphaned tool calls: both request builders insert `{"role":"tool", ..., "content":"error: no result (turn ended)"}` for any assistant tool call that has no matching tool message before the next user/assistant message, so a request never carries an unanswered call.
- Retries (`_HttpClient`): only HTTP 429 and 5xx (and connection errors/timeouts of the HTTP request itself) are retried with backoff; 408/409 and every 4xx are not. Harness errors and evaluation timeouts are never retried anywhere.
- Tests to add: cache fields parsed from both fixtures (add `cache_read_input_tokens` to `anthropic_tool_use.json` and `prompt_tokens_details` to `openai_tool_calls.json`); a `max_tokens` fixture yields `stop_reason == "length"`; a thinking block round-trips into the next request body; an orphaned call gets the inserted error result in both wire formats; `_HttpClient` retries a 503 then succeeds and does not retry a 400 (fake `urlopen`); `cost_usd` arithmetic and `None` for unknown prices. Mutation: let 400 retry -> the no-retry test FAILS.

**Spec §11 additions (keel retry):** `_HttpClient` honours `Retry-After` (seconds or HTTP date, capped at 60 s); otherwise backoff `min(10, 0.5 * 2**attempt)` seconds with +/-10% jitter from an injectable `random.Random`; retryable classes are exactly EMPTY_RESPONSE (200 with empty body or no choices/content), RATE_LIMIT (429), SERVER (5xx), TIMEOUT, TRANSPORT (`URLError`, connection reset) - provider calls only. Tests with a fake `urlopen` and a recording `sleep`: `Retry-After: 3` sleeps 3; jitter stays within 10%; an empty 200 body is retried; a 400 is not.

- [ ] **Step 1: Fixtures.** Write each JSON file as one full provider response. `anthropic_tool_use.json`:

```json
{"id": "msg_01", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
 "content": [{"type": "text", "text": "Reading the frontier first."},
             {"type": "tool_use", "id": "toolu_01", "name": "read", "input": {"path": "frontier.json"}},
             {"type": "tool_use", "id": "toolu_02", "name": "list", "input": {"path": "candidates"}}],
 "stop_reason": "tool_use", "usage": {"input_tokens": 812, "output_tokens": 64}}
```

`openai_tool_calls.json`:

```json
{"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-5-mini",
 "choices": [{"index": 0, "finish_reason": "tool_calls",
   "message": {"role": "assistant", "content": null,
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "grep", "arguments": "{\"pattern\": \"label\", \"path\": \"candidates\"}"}}]}}],
 "usage": {"prompt_tokens": 700, "completion_tokens": 30, "total_tokens": 730}}
```

`openai_bad_arguments.json`: same as above with `"arguments": "{\"pattern\": "`. `anthropic_text.json` / `openai_text.json`: a plain text reply with no tool calls.

- [ ] **Step 2: Write the failing test** - `tests/test_provider_tools.py`:

```python
import json
from pathlib import Path

import pytest

from meta_harness import providers
from meta_harness.providers import ToolCall, ToolSpec

FIX = Path(__file__).parent / "fixtures" / "providers"
READ = ToolSpec("read", "read a file", {"type": "object", "properties": {"path": {"type": "string"}},
                                         "required": ["path"]})


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_parse_anthropic_mixed_text_and_tools():
    turn = providers.parse_anthropic(load("anthropic_tool_use.json"))
    assert turn.text == "Reading the frontier first."
    assert [c.name for c in turn.tool_calls] == ["read", "list"]
    assert turn.tool_calls[0].arguments == {"path": "frontier.json"}
    assert (turn.usage.input, turn.usage.output, turn.usage.input_includes_cache) == (812, 64, False)


def test_anthropic_groups_tool_results_into_one_user_message():
    calls = [ToolCall("toolu_01", "read", {"path": "a"}), ToolCall("toolu_02", "list", {"path": "."})]
    body = providers.anthropic_request("m", "sys", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "toolu_01", "name": "read", "content": "A"},
        {"role": "tool", "tool_call_id": "toolu_02", "name": "list", "content": "B"},
    ], [READ], 1024)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in body["messages"][2]["content"]] == ["toolu_01", "toolu_02"]
    assert body["system"] == "sys" and body["tools"][0]["input_schema"]["required"] == ["path"]


def test_parse_openai_tool_calls_decodes_arguments():
    turn = providers.parse_openai(load("openai_tool_calls.json"))
    assert turn.text == ""
    assert turn.tool_calls == [ToolCall("call_1", "grep", {"pattern": "label", "path": "candidates"})]
    assert (turn.usage.input, turn.usage.output, turn.usage.input_includes_cache) == (700, 30, True)


def test_parse_openai_keeps_invalid_arguments_raw():
    turn = providers.parse_openai(load("openai_bad_arguments.json"))
    assert set(turn.tool_calls[0].arguments) == {"_raw"}


def test_openai_round_trip_reencodes_arguments_as_string():
    turn = providers.parse_openai(load("openai_tool_calls.json"))
    body = providers.openai_request("m", "sys", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "name": "grep", "content": "hits"},
    ], [READ], 512)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    sent = body["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(sent, str) and json.loads(sent) == {"pattern": "label", "path": "candidates"}
    assert body["messages"][3]["role"] == "tool"
    assert body["tools"][0]["type"] == "function"


@pytest.mark.parametrize("cls,fixture,url_tail", [
    (providers.AnthropicModel, "anthropic_tool_use.json", "/v1/messages"),
    (providers.OpenAICompatibleModel, "openai_tool_calls.json", "/chat/completions"),
    (providers.UnikeyModel, "openai_tool_calls.json", "/chat/completions"),
])
def test_chat_posts_and_parses(monkeypatch, cls, fixture, url_tail):
    monkeypatch.setenv(cls.API_KEY_ENV, "k")
    model = cls("m")
    sent = []
    model.client.post = lambda url, headers, body: sent.append((url, body)) or load(fixture)
    turn = model.chat("sys", [{"role": "user", "content": "go"}], [READ])
    assert sent[0][0].endswith(url_tail)
    assert turn.tool_calls


def test_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="ANTHROPIC_API_KEY"):
        providers.model_from_environment("anthropic", "m")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest -q tests/test_provider_tools.py`
Expected: FAIL - `ImportError: cannot import name 'ToolCall'`.

- [ ] **Step 4: Implement** the interfaces above. Add `chat` to `AnthropicModel` and `OpenAICompatibleModel`; the key check in each `__init__`. Update `tests/test_providers.py` fakes to set the key env var where they construct models.

- [ ] **Step 5: Run**

Run: `python -m pytest -q tests/test_provider_tools.py tests/test_providers.py` (with added cases: `GroqModel` base URL and env name; `embed` against a recorded `/embeddings` body via a fake `post`; `list_models` parsing `data[].id`)
Expected: PASS.

- [ ] **Step 6: Discriminating mutation** - in a scratch copy, make `anthropic_request` always append a new user message for each tool result: `test_anthropic_groups_tool_results_into_one_user_message` must FAIL. Separately, send `json.dumps` of the `{"_raw": ...}` dict in `openai_request`: no test catches it - add `test_openai_invalid_arguments_are_resent_verbatim` asserting the raw string is resent unchanged, then confirm it fails under that mutation. Restore.

- [ ] **Step 7: Commit**

```bash
git add meta_harness/providers.py meta_harness/embeddings.py tests/test_provider_tools.py tests/test_providers.py tests/fixtures/providers
git commit -m "feat(providers): tool calling for Anthropic and OpenAI-compatible APIs, groq and embeddings"
```

---

### Task 4: The `scripted` provider (Wave C)

**Files:**
- Create: `meta_harness/scripted.py` (imports only `chat_types` and `metrics.estimate_tokens`). Task 3 owns the `model_from_environment` branch that lazily imports it; this task's tests construct `ScriptedModel(path)` directly (replace `providers.model_from_environment("scripted", str(path))` in the drafted helper with `ScriptedModel(path)`).
- Test: `tests/test_scripted_provider.py`

**Interfaces:**

```python
class ScriptedModel:
    """Replays canned responses from a JSONL file. For offline tests and demos.

    Each line is one entry:
      {"match": "<regex>", "text": "..."}            reusable; first entry whose regex is found in
                                                    the prompt (call) or last user/tool message (chat)
      {"text": "...", "tool_calls": [{"name": ..., "arguments": {...}}]}
                                                    sequential; consumed once, in file order
      {"default": "..."}                             used by call() when nothing else applies
    usage is estimated as estimate_tokens(prompt) / estimate_tokens(text).
    """
    def __init__(self, path: str | Path): ...
    model: str                                       # "scripted:<file name>"
    def call(self, prompt: str, **kwargs) -> tuple[str, dict[str, int]]: ...
    def chat(self, system: str, messages: list[dict], tools: Sequence[ToolSpec],
             max_tokens: int = 4096) -> AssistantTurn: ...
    def __call__(self, prompt: str, **kwargs) -> str: ...
```

`call` order: match entries, then the next sequential entry that has no `tool_calls`, then `default`, else `ProviderError("script exhausted: <path>")`. `chat` order: next sequential entry (with or without tool calls), else a match entry against the last message's text, else `ProviderError`. Tool call ids are `f"scripted-{n}"`. A thread lock guards the cursor (candidates are evaluated concurrently). `model_from_environment("scripted", "<path>")` builds it; `--model` is the script path.

**Spec §11 additions:** entries may also be `{"error": {"status": 429, "message": "rate limited"}}`, raised as `ProviderError` carrying `.status` when reached in FIFO order (so retry and failure paths are testable offline), and `{"stop_reason": "length", "text": "..."}` to simulate truncation. Usage is returned as the Task 2 `Usage` record. Tests: an error entry raises with `.status == 429` and the next call proceeds; a length entry yields `stop_reason == "length"`.

- [ ] **Step 1: Write the failing test**

```python
import json
import threading
from pathlib import Path

import pytest

from meta_harness import providers


def script(tmp_path, rows):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return providers.model_from_environment("scripted", str(path))


def test_match_entries_are_reusable_and_first_match_wins(tmp_path):
    model = script(tmp_path, [{"match": "apple|pear", "text": "fruit"}, {"match": ".", "text": "vehicle"}])
    assert [model("an apple"), model("a car"), model("a pear")] == ["fruit", "vehicle", "fruit"]


def test_sequential_then_default_then_exhausted(tmp_path):
    model = script(tmp_path, [{"text": "one"}, {"default": "dflt"}])
    assert model("x") == "one" and model("x") == "dflt"
    bare = script(tmp_path, [{"text": "only"}])
    bare("x")
    with pytest.raises(providers.ProviderError, match="exhausted"):
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
    assert first.usage["input_tokens"] > 0


def test_cursor_is_thread_safe(tmp_path):
    model = script(tmp_path, [{"text": str(i)} for i in range(200)])
    seen, lock = [], threading.Lock()
    def worker():
        for _ in range(50):
            value = model("p")
            with lock:
                seen.append(value)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(seen, key=int) == [str(i) for i in range(200)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/test_scripted_provider.py`
Expected: FAIL - `ValueError: unsupported provider: scripted`.

- [ ] **Step 3: Implement** `ScriptedModel` and its `model_from_environment` branch.

- [ ] **Step 4: Run** `python -m pytest -q tests/test_scripted_provider.py tests/test_provider_tools.py` - PASS.

- [ ] **Step 5: Discriminating mutation** - remove the lock: `test_cursor_is_thread_safe` should FAIL (duplicates); if it passes on this machine, raise worker count to 16 and iterations so it does, then restore. Mutation 2: consume match entries once - `test_match_entries_are_reusable...` FAILS.

- [ ] **Step 6: Commit**

```bash
git add meta_harness/scripted.py tests/test_scripted_provider.py
git commit -m "feat(providers): scripted provider replays canned responses offline"
```

---

### Task 5: Harness and metric contracts and the online protocol (`protocol.py`) (Wave C)

**Files:**
- Create: `meta_harness/protocol.py` (pure; stdlib plus `metrics.math_equivalence`). Core, sandbox and metrics are wired to it in Task 10.
- Test: `tests/test_protocol.py`, `tests/test_online_protocol.py`

Also in `protocol.py` (online protocol, spec §5.3):

```python
def stream_train(harness, train_tasks, model, trace) -> None:
    for task in train_tasks:                     # labelled; prequential: predict then learn
        trace.event("train_task", {"input": task.get("input")})
        result = call_run(harness, task, model, trace)
        call_learn(harness, task, result, model)

def evaluate_frozen(harness, tasks, model, trace, metric) -> list[tuple[float, dict]]:
    out = []
    for task in tasks:
        frozen = copy.deepcopy(harness)          # one test item never teaches the next
        result = call_run(frozen, hide_target(task), model, trace)
        out.append((float(metric(task, result)), result))
    return out

def resolve_metric(spec: str, base: Path = Path.cwd(), extra: Mapping[str, Metric] = {}) -> Metric
    # `extra` = the selected domain's METRICS, checked after the built-ins
```

`tests/test_online_protocol.py`:

```python
from meta_harness import protocol

class Memorizer:
    def __init__(self): self.memory = {}
    def run(self, task, model): return {"output": self.memory.get(task["input"], "unknown")}
    def learn(self, task, result): self.memory[task["input"]] = task["label"]

class Trace:
    def event(self, *a, **k): pass

def test_test_items_are_hidden_and_never_learned():
    h = Memorizer()
    protocol.stream_train(h, [{"input": "a", "label": "x"}], None, Trace())
    tests = [{"input": "b", "label": "y"}, {"input": "b", "label": "y"}, {"input": "a", "label": "x"}]
    scores = [s for s, _ in protocol.evaluate_frozen(h, tests, None, Trace(), protocol.resolve_metric("exact"))]
    assert scores == [0.0, 0.0, 1.0]

def test_label_is_removed_before_run():
    class Peeker:
        def run(self, task, model):
            assert "label" not in task and "answer" not in task
            return {"output": ""}
    protocol.evaluate_frozen(Peeker(), [{"input": "q", "label": "l"}], None, Trace(),
                             protocol.resolve_metric("exact"))
```

Mutations for these: drop the `deepcopy` and learn after each test item -> the first test FAILS; drop `hide_target` -> the second FAILS.

**Interfaces:**

```python
TARGET_FIELDS = ("expected", "label", "answer", "target")

def expected_value(task: Mapping) -> Any            # first present TARGET_FIELDS value, else None
def hide_target(task: Mapping) -> dict              # copy without TARGET_FIELDS
def call_run(harness, task, model, trace) -> dict:
    params = [p for p in inspect.signature(harness.run).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    value = harness.run(task, model, trace) if len(params) >= 3 else harness.run(task, model)
    return dict(value) if isinstance(value, Mapping) and "output" in value else {"output": value}
def call_learn(harness, task, result, model) -> None   # no-op without learn; passes model if 3 params

Metric = Callable[[Mapping, Mapping], float]          # (task, result) -> float
def exact(task, result) -> float                      # case/space-insensitive str equality
def contains(task, result) -> float                   # expected (lowercased) is a substring of output
def numeric(task, result) -> float                    # metrics.math_equivalence(output, {"answer": expected})
BUILTIN_METRICS = {"exact": exact, "contains": contains, "numeric": numeric}

def resolve_metric(spec: str, base: Path = Path.cwd()) -> Metric:
    if spec in BUILTIN_METRICS: return BUILTIN_METRICS[spec]
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name or (len(target) == 1 and target.isalpha()):
        raise ValueError(f"metric must be one of {sorted(BUILTIN_METRICS)} or module:function, got {spec!r}")
    if target.endswith(".py"):     # file path, Windows drive letters survive rpartition
        module = _load_file((base / target) if not Path(target).is_absolute() else Path(target))
    else:
        sys.path.insert(0, str(base)) if str(base) not in sys.path else None
        module = importlib.import_module(target)
    fn = getattr(module, name, None)
    if not callable(fn): raise ValueError(f"{spec!r} is not a callable")
    return fn
```

`CandidateEvaluator(model, metric: Metric)`: per task `result = call_run(...)`, `score = float(metric(task, result))`, then `call_learn(harness, task, result, metered)` (plain protocol: prequential, as before). `metrics.METRICS` is replaced by `BUILTIN_METRICS` re-exported; `exact_match`/`math_equivalence` stay as helpers. Existing tests passing `lambda p, t:` metrics are updated to `lambda t, r:`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from meta_harness import protocol

class Old:
    def run(self, task, model, trace):
        trace.event("x")
        return "Fruit"

class New:
    def __init__(self): self.seen = []
    def run(self, task, model): return {"output": "fruit", "why": "w"}
    def learn(self, task, result): self.seen.append((task["label"], result["output"]))

class Trace:
    def event(self, *a, **k): pass

def test_both_signatures_normalise_to_output_dict():
    assert protocol.call_run(Old(), {}, None, Trace()) == {"output": "Fruit"}
    assert protocol.call_run(New(), {}, None, Trace())["why"] == "w"

def test_learn_is_optional_and_receives_result():
    h = New()
    protocol.call_learn(h, {"label": "fruit"}, {"output": "fruit"}, None)
    protocol.call_learn(Old(), {"label": "x"}, {"output": "y"}, None)
    assert h.seen == [("fruit", "fruit")]

@pytest.mark.parametrize("name,task,output,score", [
    ("exact", {"label": "Fruit "}, "fruit", 1.0), ("exact", {"expected": "a"}, "b", 0.0),
    ("contains", {"answer": "Paris"}, "It is paris.", 1.0), ("numeric", {"answer": "1/2"}, "0.5", 1.0),
])
def test_builtin_metrics(name, task, output, score):
    assert protocol.resolve_metric(name)(task, {"output": output}) == score

def test_module_and_file_metrics(tmp_path, monkeypatch):
    (tmp_path / "mymetric.py").write_text("def score(task, result):\n    return 0.25\n", encoding="utf-8")
    assert protocol.resolve_metric("mymetric:score", base=tmp_path)({}, {}) == 0.25
    assert protocol.resolve_metric(f"{tmp_path / 'mymetric.py'}:score")({}, {}) == 0.25

@pytest.mark.parametrize("bad", ["nope", "C:score", "mod:", ":fn"])
def test_bad_metric_specs(bad):
    with pytest.raises((ValueError, ModuleNotFoundError)):
        protocol.resolve_metric(bad)
```

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_protocol.py` - FAIL, module missing.
- [ ] **Step 3: Implement** `protocol.py` only.
- [ ] **Step 4: Run** `python -m pytest -q tests/test_protocol.py tests/test_online_protocol.py` - PASS.
- [ ] **Step 5: Discriminating mutation** - always pass `trace` in `call_run`: `New` raises TypeError and `test_both_signatures...` FAILS. Use `partition` instead of `rpartition`: the absolute Windows path case FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/protocol.py tests/test_protocol.py tests/test_online_protocol.py
git commit -m "feat: harness and metric contracts and the online train-then-frozen protocol"
```

---

### Task 6: Proposer tool layer confined to the run directory (Wave C)

**Files:**
- Create: `meta_harness/proposer_tools.py`
- Modify: `meta_harness/core.py` (`SearchConfig.__post_init__` rejects a `test_root` inside `root`)
- Test: `tests/test_proposer_tools.py`

**Interfaces:**

```python
class ToolRefused(Exception): ...

class ToolSandbox:
    """Read-only tools over one directory, plus the single write. Paths are POSIX-style relative."""
    MAX_READ_CHARS = 50_000
    MAX_GREP_HITS = 200

    def __init__(self, root: Path, output_dir: Path, forbidden: Sequence[Path] = ()):
        self.root = root.resolve()
        self.output_dir = output_dir          # where write_candidate puts files; NOT readable via tools
        self.forbidden = [p.resolve() for p in forbidden]   # e.g. test_root, proposals/, views/

    def resolve(self, relative: str) -> Path:
        text = str(relative or ".").replace("\\", "/")
        if text.startswith("/") or re.match(r"^[A-Za-z]:", text) or text.startswith("//"):
            raise ToolRefused("absolute paths are not allowed")
        if any(part == ".." for part in text.split("/")):
            raise ToolRefused("'..' is not allowed")
        target = (self.root / text).resolve()           # resolves symlinks
        if target != self.root and self.root not in target.parents:
            raise ToolRefused("path is outside the run directory")
        for blocked in self.forbidden:
            if target == blocked or blocked in target.parents:
                raise ToolRefused("path is not visible to the proposer")
        return target

    def list(self, path: str = ".") -> str            # "dir/" and "file  <bytes>" lines, sorted, max 500
    def read(self, path: str, offset: int = 0, limit: int = 400) -> str   # numbered lines, char cap
    def grep(self, pattern: str, path: str = ".") -> str   # "rel/path:line: text", regex, skips >2MB/binary
    def write_candidate(self, source: str, rationale: str, name: str) -> Path
        # compile(source, name, "exec") first: SyntaxError -> ToolRefused(f"syntax error: ...")
        # writes output_dir/<name>.py and output_dir/<name>.reasoning.md, utf-8

TOOL_SPECS: list[ToolSpec]   # list, read, grep, write_candidate - JSON schemas with required fields
def dispatch(sandbox: ToolSandbox, call: ToolCall, name: str) -> tuple[str, Path | None]
    # returns (result text for the model, written path or None); ToolRefused / bad args become
    # "error: <reason>" results, never exceptions; unknown tool -> "error: unknown tool <name>"
```

`grep` walks with `os.walk(followlinks=False)` and re-checks every file through `resolve` so a symlinked file inside the root that points out is skipped.

**Spec §11 additions (tool output bounds):**
- Every tool result is capped at `MAX_RESULT_TOKENS = 6000` (tokens ~ chars/4) and 2000 lines. `read` keeps the head; `grep` and `list` keep the head of their match list; text is cut on a character boundary (Python `str`) and never mid-line where avoidable.
- A truncated result ends with: `"[truncated: showing lines A-B of N. Full output saved to _spill/<id>.txt - narrow with read(path, offset, limit) or grep(pattern, path)]"`.
- The full output is spilled to `<sandbox root>/_spill/<sequence>.txt` (inside the run directory, readable through `read` with offsets; never outside the root).
- Tests to add: a 5000-line file read returns <= 2000 lines with the pointer and the spill file holds all 5000; a single 200 KB line is capped by tokens; a non-ASCII file (`"é" * 50_000`) truncates without `UnicodeError` and round-trips through the spill; the spill path is readable via `read`. Mutation: spill outside the root (e.g. `tempfile`) -> the readable-via-`read` test FAILS.

**Spec §11 additions (keel read window):** `read` returns numbered lines (`"{n}\t{line}"`), at most 2000 lines, each line cut at 2000 characters (marked `...[line truncated]`), at most 50 KB total, and always states the exact total: header `"{path}: lines {a}-{b} of {total}"`. These limits replace the 6000-token cap for `read` (grep/list keep the 6000-token cap); spill-to-file still applies when truncated. Test: a 3000-line file with one 5000-char line reports `of 3000`, shows <= 2000 lines, the long line is cut, total output <= 50 KB.

- [ ] **Step 1: Write the failing test**

```python
import os
from pathlib import Path

import pytest

from meta_harness.core import SearchConfig
from meta_harness.proposer_tools import ToolRefused, ToolSandbox, dispatch
from meta_harness.providers import ToolCall


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "store"
    (root / "candidates" / "c1").mkdir(parents=True)
    (root / "candidates" / "c1" / "harness.py").write_text("class Harness:\n    pass\n# LABEL\n", encoding="utf-8")
    (root / "candidates" / "c1" / "scores.json").write_text('{"score": 0.5}', encoding="utf-8")
    test = tmp_path / "store-test"
    test.mkdir()
    (test / "test_results.json").write_text('{"secret": 1}', encoding="utf-8")
    out = root / "proposals" / "iteration-0001"
    out.mkdir(parents=True)
    return ToolSandbox(root, out, forbidden=[test, root / "proposals"]), root, test


@pytest.mark.parametrize("path", ["../store-test/test_results.json", "candidates/../../store-test",
                                  "/etc/passwd", "C:/Windows/win.ini", "c:\\x", "\\\\server\\share",
                                  "proposals/iteration-0001"])
def test_escape_attempts_are_refused(run, path):
    sandbox, _, _ = run
    with pytest.raises(ToolRefused):
        sandbox.resolve(path)


def test_absolute_path_of_test_root_is_refused(run):
    sandbox, _, test = run
    with pytest.raises(ToolRefused):
        sandbox.resolve(str(test / "test_results.json"))


def test_symlink_out_of_root_is_refused(run, tmp_path):
    sandbox, root, test = run
    try:
        os.symlink(test, root / "candidates" / "peek", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(ToolRefused):
        sandbox.resolve("candidates/peek/test_results.json")
    assert "secret" not in sandbox.grep("secret", ".")


def test_read_list_grep_inside_root(run):
    sandbox, _, _ = run
    assert "c1/" in sandbox.list("candidates")
    assert "1\tclass Harness:" in sandbox.read("candidates/c1/harness.py")
    assert "candidates/c1/harness.py:3:" in sandbox.grep("LABEL")


def test_dispatch_turns_refusals_into_results(run):
    sandbox, _, _ = run
    text, written = dispatch(sandbox, ToolCall("1", "read", {"path": "../store-test/test_results.json"}), "cand")
    assert text.startswith("error:") and "secret" not in text and written is None
    text, _ = dispatch(sandbox, ToolCall("2", "read", {"_raw": "{bad"}), "cand")
    assert text.startswith("error:")


def test_write_candidate_rejects_syntax_errors_and_writes_valid(run):
    sandbox, _, _ = run
    text, written = dispatch(sandbox, ToolCall("1", "write_candidate", {"source": "def (", "rationale": "r"}), "cand")
    assert text.startswith("error: syntax error") and written is None
    text, written = dispatch(sandbox, ToolCall("2", "write_candidate",
                                               {"source": "class Harness:\n    pass\n", "rationale": "why"}), "cand")
    assert written.name == "cand.py" and written.with_name("cand.reasoning.md").read_text(encoding="utf-8") == "why"


def test_test_root_inside_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="test_root"):
        SearchConfig(root=tmp_path / "r", test_root=tmp_path / "r" / "test")
```

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_proposer_tools.py` - FAIL, module missing.
- [ ] **Step 3: Implement** `proposer_tools.py` only. Leave out the drafted `test_test_root_inside_root_is_rejected`: the `SearchConfig` check and that test belong to Task 10.
- [ ] **Step 4: Run** `python -m pytest -q tests/test_proposer_tools.py` - PASS.
- [ ] **Step 5: Discriminating mutation** - delete the `".."` check: the `../store-test` case must still be caught by the parent check, so ALSO delete the parent check and confirm `test_escape_attempts_are_refused` FAILS; separately remove the drive-letter regex and confirm the `C:/Windows` case FAILS on Windows. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/proposer_tools.py tests/test_proposer_tools.py
git commit -m "feat(proposer): read-only tools confined to the run directory"
```

---

### Task 7: Experience views (`views.py`: scores, summary, full, seed) (Wave C)

**Files:**
- Create: `meta_harness/views.py` (owns `build_view`, `VIEW_MODES`, `VIEW_FILES`); delete `meta_harness/agent_proposer.py` (Task 2 removed its `__init__` export)
- Test: `tests/test_views.py`

**Interfaces:**

```python
VIEW_MODES = ("scores", "summary", "full")      # user-facing; "seed" is internal (best-of-n, Task 9)
VIEW_FILES = {"scores": ("harness.py", "scores.json"),
              "summary": ("harness.py", "scores.json", "summary.md"),
              "seed": ("harness.py", "scores.json")}

def build_view(experience: FilesystemExperience, mode: str, destination: Path,
               summarizer: Callable[[str], str] | None = None) -> Path:
    """full -> experience.root itself (proposals/ and views/ are in the sandbox's forbidden list).
    scores/summary -> a fresh copy at destination with run.json, frontier.json and VIEW_FILES per candidate.
    seed -> only candidates whose id starts with 'initial-'.
    summary: summary.md is generated once per candidate into the candidate's own directory
    (cached; regenerated only when missing) by the summarizer, from source[:6000],
    scores.json and the first 12_000 chars of traces.jsonl."""
```

`AgentProposer(view=...)` builds the view per iteration under `root/views/iteration-NNNN` and uses it as `ToolSandbox.root`; `summary` with no summarizer raises `ValueError` at construction. Summarizer = the proposer model's `__call__` (wired in Task 11); its usage is added to `AgentProposer.usage`.

- [ ] **Step 1: Write the failing test** - build a store with two candidates (`initial-0000`, `iteration-0001-00`) each having `harness.py`, `scores.json`, `traces.jsonl`, `proposer_reasoning.md`, `proposer_log.jsonl`:

```python
def test_scores_view_hides_traces_and_reasoning(store):
    view = build_view(store, "scores", store.root / "views" / "i1")
    names = {p.name for p in (view / "candidates" / "iteration-0001-00").iterdir()}
    assert names == {"harness.py", "scores.json"}


def test_summary_view_adds_cached_summary(store):
    calls = []
    summarize = lambda prompt: calls.append(prompt) or "it fails on vans"
    build_view(store, "summary", store.root / "views" / "i1", summarize)
    build_view(store, "summary", store.root / "views" / "i2", summarize)
    assert len(calls) == 2, "one summary per candidate, cached across iterations"
    assert (store.root / "views" / "i2" / "candidates" / "initial-0000" / "summary.md").read_text(encoding="utf-8") == "it fails on vans"
    assert "traces" not in {p.name for p in (store.root / "views" / "i2" / "candidates" / "initial-0000").iterdir()}


def test_full_view_is_the_root(store):
    assert build_view(store, "full", store.root / "views" / "x") == store.root


def test_seed_view_only_has_initial_candidates(store):
    view = build_view(store, "seed", store.root / "views" / "s")
    assert [p.name for p in (view / "candidates").iterdir()] == ["initial-0000"]


def test_agent_in_scores_view_cannot_read_traces(tmp_path):
    # run_search from test_proposer_agent with view="scores" and a script that reads
    # candidates/initial-0000/traces.jsonl, then writes; the log entry must have ok == False.
    ...
```

(Write the last test in full by importing the `run_search` helper pattern with a `view` parameter added.)

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_views.py` - FAIL (`build_view` not in `proposer_agent`).
- [ ] **Step 3: Implement**; `git rm meta_harness/agent_proposer.py`. The drafted last test (agent in `scores` view) belongs to Task 9, which owns the agent.
- [ ] **Step 4: Run** `python -m pytest -q tests/test_views.py` - PASS.
- [ ] **Step 5: Discriminating mutation** - add `"traces.jsonl"` to `VIEW_FILES["scores"]`: `test_scores_view_hides_traces_and_reasoning` FAILS. Remove the summary cache check: the `len(calls) == 2` assertion FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/views.py tests/test_views.py
git rm -q meta_harness/agent_proposer.py
git commit -m "feat(proposer): scores, summary and full experience views for the Table 3 ablation"
```

---

### Task 7A: Sandbox protocol - temp-dir subprocess and Docker CLI, with preflight (Wave C)

Spec §11 (Sandbox). Shared by Task 18 (terminal domain) and Task 23 (isolation).

**Files:** Create `meta_harness/sandboxes.py`; Test `tests/test_sandboxes.py`.

**Interfaces:**

```python
@dataclass
class ExecResult: code: int; stdout: str; stderr: str; timed_out: bool

class Sandbox(Protocol):
    def start(self) -> None: ...
    def exec(self, command: str, timeout: float, cwd: str | None = None) -> ExecResult: ...
    def put(self, local: Path, remote: str) -> None: ...      # file or directory
    def get(self, remote: str, local: Path) -> None: ...
    def close(self) -> None: ...                              # idempotent; context manager support

class TempDirSandbox:            # subprocess in a private temp dir; remote paths are relative to it
    def __init__(self, env: Mapping[str, str] | None = None): ...   # env defaults to a minimal PATH-only env
class DockerSandbox:
    def __init__(self, image: str, workdir: str = "/app", network: bool = False,
                 memory_mb: int | None = None, cpus: float | None = None, runner=subprocess.run): ...
    # start: docker run -d [--network none] [--memory] [--cpus] -w workdir image sleep infinity
    # exec: docker exec -w cwd id bash -lc command ; put/get: docker cp ; close: docker rm -f

@dataclass
class Preflight: ok: bool; problem: str; hint: str
def preflight(runner=subprocess.run) -> Preflight:
    """`docker version --format {{.Server.Version}}`: missing binary -> hint to install Docker
    Desktop (Windows/macOS) or docker.io (Linux); daemon down -> hint to start it."""
```

All subprocess calls pass `encoding="utf-8", errors="replace"` and timeouts; `TempDirSandbox` refuses remote paths that escape its directory (same rules as `ToolSandbox.resolve`).

**Spec §11 additions (keel watchdog):** also create `meta_harness/timeouts.py`:

```python
class Watchdog:
    """Idle timeout fused with an overall deadline. kick() on progress; check() raises
    Aborted(reason) where reason is "idle" or "deadline" - whichever fired first."""
    def __init__(self, idle: float, deadline: float, clock=time.monotonic): ...
    def kick(self) -> None; def check(self) -> None; def remaining(self) -> float
class Aborted(Exception): reason: str
```

Tests with a fake clock: idle fires when no kick for `idle` seconds; deadline fires despite constant kicks; the reason names which fired. Task 9 uses it per proposal (`--proposer-idle`, `--proposer-deadline`), Task 10 per evaluation.

- [ ] **Step 1: Failing tests** - `TempDirSandbox`: `put` a file, `exec` `python -c` that reads it, `get` the output back; a command over its timeout returns `timed_out=True`; `put` to `../x` raises; `close` twice is fine. `DockerSandbox` with a fake `runner` recording argv: `--network none` by default, `docker rm -f` on close even after an exec error. `preflight` with a fake runner raising `FileNotFoundError` -> `ok=False` and an install hint; non-zero exit -> "start the daemon" hint. A Docker-gated live test (`skipif(not preflight().ok, reason=...)`) runs `echo ok` in `alpine`.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_sandboxes.py` - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Discriminating mutation** - drop `--network none`: the default-argv test FAILS. Restore.
- [ ] **Step 6: Commit** - `git add meta_harness/sandboxes.py tests/test_sandboxes.py` then `git commit -m "feat: sandbox protocol with temp-dir and Docker implementations and a preflight check"`.

---

### Task 8: Evaluation fixes 1-2 - context cost once in tokens, real resamples (Wave D)

**Files:**
- Modify: `meta_harness/core.py` (`MeteredModel`, `TraceRecorder.add_context_cost`, `CandidateEvaluator.evaluate`, `SearchRunner._evaluate_source`/`evaluate_on_test` pass `sample_index=repeat`), `meta_harness/cache.py`, `meta_harness/cls_harnesses.py` and `meta_harness/math_harnesses.py` (delete the four `trace.add_context_cost(len(...))` calls that were `harnesses.py:77,103,115,169` before Task 2's split), `meta_harness/sandbox.py` (`_NullTrace.add_context_cost` becomes a no-op)
- Test: `tests/test_eval_fixes.py`

**Interfaces:**
- `TraceRecorder.add_context_cost(amount)` no longer changes accounting: it records `trace.event("harness_reported_cost", {"amount": amount})` only (kept so old seeds don't crash).
- `MeteredModel(model, trace, sample_index: int = 0)` owns `prompt_tokens`, `completion_tokens`, `calls`. It forwards `sample_index` only to upstreams with `supports_sample_index = True` (i.e. `CachedModel`).
- `CachedModel.supports_sample_index = True`; `call(prompt, *, sample_index: int = 0, **kwargs)`; `cache_key(model_id, prompt, kwargs, sample_index=0)` includes `"sample": sample_index` in the hashed payload.
- `CandidateEvaluator.evaluate(source, tasks, candidate_id, trace_path, sample_index=0)`; `context_cost = metered.prompt_tokens / n`, `metrics["prompt_tokens"] = metered.prompt_tokens`, `metrics["completion_tokens"]`, `metrics["model_calls"]`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from meta_harness.cache import CachedModel
from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner

INFLATING = '''
class Harness:
    def run(self, task, model, trace):
        prompt = "x" * 400
        trace.add_context_cost(len(prompt))          # old habit: characters
        return model(prompt)
'''


class Usage:
    model = "u"
    def call(self, prompt, **kwargs):
        return "ok", {"prompt_tokens": 7, "completion_tokens": 1}


def test_context_cost_is_counted_once_in_tokens(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(INFLATING, encoding="utf-8")
    result = CandidateEvaluator(Usage(), lambda p, t: 1.0).evaluate(
        source, [{"input": "a"}, {"input": "b"}], "c", tmp_path / "t.jsonl")
    assert result.context_cost == 7.0, "provider tokens only; the harness's character count is ignored"
    assert result.metrics["prompt_tokens"] == 14


class Counter:
    model = "counter"
    def __init__(self):
        self.n = 0
    def call(self, prompt, **kwargs):
        self.n += 1
        return f"sample-{self.n}", {"prompt_tokens": 1}


def test_repeats_are_distinct_samples_but_reruns_hit_the_cache(tmp_path: Path):
    upstream = Counter()
    model = CachedModel(upstream, tmp_path / "cache")
    assert model.call("p", sample_index=0)[0] != model.call("p", sample_index=1)[0]
    assert model.call("p", sample_index=1)[0] == "sample-2" and upstream.n == 2


def test_search_repeats_reach_the_model(tmp_path: Path):
    upstream = Counter()
    seed = tmp_path / "s.py"
    seed.write_text("class Harness:\n    def run(self, task, model, trace):\n        return model('same')\n",
                    encoding="utf-8")
    runner = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0, repeats=3),
                          CandidateEvaluator(CachedModel(upstream, tmp_path / "c"), lambda p, t: 1.0),
                          lambda *a: [])
    runner.run([seed], [{"input": "a"}])
    assert upstream.n == 3
```

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_eval_fixes.py` - FAIL: `context_cost == 107.0` (100 chars + 7 tokens) and `upstream.n == 1`.
- [ ] **Step 3: Implement** the interfaces. Update `tests/test_core_metering.py` and `tests/test_cache.py` where they asserted the old summed cost or the old key.
- [ ] **Step 4: Run** `python -m pytest -q` - PASS.
- [ ] **Step 5: Discriminating mutation** - drop `"sample"` from `cache_key`: both repeat tests FAIL. Restore `trace.context_cost` as the evaluator's source: the context-cost test FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/core.py meta_harness/cache.py meta_harness/cls_harnesses.py meta_harness/math_harnesses.py meta_harness/sandbox.py tests/test_eval_fixes.py tests/test_core_metering.py tests/test_cache.py
git commit -m "fix(eval): count context once in provider tokens; repeats are real resamples"
```

---

### Task 9: Tool-using proposer agent, `metaharness` and `best-of-n` optimizers (Wave D)

**Files:**
- Create: `meta_harness/proposer_agent.py`, `meta_harness/proposer_prompt.md` (from `meta_harness/skill/SKILL.md`, stripped of the Claude-Code-history and `cc_harness` sections; generic text only; domain notes come from the selected domain module's `PROMPT` via `{domain_notes}`; placeholders `{iteration}`, `{count}`, `{index}`, `{domain}`, `{domain_notes}`, `{max_tool_calls}`, `{objective}`), `meta_harness/optimizers/metaharness.py` (`NAME = "metaharness"`; `build(ctx)` returns `AgentProposer(ctx.proposer_model, domain=ctx.domain.NAME, domain_notes=ctx.domain.PROMPT, view=ctx.view, max_tool_calls=ctx.max_tool_calls, summarizer=ctx.summarizer, forbidden=ctx.forbidden)`), `meta_harness/optimizers/best_of_n.py` (`NAME = "best-of-n"`; same with `view="seed"`)
- Delete: `meta_harness/skill/` (`git rm -r`)
- Not modified here: `core.py` (Task 8 owns it in this wave). The log is written to `proposals/iteration-NNNN/candidate-KK.proposer_log.jsonl`; Task 10 copies it into the candidate directory as `proposer_log.jsonl`. This task's tests read it from `proposals/`.
- Test: `tests/test_proposer_agent.py`, `tests/test_optimizers_basic.py` (both optimizers are registered; under best-of-n the iteration-2 sandbox refuses `candidates/iteration-0001-00/scores.json`; in the `scores` view a read of `traces.jsonl` is logged with `ok: false`)

**Interfaces:**

```python
PROMPT_PATH = Path(__file__).with_name("proposer_prompt.md")

def render_prompt(template: str, **values: Any) -> str:
    """Fill {name} placeholders. Unknown or leftover placeholders raise KeyError - never shipped raw."""
    return template.format_map(_Strict(values))       # _Strict.__missing__ raises KeyError

class ProposerBudgetExceeded(RuntimeError): ...

class AgentProposer:
    def __init__(self, model: Any, *, domain: str = "classification", view: str = "full",
                 max_tool_calls: int = 40, summarizer: Callable[[str], str] | None = None,
                 forbidden: Sequence[Path] = (), objective: str = "maximize score, then minimize context tokens",
                 max_tokens: int = 4096): ...
    usage: dict[str, int]       # {"calls": n, "input_tokens": n, "output_tokens": n}, cumulative
    def __call__(self, experience: FilesystemExperience, iteration: int, count: int) -> list[Path]: ...
```

Loop per candidate `index` in `range(count)` (name `candidate-{index:02d}`):

```python
messages = [{"role": "user", "content": render_prompt(...)}]
log_path = output / f"{name}.proposer_log.jsonl"
for step in range(1, self.max_tool_calls + 1):
    turn = self.model.chat(SYSTEM, messages, TOOL_SPECS, max_tokens=self.max_tokens)
    self._count(turn.usage)
    messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls})
    if not turn.tool_calls:
        messages.append({"role": "user", "content": "Use the tools; finish by calling write_candidate."})
        _log(log_path, {"step": step, "tool": None, "text": turn.text[:2000]})
        continue
    for call in turn.tool_calls:
        result, written = dispatch(sandbox, call, name)
        _log(log_path, {"step": step, "tool": call.name, "arguments": _redact(call.arguments),
                        "ok": not result.startswith("error:"), "result_chars": len(result),
                        "path": call.arguments.get("path")})
        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
        if written is not None:
            break
    if written is not None:
        paths.append(written); break
else:
    raise ProposerBudgetExceeded(f"no candidate after {self.max_tool_calls} tool turns")
```

`_redact` replaces `write_candidate.source` with its length (the source is saved as `harness.py` anyway). Every step counts toward `max_tool_calls` (one model turn = one step). The sandbox root is the view built by Task 7 (`views.build_view`), `forbidden = [config.test_root, root/"proposals", root/"views"]` plus caller extras. If the model lacks `chat`, raise `TypeError("proposer provider must support tool calling")` at construction.

**Spec §11 additions (proposer loop):**
- `before_tool(call) -> str | None` hook runs before every dispatch: it resolves every `path` argument through `ToolSandbox.resolve` and denies (returns the error text as the tool result) anything outside the root or under a forbidden path such as the test root; the tool is not executed.
- `write_candidate` terminates the turn: any later tool calls in the same assistant turn are not executed and are closed with `"error: not executed - write_candidate ended the turn"` results; on budget exhaustion every pending call is closed the same way before the error is raised (no orphaned calls in the transcript).
- Unknown tools and tool exceptions become `"error: ..."` tool results, never exceptions out of the loop; a catch-all logs `{"event": "error", ...}` and re-raises only provider failures.
- `stop_reason == "length"` is an explicit failure: logged as `{"event": "length"}`, its tool calls (possibly cut) are answered with `"error: your output was truncated - call one tool with smaller arguments"`, and three consecutive length stops raise `ProposerTruncated`.
- Per-turn batch cap: the sum of tool results in one turn is capped at 10000 tokens; results beyond it are replaced by a pointer to narrow the request.
- The proposer log is append-only, one record per message/tool call with `id`, `parent` (previous record id), `role`, `tool`, `arguments`, `ok`, `result_tokens` and per-message `usage` (`input`, `output`, `cache_read`, `cache_write`, `reasoning`) - the source for Table 8 statistics (Task 22) and cost.
- Tests to add (scripted provider): a turn with `[write_candidate, read]` writes the candidate and the `read` is logged as not executed; a `read` of `../store-test/x` is denied by the hook and the file is never opened (monkeypatch `ToolSandbox.read` to fail if called); unknown tool `rm` -> error result and the loop continues; three `length` entries raise `ProposerTruncated`; log records chain by `parent`. Mutation: execute tool calls after `write_candidate` -> the not-executed test FAILS.

**Spec §11 additions (keel: tool outcomes, fingerprint, dangling calls, redaction):**
- Tool outcome counts per proposal: `dispatched`, `denied` (before-tool hook refusals; test-split attempts counted separately as `denied_split`), `failed` (tool raised or returned an error), `aborted` (closed unexecuted). Written to `candidate-KK.outcomes.json`.
- At proposal start the agent records `fingerprint = "v1:" + sha256(sorted (candidate_id, scores.json sha256) pairs + frontier.json sha256)` of the state it may read, plus the list of candidate ids it read (`parents_read`), in `candidate-KK.proposal.json`.
- Every log record passes through `redact.bounded`/`redact_obj` before it is written; ids through `safe_id`.
- A `Watchdog(idle, deadline)` wraps the loop; on `Aborted` pending calls are closed (dangling-call repair) and the reason is logged.
- Tests to add: one denied split read + one unknown tool + one success -> `{"dispatched": 2, "denied": 1, "denied_split": 1, "failed": 1, "aborted": 0}` (adjust to the scripted turn); the fingerprint changes when any `scores.json` changes; a scripted response containing `sk-abcdefgh12345678` is logged as `[redacted]`; the idle watchdog aborts with reason `idle`.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner
from meta_harness.proposer_agent import (AgentProposer, PROMPT_PATH, ProposerBudgetExceeded,
                                         render_prompt)
from meta_harness.providers import model_from_environment

SEED = "class Harness:\n    def run(self, task, model, trace):\n        return model(task['input'])\n"
NEW = "class Harness:\n    def run(self, task, model, trace):\n        return 'fruit'\n"


def scripted(tmp_path, rows, name="p.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return model_from_environment("scripted", str(path))


def run_search(tmp_path, proposer_rows, iterations=1):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    proposer = AgentProposer(scripted(tmp_path, proposer_rows), max_tool_calls=5)
    config = SearchConfig(root=tmp_path / "store", iterations=iterations)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: "vehicle"), proposer)
    tasks = [{"input": "apple", "label": "fruit"}, {"input": "car", "label": "vehicle"}]
    runner.run([seed], tasks, [{"input": "pear", "label": "fruit"}])
    return config.root, proposer


def test_agent_reads_then_writes_and_logs_every_call(tmp_path):
    root, proposer = run_search(tmp_path, [
        {"tool_calls": [{"name": "list", "arguments": {"path": "candidates"}}]},
        {"tool_calls": [{"name": "read", "arguments": {"path": "candidates/initial-0000/scores.json"}}]},
        {"tool_calls": [{"name": "write_candidate", "arguments": {"source": NEW, "rationale": "always fruit"}}]},
    ])
    candidate = root / "candidates" / "iteration-0001-00"
    assert (candidate / "harness.py").read_text(encoding="utf-8") == NEW
    assert (candidate / "proposer_reasoning.md").read_text(encoding="utf-8") == "always fruit"
    log = [json.loads(l) for l in (candidate / "proposer_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["tool"] for e in log] == ["list", "read", "write_candidate"]
    assert log[1]["path"] == "candidates/initial-0000/scores.json"
    assert "source" not in json.dumps(log[2]["arguments"]) or isinstance(log[2]["arguments"]["source"], int)
    assert proposer.usage["calls"] == 3


def test_agent_cannot_read_the_test_split(tmp_path):
    root, _ = run_search(tmp_path, [
        {"tool_calls": [{"name": "read", "arguments": {"path": "../store-test/test_results.json"}}]},
        {"tool_calls": [{"name": "write_candidate", "arguments": {"source": NEW, "rationale": "r"}}]},
    ])
    log = [json.loads(l) for l in (root / "candidates" / "iteration-0001-00" / "proposer_log.jsonl")
           .read_text(encoding="utf-8").splitlines()]
    assert log[0]["ok"] is False


def test_budget_exhaustion_is_recorded_not_fatal(tmp_path):
    root, _ = run_search(tmp_path, [{"tool_calls": [{"name": "list", "arguments": {}}]}] * 5)
    assert "ProposerBudgetExceeded" in (root / "proposals" / "iteration-0001" / "proposer.error").read_text(encoding="utf-8")


def test_prompt_template_renders_without_leftovers():
    text = render_prompt(PROMPT_PATH.read_text(encoding="utf-8"), iteration=3, count=2, index=0,
                         domain="classification", max_tool_calls=40,
                         objective="maximize score, then minimize context tokens")
    assert "{" not in text.replace("{{", "") and "$META_HARNESS" not in text
    assert "iteration 3" in text
    with pytest.raises(KeyError):
        render_prompt("iteration {iteration} {missing}", iteration=1)
```

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_proposer_agent.py` - FAIL, module missing.
- [ ] **Step 3: Implement** `proposer_agent.py`, `proposer_prompt.md` (it must say: you may only read inside this directory; the held-out test split is not available; call `write_candidate` once with a complete single-file `class Harness`; harness contract; trade-off `{objective}`), the two optimizer modules, and `git rm -r meta_harness/skill`. In the drafted tests, read logs from `root / "proposals" / "iteration-0001" / "candidate-00.proposer_log.jsonl"`.
- [ ] **Step 4: Run** `python -m pytest -q tests/test_proposer_agent.py tests/test_core_reasoning.py` - PASS.
- [ ] **Step 5: Discriminating mutation** - skip the `_log` call for `write_candidate`: the tool-order assertion FAILS. Mutation 2: in the loop, don't count turns without tool calls toward the budget: with the 5x `list` script still exhausted - so instead script 6 empty-text turns and assert exhaustion; confirm FAIL under the mutation. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/proposer_agent.py meta_harness/proposer_prompt.md meta_harness/optimizers/metaharness.py meta_harness/optimizers/best_of_n.py tests/test_proposer_agent.py tests/test_optimizers_basic.py
git rm -r -q meta_harness/skill
git commit -m "feat(proposer): provider-agnostic tool-using proposer with a per-candidate call log"
```

---

### Task 10: Core integration - evaluation fixes 3-5, online protocol, budget, equal-budget cap (Wave E)

**Files:**
- Modify: `meta_harness/sandbox.py`, `meta_harness/core.py` (`CandidateEvaluator.validate`, `SearchRunner._evaluate_source` pass the probe), `meta_harness/proposer_agent.py` (renders through `render_prompt` - done in Task 9; this task adds the regression test against the old bug)
- Test: append to `tests/test_eval_fixes.py`

**Interfaces:**
- `validate_in_subprocess(source, timeout=20.0, probe: Mapping | None = None)`: the probe is written as JSON to a temp file and passed as `argv[2]`; `python -m meta_harness.sandbox <candidate.py> [probe.json]`. With no probe the legacy `{"input": "validation"}` row is used (no `label`).
- `SearchRunner.run` sets `self.probe = dict(search_tasks[0])` with the target fields (`expected`, `label`, `answer`, `target`) removed; `_evaluate_source` passes it.
- Fix 6: `metrics.METRICS` contains no shell-executing metric.

**Spec §11 additions (resumable runs, failures, checkpoints):**
- Every step's result is a file: proposal files under `proposals/iteration-NNNN/`, `candidates/<id>/scores.json`, test results. `SearchRunner.run(..., resume=True)` skips seeds and proposals whose `scores.json` exists (re-reading the result), skips proposing an iteration whose `proposals/iteration-NNNN/` already holds `.py` files, and continues the budget from `run.json`.
- `run.json` is checkpointed (atomic write via temp file + `replace`) after every iteration with `{"cursor": {"iteration": k}, "frontier": [...ids], "budget": {...}}`; only derived views (`views/`) are pruned on resume.
- Per-task failures are scores: a harness exception or a per-task timeout on one task records reward 0 and an `error` field on that task's `task_end` trace event; the candidate stays valid with `metrics["task_failures"]`. Only import/validation failure marks a candidate invalid. Per-evaluation wall-clock timeout (`evaluation_timeout`, default `task_timeout * len(tasks)`) is enforced; harness errors and timeouts are never retried.
- Budget records the Task 2 `Usage` totals (`input`, `output`, `cache_read`, `cache_write`, `reasoning`) separately for base model and proposer, plus `cost_usd` (or `null` when prices are unknown) - context tokens stay the Pareto axis.
- Tests to add: kill-and-resume - a proposer that raises `KeyboardInterrupt` at iteration 2; rerun with `resume=True` evaluates no seed again (counter on the model) and finishes iterations 2-3; `run.json` after iteration 1 already holds `cursor.iteration == 1`; a harness raising on 1 of 3 tasks scores 2/3 and stays valid; a harness sleeping past `evaluation_timeout` is recorded as failed tasks, not retried (model call counter). Mutation: mark the candidate invalid on any task exception -> the 2/3 test FAILS.

**Spec §11 additions (keel: receipts, acceptance gate, crash repair):**
- `candidates/<id>/receipt.json` is written first as `{"verdict": "incomplete", ...}` (atomic temp + replace) and finalised after each stage: `proposal_id`, `parents_read`, `tool_outcomes`, `fingerprint`, `validation`, `evaluation`, `fallback`, `verdict`. Verdict enum: `accepted`, `rejected_invalid_source`, `rejected_split_violation`, `stale_parent`, `budget_expired`, `failed_eval` (plus `incomplete` while running). "Accepted" means valid for the frontier, not an improvement; improvement is only claimed from held-out scores.
- Acceptance gate (host re-validation before a proposal becomes a candidate): source compiles and passes `validate_in_subprocess`; `rejected_split_violation` when the source contains the absolute test-root path or the proposer's `outcomes.json` shows `denied_split > 0` with the source referencing test data; `stale_parent` when the recorded fingerprint no longer matches the experience state it claimed (concurrent writers or resume from an older state); `budget_expired` when the equal-budget cap was hit. Rejected proposals keep their files and receipt but are excluded from the frontier.
- Crash repair on resume: torn trailing lines in `traces*.jsonl` and `proposer_log.jsonl` are tolerated (`read_jsonl(tolerate_torn=True)`); a candidate whose receipt says `incomplete` is re-evaluated; a persisted `.resume_attempts` counter in the run root caps automatic resumes at 3 (`resume=True` beyond that raises with a message to use `--force-resume`).
- `TraceRecorder.event` passes payloads through `redact.redact_obj` and string fields through `bounded` - no key ever reaches a trace or receipt.
- Evaluation uses `Watchdog(idle=task_timeout, deadline=evaluation_timeout)`; the abort reason is recorded in the receipt.
- Tests to add: receipt exists with `incomplete` when evaluation is interrupted (proposer raising mid-run), and `accepted` after; a source referencing `<root>-test` gets `rejected_split_violation` and is not in `frontier.json`; changing a prior `scores.json` between proposal and evaluation yields `stale_parent`; a torn final trace line does not break resume; a fourth automatic resume raises; a model response containing `Bearer abc123def` appears as `[redacted]` in `traces.jsonl`. Mutation: skip the fingerprint check -> the stale test FAILS.

- [ ] **Step 1: Tests**

```python
MATH_ONLY = '''
class Harness:
    def run(self, task, model, trace):
        return model("Solve: " + task["problem"])
'''


def test_validation_probe_is_the_first_search_task(tmp_path):
    seed = tmp_path / "m.py"
    seed.write_text(MATH_ONLY, encoding="utf-8")
    runner = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0),
                          CandidateEvaluator(lambda p, **_: "4", lambda p, t: float(p == t["answer"])),
                          lambda *a: [])
    [result] = runner.run([seed], [{"problem": "2+2", "answer": "4"}])
    assert result.valid, result.error           # old probe {"input": "validation"} -> KeyError 'problem'


def test_probe_does_not_carry_the_answer(tmp_path):
    peek = tmp_path / "p.py"
    peek.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                    "        assert 'answer' not in task\n        return ''\n", encoding="utf-8")
    runner = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0),
                          CandidateEvaluator(lambda p, **_: "", lambda p, t: 0.0), lambda *a: [])
    [result] = runner.run([peek], [{"problem": "x", "answer": "y"}])
    # validation passes; evaluation itself sees the answer in plain protocol, so only the probe is checked
    assert "AssertionError" not in (result.error or "") or result.valid is False and "validation" not in result.error


def test_proposer_prompt_has_no_shell_placeholders():
    from meta_harness.proposer_agent import PROMPT_PATH, render_prompt
    text = render_prompt(PROMPT_PATH.read_text(encoding="utf-8"), iteration=7, count=1, index=0,
                         domain="math", max_tool_calls=40, objective="o")
    assert "$META_HARNESS" not in text and "iteration 7" in text


def test_no_metric_runs_shell_commands():
    import inspect
    from meta_harness import metrics
    assert "subprocess" not in inspect.getsource(metrics)
```

Rewrite `test_probe_does_not_carry_the_answer` during implementation so that it asserts exactly: the run directory's validation stage passed (`result.error` does not start with `"validation"`). Keep the assertion single-purpose; the draft above shows intent.

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_eval_fixes.py -k "probe or shell"` - FAIL on the probe tests and the `subprocess` check (it is still imported in `metrics.py`).
- [ ] **Step 3: Implement.** Leave out the drafted `test_no_metric_runs_shell_commands`: the terminal metric is rebuilt to run only inside the task's container in Task 18 (spec §8), which pins fix 6. Add `test_proposer_log_is_copied_into_the_candidate` (a Task 9 style run leaves `candidates/iteration-0001-00/proposer_log.jsonl`).
- [ ] **Step 4: Run** `python -m pytest -q tests/test_eval_fixes.py tests/test_sandbox.py tests/test_baselines.py` - PASS.
- [ ] **Step 5: Discriminating mutation** - hard-code the old probe in `sandbox._check`: `test_validation_probe_is_the_first_search_task` FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/sandbox.py meta_harness/core.py meta_harness/metrics.py tests/test_eval_fixes.py
git commit -m "fix(eval): validate candidates on a real search task, not a fake classification row"
```

**Part 2 of Task 10 - protocol, online evaluation, budget and the equal-budget cap (same files, same worktree).**

**Interfaces:**

```python
# core.py
@dataclass
class Budget:
    harness_evaluations: int = 0      # one per (candidate, repeat) on the search split
    test_evaluations: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    train_prompt_tokens: int = 0      # online protocol: tokens spent streaming the train split
    proposer_calls: int = 0
    proposer_input_tokens: int = 0
    proposer_output_tokens: int = 0
    proposals: int = 0
    invalid_proposals: int = 0

class CandidateEvaluator:
    def __init__(self, model, metric: Metric, task_timeout: float = 600.0,
                 protocol: str = "plain", train_tasks: Sequence[Mapping] = ()): ...
    # plain : per task  result = call_run(...); score = metric(task, result); call_learn(...)
    # online: fresh harness -> protocol.stream_train(train_tasks) -> protocol.evaluate_frozen(tasks)

@dataclass
class SearchConfig:  # new fields
    max_evaluations: int | None = None   # equal-budget cap over harness_evaluations (search split)
    run_meta: dict = field(default_factory=dict)   # provider/model names, optimizer, view (never keys)
    # __post_init__: ValueError("test_root must be outside root") when test_root is root or inside it

class SearchRunner:
    budget: Budget                       # updated under a lock from every EvaluationResult.metrics
    # run(): stops proposing once budget.harness_evaluations >= max_evaluations (the in-flight
    # iteration is truncated to the remaining budget); after the loop adds getattr(proposer,
    # "usage", {}) into the proposer_* fields; rewrites run.json with "budget" and run_meta.
    # _sidecar_notes(): also copies <stem>.proposer_log.jsonl -> proposer_log.jsonl and
    # <stem>.parent.json -> parent.json (lineage, used by T19/T20/T22).
    # EvaluationResult.metrics gains per-task rewards: "task_rewards" stays in traces only
    # (task_end events), never in scores.json, to keep scores.json small.
```

`metrics.py` keeps `exact_match`, `math_equivalence`, `normalize_answer`, `estimate_tokens`, `terminal_metric`; `METRICS` becomes `protocol.BUILTIN_METRICS` re-exported. Existing tests that pass `(prediction, task)` lambdas switch to `(task, result)`.

- [ ] **Step 7: Tests (append to `tests/test_eval_fixes.py`)**

```python
def test_online_evaluator_learns_from_train_before_scoring(tmp_path):
    source = tmp_path / "m.py"
    source.write_text(
        "class Harness:\n"
        "    def __init__(self): self.memory = {}\n"
        "    def run(self, task, model): return {'output': self.memory.get(task['input'], 'unknown')}\n"
        "    def learn(self, task, result): self.memory[task['input']] = task['label']\n", encoding="utf-8")
    exact = protocol.resolve_metric("exact")
    tasks = [{"input": "a", "label": "x"}]
    online = CandidateEvaluator(lambda p, **_: "", exact, protocol="online", train_tasks=tasks)
    plain = CandidateEvaluator(lambda p, **_: "", exact)
    assert online.evaluate(source, tasks, "c", tmp_path / "o.jsonl").score == 1.0
    assert plain.evaluate(source, tasks, "c", tmp_path / "p.jsonl").score == 0.0


def test_budget_and_equal_budget_cap(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text("class Harness:\n    def run(self, task, model):\n        return model(task['input'])\n",
                    encoding="utf-8")
    made = []
    def proposer(experience, iteration, count):
        path = tmp_path / f"p{iteration}.py"
        path.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
        made.append(iteration)
        return [path]
    proposer.usage = {"calls": 2, "input_tokens": 10, "output_tokens": 3}
    config = SearchConfig(root=tmp_path / "r", iterations=10, repeats=2, max_evaluations=6)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: "x", protocol.resolve_metric("exact")), proposer)
    runner.run([seed], [{"input": "a", "label": "x"}, {"input": "b", "label": "x"}])
    budget = json.loads((config.root / "run.json").read_text(encoding="utf-8"))["budget"]
    assert budget["harness_evaluations"] == 6 and made == [1, 2]      # seed + 2 proposals, x2 repeats
    assert budget["model_calls"] == 12 and budget["proposer_calls"] == 2


def test_test_root_inside_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="test_root"):
        SearchConfig(root=tmp_path / "r", test_root=tmp_path / "r" / "test")
```

- [ ] **Step 8: Run** `python -m pytest -q` - PASS.
- [ ] **Step 9: Discriminating mutation** - check the cap only before each iteration but never truncate inside: still 6 here, so set `candidates_per_iteration=2, max_evaluations=4` in a second case and confirm it FAILS under that mutation; count evaluations per candidate instead of per repeat: the budget test FAILS. Restore.
- [ ] **Step 10: Commit**

```bash
git add meta_harness/core.py meta_harness/sandbox.py meta_harness/metrics.py tests
git commit -m "feat(core): online protocol, evaluation budget in run.json and an equal-budget cap"
```

---
### Task 11: `meta-harness search` CLI and offline end-to-end smoke test (Wave F)

**Files:**
- Modify: `meta_harness/__main__.py` (replace `run` with `search`), `meta_harness/demo.py` (uses scripted providers + `AgentProposer`), `meta_harness/datasets.py` (`split_tasks` takes `test_fraction`)
- Create: `tests/fixtures/smoke/data.jsonl`, `tests/fixtures/smoke/base.jsonl`, `tests/fixtures/smoke/proposer.jsonl`, `tests/fixtures/smoke/seed.py`
- Test: `tests/test_search_cli.py`; rewrite `tests/test_cli.py` and `tests/test_smoke_end_to_end.py` against `search`

**Interfaces** - `meta-harness search`:

| Flag | Default | Notes |
|---|---|---|
| `--harness PATH` (repeatable) | required | seed harness files |
| `--data PATH` | - | JSONL/JSON/CSV; split by `--test-fraction` (0.3) with `--seed` (0) |
| `--search PATH` / `--test PATH` | - | explicit split; mutually exclusive with `--data` |
| `--metric NAME|module:function` | `exact` | via `protocol.resolve_metric` |
| `--provider` | required | `anthropic|openai|compatible|unikey|scripted` |
| `--model` | required | for `scripted`, the script path |
| `--base-url` | env | passed to `compatible`/`openai`/`unikey` |
| `--proposer-provider` / `--proposer-model` | the base ones | strong proposer, cheap base model |
| `--iterations` 10, `--candidates` 1, `--repeats` 1, `--max-workers` 1 | | |
| `--view scores|summary|full` | `full` | |
| `--max-tool-calls` | 40 | |
| `--domain classification|math|generic` | `generic` | selects the prompt section |
| `--root` `.meta-harness`, `--test-root`, `--cache-dir`, `--no-cache` | | |
| `--proposer-command` | - | escape hatch: external agent via `CommandProposer` |
| `--validation-timeout` 20, `--task-timeout` 600 | | |

Output: JSON with `root`, `search_tasks`, `test_tasks`, `frontier`, `test_results`, `budget` (from `SearchRunner.budget`, Task 10). Errors (missing key, bad metric, empty data) print one line to stderr and return 2.

**Spec §11 additions:** `search --resume` passes `resume=True` (refuses with exit 2 when `--root` has no `run.json`); `--price provider:model=in,out[,cache_read,cache_write]` (repeatable) feeds `cost_usd`; the printed summary shows context tokens and billed cost as separate fields. Test: run the smoke search with `--iterations 1`, then `--resume --iterations 2` evaluates only iteration 2 (count `scores.json` mtimes or a scripted-model counter).

**Spec §11 additions (keel):** `search --force-resume` resets `.resume_attempts`; `search` prints counts by receipt verdict at the end. Test: the smoke run's summary has `"verdicts": {"accepted": 2}`.

- [ ] **Step 1: Fixtures.** `data.jsonl`: 10 rows `{"input": "...", "label": "fruit"|"vehicle"}`. `seed.py`: a harness returning `model("classify: " + input)` (new-style signature, returns dict). `base.jsonl` (harness model): `{"match": "(?i)label primed", "text": "PRIMED"}`, `{"match": "apple|pear|plum|banana|grape", "text": "fruit"}`, `{"default": "vehicle"}`. `proposer.jsonl`: `list candidates` -> `read candidates/initial-0000/scores.json` -> `grep vehicle candidates` -> `write_candidate` with a harness that maps output to one of the declared labels.
- [ ] **Step 2: Write the failing test**

```python
import json
from pathlib import Path

from meta_harness import __main__ as cli

FIX = Path(__file__).parent / "fixtures" / "smoke"


def test_search_end_to_end_offline(tmp_path, capsys):
    root = tmp_path / "store"
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--data", str(FIX / "data.jsonl"),
                     "--metric", "exact", "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
                     "--proposer-provider", "scripted", "--proposer-model", str(FIX / "proposer.jsonl"),
                     "--iterations", "1", "--root", str(root), "--no-cache"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["search_tasks"] == 7 and out["test_tasks"] == 3
    proposed = root / "candidates" / "iteration-0001-00"
    assert (proposed / "proposer_log.jsonl").is_file() and (proposed / "traces.jsonl").is_file()
    test = json.loads(Path(out["test_results"]).read_text(encoding="utf-8"))
    assert {r["split"] for r in test["results"]} == {"test"}
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    test_inputs = {json.loads(l)["input"] for l in (FIX / "data.jsonl").read_text(encoding="utf-8").splitlines()} \
        - {t["input"] for t in manifest["tasks"]}
    assert len(test_inputs) == 3, "run.json holds the search split only"
    assert "API_KEY" not in (root / "run.json").read_text(encoding="utf-8")


def test_module_metric_and_explicit_split(tmp_path, capsys):
    (tmp_path / "score.py").write_text("def always(task, result):\n    return 1.0\n", encoding="utf-8")
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--search", str(FIX / "data.jsonl"),
                     "--test", str(FIX / "data.jsonl"), "--metric", f"{tmp_path / 'score.py'}:always",
                     "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
                     "--iterations", "0", "--root", str(tmp_path / "s"), "--no-cache"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["frontier"][0]["score"] == 1.0


def test_missing_key_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--data", str(FIX / "data.jsonl"),
                     "--provider", "anthropic", "--model", "m", "--root", str(tmp_path / "s")])
    assert code == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_data_and_search_are_exclusive():
    import pytest
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["search", "--harness", "h.py", "--data", "a", "--search", "b",
                                       "--provider", "scripted", "--model", "m"])
```

- [ ] **Step 3: Run to verify it fails** - `python -m pytest -q tests/test_search_cli.py` - FAIL (`invalid choice: 'search'`).
- [ ] **Step 4: Implement** `_command_search`; port `test_cli.py`/`test_smoke_end_to_end.py` to `search` (the `--proposer-command` stub-agent tests stay, now via `search --proposer-command`). `demo` runs on bundled scripted fixtures copied to `meta_harness/demo_data/` (add to `package-data`).
- [ ] **Step 5: Run** `python -m pytest -q` - PASS; `python -m meta_harness demo --iterations 1 --root %TEMP%-free path under tmp` is covered by `test_demo_still_runs`.
- [ ] **Step 6: Discriminating mutation** - make `--proposer-provider` silently ignored (always use base provider): the smoke test FAILS (the base script has no tool calls -> budget error, no `iteration-0001-00`). Restore.
- [ ] **Step 7: Commit**

```bash
git add meta_harness/__main__.py meta_harness/demo.py meta_harness/demo_data meta_harness/datasets.py pyproject.toml tests
git commit -m "feat(cli): plug-and-play search over a harness file, a JSONL eval set and any metric"
```

---

### Task 12: Classification loaders - Symptom2Disease, LawBench, USPTO-50k (Wave F)

Gap: G12 (datasets, protocol), R2.

**Files:**
- Create: `meta_harness/loaders/symptom2disease.py`, `meta_harness/loaders/lawbench.py`, `meta_harness/loaders/uspto50k.py`, `meta_harness/domains/classification.py`, `meta_harness/commands/fetch.py`
- Create fixtures: `tests/fixtures/loaders/s2d_train_p0.json`, `s2d_train_p1.json`, `s2d_test_p0.json`, `lawbench_3-3.json` (5 records), `uspto_train_p0.json`, `uspto_validation_p0.json`
- Test: `tests/test_loaders_classification.py`

**Interfaces:**

```python
# loaders/symptom2disease.py
NAME, DOMAIN = "symptom2disease", "classification"
SOURCE = "hf:gretelai/symptom_to_diagnosis (train 853, test 212; input_text, output_text)"
def fetch(cache_dir: Path, http: Http = urllib_get, val_fraction: float = 0.2, seed: int = 0) -> dict[str, Path]:
    """train/val carved from HF train (seeded), test = HF test. Rows: {"input", "label", "labels"},
    labels = sorted label set of the HF train split (22)."""

# loaders/lawbench.py   (paper: 215 classes - LawBench charge prediction)
NAME, DOMAIN = "lawbench", "classification"
SOURCE = "https://raw.githubusercontent.com/open-compass/LawBench/main/data/zero_shot/3-3.json"
def fetch(cache_dir, http=urllib_get, val_fraction=0.2, test_fraction=0.3, seed=0) -> dict[str, Path]
    """Step 1 of this task confirms which LawBench task file holds the 215-class charge
    prediction set (record fields `instruction`, `question`, `answer`) and pins the URL above.
    The answer string is normalised to the bare charge name; labels = sorted set."""

# loaders/uspto50k.py   (paper: 180 classes)
NAME, DOMAIN = "uspto50k", "classification"
SOURCE = "hf:pingzhili/uspto-50k (train, validation)"
def fetch(cache_dir, http=urllib_get, val_fraction=0.2, seed=0) -> dict[str, Path]
    """Step 1 records the field names served for this dataset and which field gives the
    paper's 180-way label (reaction template class); validation split is the test split."""

# domains/classification.py
NAME, PROTOCOL = "classification", "online"
SEED_DIR = REPO / "baselines" / "classification"
PROMPT = ("Tasks are single-label text classification over a fixed label set given in task['labels']. "
          "The harness sees a labelled train stream through learn(task, result) and is then evaluated "
          "frozen. Optimise accuracy first, then context tokens.")
METRICS = {"accuracy": exact}          # exact from protocol
def adapt(records): ...                # requires input+label, carries labels; ValueError otherwise

# commands/fetch.py
NAME, HELP = "fetch", "download a paper dataset into data/cache/<name>"
# meta-harness fetch <loader> [--dir DIR]  -> prints {"split": "path"} JSON; unknown loader lists known ones
```

- [ ] **Step 1: Verify sources (network allowed here only, outside pytest)** - for each loader fetch 2 rows (`hf_rows(..., limit=2)` or the raw URL) in a scratchpad script, record fields and total rows in the module docstring, and save trimmed responses as the fixtures above. If a source has moved, use the closest public copy and say so in the docstring.
- [ ] **Step 2: Write the failing tests** - with a fake `http` serving the fixtures by URL (parse `split`/`offset` for HF, exact URL for raw files): each loader writes `train/val/test` JSONL with `{"input","label","labels"}`; train and val are disjoint; a second `fetch` makes zero `http` calls (cache); `fetch` CLI on an unknown name exits 2 listing `symptom2disease`; `get_domain("classification").adapt` rejects a row without `label`.
- [ ] **Step 3: Run to verify it fails** - `python -m pytest -q tests/test_loaders_classification.py` - FAIL (modules missing).
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** `python -m pytest -q tests/test_loaders_classification.py tests/test_registry.py` - PASS.
- [ ] **Step 6: Discriminating mutation** - skip the cache check: the zero-calls assertion FAILS. Carve val from test instead of train: the disjointness assertion (train vs val) still passes, so also assert `len(test) == fixture test rows` and confirm it FAILS. Restore.
- [ ] **Step 7: Commit** - `git add meta_harness/loaders meta_harness/domains/classification.py meta_harness/commands/fetch.py tests/fixtures/loaders tests/test_loaders_classification.py` then `git commit -m "feat(data): Symptom2Disease, LawBench and USPTO-50k loaders"`.

---

### Task 13: The nine out-of-distribution datasets and `ood` evaluation (Wave F)

Gap: G15 (paper Table 5, App. C.1).

**Files:**
- Create: `meta_harness/loaders/ood.py` (one module, nine registered specs via a table; `NAME = "ood"` plus `fetch_one(name, ...)`), `meta_harness/commands/ood.py`
- Fixtures: `tests/fixtures/ood/<name>.json` (3-5 rows each)
- Test: `tests/test_ood.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OodSpec:
    name: str; source: str; text: str | tuple[str, ...]; label: str
    label_names: tuple[str, ...] | None = None    # int label -> name
    split: str = "test"; limit: int = 1000         # paper evaluates a fixed subsample; seeded

OOD_SPECS = {
  "scicite":     OodSpec("scicite", "raw: allenai scicite jsonl (scicite.tar.gz)", "string", "label"),
  "finer139":    OodSpec("finer139", "hf:nlpaueb/finer-139/finer-139", "tokens", "ner_tags"),  # token task -> paper's sentence-level framing, see docstring
  "amazon5":     OodSpec("amazon5", "hf:SetFit/amazon_reviews_multi_en/default", "text", "label_text"),
  "fpb":         OodSpec("fpb", "zip: takala/financial_phrasebank FinancialPhraseBank-v1.0 Sentences_AllAgree.txt", "sentence", "label"),
  "goemotions":  OodSpec("goemotions", "hf:google-research-datasets/go_emotions/simplified", "text", "labels"),
  "banking77":   OodSpec("banking77", "raw: PolyAI task-specific-datasets banking_data/test.csv", "text", "category"),
  "agnews":      OodSpec("agnews", "hf:fancyzhx/ag_news/default", "text", "label", ("World", "Sports", "Business", "Sci/Tech")),
  "scitail":     OodSpec("scitail", "hf:allenai/scitail/tsv_format", ("premise", "hypothesis"), "label"),
  "tweet_hate":  OodSpec("tweet_hate", "hf:cardiffnlp/tweet_eval/hate", "text", "label", ("non-hate", "hate")),
}
def fetch_one(name: str, cache_dir: Path, http: Http = urllib_get) -> Path      # test.jsonl of tasks
```

`meta-harness ood --harness H [--harness H2] --datasets agnews,scitail|all --provider P --model M [--train-from loader]` evaluates each harness frozen (online protocol: trained on the named loader's train split if given, else zero-shot) per OOD set and writes `ood.json` + a table: dataset, accuracy, context tokens, n. Step 1 verifies every source exactly as in Task 12 Step 1 and fixes each spec's fields from what is actually served; multi-label GoEmotions keeps examples with exactly one label (stated in the docstring); FiNER-139 is framed as the paper does (Step 1 reads App. C.1 in `paper.pdf` and records the framing).

- [ ] **Step 1: Verify sources, record fixtures** (as Task 12 Step 1).
- [ ] **Step 2: Failing tests** - every spec in `OOD_SPECS` loads its fixture into tasks with `input`, `label`, `labels`; integer labels map to names; `scitail` joins premise and hypothesis; `limit` subsampling is seeded and stable; `ood` command with a scripted model writes `ood.json` with one row per dataset per harness.
- [ ] **Step 3: Run** `python -m pytest -q tests/test_ood.py` - FAIL.
- [ ] **Step 4: Implement.** **Step 5: Run** - PASS.
- [ ] **Step 6: Discriminating mutation** - drop `label_names` mapping: the AG News label assertion FAILS. Restore.
- [ ] **Step 7: Commit** - `git commit -m "feat(data): nine out-of-distribution text classification sets and ood evaluation"` (explicit paths).

---

### Task 14: The eight discovered classification harnesses (Wave F)

Gap: G14 (App. B.1, Table 9).

**Files:**
- Create: `baselines/classification/discovered/{label_primed_query,draft_verification,error_annotated,cot_replay,cluster_coverage,cascade,rrf_contrastive,relevance_contrastive}.py`
- Modify (owned this wave): `meta_harness/cls_harnesses.py` (LabelPrimedQuery coverage fixed to "best example per label over all memory"; memory updates move to `learn`)
- Test: `tests/test_discovered_harnesses.py`

Each file is a self-contained seed (`class Harness`, `run(task, model) -> dict`, `learn(task, result)`), importing only stdlib and `meta_harness.retrieval`. Behaviour per App. B.1 (Step 1: re-read pages 18-20 of `paper.pdf` and write each harness's one-paragraph spec into its docstring before coding):
- `label_primed_query`: label primer line, then for every label the single most similar memory example of that label (TF-IDF over all memory), then up to `pair_count` query-local contrastive pairs (nearest example with a different label).
- `draft_verification`: draft call; then a verify call with `verify_k` supporters of the draft label and `verify_k` challengers; cold start (< `draft_k` memory) returns the draft.
- `error_annotated`: memory stores each training example with the model's earlier wrong prediction when it was wrong; retrieved examples show "commonly confused with X".
- `cot_replay`: on learn, store a short model-written rationale for each example (one extra call during learn - `learn(task, result, model)`), retrieve and replay rationales.
- `cluster_coverage`: greedy k-center clustering over TF-IDF vectors of memory; one representative per cluster plus top-k nearest.
- `cascade`: cheap zero-shot pass; if the answer is not a valid label or the model's stated confidence is below a threshold, escalate to a retrieval prompt.
- `rrf_contrastive`: reciprocal rank fusion (`retrieval.reciprocal_rank_fusion`) of TF-IDF and BM25 rankings, plus contrastive pairs.
- `relevance_contrastive`: relevance-ranked examples plus the nearest example of each of the top-3 confusable labels.

- [ ] **Step 1: Specs into docstrings** (from `paper.pdf` App. B.1).
- [ ] **Step 2: Failing tests** - parametrized over the eight files: passes `validate_in_subprocess` with a classification probe; after `learn` on 12 labelled rows and a scripted model, `run` returns a declared label; context grows with memory only where the design says so (zero-shot pass of `cascade` uses no examples); `label_primed_query` includes one example of every label present in memory even when they are not in the TF-IDF top-k (the bug fixed from G14); `draft_verification` makes exactly 1 call cold and 2 warm.
- [ ] **Step 3: Run** `python -m pytest -q tests/test_discovered_harnesses.py` - FAIL.
- [ ] **Step 4: Implement.** **Step 5: Run** - PASS.
- [ ] **Step 6: Discriminating mutation** - restore the old `max(8, len(labels))` top-k coverage in `label_primed_query`: the every-label test FAILS. Restore.
- [ ] **Step 7: Commit** - `git commit -m "feat(classification): the eight discovered harnesses from the paper as seeds"`.

---

### Task 15: Classification baselines - few-shot sweep, random and dense few-shot, ACE and MCE reference behaviour (Wave F)

Gap: G13, R9.

**Files:**
- Modify (owned): `baselines/classification/{zero_shot,few_shot,ace,mce}.py` (memory updates move into `learn`), `tests/test_baselines.py`
- Create: `baselines/classification/{random_few_shot,dense_few_shot}.py`, `meta_harness/commands/sweep.py`
- Test: `tests/test_cls_baselines.py`

**Interfaces / behaviour:**
- `few_shot.py`: `N` from `META_HARNESS_FEW_SHOT_N` in {4, 8, 16, 32, all}; `all` uses the whole memory.
- `random_few_shot.py`: N examples sampled with `random.Random(hash of input)` (deterministic per task).
- `dense_few_shot.py`: top-k by cosine over embeddings from `meta_harness.embeddings.embed(texts)` (T3; provider and model from `META_HARNESS_EMBED_PROVIDER` / `META_HARNESS_EMBED_MODEL`, default `openai` / `text-embedding-3-small`); embeddings cached in memory per text; falls back with a clear `RuntimeError` when no embedding provider is configured (the seed then fails validation visibly, never silently degrades to TF-IDF).
- `ace.py`: generator / reflector / curator roles per ACE (Zhang et al.): the reflector writes tagged insights from each error, the curator applies incremental delta updates (`ADD`/`UPDATE`/`REMOVE` on numbered bullets with helpful/harmful counters) instead of rewriting the playbook; bullets with harmful > helpful are pruned. Step 1: read the ACE description in `paper.pdf` (refs and Table 2 notes) and write the exact update rule into the docstring.
- `mce.py`: per its reference behaviour as described in the paper (Step 1 records it in the docstring): skills evolved from errors with retention by measured usefulness.
- `meta-harness sweep --harness baselines/classification/few_shot.py --env META_HARNESS_FEW_SHOT_N=4,8,16,32,all <search data/provider args>` evaluates each setting (no proposer) and writes `sweep.json`.

- [ ] **Step 1: Docstrings from the paper** (ACE, MCE).
- [ ] **Step 2: Failing tests** - `run` never mutates memory (call `run` twice on labelled tasks, memory empty); `learn` adds; few-shot `all` uses every example; random few-shot is deterministic per input; dense few-shot with a fake `embed` ranks the semantically planted neighbour first and raises without a provider; ACE curator applies `UPDATE 2: ...` to bullet 2 only and prunes a bullet whose harmful count exceeds helpful; sweep writes five rows.
- [ ] **Step 3: Run** `python -m pytest -q tests/test_cls_baselines.py tests/test_baselines.py` - FAIL.
- [ ] **Step 4: Implement.** **Step 5: Run** - PASS.
- [ ] **Step 6: Discriminating mutation** - make the ACE curator rewrite the whole playbook: the single-bullet update test FAILS. Restore.
- [ ] **Step 7: Commit** - `git commit -m "feat(classification): few-shot sweep, random and dense few-shot, ACE and MCE update rules"`.

---

### Task 16: Math data - corpus build, decontamination, search and IMO-style test loaders (Wave F)

Gap: G18, G19, R2.

**Files:**
- Create: `meta_harness/loaders/math_corpus.py`, `meta_harness/loaders/math_search.py`, `meta_harness/loaders/imo.py`, `meta_harness/decontam.py`, `meta_harness/commands/math_corpus.py`
- Fixtures: `tests/fixtures/math/*.json`
- Test: `tests/test_math_data.py`

**Interfaces:**

```python
# decontam.py
def normalise(text: str) -> str                      # lowercase, collapse whitespace, strip LaTeX wrappers
def shingles(text: str, n: int = 5) -> set[str]       # word n-grams of normalise(text)
def jaccard(a: set[str], b: set[str]) -> float
def decontaminate(corpus: Iterable[dict], held_out: Sequence[str], prefix_chars: int = 60,
                  threshold: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Drop a corpus item when its normalised problem shares the first `prefix_chars` with any
    held-out problem, or its shingle Jaccard with any held-out problem is >= threshold.
    Candidate pairs come from an inverted shingle index (no O(N*M) scan).
    Returns (kept, removed) and the caller writes removed ids to decontam_report.json."""
def dedup(corpus: Iterable[dict], threshold: float = 0.8) -> list[dict]   # same test, within corpus

# loaders/math_corpus.py
NAME, DOMAIN = "math_corpus", "math"
SOURCES: dict[str, str]     # the eight Table 10 datasets -> HF ids; Step 1 reads Table 10 in paper.pdf
def fetch(cache_dir, http=urllib_get, per_source_limit: int | None = None) -> dict[str, Path]
    # -> {"corpus": corpus.jsonl} rows {"id","problem","solution","source","difficulty"}
# commands/math_corpus.py: meta-harness math-corpus build [--limit-per-source N]
#   fetch -> dedup -> decontaminate against math_search + imo problems -> data/cache/math_corpus/corpus.clean.jsonl
#   + decontam_report.json; the BM25 index is built lazily by the harness (T17), not stored.

# loaders/math_search.py  (paper: 250 hard problems from OlympiadBench + Omni-MATH)
NAME, DOMAIN = "math_search", "math"
def fetch(cache_dir, http=urllib_get, size: int = 250, seed: int = 0) -> dict[str, Path]  # {"search": ...}

# loaders/imo.py  (paper: 200 IMO-level test problems, Table 11)
NAME, DOMAIN = "imo", "math"
def fetch(cache_dir, http=urllib_get) -> dict[str, Path]   # {"test": ...}; answer-gradable items only,
    # proof-only items are listed in skipped.json (proof grading is out of reach of a string metric; README says so)
```

- [ ] **Step 1: Sources** - read Tables 10-11 and §4.2 in `paper.pdf`; pin each HF id or URL; record field names and counts in docstrings; save fixtures.
- [ ] **Step 2: Failing tests** - `decontaminate` removes an exact copy, a prefix match and a near-duplicate with one word changed (Jaccard >= 0.8), keeps an unrelated problem; inverted index gives the same result as a brute-force reference on 50 random items; `dedup` keeps one of two near-duplicates; loaders map fixture rows to `{"problem","answer"}` (search/test) and `{"problem","solution","difficulty"}` (corpus); the build command over fixtures writes `corpus.clean.jsonl` without any held-out problem.
- [ ] **Step 3: Run** `python -m pytest -q tests/test_math_data.py` - FAIL.
- [ ] **Step 4: Implement.** **Step 5: Run** - PASS.
- [ ] **Step 6: Discriminating mutation** - compare raw text instead of `normalise`d text: the whitespace/LaTeX variant test FAILS. Restore.
- [ ] **Step 7: Commit** - `git commit -m "feat(math): corpus build with dedup and decontamination; search and IMO loaders"`.

---

### Task 17: Math harness - full router and math seeds (Wave F)

Gap: G20, G19.

**Files:**
- Modify (owned): `meta_harness/math_harnesses.py`
- Create: `meta_harness/domains/math.py`, `baselines/math/{router,random_few_shot,dense,ace}.py`
- Test: `tests/test_math_harness.py`

**Behaviour** (App. B.2 / C.2; Step 1 re-reads pages 19-21, 23 and writes the spec into the docstring):
- Router order exactly as the paper's; routes geometry / number theory / combinatorics / default.
- Geometry uses a **separate hard index** built from corpus items with `difficulty > 6` (NuminaMath-sourced), not a filter over the main top-10; returns 1 hard + up to 2 main, deduplicated by id.
- Per-route BM25 depth (combinatorics 20 -> dedup -> 8; number theory 12; default 10) then difficulty rerank.
- **Adaptive K** on the default route: K grows from 1 to `max_k` while the next score is within a ratio of the top score.
- Technique check: a solution "states its technique early" when one of a technique lexicon (`induction`, `pigeonhole`, `invariant`, `extremal`, `vieta`, `am-gm`, `cauchy`, `modulo`, `bijection`, ...) appears in its first 300 characters - replaces the `"therefore"` stand-in.
- Corpus items longer than 4000 characters are excluded; solutions are truncated at 3000.
- `domains/math.py`: `NAME="math"`, `PROTOCOL="plain"`, `METRICS={"math_equivalence": ...}` wrapping `metrics.math_equivalence` in `(task, result)` form, `SEED_DIR=baselines/math`, `PROMPT` with the route/retrieval notes, `adapt` requiring `problem`.
- Seeds: `router.py` (the above, corpus path from `META_HARNESS_CORPUS`), `random_few_shot.py`, `dense.py` (k=1 and k=5 via `META_HARNESS_RETRIEVAL_K`, embeddings from T3), `ace.py` (math playbook).

- [ ] **Step 1: Spec docstring.**
- [ ] **Step 2: Failing tests** - route table over 8 crafted problems; geometry returns a hard item that is outside the main top-10; no duplicate ids; adaptive K returns 1 when the top score dominates and 3 when three are close; technique check true/false cases; >4000-char items never retrieved; `numeric`/`math_equivalence` metric through the domain.
- [ ] **Step 3: Run** `python -m pytest -q tests/test_math_harness.py` - FAIL. **Step 4: Implement.** **Step 5: Run** - PASS.
- [ ] **Step 6: Discriminating mutation** - filter geometry from the main top-10 again: the hard-index test FAILS. Restore.
- [ ] **Step 7: Commit** - `git commit -m "feat(math): full lexical router with a hard geometry index and adaptive K"`.

---

### Task 18: TerminalBench-2 style agentic coding - searchable harness, Terminus-2 and KIRA seeds, Docker (Wave F)

Gap: G21, R2; spec §8.

**Files:**
- Modify (owned): `meta_harness/terminal.py`, `meta_harness/terminal_harness.py`, `meta_harness/metrics.py` (`terminal_metric` only)
- Create: `meta_harness/domains/terminal.py`, `meta_harness/loaders/terminalbench.py`, `baselines/terminal/{terminus2,terminus_kira}.py`; delete `baselines/terminal/basic.py`
- Test: `tests/test_terminal_domain.py`, update `tests/test_terminal_harness.py`

**Interfaces / behaviour:**

```python
# terminal.py
class Container:
    """docker run -d --network none? (task decides) image; exec; copy; rm. All subprocess calls use
    encoding="utf-8", errors="replace" and timeouts. available() -> bool checks `docker version`."""
    def __init__(self, image: str, workdir: str = "/app", network: bool = False): ...
    def start(self) -> str; def exec(self, command: str, timeout: float) -> tuple[int, str]; def remove(self) -> None

# The harness contract for this domain: the harness's run(task, model) receives task["container"]
# (a started Container's id, created by the DOMAIN, not the harness) and returns {"output": summary}.
# domains/terminal.py wraps evaluation: start container -> put id on the task -> harness.run ->
# terminal_metric(task) runs test_command INSIDE that same container -> remove container.
```

- `terminal_metric(task, result)`: requires `task["container"]`; runs `docker exec -w <workdir> <id> bash -lc <test_command>`; no container -> score 0.0 and never runs anything on the host (fixes G21's host-execution bug).
- `terminus2.py`: a port of the Terminus-2 behaviour (App. B.3): JSON command batches, tmux-style keystroke execution through `Container.exec`, output capped at 30KB per observation, explicit completion confirmation turn.
- `terminus_kira.py`: Terminus-KIRA additions per App. B.3 (native tool calling through `model.chat` when available, completion checklist, environment snapshot injected at start from inside the container).
- `loaders/terminalbench.py`: reads a local TerminalBench-2 checkout (`--tb-dir`; Step 1 records the task directory format: `task.yaml`, Dockerfile, tests) into tasks `{"instruction","image","test_command","workdir","timeout","id"}`; no network in tests. A regex leakage audit (`audit_leakage(harness_source, tasks)`) flags harness source containing task ids, test file names or expected outputs (paper's audit).
- Search on this domain uses search = test (paper) with `--test` equal to `--search`; README states it.

**Spec §11 additions (Terminus-style seeds):** the seeds run commands through the `Sandbox` protocol (Task 7A: `DockerSandbox` when available, `TempDirSandbox` otherwise for local tests) instead of a private container class - `terminal.Container` becomes a thin wrapper over `DockerSandbox` kept for the domain. Both seeds get (a) an `edit(path, old, new)` tool: exact match first; if absent, a fuzzy match (whitespace-normalised, then the best `difflib.SequenceMatcher` window with ratio >= 0.9, refusing ambiguous matches) and edits to one file are serialised; (b) context compaction: when the conversation exceeds `0.8 * context_window`, keep the most recent ~20k tokens cut at a turn boundary, replace older turns with an LLM summary, and continue with "Continue from the summary; do not redo work already done." Tests: exact edit, fuzzy edit across changed indentation, ambiguous fuzzy match refused; compaction triggers at the threshold, cuts at a turn boundary, inserts the continuation line, and the scripted model receives the summary request.

- [ ] **Step 1: Read App. B.3 and the TB-2 task format; write specs into docstrings.**
- [ ] **Step 2: Failing tests** - with a fake `Container` (records exec calls): the metric runs the test command via the same container id the harness used; without a container the metric returns 0.0 and `subprocess.run` is never called (monkeypatched to raise); the domain removes the container even when the harness raises; terminus2 caps an observation at 30KB; the leakage audit flags a source containing a task id. A Docker-gated test (`pytest.mark.skipif(not Container.available(), reason="docker not installed")`) runs `alpine` with `echo ok > /app/x` and a test command `test -f /app/x`.
- [ ] **Step 3: Run** `python -m pytest -q tests/test_terminal_domain.py tests/test_terminal_harness.py` - FAIL. **Step 4: Implement.** **Step 5: Run** - PASS.
- [ ] **Step 6: Discriminating mutation** - fall back to `shell=True` on the host when no container: the never-on-host test FAILS. Restore.
- [ ] **Step 7: Commit** - `git commit -m "feat(terminal): searchable TerminalBench-2 harness with Terminus-2 and KIRA seeds in Docker"`.

---

### Task 19: OpenEvolve-style and TTT-Discover-style optimizers (Wave F)

Gap: G16, R5.

**Files:**
- Create: `meta_harness/optimizers/openevolve.py`, `meta_harness/optimizers/ttt_discover.py`
- Test: `tests/test_optimizers_evolve.py`

**Interfaces:** both are `Proposer`s built by `build(ctx)`; they use `ctx.proposer_model(prompt)` (text, no tools), read only `experience.results()` and each candidate's `harness.py` (never traces of other runs, never `test_root`), write `proposals/iteration-NNNN/candidate-KK.py` plus `candidate-KK.parent.json` (`{"parents": [...], "operator": "..."}`) and `.reasoning.md`.

- `openevolve`: MAP-Elites archive keyed by (score decile, context-cost decile) with `islands=3` (candidate i belongs to island `hash(id) % 3`; migration of the island best every 5 iterations); parent sampling: with p=0.7 exploit (an elite of the current island, weighted by score), else explore (uniform archive member); 2 inspiration programs (top-2 by score not the parent); prompt shows parent source + scores + inspirations' sources/scores and asks for a full rewrite; `random.Random(ctx.seed + iteration)`.
- `ttt_discover`: a tree whose nodes are candidates (`parent.json` lineage, seeds are roots); selection by PUCT `Q(n) + c * P(n) * sqrt(N(parent)) / (1 + N(n))` with `Q` = best score in the subtree normalised to [0,1], `P` uniform over siblings, `N` visit counts persisted in `proposals/ttt_state.json`, `c = 1.0`; the selected node is expanded by one LLM mutation conditioned on its source, score and its best child's source. Docstring states that the test-time weight-training half is not reproducible via hosted APIs.

- [ ] **Step 1: Failing tests** - with a fake experience of 6 scored candidates and a scripted text model: openevolve's parent is always from the archive, `parent.json` written, exploit/explore ratio within 0.6-0.8 over 200 seeded draws, migration happens at iteration 5; ttt picks the unvisited child over a visited sibling with equal Q, then the higher-Q branch once visits equalise; `ttt_state.json` round-trips; neither opens any path under `test_root` (monkeypatch `Path.open`/`read_text` to record paths).
- [ ] **Step 2: Run** `python -m pytest -q tests/test_optimizers_evolve.py` - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Discriminating mutation** - drop the `sqrt(N(parent))` term: the unvisited-child test FAILS. Restore.
- [ ] **Step 6: Commit** - `git commit -m "feat(optimizers): OpenEvolve-style and TTT-Discover-style comparison loops"`.

---

### Task 20: GEPA-style reflective optimizer (Wave F)

Gap: G16, R5.

**Files:** Create `meta_harness/optimizers/gepa.py`; Test `tests/test_optimizers_gepa.py`.

**Behaviour:** per-task rewards are read from each candidate's `traces.jsonl` `task_end` events; the candidate pool's Pareto set is the candidates that are best on at least one task; the parent is sampled with probability proportional to the number of tasks it wins; a minibatch of 3 tasks where the parent scored lowest is shown with input, output and reward (from the trace); the model writes a reflection then a revised harness (reflective mutation); every 4th proposal is a merge of two Pareto parents with disjoint task wins. `parent.json` records parents and operator (`reflect` / `merge`).

- [ ] **Step 1: Failing tests** - Pareto set over a crafted 3x4 reward matrix; sampling proportional to wins (seeded, 300 draws); minibatch picks the 3 lowest-reward tasks; proposal 4 is a merge with two parents; reads only `candidates/*/traces.jsonl` of this run.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_optimizers_gepa.py` - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Discriminating mutation** - pick the overall-best candidate as parent: the proportional-sampling test FAILS. Restore.
- [ ] **Step 6: Commit** - `git commit -m "feat(optimizers): GEPA-style reflective optimizer over per-task Pareto parents"`.

---

### Task 21: Multi-model transfer (Wave F)

Gap: G18 (Table 6), R7.

**Files:** Create `meta_harness/commands/transfer.py`, `meta_harness/transfer.py`; Test `tests/test_transfer.py`.

**Interfaces:**

```python
def transfer(harnesses: Sequence[Path], tasks: Sequence[Mapping], models: Sequence[tuple[str, str]],
             metric: Metric, root: Path, repeats: int = 1, protocol: str = "plain",
             train_tasks: Sequence[Mapping] = ()) -> dict:
    """Evaluate each fixed harness with each (provider, model); no proposer. Writes
    root/transfer.json {"rows": [{"harness","provider","model","score","score_std","context_cost","n"}]}
    and per-cell traces under root/<harness-stem>/<provider>-<model-slug>/."""
```

CLI: `meta-harness transfer --harness A --harness B --data test.jsonl --metric M --models unikey:gpt-5-mini,groq:llama-3.3-70b-versatile [--domain math]` prints a harness x model table.

- [ ] **Step 1: Failing tests** - two harnesses x two scripted models (the `scripted` provider with different scripts) produce four rows with the expected scores; a model whose provider construction fails yields a row with `"error"` and the run continues; slugs are Windows-safe (no `:`/`/`).
- [ ] **Step 2: Run** `python -m pytest -q tests/test_transfer.py` - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Discriminating mutation** - reuse one harness instance across models (memory leaks between cells): add the assertion that a memorising harness scores the same for both models and confirm it FAILS. Restore.
- [ ] **Step 6: Commit** - `git commit -m "feat: transfer evaluates fixed harnesses across models"`.

---

### Task 22: Experience query CLI and proposer read statistics (Wave F)

Gap: G5, G22, R8.

**Files:** Create `meta_harness/experience_query.py`, `meta_harness/commands/experience.py`; Test `tests/test_experience_query.py`.

**Interfaces:**

```python
def top_k(root: Path, k: int = 5, by: str = "score") -> list[dict]        # valid only; ties by context_cost
def diff_candidates(root: Path, a: str, b: str) -> str                    # unified diff of harness.py + score/cost deltas
def diff_runs(root_a: Path, root_b: Path) -> dict                         # frontier, best score, budget side by side
def import_candidates(source_root: Path, dest_seeds: Path, k: int = 3) -> list[Path]
    # copies the top-k harness.py files as seed files (warm start); `search --warm-start ROOT --warm-k K`
    # is NOT edited here: the command prints the --harness flags to use.
CATEGORIES = ("source", "scores", "traces", "reasoning", "summary", "run_meta", "other")
def categorise(path: str) -> str      # harness.py->source, scores.json->scores, traces*.jsonl->traces, ...
def read_stats(root: Path) -> dict:
    """From every candidate's proposer_log.jsonl: per iteration, files read (read calls with ok),
    distinct files, grep calls, list calls, and the category breakdown (paper Table 8);
    plus run-level medians."""
```

CLI: `meta-harness experience top|diff|diff-runs|import|reads ...`.

**Spec §11 additions (keel):** `experience report --root R` aggregates `receipt.json` verdicts and stages (counts by verdict, by optimizer iteration) and joins `tool_outcomes` to scores (Table 8 with denied/failed/aborted); `experience export --root R --out cases.jsonl` writes one read-only case per receipt (`id`, verdict, parents, fingerprint, scores, redacted rationale). Readers use `read_jsonl(tolerate_torn=True)`. Tests: report counts for a fake run with one receipt of each verdict; export lines equal receipts in number and contain no `sk-` strings.

- [ ] **Step 1: Failing tests** - build a fake run with 3 candidates and two `proposer_log.jsonl` files; `top_k` order; `diff_candidates` contains `-`/`+` lines and the score delta; `read_stats` counts only `ok` reads, categories sum to the read count, medians correct; `import_candidates` copies files whose content equals the sources.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_experience_query.py` - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Discriminating mutation** - count refused reads: the `ok`-only assertion FAILS. Restore.
- [ ] **Step 6: Commit** - `git commit -m "feat: experience query CLI and proposer read statistics (Table 8)"`.

---

### Task 22A: Benchmark matrix runner and blind LLM-judge metric (Wave F)

Spec §11 (Benchmark runner).

**Files:** Create `meta_harness/bench.py`, `meta_harness/judge.py`, `meta_harness/commands/bench.py`; Test `tests/test_bench.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class Arm: name: str; harness: Path; provider: str; model: str
def run_matrix(arms: Sequence[Arm], tasks: Sequence[Mapping], trials: int, metric: Metric, out: Path,
               cell_timeout: float = 600.0, retries: int = 2) -> Path:
    """Resumable {arm x task x trial} matrix in out/matrix.jsonl, one line per finished cell
    {"arm","task_id","trial","score","error","usage","seconds"}; on restart, cells already in the
    file are skipped. Each cell runs in a subprocess (Task 7A TempDirSandbox) with a hard timeout;
    provider 429/5xx are retried up to `retries`; harness errors and timeouts are not retried and
    are kept with score 0. Writes out/summary.json: per-arm mean, std over trials, failures."""

# judge.py
JUDGE_PROMPT = ("You are a strict grader. Criteria:\n{criteria}\n\nAnswer:\n{answer}\n\n"
                "Reason briefly, then end with exactly one line: VERDICT: PASS or VERDICT: FAIL")
def verdict(task, result) -> float:
    """Blind: the judge sees task['criteria'] (or the expected value) and result['output'] only -
    never the harness, its trace or other candidates. Judge model from META_HARNESS_JUDGE_PROVIDER /
    META_HARNESS_JUDGE_MODEL. Parses the LAST 'VERDICT:' line; anything else scores 0.0."""
```

Use as `--metric meta_harness.judge:verdict`. CLI: `meta-harness bench --arm name=harness.py@provider:model ... --data tasks.jsonl --trials 3 --metric M --out DIR [--resume]`.

- [ ] **Step 1: Failing tests** - 2 arms x 3 tasks x 2 trials with scripted models produce 12 lines; killing after 5 cells (fake runner raising) and rerunning produces exactly 12 lines with no duplicates; a harness raising scores 0 with `error` and is not retried (call counter); a scripted 429 is retried once then succeeds; `verdict` returns 1.0 for "...\nVERDICT: PASS", 0.0 for "VERDICT: PASS\nVERDICT: FAIL" (last wins) and for no verdict line; the judge prompt contains no harness source.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_bench.py` - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Discriminating mutation** - parse the first `VERDICT:` line: the last-wins test FAILS. Restore.
- [ ] **Step 6: Commit** - `git add meta_harness/bench.py meta_harness/judge.py meta_harness/commands/bench.py tests/test_bench.py` then `git commit -m "feat: resumable benchmark matrix and a blind LLM-judge metric"`.

---

### Task 23: Sandboxing - subprocess evaluation with resource limits, Docker when available (Wave G)

Gap: G6, G7 (post-hoc timeout), R10; spec §8.

**Files:**
- Create: `meta_harness/isolation.py`
- Modify (owned this wave): `meta_harness/core.py` (`CandidateEvaluator(isolation="none"|"process"|"docker")`, default `process`), `meta_harness/sandbox.py` (validation reuses the same runner), `meta_harness/commands/search.py` (`--isolation`, `--memory-mb`, `--cpu-seconds`)
- Test: `tests/test_isolation.py`

**Interfaces:**

```python
# Parent <-> child protocol over the child's stdin/stdout, one JSON object per line:
#   parent -> child: {"op":"load","path":...} | {"op":"run","task":{...},"learn":bool} | {"op":"model_result","text":...,"usage":{...}} | {"op":"exit"}
#   child  -> parent: {"op":"model_call","prompt":...,"kwargs":{...}} | {"op":"trace","event":...,"payload":...}
#                     | {"op":"result","result":{...}} | {"op":"error","error":"..."}
# Model calls are served by the PARENT, so the child needs no network and no keys.
class IsolatedHarness:
    def __init__(self, source: Path, mode: str = "process", memory_mb: int = 2048, cpu_seconds: int = 600,
                 image: str = "python:3.12-slim"): ...
    def run(self, task, model, trace, learn: bool = False) -> dict    # preemptive per-task timeout: kill child
    def close(self) -> None
```

- `process`: `python -I -m meta_harness.isolation <path>` with `env` stripped of every `*_API_KEY`/`*_TOKEN`; POSIX limits via `resource.setrlimit(RLIMIT_AS, RLIMIT_CPU, RLIMIT_NOFILE)` in `preexec_fn`; Windows: a Job Object via `ctypes` with `JOB_OBJECT_LIMIT_PROCESS_MEMORY` and kill-on-close; if `ctypes` setup fails, timeout-only with a warning in the trace.
- `docker`: `docker run -i --rm --network none --memory <mb>m --cpus 1 -v <repo>:/mh:ro -v <candidate dir>:/c:ro python:3.12-slim python -I -m meta_harness.isolation /c/harness.py` with `PYTHONPATH=/mh`.
- Online protocol: the child keeps harness state across `run`/`learn` messages; `evaluate_frozen` sends a `{"op":"snapshot"}`/`{"op":"restore"}` pair (child-side `deepcopy`) per test item.

**Spec §11 additions:** `docker` mode is built on Task 7A's `DockerSandbox` (`put` the candidate and repo into the container, `exec` the child) and runs `preflight()` first; a failed preflight aborts the search with its install hint instead of falling back silently.

- [ ] **Step 1: Failing tests** - a harness that sleeps 30 s on one task is killed by a 2 s task timeout and the result is invalid with "timeout" (preemptive, total wall time < 10 s); a harness reading `os.environ["UNIKEY_API_KEY"]` sees nothing; a harness allocating 4 GB fails under `memory_mb=256` (skip on platforms where the limit cannot be applied, with the reason); model calls round-trip through the parent (scripted model sees the prompt); state persists between `run` and `learn`; Docker-gated test runs one task with `mode="docker"`.
- [ ] **Step 2: Run** `python -m pytest -q tests/test_isolation.py` - FAIL. **Step 3: Implement.** **Step 4: Run** `python -m pytest -q` - PASS.
- [ ] **Step 5: Discriminating mutation** - pass the parent environment through unchanged: the API-key test FAILS. Restore.
- [ ] **Step 6: Commit** - `git commit -m "feat: evaluate candidates in a limited subprocess or a Docker container"`.

---
### Task 24: `meta-harness ablate` - Table 3 shape (Wave G)

**Files:**
- Create: `meta_harness/ablate.py`
- Modify: `meta_harness/__main__.py`
- Test: `tests/test_ablate.py`

**Interfaces:**

```python
def run_ablation(search_argv: list[str], views: Sequence[str], runs: int, root: Path,
                 search: Callable[[list[str]], int]) -> dict:
    """For each view and run i, calls search(search_argv + ["--view", v, "--root", root/v/f"run-{i}"]).
    Reads each run's frontier.json (best search score) and test_results.json (best test score).
    Returns {"views": {v: {"runs": n, "search_best": [...], "test_best": [...],
             "median_search": m, "best_search": b, "median_test": m, "best_test": b,
             "above_seed": k}}}  # above_seed: runs whose best search score beats the best initial-*
    and writes it to root/ablation.json."""
```

CLI: `meta-harness ablate --views scores,summary,full --runs 3 --root .meta-harness-ablate -- <search args>`; everything after `--` is passed to `search` (which must not include `--view` or `--root`; refused with exit 2). Output also prints a plain-text table: view, median, best, >seed.

- [ ] **Step 1: Test** - with a fake `search` callable that writes canned `frontier.json`/`test_results.json`/`candidates/initial-0000/scores.json` per root, assert medians, bests and `above_seed` per view, that roots are distinct per view/run, and that passing `--view` inside the search args exits 2. Plus one real offline run: `views=["scores","full"], runs=1` with the scripted smoke fixtures through `cli.main`, asserting `ablation.json` has both views.
- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_ablate.py` - FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `python -m pytest -q` - PASS.
- [ ] **Step 5: Discriminating mutation** - use `mean` instead of `median`: the canned scores `[0.1, 0.2, 0.9]` assertion FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add meta_harness/ablate.py meta_harness/__main__.py tests/test_ablate.py
git commit -m "feat: ablate runs the search per experience view and reports median and best"
```

---

### Task 25: README rewrite and `examples/` (Wave H)

**Files:**
- Rewrite: `README.md`
- Modify: `baselines/README.md` (contract: `run(task, model[, trace])`, optional `learn`)
- Create: `examples/sentiment/harness.py`, `examples/sentiment/data.jsonl` (30 hand-written rows, 3 labels), `examples/sentiment/metric.py` (`def score(task, result)`), `examples/sentiment/base_script.jsonl`, `examples/sentiment/proposer_script.jsonl`, `examples/README.md`
- Test: `tests/test_examples.py`

README sections, in order:
1. **What** - one paragraph: a standalone implementation of Meta-Harness (cite arXiv 2603.28052); harness = code around a model; search = a tool-using proposer reading every prior candidate's source, scores and traces.
2. **Install and quick start** - `pip install -e .`; the offline example command (scripted, runs with no key); then the real one (spec §1) and a Unikey variant (`--provider unikey`, `UNIKEY_API_KEY` from env).
3. **Plug-and-play contracts** - harness file, data, metric, providers, proposer provider; table of `search` flags.
4. **The experience directory** - layout of `candidates/<id>/{harness.py,scores.json,traces.jsonl,proposer_log.jsonl,proposer_reasoning.md}`, `run.json` (budget), `frontier.json`, `<root>-test/test_results.json`; the proposer can never read the test directory.
5. **Paper domain: online text classification** - `fetch symptom2disease`, the exact `search --protocol online` command, what the protocol does.
6. **Ablation and baselines** - `ablate`, `--optimizer best-of-n`; note on the `scores` view including code.
7. **How it differs** - from the paper's setup (no Claude Code; proposer is any tool-calling model; Pareto over provider-reported tokens), from prompt optimizers (GEPA/OpenEvolve: code-level, full-history filesystem rather than summaries).
8. **Out of scope / roadmap** - spec §8 list verbatim, each as "not implemented".
9. **Results** - only numbers produced by commands actually run, each with the command, date, model and `run.json` budget. Until Task 26 runs, the section says: "No paper result has been reproduced yet. The offline smoke run checks the pipeline, not model quality." Never copy paper numbers as ours.

- [ ] **Step 1: Test**

```python
def test_example_runs_offline(tmp_path, capsys):
    ex = Path(__file__).resolve().parent.parent / "examples" / "sentiment"
    assert cli.main(["search", "--harness", str(ex / "harness.py"), "--data", str(ex / "data.jsonl"),
                     "--metric", f"{ex / 'metric.py'}:score", "--provider", "scripted",
                     "--model", str(ex / "base_script.jsonl"), "--proposer-provider", "scripted",
                     "--proposer-model", str(ex / "proposer_script.jsonl"), "--iterations", "1",
                     "--root", str(tmp_path / "s"), "--no-cache"]) == 0


def test_readme_commands_match_the_cli():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    parser = cli.build_parser()
    for line in re.findall(r"^meta-harness (\w+)", readme, re.M):
        assert line in parser._subparsers._group_actions[0].choices
    assert "claude-cli" not in readme and "/plugin" not in readme


def test_readme_results_cite_only_produced_runs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = readme.split("## Results", 1)[1]
    for number in re.findall(r"\b0\.\d{2,}\b|\b\d{2}\.\d%", results):
        assert "run.json" in results, f"result {number} has no run.json provenance"
```

- [ ] **Step 2: Run to verify it fails** - `python -m pytest -q tests/test_examples.py` - FAIL (no `examples/`).
- [ ] **Step 3: Write** the README and examples.
- [ ] **Step 4: Run** `python -m pytest -q` - PASS. Read the README top to bottom once against the spec §1 success list.
- [ ] **Step 5: Discriminating mutation** - add `meta-harness learn` to a README code block: `test_readme_commands_match_the_cli` FAILS. Restore.
- [ ] **Step 6: Commit**

```bash
git add README.md baselines/README.md examples tests/test_examples.py
git commit -m "docs: README for the standalone engine and a runnable offline example"
```

---

### Task 26: Live verification (owner keys; not part of the test suite) (Wave I)

Keys come from the environment or the git-ignored `.env` (loaded by the CLI). They are never echoed, written to a file, passed on a command line or stored in `run.json`/traces. Nothing here is a pytest test; outputs live under git-ignored `.meta-harness-*` roots. Provider order: **unikey first**, then **groq**, then **openai** as fallbacks.

**Files:**
- Create: `tests/fixtures/providers/live_tool_calls.json` (one recorded real response, no headers)
- Modify: `README.md` (Results section, only numbers produced here), `tests/test_provider_tools.py` (one parametrized case for the recorded fixture)

- [ ] **Step 1: Which keys are present (names only)**

```bash
python -c "from meta_harness.envfile import load_env; import os; load_env(); print([k for k in ('UNIKEY_API_KEY','GROQ_API_KEY','OPENAI_API_KEY') if os.environ.get(k)])"
```

Use the first present in order unikey, groq, openai as `$P`.

- [ ] **Step 2: List models** - `python -m meta_harness models --provider $P` (implemented for unikey, groq and openai: `GET <base>/models`). Pick a cheap base model and a strong tool-capable proposer model; record both ids.

- [ ] **Step 3: Record one real tool turn and diff its shape** - a scratchpad script (not in the repo) calls `model_from_environment(P, proposer_model).chat("You are a test.", [{"role": "user", "content": "Call list with path '.'"}], TOOL_SPECS)` through a wrapped `client.post` that saves the raw response to `tests/fixtures/providers/live_tool_calls.json`. Add the parametrized parse case; it must yield one `list` call. A shape difference is fixed in `parse_openai`/`parse_anthropic` with a regression test before continuing. If `$P` fails (auth, 404 model, no tool support), record the one-line error (no key material) and move to the next provider.

- [ ] **Step 4: Small real search on classification**

```bash
python -m meta_harness fetch symptom2disease
python -m meta_harness search \
  --harness baselines/classification/zero_shot.py --harness baselines/classification/few_shot.py \
  --train data/cache/symptom2disease/train.jsonl --search data/cache/symptom2disease/val.jsonl \
  --test data/cache/symptom2disease/test.jsonl --protocol online --train-limit 100 \
  --domain classification --metric accuracy \
  --provider $P --model <base-id> --proposer-provider $P --proposer-model <proposer-id> \
  --iterations 3 --root .meta-harness-live-s2d
```

Check: `run.json` budget non-zero; no key in any file under the root (a `python -c` scan comparing against the env values, printing only `True`/`False`); every `iteration-*` candidate has a `proposer_log.jsonl` with at least one ok `read`; no log path contains `-test`; `test_results.json` exists; `meta-harness experience reads --root .meta-harness-live-s2d` prints Table 8 style counts.

- [ ] **Step 5: Optional same-budget comparison** - repeat Step 4 with `--optimizer best-of-n` and the same `--max-evaluations`; record both.

- [ ] **Step 6: README Results** - one row per frontier candidate (search accuracy, test accuracy, context tokens), the provider and model ids, date, `--train-limit`, iterations, budget totals and the exact command. State the scale (3 iterations, 100-example stream, one dataset) and that it is not comparable to the paper's Table 2.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/providers/live_tool_calls.json tests/test_provider_tools.py README.md
git commit -m "docs: first live Symptom2Disease run with recorded provider fixture"
```

---

## Self-review against the spec and the gap analysis

| Spec item / gap | Task |
|---|---|
| §1.1 stdlib + API key only | T1, T2 (`.env`), T3, T11 |
| §1.2 tool-using proposer, any provider (G2, R1) | T3, T4, T6, T9 |
| §1.3 evaluation bugs pinned by tests (G9, G11, G12, G6, G21; R3, R4) | T8 (fixes 1, 2), T10 (fixes 3, 4, 5), T18 (terminal scoring inside the container) |
| §1.4 real dataset + offline scripted run | T11 (offline smoke), T12, T26 |
| §1.5 Table 3 ablation + Best-of-N (G17, R6) | T7, T9, T24 |
| §1.6 README | T25 |
| §2 deletions, `__main__` engine-only, `keel/` untouched | T1, T2 (skill moved in T9) |
| §3 contracts, providers (+ unikey, groq, scripted), proposer provider | T2, T3, T4, T5, T11 |
| §4 tools, 40-call bound, `proposer_log.jsonl`, refusal of outside paths/test split, domain prompt sections, views (G3, G8) | T6, T7, T9, domain `PROMPT`s in T12/T17/T18 |
| §5.1-5.6 | T8, T8, T5+T10, T9+T10, T10, T18 |
| §6 Symptom2Disease, seeds, protocol | T12, T15, T5, T10 |
| §7 ablate, best-of-n, budget in `run.json` (G10) | T24, T9, T10 |
| §8 all classification datasets (G12) + 9 OOD (G15) | T12, T13 |
| §8 eight discovered harnesses (G14), dense/random few-shot, N sweep, ACE/MCE (G13) | T14, T15 |
| §8 math corpus, decontamination, IMO loader, full router, math baselines (G18-G20) | T16, T17 |
| §8 TerminalBench-2 harness, Terminus-2/KIRA, Docker, container on task, leakage audit (G21) | T18 |
| §8 Best-of-N, OpenEvolve, TTT-Discover, GEPA under equal budget (G16) | T9, T19, T20, T10 (`max_evaluations`) |
| §8 multi-model transfer (G23, Table 6) | T21 |
| §8 Table 8 read stats + experience CLI with top-k, diff, reuse (G5, G22) | T22 |
| §8 sandboxing (G7 preemptive timeout, R10) | T23 |
| Gap G23 no reasoning-effort config | T3 (`--reasoning-effort` passed through as provider kwargs where supported) |
| Gap G1, G4 (already implemented) | kept; covered by existing core tests |
| §11 proposer loop: terminate on write, before-tool hook, cap, orphan closing, unknown tools, `length` failure | T9 (T3 closes orphans in the wire format) |
| §11 tool output caps + spill | T6, T9 (per-turn batch cap) |
| §11 provider normalisation, usage record, billed cost separate, scripted text/tool/error | T2, T3, T4, T10, T11 |
| §11 resume, checkpoints, per-task failures as scores, timeouts, retry policy | T10, T11, T3 (`_HttpClient`), T22A |
| §11 Sandbox protocol + Docker preflight | T7A, T18, T23 |
| §11 benchmark matrix runner + blind judge metric | T22A |
| §11 Terminus seeds: exact-then-fuzzy edit, compaction | T18 |
| §11 keel: receipt.json, verdict enum, acceptance gate, sha256 fingerprint | T9 (fingerprint, parents), T10 (receipt, gate), T22 (report, export) |
| §11 keel: tool outcome counts | T9, T22 |
| §11 keel: crash repair (torn lines, dangling calls, resume-attempt cap) | T2, T9, T10, T11 |
| §11 keel: Retry-After + jitter, retryable classes; idle watchdog + deadline | T3, T7A, T9, T10 |
| §11 keel: redaction in every log, trace and receipt; safe ids | T2, T9, T10, T22 |
| §11 keel: read window | T6 |
| §9 constraints | Global Constraints; network guard (T1); key stripping in isolation (T23) |
| §10 risks: no key, format drift, cost | T26, T3 + T26 fixture, T9/T10 bounds and budget |

Residual, stated in the README rather than claimed: proof grading for IMO-ProofBench items, TTT-Discover weight training, exact LawBench/USPTO label framing pending Step 1 source checks, and any full-scale paper-number reproduction (documented commands only).

---

## Addendum A: coding-agent harness paper (arXiv:2609.20804)

Source notes with page refs and test ideas: `.superpowers/pivot/refs-coding-agent-harness-paper.md`
§2 (ideas 1-9). Each owning task adds these requirements, with the listed test, to its own files.

| Idea | Requirement (see notes §2 for exact values and tests) | Owner |
|---|---|---|
| 1 | `meta_harness/stats.py`: `mcnemar_exact(b, c)`, `bh(pvals, q=0.05)`; reports emit paired win/loss/tie, p, q, `significant` | T24 (creates), T21, T22A (use) |
| 2 | Stuck detection: reminder at 5 identical calls/failures, stop at 8 identical failures with `stop_reason="stuck"` | T9 (proposer loop), T18 (seeds) |
| 3 | Staged compaction B1=0.6 elide / B2=0.85 fixed-heading summary; thresholds as seed constants; no recall tool | T18 |
| 4 | Normalised `stop_reason` ∈ {done, step_limit, overflow, stuck, timeout, error}; histogram in `scores.json` | T10 (histogram), T18 (seeds emit) |
| 5 | `planning` and `bash_only` seed toggles; proposer domain notes name them as model-dependent axes | T18, T9 (prompt notes) |
| 6 | Rule-based Understand/Write/Verify phase labels; "ended without writing code" rate in `scores.json` | T18 (labels), T22 (stats) |
| 7 | Seed `edit`: read-before-write with content hash; `py_compile` diagnostics appended | T18 |
| 8 | Cost per task in reports; optional `--context-window` sweep | T24, T21 |
| 9 | Up to 8 read-only proposer calls dispatched in one turn, each counted against the cap | T9 |
| — | TerminalBench dataset version is a loader parameter recorded in `run.json` | T18 |
