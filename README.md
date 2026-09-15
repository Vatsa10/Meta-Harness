# Meta-Harness

This repository implements the paper's complete executable workflow: an agent
proposes Python harnesses, candidates are validated and evaluated on a search
set, and every source file, score, prompt, retrieval decision, model output,
and error is retained in a filesystem experience store.

## Quick start

```powershell
python -m meta_harness demo --iterations 3
python -m meta_harness inspect .meta-harness-demo
```

The concrete harness implementations are available directly:

```python
from meta_harness import (
    DraftVerificationHarness,
    LabelPrimedQueryHarness,
    MathRetrievalHarness,
)
```

`LabelPrimedQueryHarness` implements the paper's strongest classification
strategy. `DraftVerificationHarness` implements the low-context two-call
variant. `MathRetrievalHarness` implements the four-route BM25 policy with
deduplication, difficulty reranking, and solution-technique bonuses.

The demo uses a deterministic classifier and proposer, so it needs no model
provider or network access. For a real run, implement a proposer callback or
use `CommandProposer` with a coding agent command.

## Harness contract

Every candidate is a Python file exposing either `build_harness()` or a
`Harness` class. The resulting object must implement:

```python
run(task, model, trace) -> prediction
```

`model(prompt, **kwargs)` is deliberately just a callable. A harness can use
any model adapter, retrieval index, or state it needs. `trace` is a recorder;
calling `trace.event(name, payload)` writes the raw diagnostic information
that the proposer needs to diagnose failures.

The public API is in `meta_harness.core`: `SearchConfig`, `SearchRunner`,
`FilesystemExperience`, `CandidateEvaluator`, `ParetoFrontier`, and
`CommandProposer`.

## Running a real search

Create a baseline candidate with the contract above and configure a proposer:

```python
from meta_harness import CandidateEvaluator, CommandProposer, SearchConfig, SearchRunner

runner = SearchRunner(
    SearchConfig(root=".run", iterations=20, candidates_per_iteration=2),
    CandidateEvaluator(model=my_model),
    CommandProposer(["python", "propose.py"]),
)
frontier = runner.run(["baseline.py"], search_tasks)
```

The proposer receives `META_HARNESS_ROOT`, `META_HARNESS_OUTPUT`,
`META_HARNESS_ITERATION`, and `META_HARNESS_COUNT`. It can inspect every prior
candidate under `ROOT/candidates`, including `harness.py`, `scores.json`, and
`traces.jsonl`, then write new candidate files into `OUTPUT`.
