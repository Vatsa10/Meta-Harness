# Meta-Harness

A small, dependency-free implementation of the search loop described in
`paper.pdf`: an agent proposes executable harnesses, candidates are validated
and evaluated on a search set, and all source, scores, and raw traces are kept
in a filesystem experience store for later inspection.

## Quick start

```powershell
python -m meta_harness demo --iterations 3
python -m meta_harness inspect .meta-harness-demo
```

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
