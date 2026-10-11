# Seed harnesses

Seed harnesses for `meta-harness search --harness <file>` (repeat `--harness` for several).
Contract: each file exposes `class Harness` with `run(task, model[, trace]) -> {"output": ...}`
(or a `build_harness()` factory), and optionally `learn(task, result)`, which online domains call
with each train task.

| File | Domain | Paper reference | Tunable |
|---|---|---|---|
| `classification/zero_shot.py` | classification | Table 2, Zero-Shot | — |
| `classification/few_shot.py` | classification | Table 2, Few-Shot (N) | `META_HARNESS_FEW_SHOT_N` (8) |
| `classification/ace.py` | classification | Table 2, ACE [59] | `META_HARNESS_ACE_BULLETS` (24) |
| `classification/mce.py` | classification | Table 2, MCE [52] | `META_HARNESS_MCE_SKILLS` (12) |
| `classification/dense_few_shot.py` | classification | dense-retrieval few-shot baseline | - |
| `classification/random_few_shot.py` | classification | random few-shot baseline | - |
| `classification/discovered/*.py` | classification | the eight discovered harnesses (App. B, Table 9) | - |
| `math/zero_shot.py` | math | Table 6, No Retriever | — |
| `math/bm25.py` | math | Table 6, BM25 Retrieval | `META_HARNESS_CORPUS`, `META_HARNESS_RETRIEVAL_K` (3) |
| `math/dense.py`, `math/random_few_shot.py`, `math/ace.py` | math | Table 6 baselines | - |
| `math/router.py` | math | discovered four-route router (App. B.2) | - |
| `terminal/terminus2.py` | terminal | §4.3, App. B.3 Terminus-2 | `PLANNING`, `BASH_ONLY`, `B1`/`B2`, `CONTEXT_WINDOW` |
| `terminal/terminus_kira.py` | terminal | App. B.3 Terminus-KIRA | `PLANNING`, `BASH_ONLY`, `B1`/`B2`, `CONTEXT_WINDOW` |

These are seeds, not frozen references: the proposer is expected to rewrite them.
