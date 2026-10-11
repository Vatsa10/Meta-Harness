# Seed harnesses

Starting population for `meta-harness run --baseline ...`. Each file exposes `class Harness`
with `run(task, model, trace)` (or a `build_harness()` factory).

| File | Domain | Paper reference | Tunable |
|---|---|---|---|
| `classification/zero_shot.py` | classification | Table 2, Zero-Shot | — |
| `classification/few_shot.py` | classification | Table 2, Few-Shot (N) | `META_HARNESS_FEW_SHOT_N` (8) |
| `classification/ace.py` | classification | Table 2, ACE [59] | `META_HARNESS_ACE_BULLETS` (24) |
| `classification/mce.py` | classification | Table 2, MCE [52] | `META_HARNESS_MCE_SKILLS` (12) |
| `math/zero_shot.py` | math | Table 6, No Retriever | — |
| `math/bm25.py` | math | Table 6, BM25 Retrieval | `META_HARNESS_CORPUS`, `META_HARNESS_RETRIEVAL_K` (3) |
| `terminal/basic.py` | terminal | §4.3, Terminus-2 shape | — |

These are seeds, not frozen references: the proposer is expected to rewrite them.
