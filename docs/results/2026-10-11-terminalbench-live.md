# TerminalBench-2 and Docker isolation: live verification (2026-10-11/12)

Scale: 3 TerminalBench-2 tasks, 1 trial each, `MAX_STEPS` 30, no search. This checks that the
real Docker path works. It is **not a benchmark** and **not comparable to the paper**.

Setup: TB2 checkout `laude-institute/terminal-bench-2` at `2fd12b88`, cloned with
`core.autocrlf=false`; loader `dataset_version` 2.0. Prebuilt task images (`alexgshaw/<task>:20251031`).
Docker Desktop engine 28.4.0 on Windows 11. Driver: `terminalbench.load` -> `terminal.adapt` ->
`terminal.evaluate_task` with the `baselines/terminal/terminus2.py` seed (`max_steps=30`). Every docker
argv the engine ran was logged.

## Seed runs (terminus2, groq `openai/gpt-oss-120b`)

| task (difficulty easy) | workdir | score | stop_reason | steps |
|---|---|---|---|---|
| fix-git | /app/personal-site | 1.0 | error (groq 8000 TPM cap hit after the fix) | 6 |
| cobol-modernization | /app | 1.0 | error (same TPM cap) | 6 |
| prove-plus-comm | /workspace | 1.0 | done | 7 |

3/3 on 3 easy tasks (37,219 input / 13,835 output tokens, 17 model calls). The `error` stops are the
groq account's tokens-per-minute cap rejecting a request (HTTP 413), after the agent had already solved
the task. A repeat run of fix-git gave the same result: 1.0, stopped by the cap.

Earlier gpt-4o-mini run on fix-git (before the workdir fix below): 0.0, done after 21 steps
(90,106 / 4,359 tokens, about $0.016). After that, OpenAI returned "no credits".

Reference check: fix-git's own `solution/solve.sh`, replayed by the scripted provider through the
same path, scored 1.0 once the bugs below were fixed. Before the fixes it scored 0.

## Container lifecycle checks (seen in the logged argv on real Docker)

- `docker run -d ... --security-opt no-new-privileges --pids-limit 512 --memory 2048m --cpus 2.0
  -w <workdir> --entrypoint sleep <image> infinity`. There is no `--network none` because these tasks
  allow internet (harbor default), and their tests download `uv`.
- All agent commands ran as `docker exec -w <workdir> <id> timeout -s KILL ...`. Nothing ran on the host.
- Before scoring: `docker restart -t 0`, then `docker exec <id> true` succeeded, then `docker diff`, the
  symlink check, and `rm -rf /tests /logs/verifier`. After that the tests were copied in with `docker cp`,
  and the test command ran under `/bin/bash`.
- After every task, `docker rm -f <id>` ran, and `docker ps -a --filter name=meta-harness` was empty.
- Key-leak scan of every run root for loaded `*_API_KEY` values: `False`.

## Forged-score attempt (scripted provider, no API cost)

The Terminus-2 loop was given a scripted model on fix-git. The script wrote `/logs/verifier/reward.txt=1`,
planted `/tests/test.sh` (which writes reward 1), and started a `setsid nohup` background loop that kept
rewriting both. Inside the container the plant was confirmed (`cat` printed 1, `/tests/test.sh` existed).
Result: score 0.0, `score_reason: tampered`. `docker diff` flagged
`A /logs, A /logs/verifier, A /logs/verifier/reward.txt, A /tests, A /tests/test.sh`. The container
was removed.

## Docker isolation mode (`search --isolation docker`, examples/sentiment, scripted provider)

This works on real Docker. The candidate argv was `docker run -i --rm --network none --cap-drop ALL
--read-only --tmpfs /tmp --user 65534:65534 --memory 2048m --memory-swap 2048m --cpus 1
--pids-limit 256 --security-opt no-new-privileges --ulimit cpu=600`, with the package and the
candidate mounted `:ro` on `python:3.12-slim`. Scores: `initial-0000` 0.0, `iteration-0001-00` 1.0
(held-out 1.0). No containers were left behind.

## Bugs found and fixed (commit "fix(terminalbench): ...")

1. **Workdir**: the loader hard-coded `workdir=/app`, and `-w /app` overrides the image WORKDIR.
   fix-git (`/app/personal-site`), prove-plus-comm (`/workspace`) and one more TB2 task broke: even
   the reference solution scored 0. The loader now uses the Dockerfile's last absolute `WORKDIR`.
2. **CRLF**: a Windows checkout with `core.autocrlf` turns `tests/*.sh` into CRLF. Bash then fails in
   the container and every task silently scores 0. The loader now refuses CRLF shell scripts and names
   the re-clone flag.

Regression tests: `test_workdir_follows_dockerfile` and `test_crlf_test_scripts_are_refused` in
`tests/test_terminal_domain.py`.

Cost: about $0.035 in total (gpt-4o-mini about $0.016; groq gpt-oss-120b about $0.019 at
$0.15/$0.60 per M tokens, an assumed list price).
