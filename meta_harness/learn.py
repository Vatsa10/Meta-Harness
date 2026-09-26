"""The learn cycle: pick the failure that costs most, propose one artifact, score it by replay."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cc_harness import AgentConfig, ClaudeCodeHarness, prepare_workspace, run_claude_code
from .cc_history import FailureEpisode, Session, failure_episodes
from .core import TraceRecorder
from .harness_store import ARTIFACT_TYPES, Artifact, HarnessStore, harness_home
from .replay import (  # noqa: F401  (_cause is re-exported for the CLI)
    THRASH_THRESHOLD, _cause, build_replay, episode_signature, load_agent_steps,
    verify_expectation)

PROMPT_PATH = Path(__file__).resolve().parent / "learn_prompt.md"


@dataclass
class FailureClass:
    signature: str
    kind: str = ""
    tool: str = ""
    count: int = 0
    episodes: list[FailureEpisode] = field(default_factory=list)


def rank_failures(sessions: Sequence[Session]) -> list[FailureClass]:
    """Every observed failure class, most frequent first."""
    grouped: dict[str, FailureClass] = {}
    for session in sessions:
        for episode in failure_episodes(session):
            signature = episode_signature(episode)
            tools = list(episode.tools or [])
            entry = grouped.get(signature)
            if entry is None:
                entry = FailureClass(signature=signature, kind=episode.kind,
                                     tool=tools[0] if tools else "unknown")
                grouped[signature] = entry
            entry.count += 1
            entry.episodes.append(episode)
    return sorted(grouped.values(), key=lambda f: (-f.count, f.signature))


def observed_failures(home: Path | None = None) -> list[FailureClass]:
    """Failure classes the hook recorded live, fresher evidence than mined transcripts.

    Task 10's observer writes one file per session (`observed-<sessionId>.jsonl`), not a single
    shared log: the hook filesystem capability has no append or lock primitive, so per-session
    files avoid a read-modify-write race between concurrent sessions. This reads every such file
    under the harness home and merges them into one namespace of signatures, in the same
    `tool_error:<tool>:<cause>` / `thrash:<tool>` format `episode_signature` produces for mined
    history, so `HarnessStore.covered()` can dedupe across both sources.

    Fails open: a missing directory, a missing/corrupt file, or a half-written trailing line (a
    session may still be writing while this reads) degrades to "no evidence for that line", never
    an exception.
    """
    base = Path(home or harness_home())
    grouped: dict[str, FailureClass] = {}
    try:
        paths = sorted(base.glob("observed-*.jsonl"))
    except OSError:
        return []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # Either a corrupt line, or the trailing line of a file still being written.
                continue
            if not isinstance(record, dict):
                continue
            tool = str(record.get("tool", "unknown"))
            kind = record.get("kind")
            if kind == "repeat":
                signature, klass = f"thrash:{tool}", "thrash"
            elif kind == "tool_error":
                cause = record.get("cause") or _cause(str(record.get("text", "")))
                signature = f"tool_error:{tool}:{cause}"
                klass = "tool_error"
            else:
                continue
            entry = grouped.setdefault(
                signature, FailureClass(signature=signature, kind=klass, tool=tool))
            entry.count += 1
    return sorted(grouped.values(), key=lambda f: (-f.count, f.signature))


def merge_failures(observed: Sequence[FailureClass],
                   mined: Sequence[FailureClass],
                   weights: Mapping[str, float] | None = None) -> list[FailureClass]:
    """Combine live and mined evidence into one ranking, keyed by signature.

    Both sources write signatures in the same `tool_error:<tool>:<cause>` / `thrash:<tool>`
    namespace, so a signature seen in both is one failure, not two: its counts are summed, and it
    outranks either source alone. Live evidence is not preferred by source order — a single noisy
    observation must not preempt a mined signature seen fifty times; it only adds weight to
    whichever bucket its signature falls into.

    `episodes` (and the `kind`/`tool` they imply) come only from the mined side: an observed
    record has no `FailureEpisode`, so it cannot seed a replay on its own. A signature that is
    mined keeps its episodes regardless of how many times it was also observed live; a
    signature that is *only* observed carries no episodes, exactly as it did before this merge,
    and `_command_learn` already handles "no replayable episode for this failure class" for that
    case.

    `weights` ages the evidence: `None` (the default) means every signature keeps weight 1.0, so
    the ranking - and the existing two-argument call site - is unchanged. When given, each
    signature's summed count is multiplied by `weights.get(signature, 1.0)` before sorting, so a
    signature seen only in old sessions can rank below an equal-count signature seen recently.

    The tie-break (`-count`, then `signature`) is unchanged from the single-source ranking, so
    equal weighted counts order the same way on every run.
    """
    grouped: dict[str, FailureClass] = {}
    for failure in mined:
        grouped[failure.signature] = FailureClass(
            signature=failure.signature, kind=failure.kind, tool=failure.tool,
            count=failure.count, episodes=list(failure.episodes))
    for failure in observed:
        entry = grouped.get(failure.signature)
        if entry is None:
            grouped[failure.signature] = FailureClass(
                signature=failure.signature, kind=failure.kind, tool=failure.tool,
                count=failure.count)
        else:
            entry.count += failure.count

    def sort_key(failure: FailureClass) -> tuple[float, str]:
        weight = weights.get(failure.signature, 1.0) if weights is not None else 1.0
        return (-failure.count * weight, failure.signature)

    return sorted(grouped.values(), key=sort_key)


def _signature_weights(sessions: Sequence[Session]) -> dict[str, float]:
    """Age each signature by the newest session that showed it, and by version distance."""
    from .temporal import recency_weight, version_weight

    current = max((s.version for s in sessions if s.version), default="")
    weights: dict[str, float] = {}
    for session in sessions:
        stamp = session.ended or session.started
        weight = recency_weight(stamp) * version_weight(session.version, current)
        for episode in failure_episodes(session):
            signature = episode_signature(episode)
            weights[signature] = max(weights.get(signature, 0.0), weight)
    return weights


def select_target(sessions: Sequence[Session], store: HarnessStore,
                  home: Path | None = None) -> FailureClass | None:
    """The most frequent failure not already covered by an installed artifact or a tombstone.

    Live observations (what the hook actually saw this session) and mined history (past
    transcripts) are merged into one ranking by signature before selection, so a failure seen in
    both outranks either source alone, and a high-count mined signature is not starved by a
    single noisy live observation. The ranking is also weighted by how recently and on how
    current a Claude Code version each signature's evidence was seen, so a failure class fixed
    long ago stops outranking one seen this week.
    """
    covered = store.covered()
    weights = _signature_weights(sessions)
    for failure in merge_failures(observed_failures(home), rank_failures(sessions), weights=weights):
        if failure.signature not in covered:
            return failure
    return None


def build_proposal_prompt(failure: FailureClass, replay: Mapping[str, Any],
                          installed: Sequence[str]) -> str:
    sample = failure.episodes[0] if failure.episodes else None
    evidence = (sample.assistant_text or sample.detail or "") if sample else ""
    parts = [PROMPT_PATH.read_text(encoding="utf-8"),
             "\n## The failure\n",
             f"signature: {failure.signature}",
             f"kind: {failure.kind}   tool: {failure.tool}",
             f"observed {failure.count} times",
             f"error text: {evidence[:400]}" if evidence else "",
             "\n## The replay it must fix\n",
             f"instruction: {str(replay.get('instruction', ''))[:400]}",
             f"files: {sorted((replay.get('files') or {}))}",
             f"expectation: {replay.get('expect')}"]
    if installed:
        parts.append("\n## Already installed - do not duplicate\n" + "\n".join(
            f"- {name}" for name in installed))
    return "\n".join(part for part in parts if part)


def parse_proposal(text: str) -> tuple[str, str]:
    match = re.search(r"TYPE:\s*([a-z]+)", text, re.IGNORECASE)
    if not match:
        raise ValueError("no TYPE in proposal")
    artifact_type = match.group(1).lower()
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    body = text.split("PAYLOAD:", 1)
    if len(body) != 2 or not body[1].strip():
        raise ValueError("no payload in proposal")
    payload = body[1].strip()
    # Greedy match anchored to the LAST closing fence (end of payload), not the first one:
    # a doctrine/skill payload is prose and may itself contain a nested code fence, and a
    # non-greedy match would truncate the payload there with no error.
    fenced = re.match(r"```[a-zA-Z]*\s*\n(.*)```\s*$", payload, re.DOTALL)
    if fenced:
        payload = fenced.group(1)
    return artifact_type, payload.strip()


# Derived here, in the producer, rather than in the hooks. The hooks are a lookup on a hot path
# (spec section 7: "enforcement is a lookup"), they have no access to the episode a signature came
# from, and two independent derivations - one in Python, one in TypeScript - is exactly the drift
# that let `origin.kind` mean two different things on the two sides of this seam. One producer,
# written once, read verbatim by both `hooks/rules.ts` and `hooks/harness.ts`.
def enforcement_keys(failure: FailureClass, artifact_type: str) -> dict[str, Any]:
    """The `origin` fields the hook layer matches on, for one failure and one artifact type.

    `rule` gets `tools` (which tool.check calls this rule may deny) and `rule_kind` (which
    matcher branch in `evaluateRule` decides it). `rule_kind` is deliberately NOT the episode
    kind: `tool_error`/`thrash` describe where the artifact came from, not what the matcher
    should do, and `origin["kind"]` keeps carrying them for provenance.

    `injection` gets `triggers`: lowercase substrings matched against the turn. They are coarse
    by construction - the tool name plus the words of the cause slug - because a staged artifact
    is read by a human before it can fire at all.
    """
    tool = failure.tool or "unknown"
    if artifact_type == "rule":
        # The only condition decidable from the call and the session state alone, given what a
        # signature carries (a tool and a cause, never the call's arguments): the identical call
        # repeated to the point that it is the thrash the artifact was born from.
        return {"tools": [tool], "rule_kind": "repeat-call", "threshold": THRASH_THRESHOLD}
    if artifact_type == "injection":
        parts = failure.signature.split(":")
        words = [word for word in re.split(r"[^a-z0-9]+", parts[-1].lower()) if len(word) >= 4]
        triggers = [tool.lower()] + [word for word in words if word != tool.lower()]
        return {"triggers": sorted(set(triggers))}
    return {}


def propose_artifact(failure: FailureClass, replay: Mapping[str, Any],
                     installed: Sequence[str], model: Callable[..., str]) -> Artifact:
    artifact_type, payload = parse_proposal(model(
        build_proposal_prompt(failure, replay, installed)))
    origin = dict(replay.get("_origin") or {})
    origin.setdefault("signature", failure.signature)
    origin.update(enforcement_keys(failure, artifact_type))
    prefix = re.sub(r"[^A-Za-z0-9]+", "-", failure.signature).strip("-") or "artifact"
    digest = hashlib.sha256(
        f"{failure.signature}|{origin.get('session', '')}|{origin.get('turn', '')}"
        .encode("utf-8")).hexdigest()[:8]
    return Artifact(
        id=f"{prefix}-{digest}",
        type=artifact_type,
        origin=origin,
        payload=payload,
        replay=dict(replay),
        sources=[f"{origin.get('session', '')}#{origin.get('turn', '')}"],
    )


def config_with(artifact: Artifact, base: AgentConfig | None = None) -> AgentConfig:
    """Apply one artifact to an agent configuration.

    A `rule` acts through tool.check and therefore changes nothing here; it is scored by
    running the replay with the rule installed in the hook layer.
    """
    config = base or AgentConfig()
    if artifact.type == "doctrine":
        joined = "\n\n".join(part for part in (config.append_system_prompt, artifact.payload) if part)
        return replace_config(config, append_system_prompt=joined)
    if artifact.type == "skill":
        skills = dict(config.skills or {})
        skills[artifact.id] = artifact.payload
        return replace_config(config, skills=skills)
    if artifact.type == "injection":
        return replace_config(config, prompt_template=config.prompt_template + "\n\n" + artifact.payload)
    return config


def replace_config(config: AgentConfig, **changes: Any) -> AgentConfig:
    from dataclasses import replace

    return replace(config, **changes)


def decide_retention(origin_fixed: bool | None, candidate_score: float, baseline_score: float,
                     candidate_context: float, baseline_context: float) -> dict[str, Any]:
    """Keep an artifact only if it fixes what it was born from and regresses nothing.

    `origin_fixed=None` means the replay could not apply the artifact at all, which is not
    the same as a failing score and must not be silently read as one.
    """
    context_delta = round(candidate_context - baseline_context, 2)
    if origin_fixed is None:
        return {"kept": False, "reason": "origin replay was not scored",
                "context_delta": context_delta}
    if not origin_fixed:
        return {"kept": False, "reason": "origin replay still fails", "context_delta": context_delta}
    if candidate_score < baseline_score:
        return {"kept": False,
                "reason": f"regressed the task set ({candidate_score:.3f} < {baseline_score:.3f})",
                "context_delta": context_delta}
    return {"kept": True, "reason": "origin fixed, no regression", "context_delta": context_delta}


PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def install_rule_for_replay(artifact: Artifact, root: Path) -> Path | None:
    """Write a `rule` artifact into a throwaway harness home the hook layer can read.

    A rule changes nothing about the prompt, so `config_with` has nothing to apply: it acts at
    `tool.check`, inside this repository's own function-hook plugin. Scoring one therefore means
    running the replay with that plugin loaded and `META_HARNESS_HOME` pointed here, which is
    exactly the shape `hooks/rules.ts::loadInstalled` reads. Returns None when the plugin cannot
    be located, so the caller can report "not scored" instead of a number that means nothing.
    """
    if not (PLUGIN_ROOT / "hooks" / "hooks.json").is_file():
        return None
    if not (PLUGIN_ROOT / ".claude-plugin" / "plugin.json").is_file():
        return None
    home = Path(root) / f"{artifact.id}-home"
    (home / "artifacts" / artifact.id).mkdir(parents=True, exist_ok=True)
    (home / "artifacts" / artifact.id / "artifact.json").write_text(
        json.dumps({"id": artifact.id, "type": artifact.type, "origin": artifact.origin,
                    "payload": artifact.payload}, indent=2), encoding="utf-8")
    (home / "installed.json").write_text(
        json.dumps([{"id": artifact.id, "type": artifact.type,
                     "signature": artifact.signature}], indent=2), encoding="utf-8")
    return home


def run_replay(artifact: Artifact, workspace_root: Path, binary: str = "claude",
               timeout: float = 900.0) -> tuple[bool | None, dict[str, Any]]:
    """Run the artifact's replay and report whether the original failure recurred.

    Returns `(None, ...)` - never a bool - when the artifact could not actually be applied to the
    run. A `rule` that was not installed would otherwise be "scored" by a replay identical to the
    baseline, and its `origin_fixed` would be run-to-run noise wearing the costume of evidence.
    """
    replay = artifact.replay or {}
    task = {"instruction": replay.get("instruction", ""), "files": replay.get("files", {})}
    config = config_with(artifact)
    applied = "prompt"
    if artifact.type == "rule":
        home = install_rule_for_replay(artifact, workspace_root)
        if home is None:
            return None, {"scored": False,
                          "reason": ("rule not scored: the meta-harness plugin directory could "
                                     f"not be found at {PLUGIN_ROOT}, so the tool.check layer "
                                     "that enforces this artifact was not loaded into the replay")}
        config = replace_config(
            config, plugin_dirs=[str(PLUGIN_ROOT)],
            extra_env={"META_HARNESS_HOME": str(home),
                       "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1"})
        applied = f"tool.check via {home}"
    workspace = prepare_workspace(task, workspace_root)
    trace_path = Path(workspace_root) / f"{artifact.id}-replay.jsonl"
    with TraceRecorder(trace_path) as trace:
        run = run_claude_code(workspace, config, task, trace, binary=binary, timeout=timeout)
    steps = load_agent_steps(trace_path)
    fixed = verify_expectation(replay.get("expect") or {}, steps)
    return fixed, {"scored": True, "applied": applied, "turns": run.turns,
                   "input_tokens": run.input_tokens,
                   "workspace": str(workspace), "trace": str(trace_path)}


def score_task_set(config: AgentConfig, tasks: Sequence[Mapping[str, Any]], workspace_root: Path,
                   binary: str = "claude", timeout: float = 900.0) -> tuple[float, float]:
    """Mean score and mean input tokens for one configuration over a task set.

    Each task is scored by its own `test_command`, run afterwards and outside the agent's reach,
    which is the same bar the search path uses. An empty task set scores 0.0/0.0; the caller,
    not this function, decides what an absent task set means.
    """
    from .metrics import agent_metric

    if not tasks:
        return 0.0, 0.0
    scores: list[float] = []
    tokens: list[float] = []
    for index, task in enumerate(tasks):
        workspace = prepare_workspace(task, workspace_root)
        trace_path = Path(workspace_root) / f"taskset-{index}.jsonl"
        with TraceRecorder(trace_path) as trace:
            run = run_claude_code(workspace, config, task, trace, binary=binary, timeout=timeout)
        scores.append(float(agent_metric(str(workspace), task)))
        tokens.append(float(run.input_tokens))
    return sum(scores) / len(scores), sum(tokens) / len(tokens)


__all__ = ["FailureClass", "rank_failures", "observed_failures", "merge_failures", "select_target",
           "build_proposal_prompt", "enforcement_keys", "parse_proposal", "propose_artifact",
           "score_task_set", "config_with", "decide_retention", "install_rule_for_replay",
           "replace_config", "run_replay"]
