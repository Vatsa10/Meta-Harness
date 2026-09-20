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
from .replay import _cause, build_replay, episode_signature, load_agent_steps, verify_expectation  # noqa: F401  (re-exported for the CLI)

PROMPT_PATH = Path(__file__).resolve().parent / "learn_prompt.md"


@dataclass
class FailureClass:
    signature: str
    kind: str
    tool: str
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
                   mined: Sequence[FailureClass]) -> list[FailureClass]:
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

    The tie-break (`-count`, then `signature`) is unchanged from the single-source ranking, so
    equal-count signatures order the same way on every run.
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
    return sorted(grouped.values(), key=lambda f: (-f.count, f.signature))


def select_target(sessions: Sequence[Session], store: HarnessStore,
                  home: Path | None = None) -> FailureClass | None:
    """The most frequent failure not already covered by an installed artifact or a tombstone.

    Live observations (what the hook actually saw this session) and mined history (past
    transcripts) are merged into one ranking by signature before selection, so a failure seen in
    both outranks either source alone, and a high-count mined signature is not starved by a
    single noisy live observation.
    """
    covered = store.covered()
    for failure in merge_failures(observed_failures(home), rank_failures(sessions)):
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


def propose_artifact(failure: FailureClass, replay: Mapping[str, Any],
                     installed: Sequence[str], model: Callable[..., str]) -> Artifact:
    artifact_type, payload = parse_proposal(model(
        build_proposal_prompt(failure, replay, installed)))
    origin = dict(replay.get("_origin") or {})
    origin.setdefault("signature", failure.signature)
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


def decide_retention(origin_fixed: bool, candidate_score: float, baseline_score: float,
                     candidate_context: float, baseline_context: float) -> dict[str, Any]:
    """Keep an artifact only if it fixes what it was born from and regresses nothing."""
    context_delta = round(candidate_context - baseline_context, 2)
    if not origin_fixed:
        return {"kept": False, "reason": "origin replay still fails", "context_delta": context_delta}
    if candidate_score < baseline_score:
        return {"kept": False,
                "reason": f"regressed the task set ({candidate_score:.3f} < {baseline_score:.3f})",
                "context_delta": context_delta}
    return {"kept": True, "reason": "origin fixed, no regression", "context_delta": context_delta}


def run_replay(artifact: Artifact, workspace_root: Path, binary: str = "claude",
               timeout: float = 900.0) -> tuple[bool, dict[str, Any]]:
    """Run the artifact's replay and report whether the original failure recurred."""
    replay = artifact.replay or {}
    task = {"instruction": replay.get("instruction", ""), "files": replay.get("files", {})}
    workspace = prepare_workspace(task, workspace_root)
    trace_path = Path(workspace_root) / f"{artifact.id}-replay.jsonl"
    with TraceRecorder(trace_path) as trace:
        run = run_claude_code(workspace, config_with(artifact), task, trace,
                              binary=binary, timeout=timeout)
    steps = load_agent_steps(trace_path)
    fixed = verify_expectation(replay.get("expect") or {}, steps)
    return fixed, {"turns": run.turns, "input_tokens": run.input_tokens,
                   "workspace": str(workspace), "trace": str(trace_path)}


__all__ = ["FailureClass", "rank_failures", "observed_failures", "merge_failures", "select_target",
           "build_proposal_prompt", "parse_proposal", "propose_artifact",
           "config_with", "decide_retention", "replace_config", "run_replay"]
