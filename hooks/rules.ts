/**
 * Installed `rule` artifacts, applied at tool.check.
 *
 * A rule is the strongest layer available: it costs no standing tokens and cannot be talked
 * around. `read-before-edit` is built in because it is the discovered doctrine's top clause,
 * and a clause a mechanism can enforce should not be a paragraph.
 */

export type Rule = {
  artifactId: string;
  kind: string;
  tools: string[];
  reason: string;
  threshold?: number;
};

export type SessionState = {
  readPaths: Set<string>;
  /** How many times each exact tool+input call has already been allowed this session. */
  callCounts: Map<string, number>;
  /** Calls the human has explicitly rejected this session, keyed by callKey, valued by a
   * searchable snippet of the input so a later mention of the same command can clear it. */
  rejected?: Map<string, string>;
  /** "Stop doing X" rules added from the human's own words, in effect for this session only. */
  sessionRules?: Array<{ tool: string; pattern: string }>;
};

/** The identity of one tool call, for counting exact repeats. Input order is whatever the
 * engine sends; the same call in the same session serializes the same way, which is all the
 * repeat counter needs. */
export function callKey(event: { tool?: string; input?: Record<string, unknown> }): string {
  let input = '';
  try {
    input = JSON.stringify(event.input ?? {});
  } catch {
    input = String(event.input ?? '');
  }
  return `${String(event.tool ?? '')}:${input.slice(0, 400)}`;
}

/**
 * Default repeat count at which a `repeat-call` rule denies. Mirrors
 * meta_harness.replay.THRASH_THRESHOLD, so the rule that enforces a thrash fix and the replay
 * expectation that verifies it are talking about the same number.
 */
export const DEFAULT_REPEAT_THRESHOLD = 4;

export const BUILT_IN_RULES: Rule[] = [
  {
    artifactId: 'read-before-edit',
    kind: 'read-before-edit',
    tools: ['Edit', 'Write'],
    reason: 'Read this file in the session before editing it: an edit written from an assumption about its contents lands in the wrong place.',
  },
];

export type InstalledArtifact = {
  id: string;
  type: string;
  origin?: Record<string, unknown>;
  payload?: unknown;
};

/**
 * Reads `installed.json`, keeps only the rows of the given `type`, and loads each one's
 * `artifact.json`. The single reader every layer (`rule`, `injection`, and whichever of
 * `skill`/`doctrine` follows) shares, so the fail-open semantics live in exactly one place: a
 * missing or corrupt registry, or a missing/malformed artifact.json, yields fewer rows rather
 * than throwing, and every caller gets that behaviour identically instead of re-deriving it.
 */
export async function loadInstalled(dollar: any, home: string, type: string): Promise<InstalledArtifact[]> {
  const registry = `${home}/installed.json`;
  if (!(await dollar.fs.exists(registry))) return [];
  let entries: Array<{ id: string; type: string }> = [];
  try {
    entries = JSON.parse(await dollar.fs.read(registry));
  } catch {
    return [];
  }
  const out: InstalledArtifact[] = [];
  for (const entry of entries) {
    if (entry.type !== type) continue;
    const path = `${home}/artifacts/${entry.id}/artifact.json`;
    if (!(await dollar.fs.exists(path))) continue;
    try {
      out.push(JSON.parse(await dollar.fs.read(path)));
    } catch {
      continue;
    }
  }
  return out;
}

/** Installed rule artifacts, plus the built-ins. Missing or malformed files are ignored. */
export async function loadRules(dollar: any, home: string): Promise<Rule[]> {
  const rules = [...BUILT_IN_RULES];
  const artifacts = await loadInstalled(dollar, home, 'rule');
  for (const artifact of artifacts) {
    rules.push({
      artifactId: String(artifact.id),
      // `origin.rule_kind` is what meta_harness.learn.propose_artifact emits for the matcher to
      // dispatch on; `origin.kind` is the EPISODE kind ('tool_error'/'thrash'), kept for
      // provenance and never a matcher name. Falling back to it keeps a hand-written artifact
      // working, and an unknown kind simply never denies.
      kind: (artifact.origin as any)?.rule_kind ?? (artifact.origin as any)?.kind ?? 'custom',
      tools: (artifact.origin as any)?.tools ?? [],
      reason: String(artifact.payload ?? '').slice(0, 400),
      threshold: Number((artifact.origin as any)?.threshold) || undefined,
    });
  }
  return rules;
}

/** Evaluates one rule against one tool.check event. `deny: true` means the call must not proceed. */
export function evaluateRule(
  rule: Rule,
  event: { tool?: string; input?: Record<string, unknown> },
  state: SessionState,
): { deny: boolean; reason?: string } {
  const tool = String(event.tool ?? '');
  if (!rule.tools.includes(tool)) return { deny: false };

  if (rule.kind === 'repeat-call') {
    // The only condition a learned rule can decide from the call and the session state alone:
    // this exact call has already been made enough times to be the thrash the artifact was born
    // from. A first attempt is never blocked, so the rule cannot break work that is going fine.
    const threshold = rule.threshold && rule.threshold > 1 ? rule.threshold : DEFAULT_REPEAT_THRESHOLD;
    const seen = state.callCounts.get(callKey(event)) ?? 0;
    if (seen >= threshold - 1) {
      return {
        deny: true,
        reason: `${rule.reason} [${rule.artifactId}] (identical ${tool} call already made ${seen} times this session)`,
      };
    }
    return { deny: false };
  }

  if (rule.kind === 'read-before-edit') {
    const path = String((event.input as any)?.file_path ?? '');
    if (!path || state.readPaths.has(path)) return { deny: false };
    return { deny: true, reason: `${rule.reason} [${rule.artifactId}]` };
  }

  return { deny: false };
}

/** A short, human-searchable string standing in for a tool call's input: the command or path
 * a person would actually type or say when talking about this call, falling back to its JSON. */
function commandSnippet(input: unknown): string {
  if (input && typeof input === 'object') {
    const record = input as Record<string, unknown>;
    if (typeof record.command === 'string') return record.command;
    if (typeof record.file_path === 'string') return record.file_path;
  }
  try {
    return JSON.stringify(input ?? '');
  } catch {
    return String(input ?? '');
  }
}

/** Records that this exact call was rejected by the human this session. */
export function rememberRejection(state: SessionState, tool: string, input: unknown): void {
  if (!state.rejected) state.rejected = new Map<string, string>();
  const key = callKey({ tool, input: input as Record<string, unknown> });
  state.rejected.set(key, commandSnippet(input));
}

/** True when this exact call was rejected earlier this session and has not since been cleared. */
export function wasRejected(state: SessionState, tool: string, input: unknown): boolean {
  if (!state.rejected) return false;
  const key = callKey({ tool, input: input as Record<string, unknown> });
  return state.rejected.has(key);
}

/**
 * A conservative leading-verb parse: only text that OPENS with "stop"/"don't"/"never" (optionally
 * followed by a gerund like "running") is treated as an instruction. Matching anywhere in the
 * text would catch ordinary questions and descriptions that merely contain the word "stop"
 * somewhere, which is exactly the false-positive this must not produce.
 */
const STOP_INSTRUCTION = /^\s*(?:stop|don'?t|never)\s+(?:running|using|doing|calling)?\s*(.+)/i;

/** Tools whose surface is a shell-like command line; anything else defaults to Bash, the most
 * common subject of a "stop running X" instruction. */
const EDIT_LIKE = /\b(writ(?:e|ing)|edit(?:ing)?)\b/i;

/** Parses a "stop doing X" / "don't run X again" instruction out of free text, or returns null
 * for anything that is not unambiguously such an instruction (a question, a description, etc). */
export function parseStopInstruction(text: string): { tool: string; pattern: string } | null {
  const match = STOP_INSTRUCTION.exec(text ?? '');
  if (!match) return null;
  const pattern = match[1]?.replace(/["'.?!]+$/g, '').trim();
  if (!pattern) return null;
  const tool = EDIT_LIKE.test(pattern) ? 'Edit' : 'Bash';
  return { tool, pattern };
}

/** Adds a session-scoped rule parsed from the human's own words. Never touches the installed store. */
export function addSessionRule(state: SessionState, rule: { tool: string; pattern: string }): void {
  if (!state.sessionRules) state.sessionRules = [];
  state.sessionRules.push(rule);
}
