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

export type SessionRule = {
  tool: string;
  /** The command PREFIX the instruction names, e.g. "git push" from "stop using git push
   * --force", or "pytest" from "don't run pytest". Matched against the START of the tokenised
   * command string (whole tokens), never a substring of the whole JSON input — otherwise a
   * rule for "git push" would also fire on an unrelated call that merely mentions "git push"
   * inside a description or a file path. */
  pattern: string;
  /** The one flag/token that, if PRESENT in the call, means the rule allows it. Captures a
   * "without <flag>" qualifier ("stop running pytest without -q" denies pytest unless the call
   * also contains "-q"): the qualifier is representable, so it narrows the rule instead of
   * being dropped into a blanket denial. Absent when the human's qualifier (if any) could not
   * be represented this way, in which case the rule denies its pattern unconditionally. */
  requires?: string;
  /** The one flag/token that, if PRESENT in the call, means the rule DENIES it — the opposite
   * polarity of `requires`. Captures a flag named directly in the instruction's object ("stop
   * using git push --force" denies calls that start with "git push" AND carry "--force" as a
   * whole token, but allows a plain "git push origin main"). Mutually exclusive with
   * `requires` in practice, since an instruction supplies at most one qualifier. */
  flag?: string;
};

export type SessionState = {
  readPaths: Set<string>;
  /** How many times each exact tool+input call has already been allowed this session. */
  callCounts: Map<string, number>;
  /** Calls the human has explicitly rejected this session, keyed by callKey, valued by a
   * searchable snippet of the input so a later mention of the same command can clear it. */
  rejected?: Map<string, string>;
  /** "Stop doing X" rules added from the human's own words, in effect for this session only. */
  sessionRules?: SessionRule[];
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
export async function loadInstalled(io: any, home: string, type: string): Promise<InstalledArtifact[]> {
  const registry = `${home}/installed.json`;
  if (!(await io.fs.exists(registry))) return [];
  let entries: Array<{ id: string; type: string }> = [];
  try {
    entries = JSON.parse(await io.fs.read(registry));
  } catch {
    return [];
  }
  const out: InstalledArtifact[] = [];
  for (const entry of entries) {
    if (entry.type !== type) continue;
    const path = `${home}/artifacts/${entry.id}/artifact.json`;
    if (!(await io.fs.exists(path))) continue;
    try {
      out.push(JSON.parse(await io.fs.read(path)));
    } catch {
      continue;
    }
  }
  return out;
}

/** Installed rule artifacts, plus the built-ins. Missing or malformed files are ignored. */
export async function loadRules(io: any, home: string): Promise<Rule[]> {
  const rules = [...BUILT_IN_RULES];
  const artifacts = await loadInstalled(io, home, 'rule');
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
 * A conservative leading-verb parse: text must OPEN with "stop"/"don't"/"never" AND be
 * immediately followed by one of a fixed, small set of ACTION verbs — a gerund (running/using/
 * calling/doing/touching/editing/deleting/pushing) OR the plain imperative (run/use/call/do/
 * touch/edit/delete/push) — before anything counts as an instruction's object.
 *
 * Matching anywhere in the text, or treating the action verb as optional, both let ordinary
 * prose through as if it were an instruction: "don't worry about it", "don't know why this
 * fails", "never mind" and "Don't forget to update the README" all open with a trigger word but
 * name no actionable target, and "stop" alone or "stop, that's wrong" have no object at all.
 * Requiring one of these specific verbs, immediately after the trigger word, is what tells an
 * instruction ("stop running X") apart from those. Accepting the bare imperative alongside the
 * gerund is what lets "don't use git push --force", "never call the deploy script" and "don't
 * run pytest" parse at all — an earlier version only accepted the -ing form and silently missed
 * every plain-imperative instruction.
 */
const STOP_VERB =
  /^\s*(?:stop|don'?t|never)\s+(running|using|calling|doing|touching|editing|deleting|pushing|run|use|call|do|touch|edit|delete|push)\s+(.+)/i;

/** Verbs whose object is a file or a piece of code rather than a shell command. */
const EDIT_LIKE_VERBS = new Set(['editing', 'touching', 'deleting', 'edit', 'touch', 'delete']);

/** A determiner is never itself the object ("the deploy script" -> "deploy script"). */
const LEADING_DETERMINER = /^(?:the|a|an)\s+(?=\S)/i;

/** Function words — pronouns, determiners, catch-all nouns — that name no actual command, tool
 * or file: "Stop doing that" and "stop using the" must deny nothing, not deny every call whose
 * input happens to contain the English word "that" or "the". */
const FUNCTION_WORDS = new Set([
  'that', 'this', 'it', 'the', 'a', 'an', 'those', 'these', 'them', 'they', 'there', 'here',
  'everything', 'something', 'anything', 'nothing', 'things', 'stuff', 'that\'s', 'it\'s',
]);

/** A plausible command name, flag or path: word/path characters only, and not a bare function
 * word — "pytest", "git", "-q" and "deploy" all pass; "that" and "the" do not. */
function isPlausibleObject(token: string): boolean {
  if (!token) return false;
  if (FUNCTION_WORDS.has(token.toLowerCase())) return false;
  return /^[A-Za-z0-9\-][\w.\-/]*$/.test(token);
}

/** Words introducing a qualifier this mechanism cannot represent as a `requires` flag ("pytest
 * UNLESS it's urgent"): the pattern is cut before the qualifier rather than including words that
 * would make the pattern match nothing real, and `tool.check`'s denial reason says so. */
const UNREPRESENTABLE_QUALIFIER = /\s+(?:unless|except|only if|if)\b/i;

/** A "without <flag>" qualifier IS representable: it becomes `requires`, so `tool.check` can
 * deny the pattern only when that flag is absent, instead of denying it in every form — denying
 * `pytest -q` outright, the exact command the human asked to KEEP, was the wrong call. */
const WITHOUT_QUALIFIER = /\s+without\s+(\S+)/i;

/** The rule's object: the command PREFIX up to its first flag (a plausible command/path only —
 * see `isPlausibleObject`), plus whichever one qualifier the human's phrasing carried:
 *
 * - a "without X" qualifier becomes `requires` (X must be PRESENT to ALLOW the call) — unchanged
 *   from fix round 2.
 * - a flag named directly in the object ("git push --force") becomes `flag` (X must be PRESENT
 *   to DENY the call) — new in fix round 3, so "don't use git push --force" denies only calls
 *   that both start with "git push" AND carry "--force", not every "git" command.
 *
 * When the object has no flag at all ("never run git push"), the whole object (up to any
 * qualifier) is the prefix and the rule denies that subcommand unconditionally.
 *
 * Returns null when no plausible object can be found at all, or when the object is nothing but
 * a bare flag with no leading subcommand token. */
function parseObject(rest: string): { pattern: string; requires?: string; flag?: string } | null {
  const cleaned = rest.trim().replace(LEADING_DETERMINER, '');

  const withoutMatch = WITHOUT_QUALIFIER.exec(cleaned);
  const requires = withoutMatch ? withoutMatch[1].replace(/["'.,?!]+$/g, '') : undefined;
  const beforeQualifier = withoutMatch
    ? cleaned.slice(0, withoutMatch.index)
    : (cleaned.split(UNREPRESENTABLE_QUALIFIER)[0] ?? cleaned);

  const rawTokens = beforeQualifier.trim().split(/\s+/).filter(Boolean);
  if (rawTokens.length === 0) return null;
  const tokens = rawTokens.map((token, i) =>
    i === rawTokens.length - 1 ? token.replace(/["'.,?!]+$/g, '') : token);

  if (!isPlausibleObject(tokens[0])) return null;

  const prefixTokens: string[] = [];
  let flag: string | undefined;
  for (const token of tokens) {
    if (token.length > 1 && token.startsWith('-')) {
      flag = token;
      break;
    }
    prefixTokens.push(token);
  }
  if (prefixTokens.length === 0) return null;

  const pattern = prefixTokens.join(' ');
  const result: { pattern: string; requires?: string; flag?: string } = { pattern };
  if (requires) result.requires = requires;
  else if (flag) result.flag = flag;
  return result;
}

/** Parses a "stop doing X" / "don't run X again" instruction out of free text, or returns null
 * for anything that is not unambiguously such an instruction (a question, a description, an
 * acknowledgement with no actionable target, an object that is a pronoun/determiner/catch-all
 * rather than a command, etc). The returned `pattern` is deliberately just the object's leading
 * token (e.g. "pytest", not "pytest without -q"): a "without X" qualifier becomes `requires`
 * instead (see `parseObject`), and any other unrepresentable qualifier is dropped rather than
 * baked into a pattern that would deny nothing real — `tool.check`'s denial reason says so
 * explicitly either way, so the human sees exactly what is actually blocked. */
export function parseStopInstruction(text: string): SessionRule | null {
  const match = STOP_VERB.exec(text ?? '');
  if (!match) return null;
  const verb = match[1].toLowerCase();
  const object = parseObject(match[2] ?? '');
  if (!object) return null;
  const tool = EDIT_LIKE_VERBS.has(verb) ? 'Edit' : 'Bash';
  return { tool, ...object };
}

/** Adds a session-scoped rule parsed from the human's own words. Never touches the installed store. */
export function addSessionRule(state: SessionState, rule: SessionRule): void {
  if (!state.sessionRules) state.sessionRules = [];
  state.sessionRules.push(rule);
}
