/**
 * Meta-Harness function hooks: observe failures, enforce learned artifacts.
 *
 * Every handler is wrapped in `safely` or `afterCall`, both of which swallow a throw and fall
 * through to `next` rather than break the turn. A learning system that can break a session is
 * worse than no learning system.
 *
 * Per-turn features (rejection memory, session rules, injections, the first-run message) live on
 * `prompt.submit`, never `prompt.section`. `prompt.section` fires once per NAMED SECTION of the
 * system prompt, its sections are cached for the whole session, and returning `{ text }` REPLACES
 * that section — there is no `event.prompt`/`event.text` carrying the user's words on it, and a
 * handler with no section-name filter fires for and can overwrite every section that exists,
 * including ones it has never heard of. `prompt.submit` carries the user's actual turn as
 * `e.text`, and a hook adds anything the model should see via `next({ ...e, context: [...] })`
 * without touching the system prompt at all.
 */

import type { On, PluginOptions, Register } from 'claude-code';
import {
  addSessionRule,
  callKey,
  evaluateRule,
  loadInstalled,
  loadRules,
  parseStopInstruction,
  rememberRejection,
  type Rule,
  type SessionState,
  wasRejected,
} from './rules.js';

export type Fallible<E, R> = (dollar: any, event: E, next: (e: E) => Promise<R>) => Promise<R>;

/** Wrap a handler so a throw becomes a pass-through instead of a broken turn. */
export function safely<E, R>(name: string, handler: Fallible<E, R>): Fallible<E, R> {
  return async (dollar, event, next) => {
    try {
      return await handler(dollar, event, next);
    } catch (error) {
      try {
        dollar.ui.log(
          `meta-harness ${name} skipped (${error instanceof Error ? error.message : String(error)})`,
        );
      } catch {
        // logging must never be the thing that breaks the turn either
      }
      return next(event);
    }
  };
}

/**
 * Wrap a handler whose work happens AFTER the tool has already run.
 *
 * `safely` recovers by calling `next(event)`, which is correct only for a handler that calls
 * `next` last: for a post-`next` handler that would run the tool a SECOND time, duplicating the
 * side effect of a Bash or Write. Here `next` is called exactly once, up front, and the fallible
 * work is what gets swallowed — so an observer that throws costs an observation, never a repeated
 * command.
 */
export function afterCall<E, R>(
  name: string,
  handler: (dollar: any, event: E, outcome: R) => Promise<void>,
): Fallible<E, R> {
  return async (dollar, event, next) => {
    const outcome = await next(event);
    try {
      await handler(dollar, event, outcome);
    } catch (error) {
      try {
        dollar.ui.log(
          `meta-harness ${name} skipped (${error instanceof Error ? error.message : String(error)})`,
        );
      } catch {
        // logging must never be the thing that breaks the turn either
      }
    }
    return outcome;
  };
}

const REPEAT_WINDOW = 6;
const REPEAT_THRESHOLD = 4;

type Recent = { tool: string; key: string };

/** The harness's home directory: overridable for tests, otherwise under the user's profile. */
function harnessHome(dollar: any): string {
  const override = dollar.env?.get?.('META_HARNESS_HOME');
  return override || `${dollar.env?.get?.('USERPROFILE') || dollar.env?.get?.('HOME')}/.claude/harness`;
}

/**
 * Mirrors meta_harness.replay.ERROR_PATTERNS: same regex text and slug, in the same order, so a
 * `cause` recorded here dedupes against what Python's `_cause()` computes from the same text.
 * `tests/test_hook_assets.py::test_ts_cause_patterns_match_python_error_patterns` asserts this
 * list and the Python one cannot drift apart silently.
 *
 * UnicodeEncodeError must stay ordered before the decode/charmap/cp1252 pattern: an encode
 * failure's message can also contain the word "decode" in surrounding text, so checking decode
 * first would misclassify it. That ordering was fixed on the Python side in Task 2.
 */
const CAUSE_PATTERNS: ReadonlyArray<readonly [string, string]> = [
  ['UnicodeEncodeError', 'unicode-encode'],
  ['UnicodeDecodeError|charmap|cp1252', 'unicode-decode'],
  ['No such file or directory|cannot find the (file|path)', 'missing-path'],
  ['Permission denied|EACCES', 'permission'],
  ['command not found|is not recognized as', 'missing-command'],
  ['timed out|TimeoutExpired', 'timeout'],
  ['has not been read yet|must read.*before', 'unread-edit'],
  ['String to replace not found|old_string', 'edit-mismatch'],
  ['SyntaxError|unterminated', 'syntax'],
];

/** Classifies failure text into the same slug meta_harness.replay._cause() would produce. */
export function cause(text: string): string {
  const value = text ?? '';
  for (const [pattern, slug] of CAUSE_PATTERNS) {
    if (new RegExp(pattern, 'i').test(value)) return slug;
  }
  return 'other';
}

/** This session's id, or 'unknown' when the engine cannot supply one; never throws. */
async function currentSessionId(dollar: any): Promise<string> {
  try {
    const id = await dollar.session?.id?.();
    return id ? String(id) : 'unknown';
  } catch {
    return 'unknown';
  }
}

/**
 * Appends one JSON line to this session's own observation file.
 *
 * One file per session, not one shared file: `$.fs` exposes no append or lock primitive, so a
 * shared file would need a non-atomic exists/read/write cycle that two concurrent sessions can
 * race and clobber each other on. Per-session files remove the race instead of trying to guard
 * it; a reader globs `observed-*.jsonl` under the harness home. `$.fs.write` creates missing
 * parent directories itself, so a fresh harness home on a new machine needs no separate mkdir.
 */
async function observe(dollar: any, sessionId: string, record: Record<string, unknown>): Promise<void> {
  const path = `${harnessHome(dollar)}/observed-${sessionId}.jsonl`;
  const line = `${JSON.stringify({ ts: new Date().toISOString(), ...record })}\n`;
  const existing = (await dollar.fs.exists(path)) ? await dollar.fs.read(path) : '';
  await dollar.fs.write(path, existing + line);
}

function resultText(result: unknown): string {
  return typeof result === 'string' ? result : JSON.stringify(result ?? '');
}

/**
 * All the text a real `tool.call` outcome (ToolCallResult) can carry, across every shape it can
 * take: `deny` (the refusal string, present only on the deny branch), `text` (the model-facing
 * joined text core sets) and `result` (the tool's own record). A human's rejection can be
 * announced in any of the three depending on where in the chain it was decided — checking
 * `result` alone (as an earlier version here did) missed a rejection that surfaced as `deny` or
 * `text` instead.
 */
function outcomeText(outcome: any): string {
  const parts = [outcome?.deny, outcome?.text, resultText(outcome?.result)];
  return parts.filter((part): part is string => typeof part === 'string' && part.length > 0).join(' ');
}

/**
 * Unambiguous tool-error phrases: strings a tool emits when it refuses, which do not plausibly
 * appear as the FIRST thing in a successful result. Anything weaker (a bare `error:` anywhere in
 * the text) matched a successful `Grep` for the word "error:" and recorded it as a failure;
 * those counts feed selection ranking, so the noise became the thing the learn loop chased.
 */
const ERROR_PHRASES =
  /has not been read yet|String to replace not found|is not recognized as an internal or external command|No such file or directory/i;

/** True when the tool call actually failed. */
function isError(result: unknown): boolean {
  if (result && typeof result === 'object') {
    const flag = (result as any).is_error ?? (result as any).isError;
    // The engine's own verdict is authoritative; never second-guess it by grepping the text.
    if (typeof flag === 'boolean') return flag;
  }
  const text = resultText(result);
  // Otherwise only an error ANNOUNCED at the start of the result counts, plus a short list of
  // phrases a tool only ever emits when it refused.
  return /^\s*"?(error|[A-Za-z.]*Error:|Traceback \(most recent call last\))/i.test(text)
    || ERROR_PHRASES.test(text);
}

/** Observes tool.call outcomes: records errors and repeated identical calls to observed-<session>.jsonl. */
export function registerObserver(on: On): void {
  const recent: Recent[] = [];

  // afterCall, not safely: this handler's work runs after the tool has already executed, so a
  // recovery that re-entered next() would run the tool twice.
  on('tool.call', afterCall('tool.call', async (dollar, event: any, outcome: any) => {
    const tool = String(event?.tool ?? 'unknown');
    const key = `${tool}:${JSON.stringify(event?.input ?? {}).slice(0, 200)}`;

    recent.push({ tool, key });
    if (recent.length > REPEAT_WINDOW) recent.shift();
    const repeats = recent.filter((entry) => entry.key === key).length;

    const sessionId = await currentSessionId(dollar);
    if (repeats >= REPEAT_THRESHOLD) {
      await observe(dollar, sessionId, { kind: 'repeat', tool, cause: 'repeat', input: event?.input });
    }
    const text = resultText((outcome as any)?.result);
    if (isError((outcome as any)?.result)) {
      await observe(dollar, sessionId, {
        kind: 'tool_error',
        tool,
        cause: cause(text),
        input: event?.input,
        text: text.slice(0, 400),
      });
    }
  }));
}

/** Unambiguous phrases the engine emits when the human declines a tool call outright, as opposed
 * to the tool itself failing. Recording on these two phrases only (never a bare "no" in the
 * conversation) keeps rejection memory from firing on an ordinary declined suggestion in prose. */
const REJECTION_PATTERN = /doesn't want to proceed|tool use was rejected/i;

/**
 * Enforces installed `rule` artifacts at tool.check: the layer that costs no standing tokens and
 * cannot be talked around, because it runs before the tool call, not as prose in a prompt.
 *
 * Also carries this session's rejection memory (task 13) and session-scoped "stop doing X" rules
 * (task 14), both stored on the same `SessionState` so `tool.check` can consult them ahead of the
 * installed-rule loop. Rejection detection lives in the same `tool.call` handler as read-tracking
 * below; the real hook declarations confirm multiple listeners on one event DO compose in
 * production (each runs, `next` chains through them), so a second, independent registration would
 * have worked too — they were folded into one handler here for simplicity, not because
 * composition needed it. `prompt.submit` (not `prompt.section`) is where the human's own words
 * are read, since only `prompt.submit`'s event carries `text`.
 */
export function registerRules(on: On): void {
  const state: SessionState = {
    readPaths: new Set<string>(),
    callCounts: new Map<string, number>(),
    rejected: new Map<string, string>(),
    sessionRules: [],
  };
  let rules: Rule[] | null = null;

  // afterCall, not safely: rejection detection reads the outcome, which only exists AFTER the
  // tool has already run (or been denied), so a `safely` recovery re-entering next() would run
  // the tool a second time. `next` resolves before this handler's own body runs at all — both
  // the read-tracking and the rejection check below run AFTER the call, not "first"; read-tracking
  // simply doesn't care about the outcome, so its ordering relative to `next` has no effect either
  // way.
  on('tool.call', afterCall('tool.call:read-tracking', async (dollar, event: any, outcome: any) => {
    if (event?.tool === 'Read') {
      const path = String(event?.input?.file_path ?? '');
      if (path) state.readPaths.add(path);
    }
    const text = outcomeText(outcome);
    if (REJECTION_PATTERN.test(text)) {
      rememberRejection(state, String(event?.tool ?? ''), event?.input);
    }
  }));

  // prompt.submit, not prompt.section: only prompt.submit's event carries the human's actual
  // words (`e.text`). This handler mutates session state only (never the model-visible prompt),
  // so it always passes `event` through to `next` unchanged.
  on('prompt.submit', safely('prompt.submit:nlrules', async (dollar, event: any, next) => {
    const text = String(event?.text ?? '');
    const instruction = parseStopInstruction(text);
    if (instruction) {
      addSessionRule(state, instruction);
      const path = `${harnessHome(dollar)}/pending-session-rules.json`;
      await dollar.fs.write(path, JSON.stringify(state.sessionRules, null, 2));
    }
    if (text) {
      const lower = text.toLowerCase();
      for (const [key, snippet] of [...(state.rejected ?? new Map<string, string>()).entries()]) {
        if (snippet && lower.includes(snippet.toLowerCase())) {
          state.rejected!.delete(key);
        }
      }
    }
    return next(event);
  }));

  on('tool.check', safely('tool.check', async (dollar, event: any, next) => {
    const tool = String(event?.tool ?? '');
    if (wasRejected(state, tool, event?.input)) {
      return {
        decision: 'deny',
        reason: 'This exact call was already rejected earlier this session [rejection-memory]',
      };
    }
    for (const rule of state.sessionRules ?? []) {
      if (rule.tool !== tool) continue;
      const haystack = JSON.stringify(event?.input ?? {}).toLowerCase();
      if (haystack.includes(rule.pattern.toLowerCase())) {
        return {
          decision: 'deny',
          // Says exactly what was blocked: the rule denies any call whose input contains this
          // one token, nothing narrower — a qualifier the human's own phrasing carried (like
          // "without -q") could not be represented, so the rule is honest about denying the
          // command in every form rather than pretending to be more precise than it is.
          reason: `Session rule from this conversation: this blocks any ${tool} call whose input `
            + `contains "${rule.pattern}", in any form [session-rule]`,
        };
      }
    }
    if (rules === null) rules = await loadRules(dollar, harnessHome(dollar));
    for (const rule of rules) {
      const verdict = evaluateRule(rule, event, state);
      if (verdict.deny) return { decision: 'deny', reason: verdict.reason };
    }
    // Counted here, before the call proceeds, so a `repeat-call` rule sees how many times this
    // exact call has already been allowed. Counting at tool.check rather than after the result
    // keeps the observer the only post-`next` handler on this path.
    const key = callKey(event);
    state.callCounts.set(key, (state.callCounts.get(key) ?? 0) + 1);
    return next(event);
  }));
}

type Injection = { artifactId: string; triggers: string[]; text: string };

/**
 * Per-artifact injected-text cap, matching the `reason` cap the rules layer already applies
 * (rules.ts:~50). An `injection` is paid in standing tokens on every turn it fires, unlike a
 * `rule`, which is exactly why it sits below `rule` in the layer ordering — an uncapped
 * injection would out-cost the stronger, free layer above it.
 */
const INJECTION_TEXT_CAP = 400;

/**
 * Cap on the joined text of ALL matched injections for one turn, so N installed injections
 * cannot add up past a bound even though each is individually capped. Set to three artifacts'
 * worth of INJECTION_TEXT_CAP: enough for a few unrelated matches to coexist, not enough for an
 * unbounded number of installs to dominate the prompt.
 */
const INJECTION_TOTAL_CAP = INJECTION_TEXT_CAP * 3;

const TRUNCATION_MARKER = '… [truncated]';

/** Truncates visibly rather than silently, so a capped injection cannot be mistaken for a short one. */
function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}${TRUNCATION_MARKER}`;
}

/** Installed `injection` artifacts only: a `rule`, `skill` or `doctrine` row is never surfaced here. */
async function loadInjections(dollar: any, home: string): Promise<Injection[]> {
  const artifacts = await loadInstalled(dollar, home, 'injection');
  return artifacts.map((artifact) => ({
    artifactId: String(artifact.id),
    triggers: (artifact.origin as any)?.triggers ?? [],
    text: truncate(String(artifact.payload ?? ''), INJECTION_TEXT_CAP),
  }));
}

/**
 * Injects installed `injection` artifacts as model-only context on `prompt.submit`: the layer
 * below `rule`, paid in standing tokens only on the turns where it actually fires.
 *
 * This used to live on `prompt.section` and return `{ text }` to replace a system-prompt
 * section — with no section-name filter, that both matched triggers against the WRONG text (the
 * system prompt's own section content, not the human's turn) and, the moment one injection
 * artifact was ever installed, deleted or overwrote every other section of the system prompt on
 * every turn it didn't match. `prompt.submit` fixes both: triggers match `e.text` (the human's
 * actual words) and a match is ATTACHED via `next({ ...e, context: [...] })`, never a
 * replacement. The hook itself knows what it attached, so use is recorded by construction via
 * `observe` instead of asking the model to self-report a retrieval it might forget.
 */
export function registerInjection(on: On): void {
  let injections: Injection[] | null = null;

  on('prompt.submit', safely('prompt.submit:injection', async (dollar, event: any, next) => {
    if (injections === null) injections = await loadInjections(dollar, harnessHome(dollar));
    if (injections.length === 0) return next(event);

    const haystack = String(event?.text ?? '').toLowerCase();
    const matched = injections.filter((injection) =>
      injection.triggers.some((trigger) => haystack.includes(String(trigger).toLowerCase())));
    if (matched.length === 0) return next(event);

    const sessionId = await currentSessionId(dollar);
    for (const injection of matched) {
      await observe(dollar, sessionId, { kind: 'injected', artifactId: injection.artifactId });
    }
    const joined = truncate(matched.map((injection) => injection.text).join('\n\n'), INJECTION_TOTAL_CAP);
    return next({ ...event, context: [...(event?.context ?? []), joined] });
  }));
}

/**
 * Says once, on the first `prompt.submit` after install, what wrong-direction work has cost so
 * far, then never again: `<harnessHome>/bootstrap.json` is the marker. Presence of that file
 * (not any in-memory flag) is the only thing that gates the message, so it stays correct across
 * every session after the first, not just within the process that happened to write it.
 *
 * The numbers come from a report `meta-harness waste --json` writes to
 * `<harnessHome>/waste.json` (`{ sessions, corrections: { calls_burned, ... }, ... }`, per
 * `meta_harness/waste.py`'s `waste_report()`); this hook never runs that command itself.
 *
 * The marker is written ONLY after the line has actually been attached via `next(...)`: a
 * missing `waste.json` means "no report yet, try again next turn", not "never again", and a
 * transient read/parse failure propagates to the outer `safely` wrapper rather than being
 * swallowed here — either way nothing is marked done, so the question keeps being asked (at
 * most once per turn, which costs nothing on a turn with no report to show) until it can
 * actually be answered once.
 */
export function registerBootstrap(on: On): void {
  on('prompt.submit', safely('prompt.submit:bootstrap', async (dollar, event: any, next) => {
    const home = harnessHome(dollar);
    const bootstrapPath = `${home}/bootstrap.json`;
    if (await dollar.fs.exists(bootstrapPath)) return next(event);

    const wastePath = `${home}/waste.json`;
    if (!(await dollar.fs.exists(wastePath))) return next(event);

    // Deliberately not locally try/caught: a bad read or a corrupt waste.json must propagate to
    // the outer `safely` wrapper, whose catch recovers via next(event) WITHOUT writing the
    // marker below — a transient failure means "try again next turn", never "suppressed for
    // good".
    const waste: any = JSON.parse(await dollar.fs.read(wastePath));
    const sessions = waste?.sessions;
    const callsBurned = waste?.corrections?.calls_burned;
    if (sessions == null || callsBurned == null) return next(event);

    const line = `Analyzed ${sessions} sessions. ~${callsBurned} tool calls went to wrong-direction work. `
      + '`/harness waste` for the breakdown.';
    const result = await next({ ...event, context: [...(event?.context ?? []), line] });

    // Marked done only now that the line has actually reached `next` and resolved — a throw
    // above never reaches this line, so it never marks a turn "done" that didn't actually say
    // anything.
    await dollar.fs.write(bootstrapPath, JSON.stringify({ shown: new Date().toISOString() }));
    return result;
  }));
}

export const register: Register = (on: On, options: PluginOptions) => {
  void options;
  registerObserver(on);
  registerRules(on);
  registerInjection(on);
  registerBootstrap(on);
};
