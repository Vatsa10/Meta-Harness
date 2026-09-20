/**
 * Meta-Harness function hooks: observe failures, enforce learned artifacts.
 *
 * Every handler is wrapped in `safely`, which swallows errors and falls through to `next`.
 * A learning system that can break a session is worse than no learning system.
 */

import type { On, PluginOptions, Register } from 'claude-code';

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

function isError(result: unknown): boolean {
  return /is_error|error:|Traceback|not recognized|No such file/i.test(resultText(result));
}

/** Observes tool.call outcomes: records errors and repeated identical calls to observed-<session>.jsonl. */
export function registerObserver(on: On): void {
  const recent: Recent[] = [];

  on('tool.call', safely('tool.call', async (dollar, event: any, next) => {
    const outcome = await next(event);
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
    return outcome;
  }));
}

export const register: Register = (on: On, options: PluginOptions) => {
  void options;
  registerObserver(on);
};
