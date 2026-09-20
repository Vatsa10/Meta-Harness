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

/** Append one JSON line. Append-only: a rewrite would lose concurrent sessions' records. */
async function observe(dollar: any, record: Record<string, unknown>): Promise<void> {
  const path = `${harnessHome(dollar)}/observed.jsonl`;
  const line = `${JSON.stringify({ ts: new Date().toISOString(), ...record })}\n`;
  const existing = (await dollar.fs.exists(path)) ? await dollar.fs.read(path) : '';
  await dollar.fs.write(path, existing + line);
}

function isError(result: unknown): boolean {
  const text = typeof result === 'string' ? result : JSON.stringify(result ?? '');
  return /is_error|error:|Traceback|not recognized|No such file/i.test(text);
}

/** Observes tool.call outcomes: records errors and repeated identical calls to observed.jsonl. */
export function registerObserver(on: On): void {
  const recent: Recent[] = [];

  on('tool.call', safely('tool.call', async (dollar, event: any, next) => {
    const outcome = await next(event);
    const tool = String(event?.tool ?? 'unknown');
    const key = `${tool}:${JSON.stringify(event?.input ?? {}).slice(0, 200)}`;

    recent.push({ tool, key });
    if (recent.length > REPEAT_WINDOW) recent.shift();
    const repeats = recent.filter((entry) => entry.key === key).length;
    if (repeats >= REPEAT_THRESHOLD) {
      await observe(dollar, { kind: 'repeat', tool, input: event?.input });
    }
    if (isError((outcome as any)?.result)) {
      await observe(dollar, { kind: 'tool_error', tool, input: event?.input,
                              text: String((outcome as any)?.result).slice(0, 400) });
    }
    return outcome;
  }));
}

export const register: Register = (on: On, options: PluginOptions) => {
  void options;
  registerObserver(on);
};
