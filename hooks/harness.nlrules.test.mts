/**
 * Behavioural test for session-scoped natural-language rules in harness.ts / rules.ts, run
 * directly with `node --experimental-strip-types --no-warnings hooks/harness.nlrules.test.mts`
 * (see tests/test_hook_nlrules.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove parseStopInstruction discriminates a
 * real instruction from an ordinary question, or that a session rule ever denies a call and is
 * never persisted to the installed store: a reviewer could match anywhere in the text, or write
 * the rule into installed.json, and every substring-based test would still pass. This test
 * drives the real registerRules() handlers with a fake `$` and asserts on the actual decisions
 * and on-disk writes.
 */

import assert from 'node:assert/strict';
import { registerRules, safely } from './harness.ts';
import { addSessionRule, parseStopInstruction } from './rules.ts';

void safely;

function makeFakeDollar(files: Map<string, string>) {
  return {
    env: {
      get: (key: string) => (key === 'META_HARNESS_HOME' ? 'C:/fake-harness-home' : undefined),
    },
    session: { id: async () => 'sess-abc123' },
    ui: { log: () => {} },
    fs: {
      exists: async (path: string) => files.has(path),
      read: async (path: string) => {
        if (!files.has(path)) throw new Error(`ENOENT: ${path}`);
        return files.get(path)!;
      },
      write: async (path: string, text: string) => {
        files.set(path, text);
      },
    },
  };
}

function makeOn() {
  const handlers: Record<string, any[]> = {};
  const on = (event: string, handler: any) => {
    (handlers[event] ??= []).push(handler);
  };
  return { on, handlers };
}

async function main() {
  // --- 1. a leading "stop running X" names Bash and a pattern containing the command ---
  {
    const rule = parseStopInstruction('stop running pytest without -q');
    assert.ok(rule, 'expected a rule to be parsed');
    assert.equal(rule!.tool, 'Bash');
    assert.ok(rule!.pattern.includes('pytest'), `expected the pattern to contain "pytest", got: ${rule!.pattern}`);
  }

  // --- 2. an ordinary question must not create a rule ---
  {
    const rule = parseStopInstruction('what does this function do?');
    assert.equal(rule, null, 'an ordinary question must not be parsed as a stop instruction');

    // A question that merely mentions "stop" mid-sentence, rather than opening with an
    // instruction, must also not be parsed: matching anywhere in the text (rather than only at
    // the start) would misfire on this and turn an ordinary question into a session rule.
    const midSentence = parseStopInstruction('can you help me figure out why the retries never stop?');
    assert.equal(midSentence, null, 'a mid-sentence mention of "stop" must not be parsed as an instruction');
  }

  // --- 3. a session rule added from such text denies the matching call on the next tool.check ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.section'][0](
      dollar,
      { prompt: "stop running pytest" },
      async (e: any) => ({ text: e?.fallback ?? null }),
    );

    const matching = { tool: 'Bash', input: { command: 'pytest -q tests/' } };
    const denied = await handlers['tool.check'][0](dollar, matching, async () => ({ decision: 'allow' }));
    assert.equal(denied.decision, 'deny', 'a call matching the session rule must be denied');
    assert.ok(/session rule/i.test(denied.reason ?? ''), `expected the reason to name a session rule, got: ${denied.reason}`);

    const unrelated = { tool: 'Bash', input: { command: 'ls -la' } };
    const allowed = await handlers['tool.check'][0](dollar, unrelated, async () => ({ decision: 'allow' }));
    assert.equal(allowed.decision, 'allow', 'an unrelated call must not be denied by the session rule');
  }

  // --- 4. the session rule is not written to installed.json ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.section'][0](
      dollar,
      { prompt: "stop running pytest" },
      async (e: any) => ({ text: e?.fallback ?? null }),
    );

    assert.equal(files.has('C:/fake-harness-home/installed.json'), false, 'nothing here may write the installed store');
  }

  // --- 5. it is written to pending-session-rules.json for later review ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.section'][0](
      dollar,
      { prompt: "stop running pytest" },
      async (e: any) => ({ text: e?.fallback ?? null }),
    );

    const path = 'C:/fake-harness-home/pending-session-rules.json';
    assert.ok(files.has(path), `expected a write to ${path}`);
    const written = JSON.parse(files.get(path)!);
    assert.ok(Array.isArray(written) && written.length === 1, 'expected exactly one pending rule');
    assert.equal(written[0].tool, 'Bash');
    assert.ok(String(written[0].pattern).includes('pytest'));
  }

  // --- direct addSessionRule/evaluator sanity, independent of the parser ---
  {
    const state: any = { readPaths: new Set(), callCounts: new Map(), rejected: new Map(), sessionRules: [] };
    addSessionRule(state, { tool: 'Bash', pattern: 'rm -rf' });
    assert.equal(state.sessionRules.length, 1);
  }

  console.log('hooks/harness.nlrules.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
