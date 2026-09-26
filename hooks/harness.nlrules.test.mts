/**
 * Behavioural test for session-scoped natural-language rules in harness.ts / rules.ts, run
 * directly with `node --experimental-strip-types --no-warnings hooks/harness.nlrules.test.mts`
 * (see tests/test_hook_nlrules.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove parseStopInstruction discriminates a
 * real instruction from an ordinary question or a stop-word-with-no-object, that a session rule
 * ever denies a matching call and is never persisted to the installed store, or that the text
 * is actually read from `prompt.submit` (not the nonexistent `prompt.section` `event.prompt`):
 * a reviewer could match anywhere in the text, treat the action verb as optional, write the rule
 * into installed.json, or read a field that never fires, and every substring-based test would
 * still pass. This test drives the real registerRules() handlers with a fake `$` and asserts on
 * the actual decisions and on-disk writes.
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

/** A real PromptSubmitInput-shaped event, per the recovered declarations. */
function promptSubmit(text: string, extra: Record<string, unknown> = {}) {
  return { text, wait: false, origin: { user: {} }, ...extra };
}

async function main() {
  // --- 1. the brief's literal example names Bash and a pattern that is exactly "pytest" ---
  // (not "pytest without -q" — that qualifier cannot be matched against a real Bash command, so
  // it is dropped rather than baked into a pattern that would deny nothing real).
  {
    const rule = parseStopInstruction('stop running pytest without -q');
    assert.ok(rule, 'expected a rule to be parsed');
    assert.equal(rule!.tool, 'Bash');
    assert.equal(rule!.pattern, 'pytest', `expected the pattern to be exactly "pytest", got: ${rule!.pattern}`);
  }

  // --- 2. ordinary questions and acknowledgements with no actionable object must not create a rule ---
  {
    const mustNotMatch = [
      'what does this function do?',
      "don't worry about it",
      "don't know why this fails",
      'never mind',
      "Don't forget to update the README",
      'stop',
      "stop, that's wrong",
      // A mid-sentence mention of "stop": matching anywhere in the text (rather than only at
      // the start) would misfire on this.
      'can you help me figure out why the retries never stop?',
    ];
    for (const text of mustNotMatch) {
      const rule = parseStopInstruction(text);
      assert.equal(rule, null, `must not be parsed as a stop instruction: ${JSON.stringify(text)}`);
    }
  }

  // --- 3. a session rule added from real text denies the matching call on the next tool.check,
  //        via the real prompt.submit event shape (text/wait/origin), not the nonexistent
  //        prompt.section `event.prompt` ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const forwardingNext = async (e: any) => ({ text: e.text, context: e.context, origin: e.origin });
    const returned = await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest'), forwardingNext);
    assert.equal(returned.text, 'stop running pytest', 'a handler with nothing to add must pass the prompt through unchanged');

    const unrelated = { tool: 'Bash', input: { command: 'ls -la' } };
    const allowed = await handlers['tool.check'][0](dollar, unrelated, async () => ({ decision: 'allow' }));
    assert.equal(allowed.decision, 'allow', 'an unrelated call must not be denied by the session rule');
  }

  // --- 3b. end-to-end with the brief's literal example: both `pytest tests/` and `pytest -q`
  //         are denied, because the dropped "without -q" qualifier means the rule blocks pytest
  //         entirely, in every form, and says so ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest without -q'), async (e: any) => e);

    const deniedA = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest tests/' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(deniedA.decision, 'deny', '"pytest tests/" must be denied: it contains the blocked token');
    assert.ok(/pytest/i.test(deniedA.reason ?? ''), `reason must name what was blocked, got: ${deniedA.reason}`);

    const deniedB = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest -q' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(
      deniedB.decision, 'deny',
      '"pytest -q" must ALSO be denied: the qualifier "without -q" could not be represented, so the rule ' +
      'honestly blocks pytest in every form rather than pretending to allow the qualified case',
    );
  }

  // --- 4. the session rule is not written to installed.json ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest'), async (e: any) => e);

    assert.equal(files.has('C:/fake-harness-home/installed.json'), false, 'nothing here may write the installed store');
  }

  // --- 5. it is written to pending-session-rules.json for later review ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest'), async (e: any) => e);

    const path = 'C:/fake-harness-home/pending-session-rules.json';
    assert.ok(files.has(path), `expected a write to ${path}`);
    const written = JSON.parse(files.get(path)!);
    assert.ok(Array.isArray(written) && written.length === 1, 'expected exactly one pending rule');
    assert.equal(written[0].tool, 'Bash');
    assert.equal(written[0].pattern, 'pytest');
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
