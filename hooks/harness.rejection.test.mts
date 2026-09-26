/**
 * Behavioural test for rejection memory in harness.ts / rules.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.rejection.test.mts`
 * (see tests/test_hook_rejection.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove a rejected call is ever remembered or
 * denied: a reviewer could make rememberRejection a no-op, key the memory by tool name alone, or
 * never clear it, and every substring-based test would still pass. This test drives the real
 * registerRules() handlers with a fake `$` and asserts on the actual tool.check decisions.
 */

import assert from 'node:assert/strict';
import { registerRules, safely } from './harness.ts';

void safely;

function makeFakeDollar(files: Map<string, string>, sessionId = 'sess-abc123') {
  return {
    env: {
      get: (key: string) => (key === 'META_HARNESS_HOME' ? 'C:/fake-harness-home' : undefined),
    },
    session: { id: async () => sessionId },
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

async function runToolCall(handlers: Record<string, any[]>, dollar: any, event: any, next: any) {
  let outcome = next;
  // Chain every registered tool.call handler, innermost (last-registered) first, so each one
  // wraps the next exactly the way the real engine composes multiple listeners on one event.
  let chained = next;
  for (const handler of handlers['tool.call']) {
    const inner = chained;
    chained = (e: any) => handler(dollar, e, inner);
  }
  outcome = await chained(event);
  return outcome;
}

async function main() {
  const rejectedResult = { result: "The user doesn't want to proceed with this tool use." };
  const okResult = { result: 'ok' };

  // --- 1. a call rejected once, then re-proposed identically, is denied naming the rejection ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const event = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlers, dollar, event, async () => rejectedResult);

    const denied = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(denied.decision, 'deny', 'a re-proposed rejected call must be denied');
    assert.ok(
      /rejected/i.test(denied.reason ?? ''),
      `expected the deny reason to name the earlier rejection, got: ${denied.reason}`,
    );
  }

  // --- 2. a different call is not denied ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const rejected = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlers, dollar, rejected, async () => rejectedResult);

    const different = { tool: 'Bash', input: { command: 'ls -la' } };
    const result = await handlers['tool.check'][0](dollar, different, async () => ({ decision: 'allow' }));
    assert.equal(result.decision, 'allow', 'a different call must never be denied by another call\'s rejection');
  }

  // --- 3. after the user's text mentions the rejected command again, the same call is allowed ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const event = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlers, dollar, event, async () => rejectedResult);

    const stillDenied = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(stillDenied.decision, 'deny', 'sanity check: must be denied before the mention');

    await handlers['prompt.submit'][0](
      dollar,
      { text: 'actually please go ahead and run rm -rf build again', wait: false, origin: { plugin: 'test' } },
      async (e: any) => ({ text: e?.text ?? '' }),
    );

    const allowedAfterMention = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(allowedAfterMention.decision, 'allow', 'mentioning the rejected command again must clear the memory');
  }

  // --- 4. a rejection in one session does not affect another session's state ---
  {
    const filesA = new Map<string, string>();
    const dollarA = makeFakeDollar(filesA, 'sess-A');
    const { on: onA, handlers: handlersA } = makeOn();
    registerRules(onA as any);

    const event = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlersA, dollarA, event, async () => rejectedResult);
    const deniedInA = await handlersA['tool.check'][0](dollarA, event, async () => ({ decision: 'allow' }));
    assert.equal(deniedInA.decision, 'deny');

    const filesB = new Map<string, string>();
    const dollarB = makeFakeDollar(filesB, 'sess-B');
    const { on: onB, handlers: handlersB } = makeOn();
    registerRules(onB as any);

    const allowedInB = await handlersB['tool.check'][0](dollarB, event, async () => ({ decision: 'allow' }));
    assert.equal(allowedInB.decision, 'allow', 'a fresh registerRules() instance must start with no rejection memory');
  }

  // --- 5. the tool.check handler still calls next on the allow path ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    let nextCalled = false;
    const next = async () => {
      nextCalled = true;
      return { decision: 'allow' };
    };
    const result = await handlers['tool.check'][0](dollar, { tool: 'Bash', input: { command: 'ls' } }, next);
    assert.equal(nextCalled, true, 'the allow path must still call next');
    assert.equal(result.decision, 'allow');
  }

  // --- the rejection can be announced via `deny` alone (the real ToolCallResult deny branch) ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    // Shaped exactly as ToolCallResult's deny branch: `{ deny, result: undefined, text: undefined }`.
    const denyOutcome = { deny: "The user doesn't want to proceed with this tool use." };
    const event = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlers, dollar, event, async () => denyOutcome);

    const result = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(result.decision, 'deny', 'a rejection carried only on `deny` must still be remembered');
  }

  // --- the rejection can be announced via `text` alone (the model-facing joined text) ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    // Shaped exactly as ToolCallResult's answered branch, with the rejection surfaced only in
    // `text` (the model-facing joined text), not `result`.
    const textOutcome = { result: { ok: true }, text: 'Tool use was rejected by the user.' };
    const event = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlers, dollar, event, async () => textOutcome);

    const result = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(result.decision, 'deny', 'a rejection carried only on `text` must still be remembered');
  }

  // --- a real PromptSubmitInput-shaped event (text/wait/origin/context) drives the same handler ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const event = { tool: 'Bash', input: { command: 'rm -rf build' } };
    await runToolCall(handlers, dollar, event, async () => rejectedResult);

    const realShapedPromptSubmit = {
      text: 'go ahead and run rm -rf build again please',
      wait: false,
      origin: { user: {} },
      context: ['some earlier context'],
    };
    let forwarded: any = null;
    await handlers['prompt.submit'][0](dollar, realShapedPromptSubmit, async (e: any) => {
      forwarded = e;
      return { text: e.text, context: e.context, origin: e.origin };
    });
    assert.deepEqual(forwarded, realShapedPromptSubmit, 'a handler with nothing to add must pass the real event through unchanged');

    const allowed = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(allowed.decision, 'allow', 'the real-shaped mention must still clear the rejection');
  }

  // --- a merely-failed call (not a rejection) must not be remembered ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const event = { tool: 'Bash', input: { command: 'cat missing.txt' } };
    await runToolCall(handlers, dollar, event, async () => okResult);

    const result = await handlers['tool.check'][0](dollar, event, async () => ({ decision: 'allow' }));
    assert.equal(result.decision, 'allow', 'a successful (non-rejected) call must not be treated as rejected');
  }

  console.log('hooks/harness.rejection.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
