/**
 * Behavioural test for injection-artifact attachment in harness.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.injection.test.mts`
 * (see tests/test_hook_injection.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove registerInjection() ever attaches
 * text, skips non-injection artifact types, matches against the user's actual turn, or records
 * what it injected: a reviewer could make the handler always call next unchanged, inject a
 * 'rule'/'skill' artifact's payload, match against the wrong text, or drop the observe() call,
 * and every substring-based test would still pass. This test drives the real registerInjection()
 * handler with a fake `$` and a fake installed-artifact registry and asserts on the actual
 * returned event and the observe() side effect.
 *
 * registerInjection fires on `prompt.submit`, not `prompt.section`: an earlier version lived on
 * `prompt.section` with no section-name filter, matched triggers against the wrong text (the
 * system prompt's own section content), and returned `{ text }` to REPLACE that section — with
 * one injection artifact ever installed, that deleted or overwrote every OTHER section of the
 * system prompt on every turn that didn't match (CRITICAL 3). `prompt.submit` fixes both:
 * triggers match `e.text` (the human's actual words) and a match is ATTACHED via
 * `next({ ...e, context: [...] })`, never a replacement.
 */

import assert from 'node:assert/strict';
import { registerInjection, safely } from './harness.ts';
import { setDraw } from './receipts.ts';

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
  const handlers: Record<string, any> = {};
  const on = (event: string, handler: any) => {
    handlers[event] = handler;
  };
  return { on, handlers };
}

/** A real PromptSubmitInput-shaped event. */
function promptSubmit(text: string, extra: Record<string, unknown> = {}) {
  return { text, wait: false, origin: { user: {} }, ...extra };
}

/** A `next` that forwards the real PromptSubmitResult shape core would resolve to. */
async function forwardingNext(event: any) {
  return { text: event.text, context: event.context, origin: event.origin };
}

async function main() {
  // Holdout never fires unless a test says so.
  setDraw(() => 0.99);
  // --- no installed injections: the event passes through next() completely unchanged ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerInjection(on as any);

    const event = promptSubmit('hello there');
    let forwarded: any = null;
    const next = async (e: any) => {
      forwarded = e;
      return forwardingNext(e);
    };
    const result = await handlers['prompt.submit'](dollar, event, next);
    assert.deepEqual(forwarded, event, 'with nothing installed, the event must reach next() untouched');
    assert.equal(result.text, 'hello there');
    assert.equal(result.context, undefined);
  }

  // --- installed injection whose trigger matches the user's own turn: attached as context,
  //      use recorded, and the prompt TEXT ITSELF is never touched ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify([{ id: 'timeout-tip', type: 'injection', signature: 'sig', accepted: '2026-01-01' }]),
    );
    files.set(
      'C:/fake-harness-home/artifacts/timeout-tip/artifact.json',
      JSON.stringify({
        id: 'timeout-tip',
        type: 'injection',
        origin: { triggers: ['timeout'] },
        payload: 'Long-running commands should pass an explicit timeout.',
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);

    const matched = await handlers['prompt.submit'](dollar, promptSubmit('why did my command timeout'), forwardingNext);
    assert.equal(matched.text, 'why did my command timeout', 'the prompt text itself must never be replaced');
    assert.ok(Array.isArray(matched.context) && matched.context.length === 1, 'expected one attached context entry');
    assert.equal(matched.context[0], 'Long-running commands should pass an explicit timeout.');

    const sessionPath = 'C:/fake-harness-home/observed-sess-abc123.jsonl';
    assert.ok(files.has(sessionPath), 'a matched injection must be recorded to the session observation file');
    const recorded = files.get(sessionPath)!;
    assert.ok(recorded.includes('"injected"'), 'the recorded line must mark the event as an injection');
    assert.ok(recorded.includes('timeout-tip'), 'the recorded line must name the artifact that was injected');

    // --- a turn whose text does not mention any trigger attaches nothing, costing nothing ---
    const unmatchedEvent = promptSubmit('refactor the login form');
    let forwarded: any = null;
    const next = async (e: any) => {
      forwarded = e;
      return forwardingNext(e);
    };
    const unmatched = await handlers['prompt.submit'](dollar, unmatchedEvent, next);
    assert.deepEqual(forwarded, unmatchedEvent, 'no trigger match must forward the event unchanged, costing nothing this turn');
    assert.equal(unmatched.context, undefined);
  }

  // --- a match is ATTACHED alongside existing context, never replacing it ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify([{ id: 'timeout-tip', type: 'injection', signature: 'sig', accepted: '2026-01-01' }]),
    );
    files.set(
      'C:/fake-harness-home/artifacts/timeout-tip/artifact.json',
      JSON.stringify({
        id: 'timeout-tip',
        type: 'injection',
        origin: { triggers: ['timeout'] },
        payload: 'Long-running commands should pass an explicit timeout.',
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);

    const event = promptSubmit('why did my command timeout', { context: ['earlier context entry'] });
    const result = await handlers['prompt.submit'](dollar, event, forwardingNext);
    assert.deepEqual(result.context, [
      'earlier context entry',
      'Long-running commands should pass an explicit timeout.',
    ], 'a match must be appended to existing context, never replace it');
  }

  // --- an installed 'rule' artifact (not 'injection') must never be injected ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify([{ id: 'no-rm-rf', type: 'rule', signature: 'sig', accepted: '2026-01-01' }]),
    );
    files.set(
      'C:/fake-harness-home/artifacts/no-rm-rf/artifact.json',
      JSON.stringify({
        id: 'no-rm-rf',
        type: 'rule',
        origin: { triggers: ['rm'] },
        payload: 'Do not run rm -rf in this repo.',
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const result = await handlers['prompt.submit'](dollar, promptSubmit('please rm this file'), forwardingNext);
    assert.equal(result.context, undefined, 'a rule artifact must never be surfaced as an injection');
  }

  // --- a corrupt registry: fails open, prompt.submit still resolves via next(event) unchanged ---
  {
    const files = new Map<string, string>();
    files.set('C:/fake-harness-home/installed.json', '{not json');
    const dollar = makeFakeDollar(files);

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const event = promptSubmit('still works');
    let forwarded: any = null;
    const next = async (e: any) => {
      forwarded = e;
      return forwardingNext(e);
    };
    const result = await handlers['prompt.submit'](dollar, event, next);
    assert.deepEqual(forwarded, event, 'a corrupt registry must fail open, not break the turn');
    assert.equal(result.text, 'still works');
  }

  // --- a single over-long payload is truncated to the per-artifact cap, visibly ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const longPayload = 'x'.repeat(2000);
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify([{ id: 'verbose-tip', type: 'injection', signature: 'sig', accepted: '2026-01-01' }]),
    );
    files.set(
      'C:/fake-harness-home/artifacts/verbose-tip/artifact.json',
      JSON.stringify({
        id: 'verbose-tip',
        type: 'injection',
        origin: { triggers: ['verbose'] },
        payload: longPayload,
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const result = await handlers['prompt.submit'](dollar, promptSubmit('go verbose please'), forwardingNext);
    const attached = result.context[0];
    // An injection is paid in standing tokens every turn it fires, unlike a rule, so a single
    // artifact's text must never reach the model uncapped: a 2000-char payload must come back
    // far shorter than it went in, and the cut must be visible, not silent.
    assert.ok(attached.length < 500, `expected the payload to be capped well under its 2000-char source, got ${attached.length}`);
    assert.ok(attached.includes('truncated'), 'a truncated injection must say so, not cut silently');
  }

  // --- many installed injections together stay under a total bound, not just each individually ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify(ids.map((id) => ({ id, type: 'injection', signature: 'sig', accepted: '2026-01-01' }))),
    );
    for (const id of ids) {
      files.set(
        `C:/fake-harness-home/artifacts/${id}/artifact.json`,
        JSON.stringify({
          id,
          type: 'injection',
          origin: { triggers: ['broad'] },
          payload: 'y'.repeat(300),
          replay: {},
        }),
      );
    }

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const result = await handlers['prompt.submit'](dollar, promptSubmit('this is a broad match'), forwardingNext);
    const attached = result.context[0];
    // 8 artifacts x 300 chars each would be 2400+ chars uncapped; the joined total must stay
    // bounded regardless of how many injections are installed, not just each one individually.
    assert.ok(attached.length < 1500, `expected the joined total to stay bounded, got ${attached.length}`);
  }

  // --- receipts and holdout for a matched injection ---
  {
    const HOME = 'C:/fake-harness-home';
    for (const [drawValue, decision, id] of [[0.99, 'acted', 'sess-inj-acted'], [0.01, 'held', 'sess-inj-held']] as const) {
      setDraw(() => drawValue);
      const files = new Map<string, string>();
      files.set(`${HOME}/installed.json`, JSON.stringify([
        { id: 'timeout-tip', type: 'injection', signature: 'tool_error:Bash:timeout', accepted: '2026-01-01' },
      ]));
      files.set(`${HOME}/artifacts/timeout-tip/artifact.json`, JSON.stringify({
        id: 'timeout-tip', type: 'injection', origin: { triggers: ['timeout'] },
        payload: 'Long-running commands should pass an explicit timeout.',
      }));
      const dollar = { ...makeFakeDollar(files), session: { id: async () => id } };
      const { on, handlers } = makeOn();
      registerInjection(on as any);
      const event = promptSubmit('the build hit a timeout again');
      let forwarded: any = null;
      let nextCalls = 0;
      await handlers['prompt.submit'](dollar, event, async (e: any) => {
        nextCalls += 1;
        forwarded = e;
        return forwardingNext(e);
      });
      assert.equal(nextCalls, 1);
      if (decision === 'acted') {
        assert.ok(forwarded.context?.[0]?.includes('explicit timeout'), 'an acted injection attaches its text');
      } else {
        assert.equal(forwarded, event, 'a held injection passes the event through unchanged');
        assert.equal(forwarded.context, undefined, 'a held injection attaches no context');
        assert.equal(files.has(`${HOME}/observed-${id}.jsonl`), false, 'a held injection is not recorded as injected');
      }
      const text = files.get(`${HOME}/receipts-${id}.jsonl`);
      assert.ok(text, `${id}: a receipt must be written`);
      const receipts = text!.trim().split('\n').map((line) => JSON.parse(line));
      assert.equal(receipts.length, 1);
      assert.equal(receipts[0].event, 'prompt.submit');
      assert.equal(receipts[0].source, 'learned-injection');
      assert.equal(receipts[0].decision, decision);
      assert.equal(receipts[0].artifact, 'timeout-tip');
      assert.equal(receipts[0].signature, 'tool_error:Bash:timeout');
      assert.ok(!JSON.stringify(receipts[0]).includes('timeout again'), 'a receipt carries no message text');
    }
    setDraw(() => 0.99);
  }

  console.log('hooks/harness.injection.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
