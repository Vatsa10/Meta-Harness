/**
 * Behavioural test for live drift surfacing (hooks/drift.ts + registerDrift in harness.ts), run with
 * `node --experimental-strip-types --no-warnings --import ./hooks/loaders/preload.mjs hooks/harness.drift.test.mts`
 * (see tests/test_hook_drift.py, which shells out to it).
 *
 * Every event is shaped as claude-code-2.1.283.d.ts declares it: a PromptSubmitInput carries
 * text, wait and origin (and optionally attachments, turnId, context); a ToolCallInput carries
 * the tool and its arguments beside it; core's ToolCallResult is `{ ref, result, text }`. The
 * full plugin is registered through a CHAINING fake `on`, the way the engine composes listeners,
 * so the drift handlers run alongside the observer, rules, injection and bootstrap handlers.
 */

import assert from 'node:assert/strict';
import { register, registerDrift } from './harness.ts';
import { loadDriftConfig, shouldWarn } from './drift.ts';

const HOME = 'C:/fake-harness-home';
const DRIFT = `${HOME}/drift.json`;

function makeFakeDollar(files: Map<string, string>) {
  return {
    env: { get: (key: string) => (key === 'META_HARNESS_HOME' ? HOME : undefined) },
    session: { id: async () => 'sess-drift' },
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

function makeChainingOn() {
  const handlers: Record<string, any[]> = {};
  const on = (event: string, handler: any) => {
    (handlers[event] ??= []).push(handler);
  };
  const fire = (dollar: any, event: string, input: any, terminal: (e: any) => Promise<any>) => {
    let chained = terminal;
    for (const handler of handlers[event] ?? []) {
      const inner = chained;
      chained = (e: any) => handler(dollar, e, inner);
    }
    return chained(input);
  };
  return { on, fire, handlers };
}

/** A PromptSubmitInput as the engine raises it for the user's own Enter. */
function userPrompt(text: string) {
  return { text, wait: false, origin: { kind: 'composer' } };
}

/** A PromptSubmitInput for a background task's notification delivered into a running turn. */
function notification(text: string) {
  return { text, wait: false, origin: { kind: 'task-notification' }, turnId: 'turn-7' };
}

function coreResult(e: any) {
  return { ref: 1, result: { stdout: `ran ${e.command ?? e.tool}` }, text: `ran ${e.command ?? e.tool}` };
}

/** Sets up the full plugin; returns helpers that fire real-shaped events through it. */
function plugin(files: Map<string, string>) {
  const dollar = makeFakeDollar(files);
  const { on, fire, handlers } = makeChainingOn();
  register(on as any, {} as any);
  let coreCalls = 0;
  const calls = async (n: number) => {
    for (let i = 0; i < n; i += 1) {
      await fire(dollar, 'tool.call', { tool: 'Bash', command: `echo ${i}`, tool_use_id: `tu-${i}` },
        async (e: any) => coreResult(e));
    }
  };
  const submit = async (event: any) => {
    let seen: any = null;
    await fire(dollar, 'prompt.submit', event, async (e: any) => {
      coreCalls += 1;
      seen = e;
      return { text: e.text, context: e.context };
    });
    return seen;
  };
  const notes = (seen: any) => (seen?.context ?? []).filter((c: string) => c.startsWith('meta-harness drift:'));
  return { dollar, fire, handlers, calls, submit, notes, coreCalls: () => coreCalls };
}

async function main() {
  // --- 1. enabled, min_calls 8, 10 calls: the next submit carries one line naming the count ---
  {
    const files = new Map([[DRIFT, JSON.stringify({ enabled: true, min_calls: 8, judge: 'knn' })]]);
    const p = plugin(files);
    await p.calls(10);
    const seen = await p.submit(userPrompt('carry on'));
    const found = p.notes(seen);
    assert.equal(found.length, 1, `expected exactly one drift note, got ${JSON.stringify(seen?.context)}`);
    assert.ok(found[0].includes('10 tool calls'), `the note must name the count: ${found[0]}`);
    assert.equal(p.coreCalls(), 1);

    // Under min_calls: nothing.
    const quiet = plugin(new Map(files));
    await quiet.calls(7);
    assert.equal(quiet.notes(await quiet.submit(userPrompt('ok'))).length, 0, '7 calls is under min_calls 8');
  }

  // --- 2. the same stretch never produces a second line ---
  {
    const files = new Map([[DRIFT, JSON.stringify({ enabled: true, min_calls: 8 })]]);
    const p = plugin(files);
    await p.calls(10);
    // A notification delivered into the running turn is not the user speaking: the stretch goes on.
    const first = await p.submit(notification('task done'));
    assert.equal(p.notes(first).length, 1, `first submit in the stretch must carry the note: ${JSON.stringify(first?.context)}`);
    await p.calls(5);
    const second = await p.submit(notification('another task done'));
    assert.equal(p.notes(second).length, 0, `same stretch, second note: ${JSON.stringify(second?.context)}`);
    // The user speaking ends that stretch without a second note for it.
    const ends = await p.submit(userPrompt('thanks'));
    assert.equal(p.notes(ends).length, 0, `the stretch was already noted: ${JSON.stringify(ends?.context)}`);
    // A fresh stretch that runs long is noted again, once.
    await p.calls(9);
    const fresh = await p.submit(userPrompt('next'));
    assert.equal(p.notes(fresh).length, 1, 'a new stretch past min_calls is noted');
    assert.ok(p.notes(fresh)[0].includes('9 tool calls'), `the new stretch's own count: ${p.notes(fresh)[0]}`);
    // And the user speaking reset the count: no carry-over into the next stretch.
    await p.calls(3);
    assert.equal(p.notes(await p.submit(userPrompt('more'))).length, 0, 'the count must reset when the user speaks');
  }

  // --- 3. enabled false: nothing, however many calls ---
  {
    const p = plugin(new Map([[DRIFT, JSON.stringify({ enabled: false, min_calls: 1 })]]));
    await p.calls(60);
    assert.equal(p.notes(await p.submit(userPrompt('go'))).length, 0);
    await p.calls(60);
    assert.equal(p.notes(await p.submit(notification('n'))).length, 0);
  }

  // --- 4. no drift.json (and an empty one): nothing ---
  {
    for (const files of [new Map<string, string>(), new Map([[DRIFT, '']]), new Map([[DRIFT, '   \n']])]) {
      const p = plugin(files);
      await p.calls(40);
      const seen = await p.submit(userPrompt('go'));
      assert.equal(p.notes(seen).length, 0, `missing/empty drift.json must mean disabled: ${JSON.stringify(seen?.context)}`);
      assert.equal(p.coreCalls(), 1);
      assert.ok(!files.has(DRIFT) || files.get(DRIFT)!.trim() === '', 'nothing may create an enabling drift.json');
    }
    assert.equal(await loadDriftConfig(makeFakeDollar(new Map()), HOME), null);
    assert.equal(shouldWarn(1000, null), false);
  }

  // --- 5. corrupt drift.json: nothing, and the turn proceeds exactly once ---
  {
    const corrupt = ['{not json', '[]', 'null', '"enabled"', '{"enabled":"true","min_calls":8}',
      '{"enabled":true}', '{"enabled":true,"min_calls":"8"}', '{"enabled":true,"min_calls":0}',
      '{"enabled":true,"min_calls":8,"judge":"vibes"}'];
    for (const text of corrupt) {
      const p = plugin(new Map([[DRIFT, text]]));
      await p.calls(20);
      const seen = await p.submit(userPrompt('go'));
      assert.ok(seen, `the turn must proceed with drift.json=${text}`);
      assert.equal(seen.text, 'go');
      assert.equal(p.notes(seen).length, 0, `corrupt drift.json ${text} must mean disabled`);
      assert.equal(p.coreCalls(), 1, `core must see the prompt exactly once with drift.json=${text}`);
    }
    // An fs that throws on the drift read is also just "disabled".
    const throwing = makeFakeDollar(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
    throwing.fs.read = async () => { throw new Error('EACCES'); };
    assert.equal(await loadDriftConfig(throwing, HOME), null);
  }

  // --- 6. the drift tool.call handler calls next exactly once and returns its outcome unchanged ---
  {
    const { on, handlers } = makeChainingOn();
    registerDrift(on as any);
    assert.equal(handlers['tool.call'].length, 1, 'registerDrift registers exactly one tool.call handler');
    assert.equal(handlers['prompt.submit'].length, 1, 'registerDrift registers exactly one prompt.submit handler');
    assert.equal(handlers['prompt.section'], undefined, 'drift must never listen on prompt.section');
    const handler = handlers['tool.call'][0];
    const dollar = makeFakeDollar(new Map());
    for (const outcome of [
      { ref: 3, result: { stdout: 'hi' }, text: 'hi' },
      { ref: 4, result: 'boom', text: 'boom', isError: true },
      { deny: 'refused by a rule' },
    ]) {
      let nextCalls = 0;
      const event = { tool: 'Bash', command: 'echo hi', tool_use_id: 'tu-1' };
      const got = await handler(dollar, event, async (e: any) => {
        nextCalls += 1;
        assert.equal(e, event, 'the tool.call event must be passed on as received');
        return outcome;
      });
      assert.equal(nextCalls, 1, 'tool.call must call next exactly once');
      assert.equal(got, outcome, 'tool.call must return the outcome of next unchanged (same object)');
    }
    // A rejecting tool is not retried.
    let nextCalls = 0;
    await assert.rejects(() => handler(dollar, { tool: 'Bash', command: 'x' }, async () => {
      nextCalls += 1;
      throw new Error('tool failed');
    }));
    assert.equal(nextCalls, 1, 'a rejecting next must not be re-entered');

    // Through the full chain too: the tool runs once and its outcome reaches the caller intact.
    const p = plugin(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
    let runs = 0;
    const outcome = { ref: 9, result: { stdout: 'ok' }, text: 'ok' };
    const got = await p.fire(p.dollar, 'tool.call', { tool: 'Bash', command: 'ls' }, async () => {
      runs += 1;
      return outcome;
    });
    assert.equal(runs, 1);
    assert.deepEqual(got, outcome);
  }

  // --- 7. a fully populated PromptSubmitInput through the drift handler: every field unchanged,
  //        prior context ahead of the note ---
  {
    const files = new Map([[DRIFT, JSON.stringify({ enabled: true, min_calls: 2 })]]);
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeChainingOn();
    registerDrift(on as any);
    for (let i = 0; i < 3; i += 1) {
      await handlers['tool.call'][0](dollar, { tool: 'Read', file_path: `f${i}` }, async () => ({ ref: i, result: {}, text: '' }));
    }
    const full = {
      text: 'what happened here?',
      attachments: [{ type: 'image', mediaType: 'image/png', filename: 'shot.png' }],
      origin: { kind: 'composer' },
      turnId: 'turn-42',
      wait: true,
      context: ['an earlier context entry', 'a second earlier entry'],
    };
    const snapshot = JSON.parse(JSON.stringify(full));
    let seen: any = null;
    let coreCalls = 0;
    const result = await handlers['prompt.submit'][0](dollar, full, async (e: any) => {
      coreCalls += 1;
      seen = e;
      return { text: e.text, context: e.context };
    });
    assert.equal(coreCalls, 1);
    for (const field of ['text', 'attachments', 'origin', 'turnId', 'wait'] as const) {
      assert.deepEqual(seen[field], snapshot[field], `${field} must reach core unchanged, got ${JSON.stringify(seen[field])}`);
    }
    assert.deepEqual(Object.keys(seen).sort(), Object.keys(snapshot).sort(), 'no field added or dropped');
    assert.deepEqual(seen.context.slice(0, 2), snapshot.context, `prior context must come first: ${JSON.stringify(seen.context)}`);
    assert.equal(seen.context.length, 3);
    assert.ok(seen.context[2].includes('3 tool calls'), `the note comes after: ${seen.context[2]}`);
    assert.deepEqual(result, { text: full.text, context: seen.context }, "core's answer is returned as is");
    assert.deepEqual(full, snapshot, "the caller's event object must not be mutated");
  }

  // --- 8. core exactly once through the full register() chain on every path ---
  {
    const states: Array<[string, string | null]> = [
      ['absent', null], ['disabled', '{"enabled":false,"min_calls":1}'],
      ['enabled', '{"enabled":true,"min_calls":1}'], ['corrupt', '{nope'],
    ];
    for (const [label, text] of states) {
      for (const origin of ['composer', 'task-notification']) {
        // a resolving core
        {
          const files = new Map<string, string>();
          if (text !== null) files.set(DRIFT, text);
          const p = plugin(files);
          await p.calls(4);
          await p.submit({ text: 'go', wait: false, origin: { kind: origin } });
          assert.equal(p.coreCalls(), 1, `core called ${p.coreCalls()} times (drift=${label}, origin=${origin})`);
        }
        // a rejecting core is never re-entered
        {
          const files = new Map<string, string>();
          if (text !== null) files.set(DRIFT, text);
          const p = plugin(files);
          await p.calls(4);
          let coreCalls = 0;
          await assert.rejects(() => p.fire(p.dollar, 'prompt.submit', { text: 'go', wait: false, origin: { kind: origin } },
            async () => {
              coreCalls += 1;
              throw new Error('core refused');
            }));
          assert.equal(coreCalls, 1, `rejecting core called ${coreCalls} times (drift=${label}, origin=${origin})`);
        }
      }
    }

    // A throwing drift handler: reading `origin` throws inside the drift handler (the other
    // prompt.submit handlers never read it), so drift fails before next and must fall through to
    // core exactly once with the event as it arrived.
    {
      const p = plugin(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
      await p.calls(5);
      let logged = '';
      p.dollar.ui.log = (m: string) => { logged += m; };
      const event: any = { text: 'go', wait: false };
      Object.defineProperty(event, 'origin', { enumerable: false, get() { throw new Error('origin exploded'); } });
      let coreCalls = 0;
      let seen: any = null;
      await p.fire(p.dollar, 'prompt.submit', event, async (e: any) => {
        coreCalls += 1;
        seen = e;
        return { text: e.text };
      });
      assert.equal(coreCalls, 1, `a throwing drift handler must reach core exactly once, got ${coreCalls}`);
      assert.equal(seen, event, 'the throwing drift handler must pass the event through as it arrived');
      assert.ok(logged.includes('prompt.submit:drift'), `the drift skip is logged: ${logged}`);
    }

    // A throwing drift handler whose next rejects: still exactly one core call.
    {
      const { on, handlers } = makeChainingOn();
      registerDrift(on as any);
      const event: any = { text: 'go', wait: false };
      Object.defineProperty(event, 'origin', { enumerable: false, get() { throw new Error('origin exploded'); } });
      let coreCalls = 0;
      await assert.rejects(() => handlers['prompt.submit'][0](makeFakeDollar(new Map()), event, async () => {
        coreCalls += 1;
        throw new Error('core refused');
      }));
      assert.equal(coreCalls, 1);
    }
  }

  console.log('hooks/harness.drift.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
