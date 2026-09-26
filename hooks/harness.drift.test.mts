/**
 * Behavioural test for live drift surfacing (hooks/drift.ts + registerDrift in harness.ts), run with
 * `node --experimental-strip-types --no-warnings --import ./hooks/loaders/preload.mjs hooks/harness.drift.test.mts`
 * (see tests/test_hook_drift.py, which shells out to it).
 *
 * The note rides on a TOOL RESULT: the answered branch of ToolCallResult carries `context`,
 * "what the model reads after the tool's result and the user never sees". Every event is shaped as
 * claude-code-2.1.283.d.ts declares it: core's ToolCallResult is `{ ref, result, text }` (plus
 * `isError`/`isReadOnly`) or `{ deny }`; a PromptSubmitInput carries text, wait and origin (and
 * optionally attachments, turnId, context). The full plugin is registered through a CHAINING fake
 * `on`, the way the engine composes listeners.
 */

import assert from 'node:assert/strict';
import { register, registerDrift } from './harness.ts';
import { loadDriftConfig, shouldWarn } from './drift.ts';

const HOME = 'C:/fake-harness-home';
const DRIFT = `${HOME}/drift.json`;
const NOTE = 'meta-harness drift:';

function makeFakeDollar(files: Map<string, string>) {
  return {
    env: { get: (key: string) => (key === 'META_HARNESS_HOME' ? HOME : undefined) } as any,
    session: { id: async () => 'sess-drift' },
    ui: { log: (_m: string) => {} },
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

function userPrompt(text: string) {
  return { text, wait: false, origin: { kind: 'composer' } };
}

function notification(text: string) {
  return { text, wait: false, origin: { kind: 'task-notification' }, turnId: 'turn-7' };
}

/** Core's answered ToolCallResult for a Bash call. */
function answered(e: any) {
  return { ref: 1, result: { stdout: `ran ${e.command}` }, text: `ran ${e.command}` };
}

const notesOf = (outcome: any): string[] =>
  (Array.isArray(outcome?.context) ? outcome.context : []).filter((c: string) => c.startsWith(NOTE));

/** The full plugin, with helpers that fire real-shaped events through it. */
function plugin(files: Map<string, string>) {
  const dollar = makeFakeDollar(files);
  const { on, fire, handlers } = makeChainingOn();
  register(on as any, {} as any);
  let toolRuns = 0;
  let coreCalls = 0;
  let serial = 0;
  /** Fires n tool calls; returns each outcome as the caller (the engine) receives it. */
  const calls = async (n: number, core: (e: any) => any = answered) => {
    const outcomes: any[] = [];
    for (let i = 0; i < n; i += 1) {
      serial += 1;
      outcomes.push(await fire(dollar, 'tool.call', { tool: 'Bash', command: `echo ${serial}`, tool_use_id: `tu-${serial}` },
        async (e: any) => {
          toolRuns += 1;
          return core(e);
        }));
    }
    return outcomes;
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
  return { dollar, fire, handlers, calls, submit, toolRuns: () => toolRuns, coreCalls: () => coreCalls };
}

/** Index of every outcome carrying a drift note. */
const noted = (outcomes: any[]) => outcomes.flatMap((o, i) => (notesOf(o).length ? [i] : []));

async function main() {
  // --- 1 + (a). enabled, min_calls 8: the note is on the result of the call that crosses it,
  //              names the count, and appears on no earlier result ---
  {
    const p = plugin(new Map([[DRIFT, JSON.stringify({ enabled: true, min_calls: 8, judge: 'knn' })]]));
    const outcomes = await p.calls(10);
    assert.deepEqual(noted(outcomes), [7], `the note must be on call 8 exactly, got calls ${JSON.stringify(noted(outcomes))}`);
    const [note] = notesOf(outcomes[7]);
    assert.ok(note.includes('8 tool calls'), `the note must name the count: ${note}`);
    assert.equal(notesOf(outcomes[7]).length, 1, 'one note, not several');
    for (let i = 0; i < 7; i += 1) {
      assert.deepEqual(outcomes[i], answered({ command: `echo ${i + 1}` }), `call ${i + 1} is before min_calls and must be untouched`);
    }
    assert.equal(p.toolRuns(), 10, 'each tool runs exactly once');
  }

  // --- 2. the same stretch never produces a second note ---
  {
    const p = plugin(new Map([[DRIFT, JSON.stringify({ enabled: true, min_calls: 8 })]]));
    let all = await p.calls(10);
    assert.deepEqual(noted(all), [7]);
    // A notification is not the user speaking: the stretch goes on, and stays noted.
    await p.submit(notification('task done'));
    all = await p.calls(20);
    assert.deepEqual(noted(all), [], `same stretch, second note on calls ${JSON.stringify(noted(all))}`);
    // The user speaking ends the stretch and resets the count: the next note is on call 8 again.
    await p.submit(userPrompt('thanks'));
    all = await p.calls(7);
    assert.deepEqual(noted(all), [], 'the count must reset when the user speaks');
    all = await p.calls(3);
    assert.deepEqual(noted(all), [0], 'a new stretch is noted once, at its own 8th call');
    assert.ok(notesOf(all[0])[0].includes('8 tool calls'), notesOf(all[0])[0]);
  }

  // --- 3. enabled false: nothing, however many calls ---
  {
    const p = plugin(new Map([[DRIFT, JSON.stringify({ enabled: false, min_calls: 1 })]]));
    const outcomes = await p.calls(60);
    assert.deepEqual(noted(outcomes), []);
    assert.ok(outcomes.every((o, i) => o.context === undefined), 'a disabled drift adds no context at all');
  }

  // --- 4. no drift.json (and an empty one): nothing, and nothing creates one ---
  {
    for (const files of [new Map<string, string>(), new Map([[DRIFT, '']]), new Map([[DRIFT, '   \n']])]) {
      const before = files.get(DRIFT);
      const p = plugin(files);
      const outcomes = await p.calls(40);
      assert.deepEqual(noted(outcomes), [], 'missing/empty drift.json must mean disabled');
      assert.equal(files.get(DRIFT), before, 'nothing may create or change drift.json');
    }
    assert.equal(await loadDriftConfig(makeFakeDollar(new Map()), HOME), null);
    assert.equal(shouldWarn(1000, null), false);
  }

  // --- 5. corrupt drift.json: nothing, and every call proceeds with core's outcome ---
  {
    const corrupt = ['{not json', '[]', 'null', '"enabled"', '{"enabled":"true","min_calls":8}',
      '{"enabled":true}', '{"enabled":true,"min_calls":"8"}', '{"enabled":true,"min_calls":0}',
      '{"enabled":true,"min_calls":8,"judge":"vibes"}'];
    for (const text of corrupt) {
      const p = plugin(new Map([[DRIFT, text]]));
      const outcomes = await p.calls(20);
      assert.equal(p.toolRuns(), 20, `each call must run once with drift.json=${text}`);
      assert.deepEqual(noted(outcomes), [], `corrupt drift.json ${text} must mean disabled`);
      outcomes.forEach((o, i) => assert.deepEqual(o, answered({ command: `echo ${i + 1}` })));
    }
    const throwing = makeFakeDollar(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
    throwing.fs.read = async () => { throw new Error('EACCES'); };
    assert.equal(await loadDriftConfig(throwing, HOME), null);
  }

  // --- 6. the drift tool.call handler calls next exactly once; below the threshold or disabled it
  //        returns next's outcome unchanged (the same object) ---
  {
    const { on, handlers } = makeChainingOn();
    registerDrift(on as any);
    assert.equal(handlers['tool.call'].length, 1, 'registerDrift registers exactly one tool.call handler');
    assert.equal(handlers['prompt.submit'].length, 1, 'registerDrift registers exactly one prompt.submit handler');
    assert.equal(handlers['prompt.section'], undefined, 'drift must never listen on prompt.section');
    const handler = handlers['tool.call'][0];
    const dollar = makeFakeDollar(new Map([[DRIFT, '{"enabled":true,"min_calls":100}']]));
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
    let nextCalls = 0;
    await assert.rejects(() => handler(dollar, { tool: 'Bash', command: 'x' }, async () => {
      nextCalls += 1;
      throw new Error('tool failed');
    }));
    assert.equal(nextCalls, 1, 'a rejecting next must not be re-entered');

    // Next called once on the crossing call too, through the full chain.
    const p = plugin(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
    const [crossing] = await p.calls(1);
    assert.equal(p.toolRuns(), 1);
    assert.equal(notesOf(crossing).length, 1);
  }

  // --- (b). a deny is returned untouched even past the threshold; the note waits for the next
  //          answered call ---
  {
    const p = plugin(new Map([[DRIFT, '{"enabled":true,"min_calls":2}']]));
    await p.calls(1);
    const denies = [{ deny: 'refused once' }, { deny: 'refused twice' }];
    const got: any[] = [];
    for (const deny of denies) {
      const [outcome] = await p.calls(1, () => deny);
      got.push(outcome);
    }
    got.forEach((outcome, i) => {
      assert.equal(outcome, denies[i], `a deny past the threshold must come back as the same object, got ${JSON.stringify(outcome)}`);
      assert.deepEqual(Object.keys(outcome), ['deny'], 'no field may be added to a deny');
    });
    const [next] = await p.calls(1);
    assert.equal(notesOf(next).length, 1, 'the note goes on the first answered call past the threshold');
    assert.ok(notesOf(next)[0].includes('4 tool calls'), notesOf(next)[0]);
  }

  // --- (c). every outcome field survives, prior context stays ahead of the note, and core's
  //          object is never mutated ---
  {
    const p = plugin(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
    const core = { ref: 12, result: { stdout: '', stderr: 'bad' }, text: 'bad', isError: true, isReadOnly: true,
      context: ['an earlier reminder', 'a second reminder'] };
    const snapshot = JSON.parse(JSON.stringify(core));
    const [outcome] = await p.calls(1, () => core);
    assert.notEqual(outcome, core, 'a noted outcome must be a copy, not core\'s object');
    assert.deepEqual(core, snapshot, 'core\'s outcome object must not be mutated');
    for (const field of ['ref', 'result', 'text', 'isError', 'isReadOnly'] as const) {
      assert.deepEqual(outcome[field], snapshot[field], `${field} must survive, got ${JSON.stringify(outcome[field])}`);
    }
    assert.deepEqual(Object.keys(outcome).sort(), Object.keys(snapshot).sort(), 'no field added or dropped');
    assert.deepEqual(outcome.context.slice(0, 2), snapshot.context, `prior context must come first: ${JSON.stringify(outcome.context)}`);
    assert.equal(outcome.context.length, 3, `appended, not replaced: ${JSON.stringify(outcome.context)}`);
    assert.ok(outcome.context[2].startsWith(NOTE));
  }

  // --- (d). no prompt.submit event gains context from drift, in any drift state or origin; a
  //          fully populated PromptSubmitInput reaches core field-for-field ---
  {
    const full = {
      text: 'what happened here?',
      attachments: [{ type: 'image', mediaType: 'image/png', filename: 'shot.png' }],
      origin: { kind: 'composer' },
      turnId: 'turn-42',
      wait: true,
      context: ['an earlier context entry', 'a second earlier entry'],
    };
    const snapshot = JSON.parse(JSON.stringify(full));
    for (const text of [null, '{"enabled":true,"min_calls":1}', '{"enabled":true,"min_calls":8}', '{"enabled":false}']) {
      for (const kind of ['composer', 'task-notification']) {
        const files = new Map<string, string>();
        if (text !== null) files.set(DRIFT, text);
        const p = plugin(files);
        await p.calls(10);
        const event = { ...JSON.parse(JSON.stringify(snapshot)), origin: { kind } };
        const seen = await p.submit(event);
        assert.equal(p.coreCalls(), 1);
        assert.deepEqual(seen, event, `prompt.submit must reach core unchanged (drift=${text}, origin=${kind}): ${JSON.stringify(seen)}`);
      }
    }
    // The drift prompt.submit handler alone passes the very same object through.
    const { on, handlers } = makeChainingOn();
    registerDrift(on as any);
    const dollar = makeFakeDollar(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
    for (let i = 0; i < 5; i += 1) await handlers['tool.call'][0](dollar, { tool: 'Read', file_path: `f${i}` }, async () => ({ ref: i, result: {}, text: '' }));
    let seen: any = null;
    await handlers['prompt.submit'][0](dollar, full, async (e: any) => {
      seen = e;
      return { text: e.text, context: e.context };
    });
    assert.equal(seen, full, 'the drift prompt.submit handler must pass the event through as the same object');
    assert.deepEqual(full, snapshot);
  }

  // --- 8. exactly once on every path through the full register() chain ---
  {
    const states: Array<[string, string | null]> = [
      ['absent', null], ['disabled', '{"enabled":false,"min_calls":1}'],
      ['enabled', '{"enabled":true,"min_calls":1}'], ['corrupt', '{nope'],
    ];
    for (const [label, text] of states) {
      for (const origin of ['composer', 'task-notification']) {
        const files = new Map<string, string>();
        if (text !== null) files.set(DRIFT, text);
        const p = plugin(files);
        await p.calls(4);
        assert.equal(p.toolRuns(), 4, `tools ran ${p.toolRuns()} times for 4 calls (drift=${label})`);
        await p.submit({ text: 'go', wait: false, origin: { kind: origin } });
        assert.equal(p.coreCalls(), 1, `core called ${p.coreCalls()} times (drift=${label}, origin=${origin})`);

        let coreCalls = 0;
        await assert.rejects(() => p.fire(p.dollar, 'prompt.submit', { text: 'go', wait: false, origin: { kind: origin } },
          async () => {
            coreCalls += 1;
            throw new Error('core refused');
          }));
        assert.equal(coreCalls, 1, `rejecting core called ${coreCalls} times (drift=${label}, origin=${origin})`);

        let runs = 0;
        await assert.rejects(() => p.fire(p.dollar, 'tool.call', { tool: 'Bash', command: 'x' }, async () => {
          runs += 1;
          throw new Error('tool crashed');
        }));
        assert.equal(runs, 1, `a crashing tool ran ${runs} times (drift=${label})`);
      }
    }

    // A throwing drift tool.call handler (harness home lookup explodes after the tool ran): the
    // tool runs once and core's outcome reaches the caller intact.
    {
      const p = plugin(new Map([[DRIFT, '{"enabled":true,"min_calls":1}']]));
      let logged = '';
      p.dollar.ui.log = (m: string) => { logged += m; };
      p.dollar.env = { get: () => { throw new Error('env exploded'); } };
      const core = { ref: 5, result: { stdout: 'ok' }, text: 'ok' };
      const [outcome] = await p.calls(1, () => core);
      assert.equal(p.toolRuns(), 1, 'a throwing drift handler must not re-run the tool');
      assert.equal(outcome, core, 'core\'s outcome must come back as it was');
      assert.ok(logged.includes('tool.call:drift'), `the drift skip is logged: ${logged}`);
    }

    // A throwing drift prompt.submit handler: falls through to core once, event as it arrived.
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
  }

  console.log('hooks/harness.drift.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
