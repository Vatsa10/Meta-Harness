/**
 * Behavioural test for the first-run bootstrap message in harness.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.bootstrap.test.mts`
 * (see tests/test_hook_bootstrap.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove registerBootstrap() actually gates on
 * bootstrap.json, reads the real waste.json schema, attaches (rather than replaces) the line
 * while preserving every other field, calls `next` at most once no matter what fails, or marks
 * itself done only once the line has genuinely reached core: a reviewer could make it always
 * attach the line, never write the marker, mark itself done before the line is ever attached,
 * rebuild the outgoing event from scratch (dropping fields or replacing prior context), or wrap
 * itself in something that retries `next` on a failure, and every substring-based test would
 * still pass. This test drives the real registerBootstrap() handler with a fake `$` and asserts
 * on the actual returned event, how many times core (`next`) was actually called, and the
 * marker file it writes.
 *
 * registerBootstrap fires on `prompt.submit`, not `prompt.section`: `prompt.section` has no
 * `event.text`/`event.prompt` at all, and returning `{ text }` from it would REPLACE a
 * system-prompt section rather than add anything — see the CRITICAL 1 defect this file's tests
 * would have caught had they driven the real, recovered event shape from the start.
 */

import assert from 'node:assert/strict';
import { registerBootstrap } from './harness.ts';

function makeFakeDollar(files: Map<string, string>, overrides: Partial<Record<string, any>> = {}) {
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
      ...overrides,
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
function promptSubmit(extra: Record<string, unknown> = {}) {
  return { text: 'hi', wait: false, origin: { user: {} }, ...extra };
}

/** A `next` that resolves with the real PromptSubmitResult shape core would produce. */
async function forwardingNext(event: any) {
  return { text: event.text, context: event.context, origin: event.origin };
}

/** The real waste.json schema, per meta_harness/waste.py's waste_report(): `calls_burned` lives
 * under `corrections`, not at the top level. */
function wasteReport(sessions: number, callsBurned: number) {
  return JSON.stringify({ sessions, corrections: { count: 7, calls_burned: callsBurned } });
}

async function main() {
  const bootstrapPath = 'C:/fake-harness-home/bootstrap.json';
  const wastePath = 'C:/fake-harness-home/waste.json';

  // --- 1. no bootstrap.json, waste.json present: the line is ATTACHED as context, naming the
  //        count and the command, not returned as a replacement for anything ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const result = await handlers['prompt.submit'](dollar, promptSubmit(), forwardingNext);

    assert.equal(result.text, 'hi', 'the prompt text itself must pass through unchanged');
    assert.ok(Array.isArray(result.context) && result.context.length === 1, 'expected one attached context entry');
    const line = result.context[0];
    assert.ok(line.includes('340'), `expected the correction count in the attached line, got: ${line}`);
    assert.ok(line.includes('harness waste'), `expected the command name in the attached line, got: ${line}`);
  }

  // --- 2. bootstrap.json is written as a side effect, only once the line has been attached ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    await handlers['prompt.submit'](dollar, promptSubmit(), forwardingNext);
    assert.ok(files.has(bootstrapPath), 'expected bootstrap.json to be written');
  }

  // --- 3. bootstrap.json already present: passes the event through unchanged, writes nothing further ---
  {
    const files = new Map<string, string>();
    files.set(bootstrapPath, JSON.stringify({ shown: '2020-01-01T00:00:00.000Z' }));
    files.set(wastePath, wasteReport(12, 340));
    const before = new Map(files);
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const event = promptSubmit();
    let forwarded: any = null;
    const next = async (e: any) => {
      forwarded = e;
      return forwardingNext(e);
    };
    const result = await handlers['prompt.submit'](dollar, event, next);
    assert.deepEqual(forwarded, event, 'a later session must pass the event through with nothing added');
    assert.equal(result.context, undefined, 'a later session must attach no context');
    assert.deepEqual([...files.entries()], [...before.entries()], 'nothing may be written on a later session');
  }

  // --- 4. no waste.json yet: "not yet", not "never" — passes through unchanged AND does NOT
  //        write bootstrap.json, so a later turn (once a report exists) can still say the line ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const event = promptSubmit();
    let forwarded: any = null;
    const next = async (e: any) => {
      forwarded = e;
      return forwardingNext(e);
    };
    await handlers['prompt.submit'](dollar, event, next);
    assert.deepEqual(forwarded, event, 'no report yet must pass the event through with nothing added');
    assert.equal(
      files.has(bootstrapPath), false,
      'a machine with no history YET must not be marked done — it must be asked again once a report exists',
    );
  }

  // --- 5. a throwing fs.read leaves the turn proceeding (next still called exactly once),
  //        attaches nothing, and does NOT mark the bootstrap done ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, 'irrelevant');
    const dollar = makeFakeDollar(files, {
      read: async () => {
        throw new Error('EIO: disk read failure');
      },
    });
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const event = promptSubmit();
    let nextCalls = 0;
    let forwarded: any = null;
    const next = async (e: any) => {
      nextCalls += 1;
      forwarded = e;
      return forwardingNext(e);
    };
    const result = await handlers['prompt.submit'](dollar, event, next);
    assert.equal(nextCalls, 1, 'a failed read must still call next() exactly once, not zero and not twice');
    assert.deepEqual(forwarded, event, 'a failed read must not attach a half-formed line');
    assert.equal(result.context, undefined, 'a failed read must attach no context');
    assert.equal(
      files.has(bootstrapPath), false,
      'a transient read failure must not be marked done — it must be retried, not suppressed forever',
    );
  }

  // --- 6. finding 1: a marker write that THROWS must not be recovered by calling next() a
  //        second time — core must be called exactly once, and the line it already received
  //        must still be the result returned, even though the write afterward failed ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files, {
      write: async () => {
        throw new Error('EACCES: read-only harness home');
      },
    });
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    let coreCalls = 0;
    const next = async (e: any) => {
      coreCalls += 1;
      return forwardingNext(e);
    };
    const result = await handlers['prompt.submit'](dollar, promptSubmit(), next);
    assert.equal(coreCalls, 1, 'core must be called exactly once even when the marker write afterward throws');
    assert.ok(
      Array.isArray(result.context) && result.context.length === 1,
      'the line must still reach the caller even though the marker write failed',
    );
    assert.equal(files.has(bootstrapPath), false, 'a failed write leaves the marker absent, so it is retried next turn');
  }

  // --- 7. finding 1: if `next` ITSELF rejects, core must still have been called exactly once —
  //        never retried a second time with a plain, unmodified event ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    let coreCalls = 0;
    const next = async () => {
      coreCalls += 1;
      throw new Error('downstream hook exploded');
    };
    await assert.rejects(() => handlers['prompt.submit'](dollar, promptSubmit(), next));
    assert.equal(coreCalls, 1, 'core must be called exactly once even when next() itself rejects');
    assert.equal(
      files.has(bootstrapPath), false,
      'the marker must only be written once next() has actually resolved with the line attached',
    );
  }

  // --- 8. finding 3: a FULLY populated PromptSubmitInput reaches core with every field intact —
  //        attachments, origin, turnId, wait all preserved, and a PRE-EXISTING context entry
  //        survives ahead of the newly attached line, not replaced by it ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const fullEvent = {
      text: 'refactor the login form',
      attachments: [{ type: 'image', mediaType: 'image/png', filename: 'shot.png' }],
      context: ['an earlier context entry'],
      turnId: 'turn-42',
      wait: true,
      origin: { user: {} },
    };
    let forwarded: any = null;
    const next = async (e: any) => {
      forwarded = e;
      return forwardingNext(e);
    };
    const result = await handlers['prompt.submit'](dollar, fullEvent, next);

    assert.equal(forwarded.text, fullEvent.text);
    assert.deepEqual(forwarded.attachments, fullEvent.attachments, 'attachments must reach core unchanged');
    assert.equal(forwarded.turnId, fullEvent.turnId, 'turnId must reach core unchanged');
    assert.equal(forwarded.wait, fullEvent.wait, 'wait must reach core unchanged');
    assert.deepEqual(forwarded.origin, fullEvent.origin, 'origin must reach core unchanged');
    assert.deepEqual(
      forwarded.context,
      ['an earlier context entry', result.context[1]],
      'the pre-existing context entry must survive AHEAD of the newly attached line, never replaced',
    );
    assert.ok(String(result.context[1]).includes('340'), 'the attached line must still be the bootstrap message');
  }

  // --- 9. final fix wave: a report over ZERO sessions attaches nothing and is NOT marked done ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(0, 0));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const event = promptSubmit();
    let forwarded: any = null;
    await handlers['prompt.submit'](dollar, event, async (e: any) => { forwarded = e; return forwardingNext(e); });
    assert.deepEqual(forwarded, event, 'a zero-session report must add no line');
    assert.equal(files.has(bootstrapPath), false, 'a zero-session report must not mark the line as shown');
  }

  // --- 10. final fix wave: the line is hedged as an estimate, never stated as fact ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const result = await handlers['prompt.submit'](dollar, promptSubmit(), forwardingNext);
    const line = String(result.context[0]);
    assert.ok(line.includes('estimated'), `expected the line to be hedged as an estimate, got: ${line}`);
    assert.ok(!line.includes('went to wrong-direction work'), `the line must not state waste as fact: ${line}`);
  }

  console.log('hooks/harness.bootstrap.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
