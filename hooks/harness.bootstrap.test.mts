/**
 * Behavioural test for the first-run bootstrap message in harness.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.bootstrap.test.mts`
 * (see tests/test_hook_bootstrap.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove registerBootstrap() actually gates on
 * bootstrap.json, reads the real waste.json schema, attaches (rather than replaces) the line, or
 * fails open on a throw without marking itself done: a reviewer could make it always attach the
 * line, never write the marker, mark itself done before the line is ever attached, or drop the
 * `safely` wrapper, and every substring-based test would still pass. This test drives the real
 * registerBootstrap() handler with a fake `$` and asserts on the actual returned event and the
 * marker file it writes.
 *
 * registerBootstrap fires on `prompt.submit`, not `prompt.section`: `prompt.section` has no
 * `event.text`/`event.prompt` at all, and returning `{ text }` from it would REPLACE a
 * system-prompt section rather than add anything — see the CRITICAL 1 defect this file's tests
 * would have caught had they driven the real, recovered event shape from the start.
 */

import assert from 'node:assert/strict';
import { registerBootstrap, safely } from './harness.ts';

void safely;

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

    const event = promptSubmit();
    const next = async (e: any) => ({ text: e.text, context: e.context, origin: e.origin });
    const result = await handlers['prompt.submit'](dollar, event, next);

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

    const next = async (e: any) => ({ text: e.text, context: e.context, origin: e.origin });
    await handlers['prompt.submit'](dollar, promptSubmit(), next);
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
      return { text: e.text, context: e.context, origin: e.origin };
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
      return { text: e.text, context: e.context, origin: e.origin };
    };
    await handlers['prompt.submit'](dollar, event, next);
    assert.deepEqual(forwarded, event, 'no report yet must pass the event through with nothing added');
    assert.equal(
      files.has(bootstrapPath), false,
      'a machine with no history YET must not be marked done — it must be asked again once a report exists',
    );
  }

  // --- 5. a throwing fs.read leaves the turn proceeding, attaches nothing, and does NOT mark
  //        the bootstrap done — a transient failure must not suppress the line permanently ---
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
    let nextCalled = false;
    let forwarded: any = null;
    const next = async (e: any) => {
      nextCalled = true;
      forwarded = e;
      return { text: e.text, context: e.context, origin: e.origin };
    };
    const result = await handlers['prompt.submit'](dollar, event, next);
    assert.equal(nextCalled, true, 'the failure must be recovered via next(), not swallowed silently');
    assert.deepEqual(forwarded, event, 'a failed read must not attach a half-formed line');
    assert.equal(result.context, undefined, 'a failed read must attach no context');
    assert.equal(
      files.has(bootstrapPath), false,
      'a transient read failure must not be marked done — it must be retried, not suppressed forever',
    );
  }

  // --- 6. the marker is written AFTER next() resolves, not before: if attaching the line itself
  //         fails (next throws), bootstrap.json must not be written either ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, wasteReport(12, 340));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    // Throws only on the call that carries the attached line (registerBootstrap's own call);
    // succeeds on the plain, unmodified event `safely`'s recovery calls next with, so this
    // isolates "did the write happen despite the attach failing" from `safely`'s own recovery
    // path throwing a second time.
    const next = async (e: any) => {
      if (e.context) throw new Error('downstream hook exploded');
      return { text: e.text, context: e.context, origin: e.origin };
    };
    await handlers['prompt.submit'](dollar, promptSubmit(), next);
    assert.equal(
      files.has(bootstrapPath), false,
      'the marker must only be written once next() has actually resolved with the line attached',
    );
  }

  console.log('hooks/harness.bootstrap.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
