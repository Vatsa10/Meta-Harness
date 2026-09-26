/**
 * Behavioural test for the first-run bootstrap message in harness.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.bootstrap.test.mts`
 * (see tests/test_hook_bootstrap.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove registerBootstrap() actually gates on
 * bootstrap.json, reads waste.json, or fails open on a throw: a reviewer could make it always
 * return the line, never write the marker, or drop the `safely` wrapper, and every
 * substring-based test would still pass. This test drives the real registerBootstrap() handler
 * with a fake `$` and asserts on the actual returned text and the marker file it writes.
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

async function main() {
  const bootstrapPath = 'C:/fake-harness-home/bootstrap.json';
  const wastePath = 'C:/fake-harness-home/waste.json';

  // --- 1. no bootstrap.json, waste.json present: the line names the count and 'harness waste' ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, JSON.stringify({ sessions: 12, calls_burned: 340 }));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const next = async () => ({ text: null });
    const result = await handlers['prompt.section'](dollar, { prompt: 'hi' }, next);
    assert.ok(result.text, 'expected a first-run message');
    assert.ok(result.text.includes('340'), `expected the correction count in the text, got: ${result.text}`);
    assert.ok(result.text.includes('harness waste'), `expected the command name in the text, got: ${result.text}`);
  }

  // --- 2. bootstrap.json is written as a side effect ---
  {
    const files = new Map<string, string>();
    files.set(wastePath, JSON.stringify({ sessions: 12, calls_burned: 340 }));
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const next = async () => ({ text: null });
    await handlers['prompt.section'](dollar, { prompt: 'hi' }, next);
    assert.ok(files.has(bootstrapPath), 'expected bootstrap.json to be written');
  }

  // --- 3. bootstrap.json already present: {text: null}, and it writes nothing further ---
  {
    const files = new Map<string, string>();
    files.set(bootstrapPath, JSON.stringify({ shown: '2020-01-01T00:00:00.000Z' }));
    files.set(wastePath, JSON.stringify({ sessions: 12, calls_burned: 340 }));
    const before = new Map(files);
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const next = async () => ({ text: null });
    const result = await handlers['prompt.section'](dollar, { prompt: 'hi' }, next);
    assert.equal(result.text, null, 'a later session must get no message');
    assert.deepEqual([...files.entries()], [...before.entries()], 'nothing may be written on a later session');
  }

  // --- 4. no waste.json: {text: null}, but bootstrap.json is still written ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerBootstrap(on as any);

    const next = async () => ({ text: null });
    const result = await handlers['prompt.section'](dollar, { prompt: 'hi' }, next);
    assert.equal(result.text, null, 'no waste report yet must yield no message');
    assert.ok(files.has(bootstrapPath), 'a machine with no history must still be marked as seen, so it is not asked again');
  }

  // --- 5. a throwing fs.read leaves the turn proceeding and returns {text: null} ---
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

    let nextCalled = false;
    const next = async () => {
      nextCalled = true;
      return { text: null };
    };
    const result = await handlers['prompt.section'](dollar, { prompt: 'hi' }, next);
    assert.equal(result.text, null, 'a throwing fs.read must fail open to {text: null}');
    assert.equal(nextCalled, true, 'the failure must be recovered via next(), not swallowed silently');
  }

  console.log('hooks/harness.bootstrap.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
