/**
 * Behavioural test for the tool.call observer in harness.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.observer.test.mts`
 * (see tests/test_hook_assets.py, which shells out to it).
 *
 * This exists because a grep over the source text does not prove observe() ever writes
 * anything: a reviewer replaced dollar.fs.write with a no-op and every substring-based test
 * still passed. This test drives registerObserver() with a fake `$` whose fs methods record
 * calls, and asserts on the actual JSON that was written.
 */

import assert from 'node:assert/strict';
import { cause, registerObserver, safely } from './harness.ts';

// registerObserver only needs `safely` to exist as an import target; touch it so a bundler /
// type-checker never flags it as unused if this file is ever compiled instead of stripped.
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
      read: async (path: string) => files.get(path) ?? '',
      write: async (path: string, text: string) => {
        files.set(path, text);
      },
    },
  };
}

async function main() {
  // --- cause() mirrors the Python slugs, order-sensitive ---
  assert.equal(cause('No such file or directory: foo.txt'), 'missing-path');
  assert.equal(cause('UnicodeEncodeError: bogus'), 'unicode-encode');
  assert.equal(cause('nothing recognizable here'), 'other');

  // --- tool_error path writes a real, per-session JSON line ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    let handler: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler = h;
    });
    assert.equal(typeof handler, 'function', 'registerObserver must register a tool.call handler');

    const event = { tool: 'Bash', input: { command: 'cat missing.txt' } };
    const next = async () => ({ result: 'No such file or directory: missing.txt' });

    const outcome = await handler(dollar, event, next);
    assert.deepEqual(outcome, { result: 'No such file or directory: missing.txt' });

    const expectedPath = 'C:/fake-harness-home/observed-sess-abc123.jsonl';
    assert.ok(files.has(expectedPath), `expected a write to ${expectedPath}, got: ${[...files.keys()]}`);

    const lines = files.get(expectedPath)!.trim().split('\n');
    assert.equal(lines.length, 1, 'exactly one line for one failure');
    const record = JSON.parse(lines[0]);

    assert.equal(typeof record.ts, 'string');
    assert.ok(!Number.isNaN(Date.parse(record.ts)), 'ts must be a parseable timestamp');
    assert.equal(record.tool, 'Bash');
    assert.equal(record.kind, 'tool_error');
    assert.equal(record.cause, 'missing-path');
    assert.deepEqual(record.input, { command: 'cat missing.txt' });
  }

  // --- a second failure appends a second line to the SAME session's file, not a rewrite ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    let handler: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler = h;
    });

    const failingNext = async () => ({ result: 'PermissionError: Permission denied: /etc/shadow' });
    await handler(dollar, { tool: 'Read', input: { file_path: '/etc/shadow' } }, failingNext);
    await handler(dollar, { tool: 'Read', input: { file_path: '/etc/shadow2' } }, failingNext);

    const expectedPath = 'C:/fake-harness-home/observed-sess-abc123.jsonl';
    const lines = files.get(expectedPath)!.trim().split('\n');
    assert.equal(lines.length, 2, 'the second failure must append, not overwrite');
    assert.equal(JSON.parse(lines[0]).cause, 'permission');
    assert.equal(JSON.parse(lines[1]).cause, 'permission');
  }

  // --- a successful call writes nothing ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    let handler: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler = h;
    });

    const okNext = async () => ({ result: 'ok' });
    await handler(dollar, { tool: 'Read', input: { file_path: 'a.txt' } }, okNext);
    assert.equal(files.size, 0, 'a clean call must not write an observation');
  }

  // --- four identical calls in the window trip the repeat threshold ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    let handler: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler = h;
    });

    const okNext = async () => ({ result: 'ok' });
    const event = { tool: 'Bash', input: { command: 'flaky' } };
    for (let i = 0; i < 4; i += 1) {
      await handler(dollar, event, okNext);
    }

    const expectedPath = 'C:/fake-harness-home/observed-sess-abc123.jsonl';
    assert.ok(files.has(expectedPath), 'the 4th identical call must record a repeat');
    const lines = files.get(expectedPath)!.trim().split('\n');
    const kinds = lines.map((line) => JSON.parse(line).kind);
    assert.ok(kinds.includes('repeat'), `expected a repeat record, got: ${kinds}`);
  }

  // --- an observer that throws must NOT re-run the tool ---
  {
    // The whole point of `afterCall`: this handler's work happens after the tool has already
    // executed, so recovering by calling next() again would repeat a Bash command or a Write.
    const dollar = {
      env: { get: (key: string) => (key === 'META_HARNESS_HOME' ? 'C:/fake-harness-home' : undefined) },
      session: { id: async () => 'sess-abc123' },
      ui: { log: () => {} },
      fs: {
        exists: async () => false,
        read: async () => '',
        write: async () => {
          throw new Error('ENOSPC: no space left on device');
        },
      },
    };
    let handler: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler = h;
    });

    let calls = 0;
    const next = async () => {
      calls += 1;
      return { result: 'Error: No such file or directory: missing.txt' };
    };
    const outcome = await handler(dollar, { tool: 'Bash', input: { command: 'rm -rf build' } }, next);
    assert.equal(calls, 1, 'a failing observer must never run the tool a second time');
    assert.deepEqual(outcome, { result: 'Error: No such file or directory: missing.txt' },
      'the real outcome must still be returned when observation fails');
  }

  // --- isError does not fire on a successful result that merely CONTAINS error text ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    let handler: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler = h;
    });

    // A successful Grep for the literal "error:" is not a failure; counting it as one inflates
    // the ranking the learn cycle selects from.
    const grepNext = async () => ({
      result: 'src/app.ts:12:  console.log("error: something happened");',
    });
    await handler(dollar, { tool: 'Grep', input: { pattern: 'error:' } }, grepNext);
    assert.equal(files.size, 0, 'a successful search for error text must not be recorded as a failure');

    // The engine's own structured verdict is authoritative in both directions.
    const flagged = async () => ({ result: { is_error: true, content: 'nothing suspicious here' } });
    await handler(dollar, { tool: 'Bash', input: { command: 'false' } }, flagged);
    assert.equal(files.size, 1, 'an is_error result must be recorded even with innocuous text');

    const files2 = new Map<string, string>();
    const dollar2 = makeFakeDollar(files2);
    let handler2: any;
    registerObserver((event: string, h: any) => {
      if (event === 'tool.call') handler2 = h;
    });
    const notFlagged = async () => ({ result: { is_error: false, content: 'Error: in a quoted log line' } });
    await handler2(dollar2, { tool: 'Bash', input: { command: 'cat log' } }, notFlagged);
    assert.equal(files2.size, 0, 'is_error:false must win over error-looking text');
  }

  console.log('hooks/harness.observer.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
