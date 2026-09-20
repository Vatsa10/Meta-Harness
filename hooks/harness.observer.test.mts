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

  console.log('hooks/harness.observer.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
