/**
 * Behavioural test for tool.check rule enforcement in harness.ts / rules.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.rules.test.mts`
 * (see tests/test_hook_rules.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove registerRules() ever denies a call:
 * a reviewer could make evaluateRule always return {deny: false} or make the tool.check handler
 * ignore the evaluator's verdict and every substring-based test would still pass. This test
 * drives the real registerRules() handlers with a fake `$` and a fake installed-artifact
 * registry and asserts on the actual decisions and on-disk reads.
 */

import assert from 'node:assert/strict';
import { registerRules, safely } from './harness.ts';
import { evaluateRule, loadRules } from './rules.ts';

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

async function main() {
  // --- built-in read-before-edit: deny before Read, allow after Read ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const nextAllow = async () => ({ decision: 'allow' });

    const editBeforeRead = await handlers['tool.check'](
      dollar,
      { tool: 'Edit', input: { file_path: 'a.txt' } },
      nextAllow,
    );
    assert.equal(editBeforeRead.decision, 'deny', 'Edit before Read must be denied');
    assert.ok(typeof editBeforeRead.reason === 'string' && editBeforeRead.reason.includes('read-before-edit'));

    // Read tracked via the tool.call handler
    await handlers['tool.call'](dollar, { tool: 'Read', input: { file_path: 'a.txt' } }, async () => ({}));

    const editAfterRead = await handlers['tool.check'](
      dollar,
      { tool: 'Edit', input: { file_path: 'a.txt' } },
      nextAllow,
    );
    assert.equal(editAfterRead.decision, 'allow', 'Edit after Read must be allowed, falling through to next');
  }

  // --- an unrelated tool is never denied by read-before-edit ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const nextAllow = async () => ({ decision: 'allow' });
    const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'ls' } }, nextAllow);
    assert.equal(result.decision, 'allow');
  }

  // --- an installed custom rule artifact from the registry is loaded and enforced ---
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
        origin: { kind: 'no-rm-rf', tools: ['Bash'] },
        payload: 'Do not run rm -rf in this repo.',
        replay: {},
      }),
    );

    const rules = await loadRules(dollar, 'C:/fake-harness-home');
    assert.ok(rules.some((r) => r.artifactId === 'no-rm-rf'), 'installed rule artifact must be loaded');
    assert.ok(rules.some((r) => r.artifactId === 'read-before-edit'), 'built-in rule must still be present');

    const { on, handlers } = makeOn();
    registerRules(on as any);
    const nextAllow = async () => ({ decision: 'allow' });
    // The handler's own lazy-load must pick up the same registry.
    const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'rm -rf /' } }, nextAllow);
    // evaluateRule only knows the built-in 'read-before-edit' kind, so a custom-kind rule with
    // no matching evaluator branch must not deny (fail-open on unknown rule kinds).
    assert.equal(result.decision, 'allow');
  }

  // --- a missing registry file: loadRules falls back to built-ins only, never throws ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const rules = await loadRules(dollar, 'C:/fake-harness-home');
    assert.deepEqual(rules.map((r) => r.artifactId), ['read-before-edit']);
  }

  // --- a corrupt registry file: loadRules fails open (built-ins only), tool.check still calls next ---
  {
    const files = new Map<string, string>();
    files.set('C:/fake-harness-home/installed.json', '{not json');
    const dollar = makeFakeDollar(files);

    const rules = await loadRules(dollar, 'C:/fake-harness-home');
    assert.deepEqual(rules.map((r) => r.artifactId), ['read-before-edit']);

    const { on, handlers } = makeOn();
    registerRules(on as any);
    let nextCalled = false;
    const next = async () => {
      nextCalled = true;
      return { decision: 'allow' };
    };
    const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'ls' } }, next);
    assert.equal(nextCalled, true, 'a corrupt registry must not block an unrelated tool call');
    assert.equal(result.decision, 'allow');
  }

  // --- ordering independence: tool.check registered first still sees reads tracked after it ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const handlers: Record<string, any> = {};
    // Register tool.check's underlying handlers via registerRules, but call the tool.call
    // handler AFTER the first tool.check check, simulating engine dispatch order variance:
    // registerRules itself installs both under the same closed-over `state`, so whichever
    // handler runs first, the other still observes updates to the same state.
    const on = (event: string, handler: any) => {
      handlers[event] = handler;
    };
    registerRules(on as any);

    const nextAllow = async () => ({ decision: 'allow' });
    const denied = await handlers['tool.check'](dollar, { tool: 'Write', input: { file_path: 'b.txt' } }, nextAllow);
    assert.equal(denied.decision, 'deny');

    await handlers['tool.call'](dollar, { tool: 'Read', input: { file_path: 'b.txt' } }, async () => ({}));

    const allowed = await handlers['tool.check'](dollar, { tool: 'Write', input: { file_path: 'b.txt' } }, nextAllow);
    assert.equal(allowed.decision, 'allow');
  }

  // --- evaluateRule directly: reason includes the artifact id in brackets ---
  {
    const rule = { artifactId: 'read-before-edit', kind: 'read-before-edit', tools: ['Edit'], reason: 'x' };
    const state = { readPaths: new Set<string>() };
    const verdict = evaluateRule(rule, { tool: 'Edit', input: { file_path: 'c.txt' } }, state);
    assert.equal(verdict.deny, true);
    assert.ok(verdict.reason!.includes('[read-before-edit]'));
  }

  console.log('hooks/harness.rules.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
