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
import { callKey, evaluateRule, loadRules } from './rules.ts';

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
  // --- the plugin never denies an Edit of an unread file: that is the engine's job, not
  //     rules.ts/harness.ts's. A live test proved Claude Code's own "File has not been read
  //     yet" check already runs before this plugin, so a plugin-side copy only added failure
  //     modes (every Edit denied, every new-file Write denied, Write-then-Edit denied) on top
  //     of an enforcement that already existed. ---
  {
    const files = new Map<string, string>();
    files.set('a.txt', 'existing');
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const nextAllow = async () => ({ decision: 'allow' });

    // No Read has happened; the plugin must still fall through to `next` (the engine's own
    // check runs downstream of this plugin and is what would deny it live).
    const editUnread = await handlers['tool.check'](
      dollar,
      { tool: 'Edit', input: { file_path: 'a.txt' } },
      nextAllow,
    );
    assert.equal(editUnread.decision, 'allow', 'the plugin must not deny an unread Edit itself');

    // Write-then-Edit: the plugin must not deny this either (a prior bug denied it because the
    // plugin's own tracker only ever recorded Reads, never Writes).
    const write = await handlers['tool.check'](
      dollar,
      { tool: 'Write', input: { file_path: 'new.txt', content: 'hi' } },
      nextAllow,
    );
    assert.equal(write.decision, 'allow', 'the plugin must not deny a Write of a new file');
    const editAfterWrite = await handlers['tool.check'](
      dollar,
      { tool: 'Edit', input: { file_path: 'new.txt' } },
      nextAllow,
    );
    assert.equal(editAfterWrite.decision, 'allow', 'the plugin must not deny an Edit that follows a Write');
  }

  // --- a regression test for tool.check's fail-open behaviour: if `$.fs.exists` throws while
  //     loading installed rule artifacts, the call must still be allowed through, never denied. ---
  {
    const dollar = {
      env: { get: (key: string) => (key === 'META_HARNESS_HOME' ? 'C:/fake-harness-home' : undefined) },
      session: { id: async () => 'sess-throws' },
      ui: { log: () => {} },
      fs: {
        exists: async () => { throw new Error('boom: fs.exists threw'); },
        read: async () => { throw new Error('boom: fs.read threw'); },
        write: async () => { throw new Error('boom: fs.write threw'); },
      },
    };
    const { on, handlers } = makeOn();
    registerRules(on as any);
    let nextCalled = false;
    const next = async () => {
      nextCalled = true;
      return { decision: 'allow' };
    };
    const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'ls' } }, next);
    assert.equal(nextCalled, true, 'a throwing fs.exists must fail open and still reach next()');
    assert.equal(result.decision, 'allow', 'a throwing fs.exists must never turn into a deny');
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

    const { on, handlers } = makeOn();
    registerRules(on as any);
    const nextAllow = async () => ({ decision: 'allow' });
    // The handler's own lazy-load must pick up the same registry.
    const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'rm -rf /' } }, nextAllow);
    // evaluateRule only knows the built-in 'repeat-call' kind, so a custom-kind rule with
    // no matching evaluator branch must not deny (fail-open on unknown rule kinds).
    assert.equal(result.decision, 'allow');
  }

  // --- a missing registry file: loadRules yields no rules at all, never throws (no built-ins ship) ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const rules = await loadRules(dollar, 'C:/fake-harness-home');
    assert.deepEqual(rules, []);
  }

  // --- a corrupt registry file: loadRules fails open (no rules), tool.check still calls next ---
  {
    const files = new Map<string, string>();
    files.set('C:/fake-harness-home/installed.json', '{not json');
    const dollar = makeFakeDollar(files);

    const rules = await loadRules(dollar, 'C:/fake-harness-home');
    assert.deepEqual(rules, []);

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

  // --- ordering independence: tool.check registered first still sees a rejection recorded after it ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const handlers: Record<string, any> = {};
    // Register tool.check's underlying handlers via registerRules, but record the rejection via
    // the tool.call handler AFTER the first tool.check check, simulating engine dispatch order
    // variance: registerRules itself installs both under the same closed-over `state`, so
    // whichever handler runs first, the other still observes updates to the same state.
    const on = (event: string, handler: any) => {
      handlers[event] = handler;
    };
    registerRules(on as any);

    const nextAllow = async () => ({ decision: 'allow' });
    const input = { command: 'rm -rf build' };
    const beforeRejection = await handlers['tool.check'](dollar, { tool: 'Bash', input }, nextAllow);
    assert.equal(beforeRejection.decision, 'allow');

    await handlers['tool.call'](
      dollar,
      { tool: 'Bash', command: 'rm -rf build' },
      async () => ({ deny: "The user doesn't want to proceed with this tool use. The tool use was rejected." }),
    );

    const afterRejection = await handlers['tool.check'](dollar, { tool: 'Bash', input }, nextAllow);
    assert.equal(afterRejection.decision, 'deny');
    assert.match(afterRejection.reason, /rejection-memory/);
  }

  // --- evaluateRule directly: a repeat-call rule denies once the threshold is reached, and the
  //     reason includes the artifact id in brackets ---
  {
    const rule = { artifactId: 'my-repeat', kind: 'repeat-call', tools: ['Bash'], reason: 'x', threshold: 2 };
    const state = { callCounts: new Map<string, number>([[callKey({ tool: 'Bash', input: { command: 'ls' } }), 1]]) };
    const verdict = evaluateRule(rule, { tool: 'Bash', input: { command: 'ls' } }, state as any);
    assert.equal(verdict.deny, true);
    assert.ok(verdict.reason!.includes('[my-repeat]'));
  }

  console.log('hooks/harness.rules.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
