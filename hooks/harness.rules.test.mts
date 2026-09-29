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

async function main() {
  // Never hold unless a test says so: holdout draws must not make these assertions flaky.
  setDraw(() => 0.99);

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

  // --- receipts and holdout at tool.check ---
  {
    const HOME = 'C:/fake-harness-home';
    const receiptsOf = (files: Map<string, string>, session: string) => {
      const text = files.get(`${HOME}/receipts-${session}.jsonl`);
      return text ? text.trim().split('\n').map((line) => JSON.parse(line)) : [];
    };
    const withRule = (files: Map<string, string>) => {
      files.set(`${HOME}/installed.json`, JSON.stringify([
        { id: 'no-thrash', type: 'rule', signature: 'repeat:Bash:ls', accepted: '2026-01-01' },
      ]));
      files.set(`${HOME}/artifacts/no-thrash/artifact.json`, JSON.stringify({
        id: 'no-thrash', type: 'rule',
        origin: { rule_kind: 'repeat-call', tools: ['Bash'], threshold: 2 },
        payload: 'Stop repeating ls.',
      }));
    };
    const session = (id: string, files: Map<string, string>) =>
      ({ ...makeFakeDollar(files), session: { id: async () => id } });

    // loadRules carries the registry row's signature, '' when absent
    {
      const files = new Map<string, string>();
      withRule(files);
      const rules = await loadRules(makeFakeDollar(files), HOME);
      assert.equal(rules[0].signature, 'repeat:Bash:ls');
      files.set(`${HOME}/installed.json`, JSON.stringify([{ id: 'no-thrash', type: 'rule' }]));
      assert.equal((await loadRules(makeFakeDollar(files), HOME))[0].signature, '');
    }

    // draw 0.99, rate 0.1: denied, one 'acted' receipt naming the artifact and signature
    for (const [drawValue, rate, expectDeny, expectDecision, id] of [
      [0.99, null, true, 'acted', 'sess-acted'],
      [0.01, null, false, 'held', 'sess-held'],
      [0.0, 0, true, 'acted', 'sess-zero'],
    ] as const) {
      setDraw(() => drawValue);
      const files = new Map<string, string>();
      withRule(files);
      if (rate !== null) files.set(`${HOME}/receipts.json`, JSON.stringify({ holdout_rate: rate }));
      const dollar = session(id, files);
      const { on, handlers } = makeOn();
      registerRules(on as any);
      let nextCalls = 0;
      const next = async () => { nextCalls += 1; return { decision: 'allow' }; };
      const event = { tool: 'Bash', input: { command: 'ls' } };
      await handlers['tool.check'](dollar, event, next);
      assert.equal(nextCalls, 1, 'the first identical call is never denied');
      assert.equal(receiptsOf(files, id).length, 0, 'no decision, no receipt');
      const result = await handlers['tool.check'](dollar, event, next);
      if (expectDeny) {
        assert.equal(result.decision, 'deny', `${id}: must deny`);
        assert.equal(nextCalls, 1);
      } else {
        assert.equal(result.decision, 'allow', `${id}: a held rule lets the call proceed`);
        assert.equal(nextCalls, 2, 'next called exactly once for the held call');
      }
      const receipts = receiptsOf(files, id);
      assert.equal(receipts.length, 1, `${id}: exactly one receipt`);
      assert.equal(receipts[0].event, 'tool.check');
      assert.equal(receipts[0].source, 'learned-rule');
      assert.equal(receipts[0].decision, expectDecision);
      assert.equal(receipts[0].artifact, 'no-thrash');
      assert.equal(receipts[0].signature, 'repeat:Bash:ls');
      assert.equal(receipts[0].tool, 'Bash');
      assert.equal('input' in receipts[0], false, 'a receipt carries no tool input');
    }

    // a session rule is never held out, whatever the draw
    {
      setDraw(() => 0.01);
      const files = new Map<string, string>();
      const dollar = session('sess-rule', files);
      const { on, handlers } = makeOn();
      registerRules(on as any);
      await handlers['prompt.submit'](dollar, { text: 'stop running pytest' }, async (e: any) => e);
      const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'pytest tests' } },
        async () => ({ decision: 'allow' }));
      assert.equal(result.decision, 'deny', 'a session rule must never be held');
      const receipts = receiptsOf(files, 'sess-rule');
      assert.equal(receipts.length, 1);
      assert.equal(receipts[0].source, 'session-rule');
      assert.equal(receipts[0].decision, 'acted');
      assert.equal(receipts[0].artifact, null);
      assert.equal(receipts[0].signature, null);
    }

    // rejection memory is never held out, whatever the draw
    {
      setDraw(() => 0.01);
      const files = new Map<string, string>();
      const dollar = session('sess-rej', files);
      const { on, handlers } = makeOn();
      registerRules(on as any);
      await handlers['tool.call'](dollar, { tool: 'Bash', command: 'rm -rf build' },
        async () => ({ deny: "The user doesn't want to proceed with this tool use. The tool use was rejected." }));
      const result = await handlers['tool.check'](dollar, { tool: 'Bash', input: { command: 'rm -rf build' } },
        async () => ({ decision: 'allow' }));
      assert.equal(result.decision, 'deny', 'rejection memory must never be held');
      const receipts = receiptsOf(files, 'sess-rej');
      assert.equal(receipts.length, 1);
      assert.equal(receipts[0].source, 'rejection-memory');
      assert.equal(receipts[0].decision, 'acted');
    }
    setDraw(() => 0.99);
  }

  console.log('hooks/harness.rules.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
