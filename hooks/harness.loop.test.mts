/**
 * End-to-end test of the seam between the Python producer and the TypeScript consumers.
 *
 * Every other hook test hand-authors the artifact it feeds the hooks, and that is precisely what
 * let the loop ship broken: `propose_artifact` emitted an `origin` with no `tools` and no
 * `triggers`, and hand-written fixtures supplied both, so both sides passed while no learned
 * artifact could ever deny or inject. This file reads a harness home that
 * `tests/test_learn_loop_closes.py` built by running the REAL `propose_artifact` and the REAL
 * `HarnessStore.accept`, and asserts the real hook handlers act on it.
 *
 * Usage: node --experimental-strip-types hooks/harness.loop.test.mts <harness-home>
 */

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import nodePath from 'node:path';
import { registerInjection, registerRules } from './harness.ts';

const home = process.argv[2];
assert.ok(home, 'a harness home built by the Python producer must be passed as argv[2]');

function makeDollar() {
  return {
    env: { get: (key: string) => (key === 'META_HARNESS_HOME' ? home : undefined) },
    session: { id: async () => 'sess-loop' },
    ui: { log: () => {} },
    fs: {
      exists: async (path: string) => {
        try {
          await fs.access(path);
          return true;
        } catch {
          return false;
        }
      },
      read: async (path: string) => fs.readFile(path, 'utf-8'),
      write: async (path: string, text: string) => {
        await fs.mkdir(nodePath.dirname(path), { recursive: true });
        await fs.writeFile(path, text, 'utf-8');
      },
    },
  };
}

function makeOn() {
  const handlers: Record<string, any> = {};
  return { on: (event: string, handler: any) => { handlers[event] = handler; }, handlers };
}

async function main() {
  // --- the produced `rule` artifact actually denies at tool.check ---
  {
    const dollar = makeDollar();
    const { on, handlers } = makeOn();
    registerRules(on as any);
    const nextAllow = async () => ({ decision: 'allow' });
    const call = { tool: 'Bash', input: { command: 'python broken.py' } };

    // The first three identical calls must pass: a rule that blocks a first attempt would break
    // work that is going fine, which the fail-open policy forbids.
    for (let i = 0; i < 3; i += 1) {
      const verdict = await handlers['tool.check'](dollar, call, nextAllow);
      assert.equal(verdict.decision, 'allow', `attempt ${i + 1} must be allowed`);
    }
    const denied = await handlers['tool.check'](dollar, call, nextAllow);
    assert.equal(denied.decision, 'deny', 'the produced rule must deny the repeated call');
    assert.ok(
      String(denied.reason).includes('READ THE FILE FIRST'),
      `the deny reason must carry the produced payload, got: ${denied.reason}`,
    );
  }

  // --- the produced `injection` artifact actually injects on a matching turn ---
  {
    const dollar = makeDollar();
    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const next = async () => ({ text: null });

    const matched = await handlers['prompt.section'](dollar, { prompt: 'rerun the bash script' }, next);
    assert.ok(matched.text, 'the produced injection must fire on a turn naming its trigger');
    assert.ok(
      String(matched.text).includes('PASS AN EXPLICIT TIMEOUT'),
      `the injected text must be the produced payload, got: ${matched.text}`,
    );

    const unmatched = await handlers['prompt.section'](dollar, { prompt: 'zzzz qqqq' }, next);
    assert.equal(unmatched.text, null, 'an unrelated turn must still cost nothing');
  }

  console.log('hooks/harness.loop.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
