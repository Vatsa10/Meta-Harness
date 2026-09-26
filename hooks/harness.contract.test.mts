/**
 * Two cross-cutting contract tests, driven with real-shaped events per the recovered Claude
 * Code declarations (claude-code-2.1.283.d.ts), run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.contract.test.mts`
 * (see tests/test_hook_contract.py, which shells out to it).
 *
 * 1. No handler this plugin registers may ever touch a `prompt.section` it does not own. Before
 *    Lane C's fix round 1, `registerBootstrap` and `registerInjection` both listened on
 *    `prompt.section` with no section-name filter: `prompt.section` fires once per NAMED
 *    SECTION of the system prompt (env_info_simple, memory, ...) and a `{ text }` return
 *    REPLACES that section. With no filter, the first session's `bootstrap.json` write path
 *    overwrote ONE section and the `{ text: null }` fallback for every session after that
 *    deleted every OTHER section, forever, for anyone who installed the plugin — see CRITICAL 1
 *    and CRITICAL 3 in the fix-round-1 report. This test registers the real, full plugin via
 *    `register()` and fires `prompt.section` for an arbitrary section name with arbitrary text,
 *    asserting the text comes back byte-for-byte unchanged. It must (and, run against the
 *    pre-fix commit, does) fail otherwise.
 *
 * 2. `tool.call` handlers from different registerX functions genuinely COMPOSE in production:
 *    the real declarations say `next(e)` is "the rest of the chain" and multiple listeners on
 *    one event all run as long as each calls `next` once. This test builds a fake `on` that
 *    chains every registered `tool.call` handler exactly the way the real engine does (not the
 *    single-slot fake other test files here originally used, which is what let an earlier draft
 *    quietly break composition without any test catching it) and asserts every handler runs and
 *    the tool itself runs exactly once.
 *
 * 3. Case 1 repeated across every file state the handlers branch on: bootstrap.json present or
 *    absent, waste.json present or absent, drift.json absent/enabled/disabled with a long stretch
 *    of tool calls pending, and an installed injection whose trigger appears IN
 *    the section text, both before and after a prompt.submit turn has run. A handler that
 *    deleted a section only once bootstrap.json exists (the original CRITICAL 1 shape) passes
 *    case 1, which only ever runs with an empty harness home; it fails here.
 *
 * 4. A fully populated PromptSubmitInput (text, attachments, origin, turnId, wait, and a
 *    pre-existing context entry) sent through EVERY prompt.submit handler reaches core with
 *    every field unchanged and the prior context entry ahead of anything added.
 *
 * 5. `next` is called at most once: a core that rejects is never re-entered by a fail-open
 *    wrapper, through the full prompt.submit chain or through `safely` directly.
 */

import assert from 'node:assert/strict';
import { register, safely } from './harness.ts';

const HOME = 'C:/fake-harness-home';
const WASTE = JSON.stringify({ sessions: 12, corrections: { calls_burned: 340 } });

function installInjection(files: Map<string, string>, triggers: string[]) {
  files.set(`${HOME}/installed.json`, JSON.stringify([
    { id: 'timeout-tip', type: 'injection', signature: 'sig', accepted: '2026-01-01' },
  ]));
  files.set(`${HOME}/artifacts/timeout-tip/artifact.json`, JSON.stringify({
    id: 'timeout-tip',
    type: 'injection',
    origin: { triggers },
    payload: 'Long-running commands should pass an explicit timeout.',
    replay: {},
  }));
}

const PROMPT_FIELDS = ['text', 'attachments', 'origin', 'turnId', 'wait'] as const;

function makeFakeDollar(files: Map<string, string>) {
  return {
    env: {
      get: (key: string) => (key === 'META_HARNESS_HOME' ? 'C:/fake-harness-home' : undefined),
    },
    session: { id: async () => 'sess-contract' },
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

/** A chaining fake `on`: every event name maps to a LIST of handlers, composed innermost
 * (last-registered) first — exactly how the real engine's `next` chains multiple listeners. */
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

async function main() {
  // --- 1. register() must never let ANY prompt.section come back changed ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, fire } = makeChainingOn();
    register(on as any, {} as any);

    const originalText = 'ORIGINAL';
    // A PromptSectionInput-shaped event, per the recovered declarations: { name, text }.
    const event = { name: 'env_info_simple', text: originalText };
    const terminal = async (e: any) => ({ text: e.text });
    const result = await fire(dollar, 'prompt.section', event, terminal);

    assert.equal(
      result.text, originalText,
      `a section this plugin does not own must come back byte-for-byte unchanged, got: ${JSON.stringify(result)}`,
    );
  }

  // --- 1b. same, for a second, differently-named section and a second, fresh session (in case
  //         any handler's behaviour depends on which section fired first or on session state) ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, fire } = makeChainingOn();
    register(on as any, {} as any);

    const event = { name: 'memory', text: 'SOME MEMORY SECTION TEXT' };
    const terminal = async (e: any) => ({ text: e.text });
    const result = await fire(dollar, 'prompt.section', event, terminal);
    assert.equal(result.text, 'SOME MEMORY SECTION TEXT');
  }

  // --- 2. multiple tool.call handlers from different registerX functions compose: each runs,
  //         and the tool underneath runs exactly once ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, fire, handlers } = makeChainingOn();
    register(on as any, {} as any);

    // register() puts at least two independent tool.call handlers on the chain: registerObserver's
    // and registerRules'. If composition were broken (e.g. a bare event-map overwriting all but
    // the last registration), this would be 1.
    assert.ok((handlers['tool.call']?.length ?? 0) >= 2, 'expected at least two composed tool.call handlers');

    let toolRuns = 0;
    const terminal = async (e: any) => {
      toolRuns += 1;
      return { result: 'ok', text: 'ok' };
    };
    const event = { tool: 'Bash', input: { command: 'echo hi' } };
    const outcome = await fire(dollar, 'tool.call', event, terminal);

    assert.equal(toolRuns, 1, 'the tool itself must run exactly once, regardless of how many handlers observe the call');
    assert.deepEqual(outcome, { result: 'ok', text: 'ok' }, 'the outcome each handler forwards must reach the caller unchanged');
  }

  // --- 3. prompt.section unchanged across bootstrap.json x waste.json x a matching injection ---
  {
    const sections = [
      { name: 'env_info_simple', text: 'env: the default timeout is 120s' },
      { name: 'memory', text: 'remember the timeout flag' },
      { name: 'tool_use', text: 'ORIGINAL' },
    ];
    // drift.json: absent, explicitly enabled with a threshold every stretch here passes, and
    // disabled. No drift state may reach a prompt.section, even with a long stretch pending.
    const driftStates: Array<[string, string | null]> = [
      ['absent', null],
      ['enabled', JSON.stringify({ enabled: true, min_calls: 1, judge: 'overlap' })],
      ['disabled', JSON.stringify({ enabled: false, min_calls: 1 })],
    ];
    for (const hasBootstrap of [false, true]) {
      for (const hasWaste of [false, true]) {
        for (const afterTurn of [false, true]) {
          for (const [drift, driftJson] of driftStates) {
            const files = new Map<string, string>();
            installInjection(files, ['timeout', 'env', 'original']);
            if (hasBootstrap) files.set(`${HOME}/bootstrap.json`, '{"shown":"2026-01-01"}');
            if (hasWaste) files.set(`${HOME}/waste.json`, WASTE);
            if (driftJson !== null) files.set(`${HOME}/drift.json`, driftJson);
            const dollar = makeFakeDollar(files);
            const { on, fire } = makeChainingOn();
            register(on as any, {} as any);
            const toolCalls = async () => {
              for (let i = 0; i < 5; i += 1) {
                await fire(dollar, 'tool.call', { tool: 'Bash', command: `echo ${i}`, tool_use_id: `tu-${i}` },
                  async () => ({ ref: i, result: { stdout: '' }, text: '' }));
              }
            };
            await toolCalls();
            if (afterTurn) {
              await fire(dollar, 'prompt.submit', { text: 'why the timeout', wait: false, origin: { kind: 'composer' } },
                async (e: any) => ({ text: e.text, context: e.context }));
              await toolCalls();
            }
            for (const section of sections) {
              const result = await fire(dollar, 'prompt.section', { ...section }, async (e: any) => ({ text: e.text }));
              assert.equal(
                result?.text, section.text,
                `section ${section.name} changed with bootstrap.json=${hasBootstrap} waste.json=${hasWaste} `
                  + `afterTurn=${afterTurn} drift.json=${drift}: got ${JSON.stringify(result)}`,
              );
            }
          }
        }
      }
    }
  }

  // --- 4. a fully populated PromptSubmitInput survives every prompt.submit handler ---
  {
    const files = new Map<string, string>();
    files.set(`${HOME}/waste.json`, WASTE);
    installInjection(files, ['timeout']);
    const dollar = makeFakeDollar(files);
    const { on, fire, handlers } = makeChainingOn();
    register(on as any, {} as any);
    assert.ok((handlers['prompt.submit']?.length ?? 0) >= 3, 'expected nlrules, injection and bootstrap on prompt.submit');

    const full = {
      text: 'the build hit a timeout again',
      attachments: [{ type: 'image', mediaType: 'image/png', filename: 'shot.png' }],
      origin: { kind: 'composer' },
      turnId: 'turn-42',
      wait: true,
      context: ['an earlier context entry'],
    };
    const snapshot = JSON.parse(JSON.stringify(full));
    let atCore: any = null;
    let coreCalls = 0;
    await fire(dollar, 'prompt.submit', full, async (e: any) => {
      coreCalls += 1;
      atCore = e;
      return { text: e.text, context: e.context };
    });
    assert.equal(coreCalls, 1, 'core must see the prompt exactly once');
    for (const field of PROMPT_FIELDS) {
      assert.deepEqual(atCore[field], snapshot[field], `${field} must reach core unchanged, got ${JSON.stringify(atCore[field])}`);
    }
    assert.equal(atCore.context[0], 'an earlier context entry', `prior context must survive, first: ${JSON.stringify(atCore.context)}`);
    assert.equal(atCore.context.length, 3, `expected prior entry + injection + bootstrap line, got ${JSON.stringify(atCore.context)}`);
    assert.ok(atCore.context.some((c: string) => c.includes('explicit timeout')), 'the injection must be attached');
    assert.ok(atCore.context.some((c: string) => c.includes('340')), 'the bootstrap line must be attached');
    assert.deepEqual(full, snapshot, "the caller's event object must not be mutated");

    // Each handler on its own too, in a state where it adds something: none may drop a field or
    // replace the prior entry.
    for (const handler of handlers['prompt.submit']) {
      const fresh = new Map<string, string>();
      fresh.set(`${HOME}/waste.json`, WASTE);
      installInjection(fresh, ['timeout']);
      let seen: any = null;
      await handler(makeFakeDollar(fresh), JSON.parse(JSON.stringify(snapshot)), async (e: any) => {
        seen = e;
        return { text: e.text, context: e.context };
      });
      for (const field of PROMPT_FIELDS) {
        assert.deepEqual(seen[field], snapshot[field], `a single prompt.submit handler dropped or changed ${field}`);
      }
      assert.equal(seen.context?.[0], 'an earlier context entry', `a single handler replaced prior context: ${JSON.stringify(seen.context)}`);
    }
  }

  // --- 5. a rejecting core is called exactly once, never retried by a fail-open wrapper ---
  {
    for (const hasWaste of [false, true]) {
      for (const hasInjection of [false, true]) {
        const files = new Map<string, string>();
        if (hasWaste) files.set(`${HOME}/waste.json`, WASTE);
        if (hasInjection) installInjection(files, ['timeout']);
        const dollar = makeFakeDollar(files);
        const { on, fire } = makeChainingOn();
        register(on as any, {} as any);
        let coreCalls = 0;
        await assert.rejects(() => fire(
          dollar, 'prompt.submit', { text: 'a timeout again', wait: false, origin: { kind: 'composer' } },
          async () => {
            coreCalls += 1;
            throw new Error('core refused');
          },
        ));
        assert.equal(coreCalls, 1, `core called ${coreCalls} times (waste.json=${hasWaste} injection=${hasInjection})`);
      }
    }

    const quiet = { ui: { log() {} } };
    let calls = 0;
    const passThrough = safely('t', async (_d: any, e: any, next: any) => next(e));
    await assert.rejects(() => passThrough(quiet, {}, async () => {
      calls += 1;
      throw new Error('x');
    }));
    assert.equal(calls, 1, 'safely must not re-enter a next that already rejected');

    calls = 0;
    const throwsAfter = safely('t', async (_d: any, e: any, next: any) => {
      await next(e);
      throw new Error('after');
    });
    const kept = await throwsAfter(quiet, {}, async () => {
      calls += 1;
      return 'core-result';
    });
    assert.equal(calls, 1, 'safely must not call next again after a post-next throw');
    assert.equal(kept, 'core-result', 'the result core already produced must be returned');

    calls = 0;
    const throwsBefore = safely('t', async () => {
      throw new Error('before');
    });
    await throwsBefore(quiet, {}, async () => {
      calls += 1;
      return 'ok';
    });
    assert.equal(calls, 1, 'a throw before next must still fall through to next exactly once');
  }

  console.log('hooks/harness.contract.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
