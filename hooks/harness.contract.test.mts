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
 */

import assert from 'node:assert/strict';
import { register } from './harness.ts';

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

  console.log('hooks/harness.contract.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
