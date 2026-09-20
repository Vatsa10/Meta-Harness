/**
 * Behavioural test for prompt.section injection in harness.ts, run directly with
 * `node --experimental-strip-types --no-warnings hooks/harness.injection.test.mts`
 * (see tests/test_hook_injection.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove registerInjection() ever returns
 * injected text, skips non-injection artifact types, or records what it injected: a reviewer
 * could make the handler always return {text: null}, or inject a 'rule'/'skill' artifact's
 * payload, and every substring-based test would still pass. This test drives the real
 * registerInjection() handler with a fake `$` and a fake installed-artifact registry and
 * asserts on the actual returned text and the observe() side effect.
 */

import assert from 'node:assert/strict';
import { registerInjection, safely } from './harness.ts';

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
  // --- no installed injections: returns null via next(event), never {text: null} magic ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerInjection(on as any);

    const next = async (event: any) => ({ text: event?.fallback ?? null });
    const result = await handlers['prompt.section'](dollar, { fallback: 'x' }, next);
    assert.equal(result.text, 'x', 'with nothing installed, falls through to next() untouched');
  }

  // --- installed injection whose trigger matches the event: text is returned, use recorded ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify([{ id: 'timeout-tip', type: 'injection', signature: 'sig', accepted: '2026-01-01' }]),
    );
    files.set(
      'C:/fake-harness-home/artifacts/timeout-tip/artifact.json',
      JSON.stringify({
        id: 'timeout-tip',
        type: 'injection',
        origin: { triggers: ['timeout'] },
        payload: 'Long-running commands should pass an explicit timeout.',
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const next = async () => ({ text: null });

    const matched = await handlers['prompt.section'](dollar, { prompt: 'why did my command timeout' }, next);
    assert.equal(matched.text, 'Long-running commands should pass an explicit timeout.');

    const sessionPath = 'C:/fake-harness-home/observed-sess-abc123.jsonl';
    assert.ok(files.has(sessionPath), 'a matched injection must be recorded to the session observation file');
    const recorded = files.get(sessionPath)!;
    assert.ok(recorded.includes('"injected"'), 'the recorded line must mark the event as an injection');
    assert.ok(recorded.includes('timeout-tip'), 'the recorded line must name the artifact that was injected');

    // --- a turn whose text does not mention any trigger gets {text: null}, not the installed text ---
    const unmatched = await handlers['prompt.section'](dollar, { prompt: 'refactor the login form' }, next);
    assert.equal(unmatched.text, null, 'no trigger match must yield text: null, costing nothing this turn');
  }

  // --- an installed 'rule' artifact (not 'injection') must never be injected ---
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
        origin: { triggers: ['rm'] },
        payload: 'Do not run rm -rf in this repo.',
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const next = async () => ({ text: null });
    const result = await handlers['prompt.section'](dollar, { prompt: 'please rm this file' }, next);
    assert.equal(result.text, null, 'a rule artifact must never be surfaced as an injection');
  }

  // --- a corrupt registry: fails open, prompt.section still resolves via next(event) ---
  {
    const files = new Map<string, string>();
    files.set('C:/fake-harness-home/installed.json', '{not json');
    const dollar = makeFakeDollar(files);

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const next = async (event: any) => ({ text: event?.fallback ?? null });
    const result = await handlers['prompt.section'](dollar, { fallback: 'still works' }, next);
    assert.equal(result.text, 'still works', 'a corrupt registry must fail open, not break the turn');
  }

  // --- a single over-long payload is truncated to the per-artifact cap, visibly ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const longPayload = 'x'.repeat(2000);
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify([{ id: 'verbose-tip', type: 'injection', signature: 'sig', accepted: '2026-01-01' }]),
    );
    files.set(
      'C:/fake-harness-home/artifacts/verbose-tip/artifact.json',
      JSON.stringify({
        id: 'verbose-tip',
        type: 'injection',
        origin: { triggers: ['verbose'] },
        payload: longPayload,
        replay: {},
      }),
    );

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const next = async () => ({ text: null });
    const result = await handlers['prompt.section'](dollar, { prompt: 'go verbose please' }, next);
    // An injection is paid in standing tokens every turn it fires, unlike a rule, so a single
    // artifact's text must never reach the prompt uncapped: a 2000-char payload must come back
    // far shorter than it went in, and the cut must be visible, not silent.
    assert.ok(result.text.length < 500, `expected the payload to be capped well under its 2000-char source, got ${result.text.length}`);
    assert.ok(result.text.includes('truncated'), 'a truncated injection must say so, not cut silently');
  }

  // --- many installed injections together stay under a total bound, not just each individually ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    files.set(
      'C:/fake-harness-home/installed.json',
      JSON.stringify(ids.map((id) => ({ id, type: 'injection', signature: 'sig', accepted: '2026-01-01' }))),
    );
    for (const id of ids) {
      files.set(
        `C:/fake-harness-home/artifacts/${id}/artifact.json`,
        JSON.stringify({
          id,
          type: 'injection',
          origin: { triggers: ['broad'] },
          payload: 'y'.repeat(300),
          replay: {},
        }),
      );
    }

    const { on, handlers } = makeOn();
    registerInjection(on as any);
    const next = async () => ({ text: null });
    const result = await handlers['prompt.section'](dollar, { prompt: 'this is a broad match' }, next);
    // 8 artifacts x 300 chars each would be 2400+ chars uncapped; the joined total must stay
    // bounded regardless of how many injections are installed, not just each one individually.
    assert.ok(result.text.length < 1500, `expected the joined total to stay bounded, got ${result.text.length}`);
  }

  console.log('hooks/harness.injection.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
