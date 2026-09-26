/**
 * Behavioural test for the shape of a REAL `tool.call` event, run with
 * `node --experimental-strip-types --no-warnings hooks/harness.toolargs.test.mts`
 * (tests/test_hook_toolargs.py shells out to it).
 *
 * On `tool.call`, Claude Code 2.1.283 puts the tool's arguments at the TOP LEVEL of the event
 * (`e.command`, `e.file_path`) beside `tool`, `tool_use_id`, `agentId` and `consent`. Only
 * `tool.check` nests them under `input`. Earlier fakes put `input` on tool.call too, which hid a
 * live bug: read-tracking recorded nothing, so the built-in read-before-edit rule denied every
 * Edit. Every case here runs both shapes: the real top-level one, and the `input` variant.
 */

import assert from 'node:assert/strict';
import { register } from './harness.ts';
import { callKey, toolArgs } from './rules.ts';

const HOME = 'C:/fake-harness-home';

function makeFakeDollar(files: Map<string, string>) {
  return {
    env: { get: (key: string) => (key === 'META_HARNESS_HOME' ? HOME : undefined) },
    session: { id: async () => 'sess-toolargs' },
    ui: { log: () => {} },
    fs: {
      exists: async (path: string) => files.has(path),
      read: async (path: string) => {
        if (!files.has(path)) throw new Error(`ENOENT: ${path}`);
        return files.get(path)!;
      },
      write: async (path: string, text: string) => { files.set(path, text); },
    },
  };
}

function setup() {
  const files = new Map<string, string>();
  const dollar = makeFakeDollar(files);
  const handlers: Record<string, any[]> = {};
  register(((event: string, hook: any) => { (handlers[event] ??= []).push(hook); }) as any, {} as any);
  const fire = (event: string, e: any, next: (e: any) => Promise<any>) => handlers[event][0](dollar, e, next);
  return { files, fire };
}

/** A tool.call event as the engine sends it, or the `input` variant. */
function callEvent(shape: 'real' | 'nested', tool: string, args: Record<string, unknown>) {
  return shape === 'real'
    ? { tool, tool_use_id: 'toolu_1', agentId: 'main', ...args }
    : { tool, tool_use_id: 'toolu_1', input: args };
}

const allow = async () => ({ decision: 'allow' });

async function main() {
  for (const shape of ['real', 'nested'] as const) {
    // --- 1. a Read seen at tool.call lets the Edit of that file through read-before-edit ---
    {
      const { fire } = setup();
      await fire('tool.call', callEvent(shape, 'Read', { file_path: 'C:/p/notes.txt' }),
        async () => ({ result: 'hello', text: 'hello' }));
      const verdict = await fire('tool.check',
        { tool: 'Edit', input: { file_path: 'C:/p/notes.txt', old_string: 'a', new_string: 'b' } }, allow);
      assert.equal(verdict.decision, 'allow', `${shape}: an Edit after a Read of the same file must be allowed`);

      const unread = await fire('tool.check',
        { tool: 'Edit', input: { file_path: 'C:/p/other.txt', old_string: 'a', new_string: 'b' } }, allow);
      assert.equal(unread.decision, 'deny', `${shape}: an Edit of an unread file must still be denied`);
      assert.match(unread.reason, /\[read-before-edit\]/);
    }

    // --- 2. a call rejected at tool.call is denied when the same call reaches tool.check ---
    {
      const { fire } = setup();
      await fire('tool.call', callEvent(shape, 'Bash', { command: 'rm -rf build', description: 'clean' }),
        async () => ({ deny: "The user doesn't want to proceed with this tool use. The tool use was rejected." }));
      const verdict = await fire('tool.check',
        { tool: 'Bash', input: { description: 'clean', command: 'rm -rf build' } }, allow);
      assert.equal(verdict.decision, 'deny', `${shape}: the rejected call must be remembered`);
      assert.match(verdict.reason, /rejection-memory/);
    }

    // --- 3. the observer records the tool's arguments for a failing call ---
    {
      const { files, fire } = setup();
      await fire('tool.call', callEvent(shape, 'Bash', { command: 'cat missing.txt' }),
        async () => ({ result: 'cat: missing.txt: No such file or directory', isError: true, text: '' }));
      const log = files.get(`${HOME}/observed-sess-toolargs.jsonl`);
      assert.ok(log, `${shape}: an observation must be written`);
      const record = JSON.parse(log!.trim().split('\n')[0]);
      assert.deepEqual(record.input, { command: 'cat missing.txt' }, `${shape}: the record must carry the input`);
    }
  }

  // --- 4. the key computed at tool.call equals the key computed at tool.check for one call ---
  {
    const real = { tool: 'Bash', tool_use_id: 'toolu_9', agentId: 'a', consent: 'x', command: 'ls', description: 'list' };
    const check = { tool: 'Bash', input: { description: 'list', command: 'ls' } };
    assert.deepEqual(toolArgs(real), { command: 'ls', description: 'list' });
    assert.equal(callKey({ tool: real.tool, input: toolArgs(real) }), callKey(check));
    assert.deepEqual(toolArgs({ tool: 'Bash', input: { command: 'ls' } }), { command: 'ls' });
  }

  console.log('hooks/harness.toolargs.test.mts: all assertions passed');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
