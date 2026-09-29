import assert from 'node:assert/strict';
import { bumpCall, currentCall, holdoutRate, setDraw, shouldHold, writeReceipt } from './receipts.ts';

function fakeIo(files: Map<string, string>, failWrite = false) {
  return {
    fs: {
      exists: async (p: string) => files.has(p),
      read: async (p: string) => { if (!files.has(p)) throw new Error('missing'); return files.get(p)!; },
      write: async (p: string, t: string) => { if (failWrite) throw new Error('disk full'); files.set(p, t); },
    },
    log: () => {},
  };
}

// call sequence is per session
assert.equal(currentCall('s1'), 0);
assert.equal(bumpCall('s1'), 1);
assert.equal(bumpCall('s1'), 2);
assert.equal(currentCall('s2'), 0, 'sessions must not share a counter');

// holdout rate: default, clamps, corrupt
const home = '/h';
assert.equal(await holdoutRate(fakeIo(new Map()), home), 0.1, 'missing config means 0.1');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', '{"holdout_rate": 0.9}']])), home), 0.5, 'clamped to 0.5');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', '{"holdout_rate": -1}']])), home), 0, 'clamped to 0');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', 'not json']])), home), 0.1, 'corrupt means default');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', '{"holdout_rate": 0}']])), home), 0, 'zero disables');

// the draw is replaceable
setDraw(() => 0.05);
assert.equal(shouldHold(0.1), true);
setDraw(() => 0.5);
assert.equal(shouldHold(0.1), false);
assert.equal(shouldHold(0), false, 'a zero rate never holds');
setDraw(() => -1);
assert.equal(shouldHold(0), false, 'a zero rate never holds, whatever the draw');

// receipts append, one JSON object per line, no text or input
const files = new Map<string, string>();
await writeReceipt(fakeIo(files), home, 's1', { event: 'tool.check', source: 'learned-rule', artifact: 'a1', signature: 'tool_error:Bash:x', tool: 'Bash', decision: 'acted' });
await writeReceipt(fakeIo(files), home, 's1', { event: 'tool.check', source: 'learned-rule', artifact: 'a1', signature: 'tool_error:Bash:x', tool: 'Bash', decision: 'held' });
const lines = files.get('/h/receipts-s1.jsonl')!.trim().split('\n').map((l) => JSON.parse(l));
assert.equal(lines.length, 2, 'receipts must append');
assert.equal(lines[0].call, 2, 'a receipt carries the current call number');
assert.equal(lines[1].decision, 'held');
assert.ok(!('input' in lines[0]) && !('text' in lines[0]), 'receipts carry no input or text');

// fail open
await writeReceipt(fakeIo(new Map(), true), home, 's1', { event: 'x', source: 'drift-note', artifact: null, signature: null, tool: 'Bash', decision: 'acted' });
console.log('all assertions passed');
