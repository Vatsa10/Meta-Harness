/**
 * Decision receipts (spec: docs/superpowers/specs/2026-09-29-decision-receipts-design.md).
 * A receipt is written when the plugin acts - or, for a held-out case, would have acted - and
 * the outcome is attributed offline by meta_harness/receipts.py. Helpers take `io`, never `$`.
 */
export type ReceiptFields = {
  event: string; source: string; artifact: string | null;
  signature: string | null; tool: string; decision: 'acted' | 'held';
};

const DEFAULT_RATE = 0.1;
const MAX_RATE = 0.5;
const calls = new Map<string, number>();
let drawFn: () => number = Math.random;

export function bumpCall(sessionId: string): number {
  const next = (calls.get(sessionId) ?? 0) + 1;
  calls.set(sessionId, next);
  return next;
}

export function currentCall(sessionId: string): number {
  return calls.get(sessionId) ?? 0;
}

export function setDraw(fn: () => number): void { drawFn = fn; }
export function draw(): number { return drawFn(); }
export function shouldHold(rate: number): boolean { return rate > 0 && draw() < rate; }

export async function holdoutRate(io: any, home: string): Promise<number> {
  try {
    const path = `${home}/receipts.json`;
    if (!(await io.fs.exists(path))) return DEFAULT_RATE;
    const value = Number(JSON.parse(await io.fs.read(path))?.holdout_rate);
    if (!Number.isFinite(value)) return DEFAULT_RATE;
    return Math.min(MAX_RATE, Math.max(0, value));
  } catch {
    return DEFAULT_RATE;
  }
}

export async function writeReceipt(io: any, home: string, sessionId: string,
                                   record: ReceiptFields): Promise<void> {
  try {
    const path = `${home}/receipts-${sessionId}.jsonl`;
    const line = JSON.stringify({ ts: new Date().toISOString(), session: sessionId,
                                  call: currentCall(sessionId), ...record }) + '\n';
    const prior = (await io.fs.exists(path)) ? await io.fs.read(path) : '';
    await io.fs.write(path, prior + line);
  } catch (error) {
    try { io.log?.(`meta-harness: receipt not written: ${String(error)}`); } catch { /* fail open */ }
  }
}
