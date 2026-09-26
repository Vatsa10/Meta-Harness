/**
 * Live drift surfacing: counts tool calls in the current stretch (the calls since the user last
 * spoke) and, once a stretch passes `min_calls`, says so ONCE as model-only context on the next
 * `prompt.submit`.
 *
 * Configured by `<harnessHome>/drift.json`, shaped
 * `{"enabled": boolean, "min_calls": number, "judge": "knn"|"overlap"|"model"}`. A missing,
 * empty, unreadable, unparseable or ill-shaped file means DISABLED. That is how the ship gate in
 * `tools/tune_drift.py` is honoured: it refused every judge, so nothing writes an enabling file
 * and the feature ships built, tested, and off.
 *
 * The live hook only has the call count to go on; the `judge` field names which offline judge
 * justified turning it on and is validated, not executed here.
 */

export type DriftConfig = {
  enabled: true;
  min_calls: number;
  judge?: 'knn' | 'overlap' | 'model';
};

const JUDGES = new Set(['knn', 'overlap', 'model']);

/**
 * The drift config, or null (disabled) for anything short of an explicit, well-formed
 * `{"enabled": true, "min_calls": <positive number>}`. Never throws.
 */
export async function loadDriftConfig(dollar: any, home: string): Promise<DriftConfig | null> {
  try {
    const path = `${home}/drift.json`;
    if (!(await dollar.fs.exists(path))) return null;
    const raw = await dollar.fs.read(path);
    if (typeof raw !== 'string' || raw.trim() === '') return null;
    const parsed: any = JSON.parse(raw);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    if (parsed.enabled !== true) return null;
    const minCalls = parsed.min_calls;
    if (typeof minCalls !== 'number' || !Number.isFinite(minCalls) || minCalls < 1) return null;
    if (parsed.judge !== undefined && !JUDGES.has(parsed.judge)) return null;
    return { enabled: true, min_calls: minCalls, judge: parsed.judge };
  } catch {
    return null;
  }
}

/** Whether a stretch of `count` calls is long enough to mention under `config`. */
export function shouldWarn(count: number, config: DriftConfig | null): boolean {
  return config !== null && config.enabled === true && count >= config.min_calls;
}

/** The one line the model reads when a stretch has run long. */
export function driftNote(count: number): string {
  return `meta-harness drift: ${count} tool calls have run since the user last spoke. `
    + 'Before continuing, check the work still matches what was asked.';
}

/**
 * Origins that mean the user themself spoke, which ends the stretch. A notification, peer
 * message, schedule or plugin submission does not: the stretch it lands in keeps counting, and
 * the once-per-stretch latch keeps it from carrying a second note. An absent or unrecognised
 * origin is treated as the user speaking, the choice that produces fewer notes, not more.
 */
const NON_USER_ORIGINS = new Set([
  'task-notification', 'scheduled-trigger', 'peer', 'peer-send-message', 'projects-relay',
  'channel', 'coordinator', 'observer', 'observer-activity', 'auto-continuation', 'slack-ping',
  'plugin',
]);

export function userSpoke(event: any): boolean {
  const kind = event?.origin?.kind;
  return !(typeof kind === 'string' && NON_USER_ORIGINS.has(kind));
}
