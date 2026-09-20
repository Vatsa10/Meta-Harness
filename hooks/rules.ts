/**
 * Installed `rule` artifacts, applied at tool.check.
 *
 * A rule is the strongest layer available: it costs no standing tokens and cannot be talked
 * around. `read-before-edit` is built in because it is the discovered doctrine's top clause,
 * and a clause a mechanism can enforce should not be a paragraph.
 */

export type Rule = {
  artifactId: string;
  kind: string;
  tools: string[];
  reason: string;
};

export type SessionState = {
  readPaths: Set<string>;
};

export const BUILT_IN_RULES: Rule[] = [
  {
    artifactId: 'read-before-edit',
    kind: 'read-before-edit',
    tools: ['Edit', 'Write'],
    reason: 'Read this file in the session before editing it: an edit written from an assumption about its contents lands in the wrong place.',
  },
];

export type InstalledArtifact = {
  id: string;
  type: string;
  origin?: Record<string, unknown>;
  payload?: unknown;
};

/**
 * Reads `installed.json`, keeps only the rows of the given `type`, and loads each one's
 * `artifact.json`. The single reader every layer (`rule`, `injection`, and whichever of
 * `skill`/`doctrine` follows) shares, so the fail-open semantics live in exactly one place: a
 * missing or corrupt registry, or a missing/malformed artifact.json, yields fewer rows rather
 * than throwing, and every caller gets that behaviour identically instead of re-deriving it.
 */
export async function loadInstalled(dollar: any, home: string, type: string): Promise<InstalledArtifact[]> {
  const registry = `${home}/installed.json`;
  if (!(await dollar.fs.exists(registry))) return [];
  let entries: Array<{ id: string; type: string }> = [];
  try {
    entries = JSON.parse(await dollar.fs.read(registry));
  } catch {
    return [];
  }
  const out: InstalledArtifact[] = [];
  for (const entry of entries) {
    if (entry.type !== type) continue;
    const path = `${home}/artifacts/${entry.id}/artifact.json`;
    if (!(await dollar.fs.exists(path))) continue;
    try {
      out.push(JSON.parse(await dollar.fs.read(path)));
    } catch {
      continue;
    }
  }
  return out;
}

/** Installed rule artifacts, plus the built-ins. Missing or malformed files are ignored. */
export async function loadRules(dollar: any, home: string): Promise<Rule[]> {
  const rules = [...BUILT_IN_RULES];
  const artifacts = await loadInstalled(dollar, home, 'rule');
  for (const artifact of artifacts) {
    rules.push({
      artifactId: String(artifact.id),
      kind: (artifact.origin as any)?.kind ?? 'custom',
      tools: (artifact.origin as any)?.tools ?? [],
      reason: String(artifact.payload ?? '').slice(0, 400),
    });
  }
  return rules;
}

/** Evaluates one rule against one tool.check event. `deny: true` means the call must not proceed. */
export function evaluateRule(
  rule: Rule,
  event: { tool?: string; input?: Record<string, unknown> },
  state: SessionState,
): { deny: boolean; reason?: string } {
  const tool = String(event.tool ?? '');
  if (!rule.tools.includes(tool)) return { deny: false };

  if (rule.kind === 'read-before-edit') {
    const path = String((event.input as any)?.file_path ?? '');
    if (!path || state.readPaths.has(path)) return { deny: false };
    return { deny: true, reason: `${rule.reason} [${rule.artifactId}]` };
  }

  return { deny: false };
}
