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

/** Installed rule artifacts, plus the built-ins. Missing or malformed files are ignored. */
export async function loadRules(dollar: any, home: string): Promise<Rule[]> {
  const rules = [...BUILT_IN_RULES];
  const registry = `${home}/installed.json`;
  if (!(await dollar.fs.exists(registry))) return rules;
  let entries: Array<{ id: string; type: string }> = [];
  try {
    entries = JSON.parse(await dollar.fs.read(registry));
  } catch {
    return rules;
  }
  for (const entry of entries) {
    if (entry.type !== 'rule') continue;
    const path = `${home}/artifacts/${entry.id}/artifact.json`;
    if (!(await dollar.fs.exists(path))) continue;
    try {
      const artifact = JSON.parse(await dollar.fs.read(path));
      rules.push({
        artifactId: artifact.id,
        kind: artifact.origin?.kind ?? 'custom',
        tools: artifact.origin?.tools ?? [],
        reason: String(artifact.payload ?? '').slice(0, 400),
      });
    } catch {
      continue;
    }
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
