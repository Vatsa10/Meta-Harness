/**
 * Node customization hook: resolves a relative `.js` specifier to a same-named `.ts` sibling
 * when one exists, before falling through to the default resolver.
 *
 * This mirrors the actual Claude Code plugin runtime, which resolves `import ... from
 * './rules.js'` in hooks/harness.ts to the sibling hooks/rules.ts (see
 * fast-jev-compaction/hooks/fast-jev.ts importing '../src/compact.js' against only
 * src/compact.ts for a shipped precedent). Bare `node --experimental-strip-types` does not do
 * this remapping on its own, so tests that exercise the real (correct) `.js` specifier need
 * this hook registered via `--import hooks/loaders/preload.mjs`. Test-path only: no build step,
 * no dependency, nothing this loader does reaches the shipped plugin.
 */

import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export async function resolve(specifier, context, nextResolve) {
  const isRelative = specifier.startsWith('./') || specifier.startsWith('../');
  if (isRelative && specifier.endsWith('.js')) {
    const tsSpecifier = `${specifier.slice(0, -'.js'.length)}.ts`;
    const candidateUrl = new URL(tsSpecifier, context.parentURL);
    if (existsSync(fileURLToPath(candidateUrl))) {
      return nextResolve(tsSpecifier, context);
    }
  }
  return nextResolve(specifier, context);
}
