/**
 * `--import` preload that registers ts-js-resolve.mjs before the entry module's own imports are
 * linked, so a relative `.js` specifier pointing at a `.ts` sibling (as hooks/harness.ts writes
 * it, matching the plugin runtime's own resolution) resolves under plain node too.
 *
 * Test-path only: nothing here ships as part of the plugin; it exists so
 * `node --experimental-strip-types --import hooks/loaders/preload.mjs <test file>` can run the
 * real harness.ts source unmodified.
 */

import { register } from 'node:module';

register('./ts-js-resolve.mjs', import.meta.url);
