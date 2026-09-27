/**
 * Runs hooks/harness.ts's cause() over a list of texts read from a JSON file and writes the
 * resulting slugs to another JSON file, so a Python test can compare them against what
 * meta_harness.replay._cause() computes for the identical texts.
 *
 * This exists because tests/test_hook_assets.py used to compare the *source text* of
 * CAUSE_PATTERNS against ERROR_PATTERNS, which can only prove the two lists look alike -- it
 * cannot prove they classify the same way once JavaScript's string-literal escaping has had its
 * say. A regex that agrees on paper can disagree at runtime. This script makes the two
 * classifiers actually run, on the actual same inputs, so the Python side can assert on the
 * actual same outputs.
 *
 * Usage: node --experimental-strip-types --no-warnings --import ./hooks/loaders/preload.mjs
 *        hooks/harness.cause_parity.test.mts <input.json> <output.json>
 * <input.json> is a JSON array of strings. <output.json> is written as a JSON array of the
 * slug cause() returned for each one, in the same order.
 */

import fs from 'node:fs';
import { cause } from './harness.ts';

function main(): void {
  const [, , inputPath, outputPath] = process.argv;
  if (!inputPath || !outputPath) {
    throw new Error('usage: harness.cause_parity.test.mts <input.json> <output.json>');
  }
  const texts: string[] = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));
  const slugs = texts.map((text) => cause(text));
  fs.writeFileSync(outputPath, JSON.stringify(slugs), 'utf-8');
}

main();
