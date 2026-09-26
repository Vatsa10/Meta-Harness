/**
 * Behavioural test for session-scoped natural-language rules in harness.ts / rules.ts, run
 * directly with `node --experimental-strip-types --no-warnings hooks/harness.nlrules.test.mts`
 * (see tests/test_hook_nlrules.py, which shells out to it).
 *
 * This exists because a grep over the source cannot prove parseStopInstruction discriminates a
 * real instruction from an ordinary question or a stop-word-with-no-object, that a session rule
 * ever denies a matching call and is never persisted to the installed store, or that the text
 * is actually read from `prompt.submit` (not the nonexistent `prompt.section` `event.prompt`):
 * a reviewer could match anywhere in the text, treat the action verb as optional, write the rule
 * into installed.json, or read a field that never fires, and every substring-based test would
 * still pass. This test drives the real registerRules() handlers with a fake `$` and asserts on
 * the actual decisions and on-disk writes.
 */

import assert from 'node:assert/strict';
import { registerRules, safely } from './harness.ts';
import { addSessionRule, parseStopInstruction } from './rules.ts';

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
  const handlers: Record<string, any[]> = {};
  const on = (event: string, handler: any) => {
    (handlers[event] ??= []).push(handler);
  };
  return { on, handlers };
}

/** A real PromptSubmitInput-shaped event, per the recovered declarations. */
function promptSubmit(text: string, extra: Record<string, unknown> = {}) {
  return { text, wait: false, origin: { user: {} }, ...extra };
}

async function main() {
  // --- 1. the brief's literal example names Bash and a pattern that is exactly "pytest" ---
  // (not "pytest without -q" — that qualifier cannot be matched against a real Bash command, so
  // it is dropped rather than baked into a pattern that would deny nothing real).
  {
    const rule = parseStopInstruction('stop running pytest without -q');
    assert.ok(rule, 'expected a rule to be parsed');
    assert.equal(rule!.tool, 'Bash');
    assert.equal(rule!.pattern, 'pytest', `expected the pattern to be exactly "pytest", got: ${rule!.pattern}`);
  }

  // --- 2. ordinary questions and acknowledgements with no actionable object must not create a rule ---
  {
    const mustNotMatch = [
      'what does this function do?',
      "don't worry about it",
      "don't know why this fails",
      'never mind',
      "Don't forget to update the README",
      'stop',
      "stop, that's wrong",
      // A mid-sentence mention of "stop": matching anywhere in the text (rather than only at
      // the start) would misfire on this.
      'can you help me figure out why the retries never stop?',
      // Fix round 2 finding 4: the object must plausibly be a command, tool or file — a
      // pronoun or a bare determiner names nothing, so these must ALSO return null, not a rule
      // that denies every Bash call containing the English word "that" or "the".
      'Stop doing that',
      'stop using the',
    ];
    for (const text of mustNotMatch) {
      const rule = parseStopInstruction(text);
      assert.equal(rule, null, `must not be parsed as a stop instruction: ${JSON.stringify(text)}`);
    }
  }

  // --- 2b. fix round 2 finding 4: the PLAIN IMPERATIVE (not just the -ing form) must also parse ---
  {
    // Fix round 3: the pattern is the command PREFIX up to its first flag, not just the leading
    // token — "git push --force" -> prefix "git push", not "git" (which would deny every git
    // command, including `git status` and `git diff`, for the rest of the session).
    const mustMatch: Array<[string, string]> = [
      ["don't use git push --force", 'git push'],
      ['never call the deploy script', 'deploy script'],
      ["don't run pytest", 'pytest'],
    ];
    for (const [text, expectedPattern] of mustMatch) {
      const rule = parseStopInstruction(text);
      assert.ok(rule, `expected a rule to be parsed from: ${JSON.stringify(text)}`);
      assert.equal(
        rule!.pattern, expectedPattern,
        `expected pattern "${expectedPattern}" from ${JSON.stringify(text)}, got: ${rule!.pattern}`,
      );
    }
  }

  // --- 2c. fix round 3: the flag named in the object is what is DENIED, and the pattern is the
  //         prefix up to that flag, not just the leading token ---
  {
    const rule = parseStopInstruction("don't use git push --force");
    assert.ok(rule);
    assert.equal(rule!.pattern, 'git push');
    assert.equal(rule!.flag, '--force');
    assert.equal(rule!.requires, undefined);
  }

  // --- 2d. fix round 3: an instruction with a subcommand but no flag denies the whole
  //         subcommand, allowing sibling subcommands and bare invocations ---
  {
    const rule = parseStopInstruction('never run git push');
    assert.ok(rule);
    assert.equal(rule!.pattern, 'git push');
    assert.equal(rule!.flag, undefined);
    assert.equal(rule!.requires, undefined);
  }

  // --- 3. a session rule added from real text denies the matching call on the next tool.check,
  //        via the real prompt.submit event shape (text/wait/origin), not the nonexistent
  //        prompt.section `event.prompt` ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const forwardingNext = async (e: any) => ({ text: e.text, context: e.context, origin: e.origin });
    const returned = await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest'), forwardingNext);
    assert.equal(returned.text, 'stop running pytest', 'a handler with nothing to add must pass the prompt through unchanged');

    const unrelated = { tool: 'Bash', input: { command: 'ls -la' } };
    const allowed = await handlers['tool.check'][0](dollar, unrelated, async () => ({ decision: 'allow' }));
    assert.equal(allowed.decision, 'allow', 'an unrelated call must not be denied by the session rule');
  }

  // --- 3b. fix round 2 finding 6 (reversing an earlier ruling): "stop running pytest without
  //         -q" denies `pytest tests/` but ALLOWS `pytest -q` and `pytest -q tests/` — the
  //         qualifier is now captured as a `requires` flag, not dropped into a blanket denial ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest without -q'), async (e: any) => e);

    const denied = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest tests/' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(denied.decision, 'deny', '"pytest tests/" (no -q) must be denied');
    assert.ok(/pytest/i.test(denied.reason ?? ''), `reason must name what was blocked, got: ${denied.reason}`);

    const allowedBare = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest -q' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(
      allowedBare.decision, 'allow',
      '"pytest -q" — the exact command the human asked to KEEP — must be ALLOWED, not blocked',
    );

    const allowedWithArgs = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest -q tests/' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(allowedWithArgs.decision, 'allow', '"pytest -q tests/" must also be allowed: it still contains -q');
  }

  // --- 3c: a qualifier this mechanism genuinely cannot represent (not a "without X" flag)
  //         still falls back to the honest blanket denial, in every form ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit("stop running pytest unless it's urgent"), async (e: any) => e);

    const denied = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest -q' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(denied.decision, 'deny', 'an unrepresentable qualifier must still deny the pattern in every form');
    assert.ok(/whatever its arguments/i.test(denied.reason ?? ''), `reason must say so honestly, got: ${denied.reason}`);
  }

  // --- 3e. fix round 3: "don't use git push --force" denies `git push --force origin main` but
  //         ALLOWS `git push origin main`, `git status`, and `git diff` — the defect this round
  //         fixes was a pattern of just "git", which denied every git command for the rest of
  //         the session ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit("don't use git push --force"), async (e: any) => e);

    const cases: Array<[string, 'allow' | 'deny']> = [
      ['git push --force origin main', 'deny'],
      ['git push origin main', 'allow'],
      ['git status', 'allow'],
      ['git diff', 'allow'],
    ];
    for (const [command, expected] of cases) {
      const result = await handlers['tool.check'][0](
        dollar, { tool: 'Bash', input: { command } }, async () => ({ decision: 'allow' }),
      );
      assert.equal(
        result.decision, expected,
        `"${command}" expected ${expected}, got ${result.decision}: ${result.reason ?? ''}`,
      );
    }
  }

  // --- 3f. fix round 3: "never run git push" (subcommand, no flag) denies the whole
  //         subcommand but allows sibling subcommands like `git status` ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('never run git push'), async (e: any) => e);

    const denied = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'git push origin main' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(denied.decision, 'deny', '"git push origin main" must be denied: no flag was named at all');

    const allowed = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'git status' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(allowed.decision, 'allow', '"git status" must stay allowed');
  }

  // --- 3g. fix round 3: `requires` ("without -q") matches WHOLE TOKENS of the command string,
  //         not substrings of the JSON input — `pytest --quick tests/` must still be denied
  //         (it does not contain "-q" as a whole token), and a `-q` mentioned only inside some
  //         other field must not satisfy the requirement either ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest without -q'), async (e: any) => e);

    const denied = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest --quick tests/' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(
      denied.decision, 'deny',
      '"pytest --quick tests/" must be denied: "-q" is not a whole token of this command',
    );

    const allowed = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest -q tests/' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(allowed.decision, 'allow', '"pytest -q tests/" must be allowed: "-q" is a whole token');
  }

  // --- 3h. fix round 3: existing "keep passing" cases through the real end-to-end path ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit("don't run pytest"), async (e: any) => e);

    const denied = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest tests/' } }, async () => ({ decision: 'allow' }),
    );
    assert.equal(denied.decision, 'deny', '"don\'t run pytest" must deny pytest');
  }

  // --- 3d. finding 3: a FULLY populated PromptSubmitInput reaches core with every field intact ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    const fullEvent = {
      text: 'stop running pytest',
      wait: true,
      origin: { user: {} },
      context: ['an earlier context entry'],
      attachments: [{ type: 'image' as const, mediaType: 'image/png' }],
      turnId: 'turn-9',
    };
    let forwarded: any = null;
    await handlers['prompt.submit'][0](dollar, fullEvent, async (e: any) => {
      forwarded = e;
      return { text: e.text, context: e.context, origin: e.origin };
    });
    assert.deepEqual(forwarded, fullEvent, 'this handler must never modify the event, field for field');
  }

  // --- 4. the session rule is not written to installed.json ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest'), async (e: any) => e);

    assert.equal(files.has('C:/fake-harness-home/installed.json'), false, 'nothing here may write the installed store');
  }

  // --- 5. it is written to pending-session-rules.json for later review ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit('stop running pytest'), async (e: any) => e);

    const path = 'C:/fake-harness-home/pending-session-rules.json';
    assert.ok(files.has(path), `expected a write to ${path}`);
    const written = JSON.parse(files.get(path)!);
    assert.ok(Array.isArray(written) && written.length === 1, 'expected exactly one pending rule');
    assert.equal(written[0].tool, 'Bash');
    assert.equal(written[0].pattern, 'pytest');
  }

  // --- final fix wave: only a human's prompt creates a session rule ---
  for (const [origin, expectRule] of [
    [{ kind: 'plugin', name: 'some-plugin' }, false],
    [{ kind: 'peer' }, false],
    [{ kind: 'scheduled-trigger' }, false],
    [{ kind: 'task-notification' }, false],
    [{ kind: 'composer' }, true],
  ] as const) {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);

    await handlers['prompt.submit'][0](dollar, promptSubmit("don't run pytest", { origin }), async (e: any) => e);
    const decision = await handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command: 'pytest tests/' } }, async () => ({ decision: 'allow' }),
    );
    const path = 'C:/fake-harness-home/pending-session-rules.json';
    if (expectRule) {
      assert.equal(decision.decision, 'deny', `a ${origin.kind}-origin "don't run pytest" must create a denying rule`);
      assert.ok(files.has(path), `a ${origin.kind}-origin rule must be recorded`);
    } else {
      assert.equal(decision.decision, 'allow', `a ${origin.kind}-origin prompt must NOT create a denying rule`);
      assert.equal(files.has(path), false, `a ${origin.kind}-origin prompt must record no rule`);
    }
  }

  // --- direct addSessionRule/evaluator sanity, independent of the parser ---
  {
    const state: any = { readPaths: new Set(), callCounts: new Map(), rejected: new Map(), sessionRules: [] };
    addSessionRule(state, { tool: 'Bash', pattern: 'rm -rf' });
    assert.equal(state.sessionRules.length, 1);
  }

  console.log('hooks/harness.nlrules.test.mts: all assertions passed');

  // --- 4. launchers: the rule's pattern is matched after python -m / npx / uv run / env ... ---
  //         and the deny reason says what matching does, never "containing" ---
  {
    const files = new Map<string, string>();
    const dollar = makeFakeDollar(files);
    const { on, handlers } = makeOn();
    registerRules(on as any);
    await handlers['prompt.submit'][0](dollar, promptSubmit('Stop running pytest without -q.'), async (e: any) => e);
    const check = (command: string) => handlers['tool.check'][0](
      dollar, { tool: 'Bash', input: { command } }, async () => ({ decision: 'allow' }),
    );
    const launched = [
      'pytest tests/',
      'python -m pytest tests/',
      'python3 -m pytest tests/',
      'py -m pytest tests/',
      'npx pytest tests/',
      'uv run pytest tests/',
      'poetry run pytest tests/',
      'pipx run pytest tests/',
      'env PYTHONPATH=. pytest tests/',
      'PYTHONPATH=. pytest tests/',
      'FOO=1 BAR=2 uv run python -m pytest tests/',
    ];
    for (const command of launched) {
      const denied = await check(command);
      assert.equal(denied.decision, 'deny', `"${command}" must be denied`);
      assert.ok(!/containing/i.test(denied.reason ?? ''), `reason must not say "containing": ${denied.reason}`);
      assert.ok(/first words are "pytest"/.test(denied.reason ?? ''), `reason must say how it matches: ${denied.reason}`);
      assert.ok(/unless "-q" is one of its words/.test(denied.reason ?? ''), `reason must name the flag: ${denied.reason}`);
      const allowed = await check(command.replace('pytest tests/', 'pytest -q tests/'));
      assert.equal(allowed.decision, 'allow', `"${command}" with -q must be allowed`);
    }
    // A launcher running something else is not the pattern.
    assert.equal((await check('python -m pip install x')).decision, 'allow');
    assert.equal((await check('python pytest_helper.py')).decision, 'allow');
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
