"""Shared fixture of (text, expected_slug) pairs for meta_harness.replay._cause().

This is the single source of truth used by two different checks:

1. `tests/test_replay_signature.py` pins every entry of `ERROR_PATTERNS` (Python side) to a
   real sample string, so a pattern can never be silently deleted or narrowed to uselessness
   without a test failing.
2. `tests/test_hook_assets.py` feeds the same texts through both `meta_harness.replay._cause`
   and `hooks/harness.ts`'s `cause()` (via node) and asserts the two produce identical slugs for
   every one of them. That is a real parity check: it executes both classifiers on the same
   input, rather than comparing their source text (which can certify agreement that does not
   exist -- see Task 1 fix round 1).

Each pair below is either a direct sample of one `ERROR_PATTERNS` entry (comment gives its
slug) or a "divergence probe": a text engineered so that a classifier with a common escaping
bug (e.g. a JS string literal silently dropping backslashes from `\\d`, `\\S`, `\\[`, `\\]`,
`\\(`, `\\)`) would produce a different, wrong slug than the correct one on the right.
"""

from __future__ import annotations

CAUSE_FIXTURES: tuple[tuple[str, str], ...] = (
    # --- one sample per ERROR_PATTERNS entry, in slug order as they appear in the tuple ---
    ("UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'", "unicode-encode"),
    ("UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d", "unicode-decode"),
    ("No such file or directory", "missing-path"),
    ("Permission denied", "permission"),
    ("bash: foo: command not found", "missing-command"),
    ("Command timed out after 30s", "timeout"),
    ("This file has not been read yet", "unread-edit"),
    ("String to replace not found", "edit-mismatch"),
    ("SyntaxError: unexpected token", "syntax"),
    ("Compound command changes working directory (Set-Location)", "compound-shell"),
    ("This command requires approval", "needs-approval"),
    ("The tool use was rejected", "user-rejected"),
    ("<tool_use_error>Blocked: sleep 45 followed by: echo waited", "blocked-policy"),
    ("bash: -c: line 1: unexpected EOF while looking for matching `'", "shell-quoting"),
    ("Tab 3 is not in Claude's tab group for this session", "tab-target"),
    ("File has been modified since read, either by the user or by a linter", "stale-read"),
    ("EISDIR: illegal operation on a directory, read", "is-directory"),
    ('Exit code 1\nTraceback (most recent call last):\n  File "x.py", line 1', "python-traceback"),
    ("Permission to use Bash with command ls has been denied.", "permission-denied-tool"),
    ("File does not exist. Note: your current working directory is D:\\proj.", "missing-path"),
    ("claude-opus-4-8 is temporarily unavailable. Please retry.", "model-unavailable"),
    ("<tool_use_error>InputValidationError: Workflow failed due to the following issue", "workflow-error"),
    ('Failed to execute JavaScript: {"code":-32000,"message":"boom"}', "js-error"),
    ("actions[4] (find) failed: The accessibility tree does not contain a match", "browser-action-failed"),
    ("<tool_use_error>Error: No such tool available: fake_tool</tool_use_error>", "unknown-tool"),
    ("PreToolUse hook did not respond before its timeout", "hook-timeout"),
    ("<tool_use_error>Found 2 matches of the string to replace</tool_use_error>", "edit-mismatch"),
    ("pdftoppm is not installed. Install poppler-utils to enable PDF page rendering.", "missing-command"),
    ('Exit code 1\nnode:internal/modules/package_json_reader:266\n  throw new Error', "module-not-found"),
    ('{"error":{"name":"HttpException","message":"Failed to run sql query"}}', "api-error"),
    ("Exit code 128\nfatal: pathspec 'x.py' did not match any files", "git-error"),
    ("Exit code 1\nOn branch main\nYour branch is up to date with 'origin/main'.", "git-noise"),
    ("Exit code 1\nnpm error code EJSONPARSE\nnpm error JSON.parse Invalid package.json", "npm-error"),
    ("File content (35614 tokens) exceeds maximum allowed tokens (25000).", "output-too-large"),
    ("FAIL origin=http://localhost:3000: ConnectionRefusedError [WinError 1225]", "connection-refused"),
    ('Remove-Item on system path "C:\\Program" is blocked. This path is protected', "blocked-policy"),
    (
        "Exit code 1\nF...........                                       [100%]\r\n"
        "================================== FAILURES ===================================\r\n"
        "__________________________ test_removes_hesitations ___________________________",
        "test-failure",
    ),
    ("This session's tab group no longer exists (tabs were closed). Call tabs_context.", "tab-target"),
    ("DesignSync needs design-system authorization. Run /design-login to authorize.", "needs-approval"),
    ("ENAMETOOLONG: name too long, uv_spawn", "path-too-long"),
    ("app/routers/counter.py:17:1: error TS2345: Argument of type 'string' is not assignable", "ts-error"),
    ("You are not logged into any GitHub hosts. To log in, run: gh auth login", "gh-auth-error"),
    # --- divergence probes: each targets one specific backslash-escaping bug a naive JS string
    # mirror of these patterns can introduce (see Task 1 fix round 1 review). Every one of these
    # must classify identically under Python's `_cause` and TS's `cause()`. ---
    # `\d` -> `d` in a JS string literal turns `\d+` into a literal-'d' character class.
    ("this text has no digits after the word Found matches of the string", "other"),
    ("<tool_use_error>Found 3 matches of the string to replace</tool_use_error>", "edit-mismatch"),
    # `\[` / `\]` -> `[` / `]` around `\d+` forms `[d+]`, matching a bare 'd' or '+'.
    ("no action markers here, just failed prose with a d in it", "other"),
    ("[computer:wait] Waited\n\nactions[12] (click) failed: element gone", "browser-action-failed"),
    # `\(` / `\)` -> an unescaped capturing group; "Traceback (most recent call last)" must still
    # match even though the parens are no longer required to be literal.
    ("Exit code 1\nTraceback (most recent call last):\n  File \"<string>\", line 1", "python-traceback"),
    # `\.` -> `.` (any character) is a *widening* bug, not a narrowing one, but must still agree.
    ("ignored by one of your .gitignore files, use --force to add anyway", "git-error"),
    ("was blocked. For security reasons this command needs manual review", "blocked-policy"),
    # `\S` -> `S` turns "any non-whitespace run" into "a literal capital S".
    ("Exit code 1\nOn branch feature/rename-things\nYour branch is ahead of 'origin/main' by 1 commit.", "git-noise"),
    # `error TS\d+` must still fire without the digits collapsing to a literal 'd'.
    ("error TS9999: Type 'number' is not assignable to type 'string'.", "ts-error"),
    # test-failure must NOT be a catch-all for every unrelated use of the word "failed".
    ("Command failed with no output", "other"),
    ("Search failed \u2014 ripgrep rejected the pattern, glob, or file type without searching", "other"),
    ("Error: Transform failed with 1 error:\n  file.ts:1:1: expected expression", "other"),
    ("warning: Failed to hardlink files; falling back to full copy.", "other"),
    ("error: failed to push some refs to 'origin'", "other"),
    ("Failed to compile.\n./app/page.tsx\nType error: ...", "other"),
    (
        '<tool_use_error>The permission handler returned updatedInput for Workflow that failed '
        "schema validation: [{\"code\": \"custom\"}]</tool_use_error>",
        "other",
    ),
    ("Loading...Finished in 2.3s", "other"),
    ("Error: something wrong with config file", "other"),
)
