"""Claude Code with hand-written doctrine - the Terminus-shaped seed.

The paper initializes its agentic-coding search from strong open baselines (Terminus 2,
Terminus-KIRA) rather than from nothing. This is that role: a reasonable hand-engineered
harness for the proposer to beat, not a strawman.
"""

from meta_harness.cc_harness import AgentConfig, ClaudeCodeHarness

DOCTRINE = """You are completing one task autonomously. No one will answer questions.

Work in this order:
1. Read before you write. Open every file the task mentions and the tests that cover them.
2. Find the actual cause. Match the failing behaviour to a specific line before editing.
3. Make the smallest change that fixes the cause. Do not refactor around it.
4. Re-read your edit and check it against the task's stated requirement.

Rules:
- Never leave a placeholder, a TODO, or a stub in place of working code.
- Never change a test to make it pass.
- Do not stop until the stated requirement is actually satisfied in the files.
"""

FRAMING = """{instruction}

Files in the working directory: {files}
Finish the task completely before you stop."""


class Harness(ClaudeCodeHarness):
    def config(self, task):
        return AgentConfig(
            append_system_prompt=DOCTRINE,
            allowed_tools=("Read", "Write", "Edit", "Glob", "Grep"),
            max_turns=30,
            model="haiku",
            prompt_template=FRAMING,
        )
