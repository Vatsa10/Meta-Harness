"""Claude Code out of the box - the paper's Table 7 baseline row.

No appended doctrine, no reframing, default turn budget. This is the floor every proposed
harness has to beat. In the paper it scored 58.0% on Opus 4.6 and 27.5% on Haiku 4.5.
"""

from meta_harness.cc_harness import AgentConfig, ClaudeCodeHarness


class Harness(ClaudeCodeHarness):
    def config(self, task):
        return AgentConfig(
            append_system_prompt="",
            allowed_tools=("Read", "Write", "Edit", "Glob", "Grep"),
            max_turns=30,
            model="haiku",
            prompt_template="{instruction}",
        )
