"""Claude Code with a reviewer subagent and a shipped skill.

Third seed for the agent domain, alongside cc_default (Claude Code as it ships) and cc_guided
(hand-written doctrine). This one exercises the knobs beyond prompt text: a custom subagent the
session can delegate to, and a skill loaded from a generated plugin.

Whether delegation is worth its tokens is exactly the kind of question the search decides. Mined
history says delegation is already common (`Agent` was the fifth most-used tool across sessions),
so a seed that uses it is the honest starting point, not an exotic one.
"""

from meta_harness.cc_harness import AgentConfig, ClaudeCodeHarness

DOCTRINE = """You are completing one task autonomously. No one will answer questions.

Read the files you are about to change before changing them. Make the smallest change that
satisfies the stated requirement. Before you stop, delegate a review to the `verifier` agent and
act on what it reports.
"""

VERIFIER = {
    "verifier": {
        "description": "Checks a finished change against the stated requirement.",
        "prompt": (
            "You verify work. Read the files in the working directory and the requirement you "
            "were given. Report, in at most five lines: any requirement not yet satisfied, any "
            "placeholder or stub left behind, and any edit that contradicts the requirement. "
            "If everything is satisfied, say exactly: VERIFIED. Do not edit anything."
        ),
        "tools": ["Read", "Glob", "Grep"],
    }
}

READ_FIRST_SKILL = """Use when about to edit a file in this workspace.

Open the file and read the part you intend to change before editing it. An edit written from
memory of the file, rather than from its current contents, is the most common way a change lands
in the wrong place or reverts someone else's work.

After editing, re-read the edited region and check it against the requirement you were given.
"""

FRAMING = """{instruction}

Files in the working directory: {files}

Finish the task completely. The requirement above is the acceptance criterion."""


class Harness(ClaudeCodeHarness):
    def config(self, task):
        return AgentConfig(
            append_system_prompt=DOCTRINE,
            prompt_template=FRAMING,
            allowed_tools=("Read", "Write", "Edit", "Glob", "Grep"),
            max_turns=40,
            model="haiku",
            agents=VERIFIER,
            skills={"read-before-editing": READ_FIRST_SKILL},
        )
