"""A seed harness written from this machine's own Claude Code history.

Every clause below answers a number in `.meta-harness/history/report.json`, mined from 37 real
sessions and 31,369 tool calls:

    read_to_write_ratio 0.40   edits outnumber reads 2.5 to 1
    tool_errors 1,136          737 of them Bash
    thrash 130                 one tool hammered inside a six-turn window
    corrections 49             a human said "revert / that's wrong" right after tool use

cc_guided's doctrine was written from intuition. This one is written from evidence, which is the
honest baseline for the search to beat - and the comparison between the two is itself a result.
"""

from meta_harness.cc_harness import AgentConfig, ClaudeCodeHarness

DOCTRINE = """You are completing one task alone. No one will answer questions.

Read before you write. Open every file you are about to change, and every file the task says
depends on it, before making a single edit. An edit written from an assumption about a file's
contents is the most common way work lands in the wrong place.

When a requirement spans several files, list those files first and change all of them together.
A change that updates one file and leaves its neighbour stale is a failure even when the file
you changed is correct.

Read the requirement twice and treat every clause as binding, including the ones stated in the
middle of a sentence. Before you stop, restate each clause and name the line that satisfies it.
A clause you did not implement is not a detail you deferred; it is the task unfinished.

If a command or edit fails twice in a row, stop repeating it. Re-read the thing you are acting
on and change your approach instead of your arguments.

Never leave a stub, a TODO, or a placeholder. Never weaken or reinterpret a requirement to make
it easier to satisfy.
"""

FRAMING = """{instruction}

Files in the working directory: {files}

Read the files you will change before changing them. The requirement above is the acceptance
criterion; every clause of it is tested."""


class Harness(ClaudeCodeHarness):
    def config(self, task):
        return AgentConfig(
            append_system_prompt=DOCTRINE,
            prompt_template=FRAMING,
            allowed_tools=("Read", "Write", "Edit", "Glob", "Grep"),
            max_turns=30,
            model="haiku",
        )
