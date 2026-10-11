"""Best-of-n: the same agent, but it only ever sees the seed candidates (no search history)."""

from __future__ import annotations

from ..proposer_agent import AgentProposer

NAME = "best-of-n"


def build(ctx):
    return AgentProposer(ctx.proposer_model, domain=ctx.domain.NAME,
                         domain_notes=getattr(ctx.domain, "PROMPT", ""), view="seed",
                         max_tool_calls=ctx.max_tool_calls, summarizer=ctx.summarizer,
                         forbidden=ctx.forbidden, test_root=ctx.test_root)
