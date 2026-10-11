"""The Meta-Harness optimizer: a tool-using agent reading the experience filesystem."""

from __future__ import annotations

from ..proposer_agent import AgentProposer

NAME = "metaharness"


def build(ctx):
    return AgentProposer(ctx.proposer_model, domain=ctx.domain.NAME,
                         domain_notes=getattr(ctx.domain, "PROMPT", ""), view=ctx.view,
                         max_tool_calls=ctx.max_tool_calls, summarizer=ctx.summarizer,
                         forbidden=ctx.forbidden)
