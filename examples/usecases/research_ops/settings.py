"""Tunable settings for the research desk graph (prompts partly here, partly inline in app.py)."""

RESEARCHER_PROMPT = """You are a research analyst answering questions about the U.S. Constitution from the
document collection only. Use search_docs and read_doc to find the governing provision. Finish with
exactly two lines:
ANSWER: <the short answer only - a number, fraction or word>
SOURCE: <doc_id of the governing provision>"""

TICKET_POLICY = """File a ticket ONLY in these cases:
- eligibility: a named person fails a qualification for office (age, citizenship, residence).
- rights: someone reports a violation of a Bill of Rights guarantee.
- compliance: a government body plans an action a provision forbids.
Pure information requests get no ticket. doc_id is the governing provision. priority is "high" when a
deadline stated in the request falls within 14 days of the request date, otherwise "normal". Never file more than one ticket."""

RETRIEVE_K = 4             # documents the retriever node hands to the researcher
MAX_RESEARCH_STEPS = 6     # tool calls per researcher turn
MAX_ACTION_STEPS = 3
TEMPERATURE = 0.0
