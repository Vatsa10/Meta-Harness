"""Tunable settings for the retail support graph (the owner's settings module).

Some prompts live here; others are inline in `app.py` node code. Meta-Harness may change values in
this module (and the inline prompts), never the tools or the backend.
"""

AGENT_SYSTEM_PROMPT = """You are a customer-service agent for an online retail store.
Follow the store policy below exactly. Use tools to look things up; never invent ids or prices.

{policy}
"""

# Extra rules appended after the policy.
AGENT_RULES = [
    "Authenticate the customer first (find_user by email, or by first name + last name + zip).",
    "Item ids and product ids are different things; new_item_ids must be variant item ids from list_products.",
    "Collect every item for an order before calling modify_order or exchange_items: they can be called only once per order.",
]

TRIAGE_ENABLED = True
MAX_TOOL_STEPS = 30        # tool calls per agent turn
MAX_USER_TURNS = 4         # scripted customer replies before the conversation ends
TEMPERATURE = 0.0
