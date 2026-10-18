"""Tunable settings for the SQL analyst graph (prompts partly here, partly inline in app.py)."""

PLANNER_PROMPT = """You are the planning analyst. Given a business question about the Chinook music-store
database, write a short numbered plan: which tables to join, filters, grouping, ordering and the exact
output columns the question asks for. Do not write SQL."""

WRITER_PROMPT = """You are the SQL writer for a SQLite database. Use list_tables / describe_table / run_sql to
check names and values before you answer. When you are confident, reply with ONLY the final query in a
```sql fenced block. Return exactly the columns the question asks for, in that order."""

SCHEMA_HINT = True          # give the writer the table list up front
MAX_WRITER_STEPS = 10       # tool calls per writer turn
MAX_REVISIONS = 1           # verifier -> writer loops
RESULT_TABLE = "answer"     # scratch table the executor saves the final result into
TEMPERATURE = 0.0
