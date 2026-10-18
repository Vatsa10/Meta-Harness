"""SQLite backend: the Chinook DB (read-only by policy) plus a writable `scratch` schema per task.

Both files live in a per-task sandbox. A SQLite authorizer enforces the rule that writes may only
touch `scratch`: run_sql is read-only everywhere; save_result / insert_rows may create or insert into
scratch tables only. Any attempt to change `main` is denied by SQLite itself, not by string checks.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

DB_FILE = "chinook.sqlite"
SCRATCH_FILE = "scratch.sqlite"
READ_TOOLS = ("list_tables", "describe_table", "run_sql")
WRITE_TOOLS = ("save_result", "insert_rows")
TOOL_KINDS = {**{n: "read" for n in READ_TOOLS}, **{n: "write" for n in WRITE_TOOLS}}
PREVIEW_ROWS = 20
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")

_WRITE_ACTIONS = {getattr(sqlite3, n) for n in (
    "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_VIEW", "SQLITE_CREATE_TRIGGER", "SQLITE_DROP_TABLE", "SQLITE_DROP_INDEX", "SQLITE_DROP_VIEW",
    "SQLITE_DROP_TRIGGER", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_CREATE_VTABLE",
    "SQLITE_DROP_VTABLE")}
_NEVER = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA, sqlite3.SQLITE_TRANSACTION}


class AnalystDB:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.con = sqlite3.connect(str(self.root / DB_FILE), isolation_level=None)
        self.con.execute("ATTACH DATABASE ? AS scratch", (str(self.root / SCRATCH_FILE),))
        self.allow_scratch_writes = False
        self.con.set_authorizer(self._authorize)

    def _authorize(self, action, arg1, arg2, dbname, source):
        if action in _NEVER:
            return sqlite3.SQLITE_DENY
        if action in _WRITE_ACTIONS:
            if self.allow_scratch_writes and dbname == "scratch":
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def close(self) -> None:
        self.con.close()

    def _rows(self, sql: str, limit: int = PREVIEW_ROWS) -> dict:
        cur = self.con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(limit + 1)
        return {"columns": cols, "rows": [list(r) for r in rows[:limit]], "truncated": len(rows) > limit}

    def list_tables(self) -> Any:
        main = [r[0] for r in self.con.execute("SELECT name FROM main.sqlite_master WHERE type='table' ORDER BY name")]
        scratch = [r[0] for r in self.con.execute("SELECT name FROM scratch.sqlite_master WHERE type='table' ORDER BY name")]
        return {"main": main, "scratch": scratch}

    def describe_table(self, table: str) -> Any:
        schema = "scratch" if table.startswith("scratch.") else "main"
        name = table.split(".", 1)[-1]
        row = self.con.execute(f"SELECT sql FROM {schema}.sqlite_master WHERE type='table' AND name = ?", (name,)).fetchone()
        if not row:
            return f"Error: no table {table}"
        sample = self._rows(f'SELECT * FROM {schema}."{name}" LIMIT 3', 3)
        return {"create": row[0], "sample": sample}

    def run_sql(self, sql: str) -> Any:
        self.allow_scratch_writes = False
        try:
            return self._rows(sql)
        except sqlite3.Error as exc:
            return f"Error: {exc}"

    def _scratch_write(self, statement: str) -> Any:
        self.allow_scratch_writes = True
        try:
            self.con.execute(statement)
        except sqlite3.Error as exc:
            return f"Error: {exc}"
        finally:
            self.allow_scratch_writes = False
        return None

    def save_result(self, table: str, sql: str) -> Any:
        table = table.split(".", 1)[-1]
        if not _IDENT.match(table):
            return "Error: table must be a plain identifier"
        err = self._scratch_write(f'DROP TABLE IF EXISTS scratch."{table}"')
        err = err or self._scratch_write(f'CREATE TABLE scratch."{table}" AS {sql.strip().rstrip(";")}')
        if err:
            return err
        count = self.con.execute(f'SELECT COUNT(*) FROM scratch."{table}"').fetchone()[0]
        return {"saved": f"scratch.{table}", "row_count": count, "preview": self._rows(f'SELECT * FROM scratch."{table}"', 5)}

    def insert_rows(self, table: str, sql: str) -> Any:
        table = table.split(".", 1)[-1]
        if not _IDENT.match(table):
            return "Error: table must be a plain identifier"
        err = self._scratch_write(f'INSERT INTO scratch."{table}" {sql.strip().rstrip(";")}')
        if err:
            return err
        return {"inserted_into": f"scratch.{table}", "row_count": self.con.execute(f'SELECT COUNT(*) FROM scratch."{table}"').fetchone()[0]}

    def scratch_rows(self, table: str) -> list[tuple] | None:
        try:
            return [tuple(r) for r in self.con.execute(f'SELECT * FROM scratch."{table}"')]
        except sqlite3.Error:
            return None


def make_tools(db: AnalystDB, recorder) -> list:
    from usecases._common import recorded

    def list_tables() -> Any:
        """Names of the tables in the Chinook database (main) and in your scratch schema."""
        return db.list_tables()

    def describe_table(table: str) -> Any:
        """CREATE statement and 3 sample rows of a table (prefix 'scratch.' for scratch tables)."""
        return db.describe_table(table)

    def run_sql(sql: str) -> Any:
        """Run a read-only SQLite query; returns columns and up to 20 rows."""
        return db.run_sql(sql)

    def save_result(table: str, sql: str) -> Any:
        """Save a SELECT's result as scratch.<table> (replacing it). Writes are only allowed in scratch."""
        return db.save_result(table, sql)

    def insert_rows(table: str, sql: str) -> Any:
        """INSERT the rows of a SELECT into an existing scratch.<table>."""
        return db.insert_rows(table, sql)

    return [recorded(f, recorder) for f in (list_tables, describe_table, run_sql, save_result, insert_rows)]
