"""Build the sql_analyst task file from Chinook (MIT, Luis Rocha) and the authored questions.

    python examples/usecases/sql_analyst/build_data.py --fetch   # download Chinook into data/cache/chinook
    python examples/usecases/sql_analyst/build_data.py           # copy DB, compute gold rows, split
"""
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from usecases._common import split_tasks, write_jsonl  # noqa: E402
from usecases.sql_analyst.questions import QUESTIONS  # noqa: E402

REPO = HERE.parents[2]
CACHE = REPO / "data" / "cache" / "chinook"
URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
SEED = 20261018


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, path in (("Chinook_Sqlite.sqlite", "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"),
                       ("LICENSE.md", "LICENSE.md")):
        with urllib.request.urlopen(URL + path, timeout=60) as r:
            (CACHE / name).write_bytes(r.read())


def build() -> None:
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    shutil.copy2(CACHE / "Chinook_Sqlite.sqlite", data / "chinook.sqlite")
    shutil.copy2(CACHE / "LICENSE.md", data / "LICENSE-chinook.md")
    con = sqlite3.connect(str(data / "chinook.sqlite"))
    tasks = []
    for i, (q, sql) in enumerate(QUESTIONS):
        rows = [list(r) for r in con.execute(sql)]
        if not rows:
            raise SystemExit(f"question {i} has an empty gold result")
        m = re.search(r"\s+LIMIT (\d+)\s*$", sql)
        if m:  # a top-N cut-off must not split a tie on the ordering value
            unlimited = [list(r) for r in con.execute(sql[:m.start()])]
            n = int(m.group(1))
            explicit_tiebreak = "," in sql.rsplit("ORDER BY", 1)[-1]
            if len(unlimited) > n and unlimited[n][-1] == unlimited[n - 1][-1] and not explicit_tiebreak:
                raise SystemExit(f"question {i}: tie at the top-{n} cut-off")
        tasks.append({"id": f"sql-{i:03d}", "input": q, "gold_sql": sql, "gold_rows": rows})
    con.close()
    write_jsonl(HERE / "tasks.jsonl", split_tasks(tasks, SEED, (len(tasks) - 25, 10, 15)))
    print(f"{len(tasks)} tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch:
        fetch()
    build()
