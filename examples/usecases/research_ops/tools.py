"""Research desk backend: a BM25 index over the corpus (meta_harness.retrieval) and a ticket store
(a JSON file in the per-task sandbox)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from meta_harness.retrieval import BM25Index

TICKETS_FILE = "tickets.json"
QUEUES = ("eligibility", "rights", "compliance")
PRIORITIES = ("normal", "high")
READ_TOOLS = ("search_docs", "read_doc", "list_tickets")
WRITE_TOOLS = ("file_ticket",)
TOOL_KINDS = {**{n: "read" for n in READ_TOOLS}, **{n: "write" for n in WRITE_TOOLS}}


class Desk:
    def __init__(self, corpus: list[dict], root: Path):
        self.docs = {d["id"]: d for d in corpus}
        self.index = BM25Index(corpus, [f"{d['title']} {d['text']}" for d in corpus], math_mode=False)
        self.path = Path(root) / TICKETS_FILE
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def search(self, query: str, k: int = 4) -> list[dict]:
        return [{"doc_id": r.item["id"], "title": r.item["title"], "score": round(r.score, 3),
                 "snippet": r.item["text"][:300]} for r in self.index.search(query, k) if r.score > 0]

    def read(self, doc_id: str) -> Any:
        d = self.docs.get(doc_id)
        return {"doc_id": d["id"], "title": d["title"], "text": d["text"]} if d else f"Error: no document {doc_id}"

    def tickets(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def file_ticket(self, queue: str, doc_id: str, priority: str, summary: str) -> Any:
        if queue not in QUEUES:
            return f"Error: queue must be one of {list(QUEUES)}"
        if priority not in PRIORITIES:
            return f"Error: priority must be one of {list(PRIORITIES)}"
        if doc_id not in self.docs:
            return f"Error: no document {doc_id}"
        tickets = self.tickets()
        ticket = {"ticket_id": f"T-{len(tickets) + 1:04d}", "queue": queue, "doc_id": doc_id, "priority": priority,
                  "summary": summary}
        tickets.append(ticket)
        self.path.write_text(json.dumps(tickets, indent=1), encoding="utf-8")
        return ticket


def make_tools(desk: Desk, recorder, k: int = 4) -> list:
    from usecases._common import recorded

    def search_docs(query: str, top_k: Optional[int] = None) -> Any:
        """BM25 search over the constitutional corpus; returns doc_id, title and a snippet per hit."""
        return desk.search(query, top_k or k)

    def read_doc(doc_id: str) -> Any:
        """Full text of one document by doc_id (e.g. 'art1-sec3', 'amend4')."""
        return desk.read(doc_id)

    def list_tickets() -> Any:
        """Tickets already filed in this session."""
        return desk.tickets()

    def file_ticket(queue: str, doc_id: str, priority: str, summary: str) -> Any:
        """File a follow-up ticket. queue: eligibility | rights | compliance; doc_id: the governing provision;
        priority: normal | high; summary: one sentence."""
        return desk.file_ticket(queue, doc_id, priority, summary)

    return [recorded(f, recorder) for f in (search_docs, read_doc, list_tickets, file_ticket)]
