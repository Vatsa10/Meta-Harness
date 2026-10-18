"""Build the committed tau_retail subset from tau-bench's public retail data (MIT, Sierra).

Fetch once (network) into the git-ignored cache, then build offline:

    python examples/usecases/tau_retail/build_data.py --fetch      # download into data/cache/tau_bench
    python examples/usecases/tau_retail/build_data.py              # write data/*.json + tasks.jsonl

Tasks are parsed with `ast` (the upstream task file is never executed). We keep test tasks whose gold
actions use only the tools this app implements and that write at least once, take the first
`MAX_TASKS`, keep the users they touch (plus their orders) and every product, and compute the gold
final DB state by replaying the gold actions through this app's own tool functions.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # examples/ on the path

from usecases._common import split_tasks, write_jsonl  # noqa: E402
from usecases.tau_retail import tools as T  # noqa: E402

REPO = HERE.parents[2]
CACHE = REPO / "data" / "cache" / "tau_bench"
RAW = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/"
FILES = {"orders.json": "tau_bench/envs/retail/data/orders.json",
         "users.json": "tau_bench/envs/retail/data/users.json",
         "products.json": "tau_bench/envs/retail/data/products.json",
         "tasks_test.py": "tau_bench/envs/retail/tasks_test.py",
         "wiki.md": "tau_bench/envs/retail/wiki.md",
         "LICENSE": "LICENSE"}
MAX_TASKS = 50
SEED = 20261018

# upstream action name -> (our tool, arg mapper)
ACTION_MAP = {
    "find_user_id_by_name_zip": ("find_user", lambda a: {"first_name": a["first_name"], "last_name": a["last_name"], "zip": a["zip"]}),
    "find_user_id_by_email": ("find_user", lambda a: {"email": a["email"]}),
    "get_user_details": ("get_user", lambda a: a),
    "get_order_details": ("get_order", lambda a: a),
    "get_product_details": ("list_products", lambda a: {"product_id": a["product_id"]}),
    "list_all_product_types": ("list_products", lambda a: {}),
    "calculate": (None, None),
    "cancel_pending_order": ("cancel_order", lambda a: a),
    "modify_pending_order_items": ("modify_order", lambda a: a),
    "modify_pending_order_address": ("change_address", lambda a: a),
    "return_delivered_order_items": ("return_items", lambda a: a),
    "exchange_delivered_order_items": ("exchange_items", lambda a: a),
}


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        with urllib.request.urlopen(RAW + path, timeout=60) as r:
            (CACHE / name).write_bytes(r.read())
        print("fetched", name)


def parse_tasks(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assign = next(n for n in tree.body if isinstance(n, ast.Assign))
    out = []
    for call in assign.value.elts:
        kw = {k.arg: k.value for k in call.keywords}
        actions = [{"name": ast.literal_eval(next(k.value for k in a.keywords if k.arg == "name")),
                    "kwargs": ast.literal_eval(next(k.value for k in a.keywords if k.arg == "kwargs"))}
                   for a in kw["actions"].elts]
        out.append({"user_id": ast.literal_eval(kw["user_id"]), "instruction": ast.literal_eval(kw["instruction"]),
                    "actions": actions, "outputs": ast.literal_eval(kw["outputs"]) if "outputs" in kw else []})
    return out


def first_person(instruction: str) -> str:
    """The scripted customer's opening message: upstream instructions are written as 'You are X ...'."""
    text = instruction
    for a, b in (("You are ", "I am "), ("you are ", "I am "), ("You're ", "I'm "), ("you're ", "I'm "),
                 ("Your ", "My "), (" your ", " my "), ("You ", "I "), (" you ", " I "), (" yourself", " myself")):
        text = text.replace(a, b)
    return "Hi. " + text


def facts_for(gold: list[dict], users: dict, user_id: str) -> str:
    """What the scripted customer says when asked a follow-up (payment method, cancel reason)."""
    facts = []
    for act in gold:
        args = act["args"]
        pm = args.get("payment_method_id")
        if pm:
            m = users[user_id]["payment_methods"][pm]
            desc = {"credit_card": f"my {m.get('brand', 'credit')} card ending in {m.get('last_four', '')}",
                    "paypal": "my PayPal account", "gift_card": "my gift card"}.get(m["source"], pm)
            facts.append(f"For order {args['order_id']} please use {desc} for any payment or refund.")
        if act["name"] == "cancel_order":
            facts.append(f"The reason for cancelling {args['order_id']} is: {args['reason']}.")
    return " ".join(dict.fromkeys(facts))


def gold_applies(users: dict, orders: dict, products: dict, gold: list[dict]) -> bool:
    box = Path(tempfile.mkdtemp(prefix="tau-check-"))
    try:
        for name, obj in (("users.json", users), ("orders.json", orders), ("products.json", products)):
            (box / name).write_text(json.dumps(obj), encoding="utf-8")
        db = T.RetailDB(box)
        return not any(g["name"] in T.WRITE_TOOLS and str(T.dispatch(db, g["name"], g["args"])).startswith("Error")
                       for g in gold)
    finally:
        shutil.rmtree(box, ignore_errors=True)


def build() -> None:
    raw_tasks = parse_tasks(CACHE / "tasks_test.py")
    users = json.loads((CACHE / "users.json").read_text(encoding="utf-8"))
    orders = json.loads((CACHE / "orders.json").read_text(encoding="utf-8"))
    products = json.loads((CACHE / "products.json").read_text(encoding="utf-8"))
    picked = []
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / "products.json").write_text(json.dumps(products, indent=1, sort_keys=True), encoding="utf-8")
    for i, t in enumerate(raw_tasks):
        names = [a["name"] for a in t["actions"]]
        if not all(n in ACTION_MAP for n in names):
            continue
        gold = [{"name": ACTION_MAP[a["name"]][0], "args": ACTION_MAP[a["name"]][1](a["kwargs"])}
                for a in t["actions"] if ACTION_MAP[a["name"]][0]]
        if not any(g["name"] in T.WRITE_TOOLS for g in gold):
            continue
        if not gold_applies(users, orders, products, gold):
            print(f"skip upstream task {i}: a gold write errors (dataset noise)")
            continue
        picked.append((i, t, gold))
        if len(picked) == MAX_TASKS:
            break
    keep_users = sorted({t["user_id"] for _, t, _ in picked})
    sub_users = {u: users[u] for u in keep_users}
    sub_orders = {o: v for o, v in orders.items() if v["user_id"] in sub_users}
    for name, obj in (("users.json", sub_users), ("orders.json", sub_orders)):
        (data / name).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    shutil.copy2(CACHE / "wiki.md", data / "policy.md")
    shutil.copy2(CACHE / "LICENSE", data / "LICENSE-tau-bench")

    tasks = []
    for i, t, gold in picked:
        box = Path(tempfile.mkdtemp(prefix="tau-gold-"))
        for name in T.DB_FILES:
            shutil.copy2(data / name, box / name)
        db = T.RetailDB(box)
        for g in gold:
            out = T.dispatch(db, g["name"], g["args"])
            assert not (g["name"] in T.WRITE_TOOLS and str(out).startswith("Error")), (i, g, out)
        tasks.append({"id": f"tau-retail-{i:03d}", "source_index": i, "user_id": t["user_id"],
                      "input": first_person(t["instruction"]), "instruction": t["instruction"],
                      "user_facts": facts_for(gold, sub_users, t["user_id"]), "gold_actions": gold,
                      "required_outputs": t["outputs"], "gold_state_digest": db.state_digest()})
        shutil.rmtree(box, ignore_errors=True)
    write_jsonl(HERE / "tasks.jsonl", split_tasks(tasks, SEED, (len(tasks) - 30, 15, 15)))
    print(f"{len(tasks)} tasks, {len(sub_users)} users, {len(sub_orders)} orders, {len(products)} products")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch:
        fetch()
    build()
