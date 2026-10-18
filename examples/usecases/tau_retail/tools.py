"""Retail backend (JSON files in a per-task sandbox) and the agent's tools.

Write-tool semantics mirror tau-bench's retail tools (MIT, Sierra Research): the same status checks,
refund bookkeeping and gift-card balance rules, so tau-bench's gold actions reproduce its gold state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

DB_FILES = ("users.json", "orders.json", "products.json")
READ_TOOLS = ("find_user", "get_user", "get_order", "list_products")
WRITE_TOOLS = ("cancel_order", "modify_order", "change_address", "return_items", "exchange_items")
TOOL_KINDS = {**{n: "read" for n in READ_TOOLS}, **{n: "write" for n in WRITE_TOOLS}}


class RetailDB:
    """Loads the three JSON tables from `root`; every write is saved back immediately."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.users = json.loads((self.root / "users.json").read_text(encoding="utf-8"))
        self.orders = json.loads((self.root / "orders.json").read_text(encoding="utf-8"))
        self.products = json.loads((self.root / "products.json").read_text(encoding="utf-8"))

    def save(self) -> None:
        (self.root / "users.json").write_text(json.dumps(self.users, indent=1, sort_keys=True), encoding="utf-8")
        (self.root / "orders.json").write_text(json.dumps(self.orders, indent=1, sort_keys=True), encoding="utf-8")

    def state(self) -> dict:
        return {"users": self.users, "orders": self.orders}

    def state_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()

    # ---------------------------------------------------------------- reads
    def find_user(self, email: str = "", first_name: str = "", last_name: str = "", zip: str = "") -> str:
        for uid, u in self.users.items():
            if email and u["email"].lower() == email.strip().lower():
                return uid
            if (first_name and last_name and zip and u["name"]["first_name"].lower() == first_name.strip().lower()
                    and u["name"]["last_name"].lower() == last_name.strip().lower()
                    and u["address"]["zip"] == zip.strip()):
                return uid
        return "Error: user not found"

    def get_user(self, user_id: str) -> Any:
        return self.users.get(user_id, "Error: user not found")

    def get_order(self, order_id: str) -> Any:
        return self.orders.get(order_id, "Error: order not found")

    def list_products(self, product_id: str = "") -> Any:
        if not product_id:
            return {p["name"]: pid for pid, p in sorted(self.products.items(), key=lambda kv: kv[1]["name"])}
        return self.products.get(product_id, "Error: product not found")

    # ---------------------------------------------------------------- writes
    def cancel_order(self, order_id: str, reason: str) -> Any:
        order = self.orders.get(order_id)
        if order is None:
            return "Error: order not found"
        if order["status"] != "pending":
            return "Error: non-pending order cannot be cancelled"
        if reason not in ("no longer needed", "ordered by mistake"):
            return "Error: invalid reason"
        refunds = []
        for p in order["payment_history"]:
            refunds.append({"transaction_type": "refund", "amount": p["amount"], "payment_method_id": p["payment_method_id"]})
            if "gift_card" in p["payment_method_id"]:
                m = self.users[order["user_id"]]["payment_methods"][p["payment_method_id"]]
                m["balance"] = round(m["balance"] + p["amount"], 2)
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(refunds)
        self.save()
        return order

    def _swap_check(self, order: dict, item_ids: list, new_item_ids: list, payment_method_id: str):
        all_ids = [i["item_id"] for i in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_ids.count(item_id):
                return f"Error: {item_id} not found"
        if len(item_ids) != len(new_item_ids):
            return "Error: the number of items to be exchanged should match"
        diff = 0.0
        for item_id, new_id in zip(item_ids, new_item_ids):
            item = next(i for i in order["items"] if i["item_id"] == item_id)
            variants = self.products[item["product_id"]]["variants"]
            if not (new_id in variants and variants[new_id]["available"]):
                return f"Error: new item {new_id} not found or available"
            diff += variants[new_id]["price"] - item["price"]
        methods = self.users[order["user_id"]]["payment_methods"]
        if payment_method_id not in methods:
            return "Error: payment method not found"
        m = methods[payment_method_id]
        if m["source"] == "gift_card" and m["balance"] < diff:
            return "Error: insufficient gift card balance to pay for the price difference"
        return diff, m

    def modify_order(self, order_id: str, item_ids: list, new_item_ids: list, payment_method_id: str) -> Any:
        order = self.orders.get(order_id)
        if order is None:
            return "Error: order not found"
        if order["status"] != "pending":
            return "Error: non-pending order cannot be modified"
        checked = self._swap_check(order, item_ids, new_item_ids, payment_method_id)
        if isinstance(checked, str):
            return checked
        diff, m = checked
        order["payment_history"].append({"transaction_type": "payment" if diff > 0 else "refund",
                                         "amount": abs(diff), "payment_method_id": payment_method_id})
        if m["source"] == "gift_card":
            m["balance"] = round(m["balance"] - diff, 2)
        for item_id, new_id in zip(item_ids, new_item_ids):
            item = next(i for i in order["items"] if i["item_id"] == item_id)
            variant = self.products[item["product_id"]]["variants"][new_id]
            item.update(item_id=new_id, price=variant["price"], options=variant["options"])
        order["status"] = "pending (item modified)"
        self.save()
        return order

    def change_address(self, order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str) -> Any:
        order = self.orders.get(order_id)
        if order is None:
            return "Error: order not found"
        if order["status"] != "pending":
            return "Error: non-pending order cannot be modified"
        order["address"] = {"address1": address1, "address2": address2, "city": city, "state": state,
                            "country": country, "zip": zip}
        self.save()
        return order

    def return_items(self, order_id: str, item_ids: list, payment_method_id: str) -> Any:
        order = self.orders.get(order_id)
        if order is None:
            return "Error: order not found"
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be returned"
        if payment_method_id not in self.users[order["user_id"]]["payment_methods"]:
            return "Error: payment method not found"
        if "gift_card" not in payment_method_id and payment_method_id != order["payment_history"][0]["payment_method_id"]:
            return "Error: payment method should be either the original payment method or a gift card"
        all_ids = [i["item_id"] for i in order["items"]]
        if any(item_ids.count(i) > all_ids.count(i) for i in item_ids):
            return "Error: some item not found"
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        order["return_payment_method_id"] = payment_method_id
        self.save()
        return order

    def exchange_items(self, order_id: str, item_ids: list, new_item_ids: list, payment_method_id: str) -> Any:
        order = self.orders.get(order_id)
        if order is None:
            return "Error: order not found"
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be exchanged"
        checked = self._swap_check(order, item_ids, new_item_ids, payment_method_id)
        if isinstance(checked, str):
            return checked
        diff, _ = checked
        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = round(diff, 2)
        self.save()
        return order


def dispatch(db: RetailDB, name: str, args: dict) -> Any:
    if name not in TOOL_KINDS:
        return f"Error: unknown tool {name}"
    try:
        return getattr(db, name)(**args)
    except TypeError as exc:
        return f"Error: bad arguments: {exc}"


def make_tools(root: Path, recorder) -> list:
    """LangChain tools bound to the sandboxed DB at `root`, every call recorded."""
    from usecases._common import recorded

    db = RetailDB(root)

    def find_user(email: Optional[str] = None, first_name: Optional[str] = None, last_name: Optional[str] = None,
                  zip: Optional[str] = None) -> Any:
        """Authenticate a customer: return their user id from their email, or from first name + last name + zip code."""
        return db.find_user(email or "", first_name or "", last_name or "", zip or "")

    def get_user(user_id: str) -> Any:
        """Profile of a user: name, address, email, payment methods (with ids) and order ids."""
        return db.get_user(user_id)

    def get_order(order_id: str) -> Any:
        """Details of an order (id like '#W1234567'): status, items with item_id/product_id/price/options, payments."""
        return db.get_order(order_id)

    def list_products(product_id: Optional[str] = None) -> Any:
        """Without product_id: map of product type name -> product_id. With product_id: that product's variants
        (item_id -> options, availability, price)."""
        return db.list_products(product_id or "")

    def cancel_order(order_id: str, reason: str) -> Any:
        """Cancel a PENDING order. reason must be 'no longer needed' or 'ordered by mistake'."""
        return db.cancel_order(order_id, reason)

    def modify_order(order_id: str, item_ids: list[str], new_item_ids: list[str], payment_method_id: str) -> Any:
        """Swap items of a PENDING order to other variants of the same product (call once with every item).
        payment_method_id pays or receives the price difference."""
        return db.modify_order(order_id, item_ids, new_item_ids, payment_method_id)

    def change_address(order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str) -> Any:
        """Change the shipping address of a PENDING order."""
        return db.change_address(order_id, address1, address2, city, state, country, zip)

    def return_items(order_id: str, item_ids: list[str], payment_method_id: str) -> Any:
        """Request a return of items of a DELIVERED order; refund goes to the original payment method or a gift card."""
        return db.return_items(order_id, item_ids, payment_method_id)

    def exchange_items(order_id: str, item_ids: list[str], new_item_ids: list[str], payment_method_id: str) -> Any:
        """Exchange items of a DELIVERED order for other variants of the same product (call once with every item)."""
        return db.exchange_items(order_id, item_ids, new_item_ids, payment_method_id)

    fns = [find_user, get_user, get_order, list_products, cancel_order, modify_order, change_address, return_items,
           exchange_items]
    return [recorded(f, recorder) for f in fns]
