"""What the customer must have said for a reference trajectory to be the right answer.

Used only to script a perfect model offline: the workflow's form is filled in from the
reference actions, so the test checks the whole path (form, rules, plan, approval,
execution) without a live model. Returns None when the segment does not look like this
workflow's request, so the caller falls back to a free-form plan.
"""

from collections.abc import Sequence
from typing import Any, Optional




def facts_from_gold(workflow: str, writes: Sequence[Any], database: Any) -> Optional[dict[str, Any]]:
    if workflow == "debit_dispute":
        disputes = [w for w in writes if w.name == "file_debit_card_transaction_dispute_6281"]
        if not disputes:
            return None
        keys = (
            "transaction_id discovery_date disputed_amount transaction_type card_in_possession "
            "pin_compromised contacted_merchant police_report_filed written_statement_provided"
        ).split()
        return {
            "disputes": [{**{k: d.args[k] for k in keys}, "category": d.args["dispute_category"]} for d in disputes]
        }
    if workflow == "debit_replacement":
        orders = [w for w in writes if w.name == "order_debit_card_5739"]
        closes = [
            w
            for w in writes
            if w.name == "close_debit_card_4721"
            and database.debit_cards.data[w.args["card_id"]]["status"] != "CLOSED"
        ]
        if not orders and not closes:
            return None
        return {
            "closures": [{"card_id": c.args["card_id"], "reason": c.args["reason"]} for c in closes],
            "new_cards": [
                {
                    "account_id": o.args["account_id"],
                    "delivery_option": o.args["delivery_option"],
                    "card_design": o.args["card_design"],
                    "shipping_address": None
                    if o.args["shipping_address"] == database.users.data[o.args["user_id"]].get("address")
                    else o.args["shipping_address"],
                    "accepts_excess_fee": o.args.get("excess_replacement_fee") is not None,
                }
                for o in orders
            ],
        }
    return None
