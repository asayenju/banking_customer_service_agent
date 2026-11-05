"""What the customer must have said for a reference trajectory to be the right answer.

Used only to script a perfect model offline: the workflow's form is filled in from the
reference actions, so the test checks the whole path (form, rules, plan, approval,
execution) without a live model. Returns None when the segment does not look like this
workflow's request, so the caller falls back to a free-form plan.
"""

from collections.abc import Sequence
from typing import Any, Optional




def facts_from_gold(workflow: str, writes: Sequence[Any], database: Any) -> Optional[dict[str, Any]]:
    if workflow == "close_account":
        from agent.workflows.account_close import CLOSE_TOOL, DEFAULT_REASON, TRANSFER_TOOL

        closes = [w for w in writes if w.name == CLOSE_TOOL]
        if not closes:
            return None
        return {
            "closures": [
                {
                    "account_id": c.args["account_id"],
                    "move_to_account": next(
                        (w.args["destination_account_id"] for w in writes if w.name == TRANSFER_TOOL and w.args["source_account_id"] == c.args["account_id"]),
                        None,
                    ),
                    "reason": c.args.get("reason") or DEFAULT_REASON,
                    "asks_to_waive_fee": False,
                }
                for c in closes
            ]
        }
    if workflow == "open_account":
        from agent.workflows.account_open import OPEN_TOOL, TRANSFER_TOOL, new_account_id

        opens = [w for w in writes if w.name == OPEN_TOOL]
        if not opens:
            return None
        accounts = []
        for o in opens:
            destination = new_account_id(o.args["user_id"], o.args["account_type"], o.args["account_class"])
            move = next((w for w in writes if w.name == TRANSFER_TOOL and w.args["destination_account_id"] == destination), None)
            accounts.append(
                {
                    "account_type": o.args["account_type"],
                    "account_class": o.args["account_class"],
                    "fund_now": "yes" if move is not None else "no",
                    "fund_from_account": move.args["source_account_id"] if move is not None else None,
                    "fund_amount": float(move.args["amount"]) if move is not None else None,
                }
            )
        return {"accounts": accounts}
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
    if workflow == "credit_dispute":
        disputes = [w for w in writes if w.name == "file_credit_card_transaction_dispute_4829"]
        if not disputes:
            return None
        return {
            "disputes": [
                {
                    "transaction_id": d.args["transaction_id"],
                    "reason": d.args["dispute_reason"],
                    "noticed_date": d.args["issue_noticed_date"],
                    "contacted_merchant": d.args["contacted_merchant"],
                    "card_last_4_digits": d.args["card_last_4_digits"],
                    "partial_refund_amount": d.args.get("partial_refund_amount"),
                    "cancel_and_reissue": d.args["card_action"] == "cancel_and_reissue",
                }
                for d in disputes
            ]
        }
    if workflow == "credit_replacement":
        orders = [w for w in writes if w.name == "order_replacement_credit_card_7291"]
        if not orders:
            return None
        return {
            "replacements": [
                {
                    "credit_card_account_id": o.args["credit_card_account_id"],
                    "reason": o.args["reason"],
                    "expedited": bool(o.args.get("expedited_shipping")),
                    "accepts_expedite_fee": bool(o.args.get("expedited_shipping")),
                    "shipping_address": None
                    if o.args["shipping_address"] == database.users.data[o.args["user_id"]].get("address")
                    else o.args["shipping_address"],
                }
                for o in orders
            ]
        }
    if workflow == "credit_limit":
        submits = [w for w in writes if w.name == "submit_credit_limit_increase_request_7392"]
        if not submits:
            return None
        first = submits[0].args
        return {
            "credit_card_account_id": first["credit_card_account_id"],
            "requested_increase_amount": int(first["requested_increase_amount"]),
        }
    if workflow == "credit_closure":
        mine = [w for w in writes if w.args.get("credit_card_account_id")]
        if not mine:
            return None
        account = mine[0].args["credit_card_account_id"]
        names = {w.name for w in mine}
        logged = next((w.args["closure_reason"] for w in mine if w.name == "log_credit_card_closure_reason_4521"), None)
        waiver = "apply_credit_card_account_flag_6147" in names
        credit = "apply_statement_credit_8472" in names
        closing = "close_credit_card_account_7834" in names
        return {
            "credit_card_account_id": account,
            "reason": logged,
            "pay_from_checking_account_id": None,
            "waiver_answer": "yes" if waiver else None,
            "credit_answer": "yes" if credit else ("no" if closing and logged else None),
            "keeps_card": bool(logged) and not (waiver or credit or closing),
        }
    return None
