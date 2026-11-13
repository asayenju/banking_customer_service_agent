"""What the bank's debit tools really check, shown by running them (docs/debit-policy-decisions.md section 1)."""

from pathlib import Path

from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

DOC = Path(__file__).resolve().parents[3] / "docs" / "debit-policy-decisions.md"
TASKS = {t["id"]: t for t in load_tasks()}


def test_the_note_has_its_sections():
    text = DOC.read_text()
    for heading in ("## 1. The bank's tools", "## 2. Rules enforced in code", "## 3. Contradictions", "## 4. Not done yet"):
        assert heading in text, heading


def test_the_close_tool_ignores_the_card_age():
    shared = BankingSession.from_task(TASKS["task_078"])
    card = shared.database.debit_cards.data["dbc_mc78a5b9d2_green"]
    card["issue_date"] = "11/14/2025"
    assert shared.call("close_debit_card_4721", {"card_id": card["card_id"], "reason": "no_longer_needed"}).ok


def test_the_order_tool_ignores_the_account_balance_and_age():
    shared = BankingSession.from_task(TASKS["task_078"])
    shared.call("close_debit_card_4721", {"card_id": "dbc_mc78a5b9d2_green", "reason": "lost"})
    account = shared.database.accounts.data["chk_mc78a5b9d2_2"]
    account["current_holdings"] = "30.00"
    account["date_opened"] = "11/13/2025"
    args = {
        "account_id": "chk_mc78a5b9d2_2", "user_id": "mc78a5b9d2", "delivery_option": "STANDARD", "delivery_fee": 0,
        "card_design": "CLASSIC", "design_fee": 0, "shipping_address": "1 Test St",
    }
    assert shared.call("order_debit_card_5739", args).ok
