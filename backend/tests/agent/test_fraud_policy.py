"""Slice 0 of the fraud and loss composer: what the bank's tools and the task data really say, shown by running them.

``docs/fraud-policy-decisions.md`` says the tools check far less than the policy documents. Each statement in it
that the composer relies on is a test here, so the note cannot drift from the tools.
"""

from pathlib import Path

from agent.workflows.tiers import tier_of_level
from banking_tools import BankingSession
from banking_tools.catalog import get_registry
from banking_tools.manifest import load_tasks

DOC = Path(__file__).resolve().parents[3] / "docs" / "fraud-policy-decisions.md"
TASKS = {t["id"]: t for t in load_tasks()}


def world(task: str) -> BankingSession:
    return BankingSession.from_task(TASKS[task])


def test_the_policy_note_has_every_section_the_slices_depend_on():
    text = DOC.read_text()
    for heading in (
        "## 1. The bank's tools",
        "## 2. The wallet protocol",
        "## 3. Rulings",
        "## 4. Contradictions between documents and gold",
        "## 5. Where the gold departs",
        "## 6. Status",
    ):
        assert heading in text, heading


def test_a_frozen_card_cannot_be_closed_until_it_is_unfrozen():
    shared = world("task_078")
    card = "dbc_mc78a5b9d2_lg"
    assert shared.call("freeze_debit_card_3892", {"card_id": card}).ok
    refused = shared.call("close_debit_card_4721", {"card_id": card, "reason": "lost"})
    assert not refused.ok and "FROZEN" in str(refused.error)
    assert shared.call("unfreeze_debit_card_3893", {"card_id": card}).ok
    assert shared.call("close_debit_card_4721", {"card_id": card, "reason": "lost"}).ok


def test_the_close_tool_ignores_pending_transactions_so_the_composer_must_not():
    """Task 078: Light Green has two pending purchases. The bank's tool closes it anyway."""
    shared = world("task_078")
    pending = [
        t for t in shared.database.bank_account_transaction_history.data.values()
        if t["account_id"] == "chk_mc78a5b9d2_3" and t["status"] == "pending"
    ]
    assert len(pending) == 2 and all(t["type"] == "debit_card_purchase" for t in pending)
    assert shared.call("close_debit_card_4721", {"card_id": "dbc_mc78a5b9d2_lg", "reason": "lost"}).ok


def test_a_pending_refund_is_a_refund_row_not_a_purchase():
    """Task 079: Green's pending row is a refund, which the customer accepts; a purchase blocks, a refund asks."""
    shared = world("task_079")
    pending = [
        t for t in shared.database.bank_account_transaction_history.data.values() if t["status"] == "pending"
    ]
    assert [(t["type"], t["account_id"]) for t in pending] == [("refund", "chk_cr89a2b3c4_3")]


def test_the_order_tool_trusts_whatever_fee_it_is_given():
    shared = world("task_078")
    assert shared.call("close_debit_card_4721", {"card_id": "dbc_mc78a5b9d2_green", "reason": "lost"}).ok
    args = {
        "account_id": "chk_mc78a5b9d2_2", "user_id": "mc78a5b9d2", "delivery_option": "RUSH",
        "delivery_fee": 0, "card_design": "CUSTOM", "design_fee": 0, "shipping_address": "1 Test St",
    }
    assert shared.call("order_debit_card_5739", args).ok, "a RUSH CUSTOM card for nothing goes through"


def test_the_credit_replacement_closes_the_account_at_once_and_checks_nothing_else():
    shared = world("task_080")
    account = "cc_tm92c4d7e8_gold"
    args = {"credit_card_account_id": account, "user_id": "tm92c4d7e8", "shipping_address": "x", "reason": "stolen", "expedited_shipping": True}
    assert shared.call("order_replacement_credit_card_7291", args).ok
    assert shared.database.credit_card_accounts.data[account]["status"] == "CLOSED"
    again = shared.call("order_replacement_credit_card_7291", args)
    assert not again.ok, "only an identical repeat is refused (the order id is a hash of the same details)"
    other = {**args, "reason": "lost"}
    assert shared.call("order_replacement_credit_card_7291", other).ok, "a different reason for a closed card goes through"


def test_there_is_no_credit_freeze_fraud_report_or_password_tool():
    names = " ".join(get_registry())
    for missing in ("freeze_credit", "report_fraud", "password"):
        assert missing not in names


def test_the_entry_tier_is_light_blue_and_the_unknown_levels_have_no_tier():
    assert tier_of_level("Light Blue Account") == "Entry"
    assert tier_of_level("Light Green Account") is None
    assert tier_of_level("Green Fee-Free Account") is None


def test_the_five_wallet_tasks_are_the_ones_in_the_manifest_with_both_products():
    from banking_tools.manifest import build_manifest

    entries = {e.task_id: e for e in build_manifest(list(TASKS.values()))}
    assert entries["task_077"].clusters == ("credit", "debit")
    assert entries["task_078"].pilot and entries["task_079"].pilot, "078 and 079 stay in the debit pilot's tasks"
    assert entries["task_080"].clusters == ("credit", "debit") and entries["task_081"].human_transfer
