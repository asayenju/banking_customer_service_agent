"""Slice 0 of the accounts specialist: what the bank's account tools really do, shown by running them.

``docs/accounts-policy-decisions.md`` says the tools check far less than the policy documents. Each
statement there that the code relies on is a test here, so the note cannot drift from the tools:
if tau2 ever fixes one of these, the test fails and the note (and the validator that covers it)
is revisited.
"""

from pathlib import Path

from banking_tools import BankingSession, Fidelity

DOC = Path(__file__).resolve().parents[3] / "docs" / "accounts-policy-decisions.md"

MIA = "123"  # checking 01 (Blue, opened 11/08/2025, six days before the clock) and savings 02 (Bronze, $129.02)
MIA_CHECKING, MIA_SAVINGS = "01", "02"
SAM = "125"  # checking 03 (Green, $3,400, opened 2023) and savings 04 (Silver, $900)
SAM_CHECKING, SAM_SAVINGS = "03", "04"


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def rows(shared: BankingSession) -> dict:
    return shared.database.accounts.data


# ---------------------------------------------------------------- the note exists and says what the code relies on

def test_the_policy_note_has_every_section_the_slices_depend_on():
    text = DOC.read_text()
    for heading in (
        "## 1. Rules by product",
        "## 2. What the bank's tools enforce",
        "## 3. Early-closure fee",
        "## 4. Transfers",
        "## 5. Customer-side steps in the gold",
        "## 6. Boundaries with other specialists",
        "## 7. Where the gold departs from policy",
        "## 8. Not derivable",
    ):
        assert heading in text, heading


# ---------------------------------------------------------------- transfer

def test_transfer_does_not_check_that_both_accounts_have_the_same_owner():
    shared = world()
    assert rows(shared)[SAM_CHECKING]["user_id"] != rows(shared)[MIA_CHECKING]["user_id"]
    result = shared.call(
        "transfer_funds_between_bank_accounts_7291",
        {"source_account_id": SAM_CHECKING, "destination_account_id": MIA_CHECKING, "amount": 10},
    )
    assert result.ok, "the bank moved money to another customer's account: the ownership rule is ours"


def test_transfer_writes_no_transaction_row_so_the_read_back_must_use_balances():
    shared = world()
    before = len(shared.database.bank_account_transaction_history.data)
    result = shared.call(
        "transfer_funds_between_bank_accounts_7291",
        {"source_account_id": SAM_CHECKING, "destination_account_id": SAM_SAVINGS, "amount": 100},
    )
    assert result.ok
    assert len(shared.database.bank_account_transaction_history.data) == before
    assert rows(shared)[SAM_CHECKING]["current_holdings"] == "$3300.00"
    assert rows(shared)[SAM_SAVINGS]["current_holdings"] == "$1000.00"


def test_transfer_has_no_duplicate_guard():
    shared = world()
    args = {"source_account_id": SAM_CHECKING, "destination_account_id": SAM_SAVINGS, "amount": 100}
    assert shared.call("transfer_funds_between_bank_accounts_7291", args).ok
    assert shared.call("transfer_funds_between_bank_accounts_7291", args).ok, "the same call moves the money twice"
    assert rows(shared)[SAM_SAVINGS]["current_holdings"] == "$1100.00"


# ---------------------------------------------------------------- opening

def test_open_accepts_any_class_string():
    shared = world()
    result = shared.call(
        "open_bank_account_4821", {"user_id": SAM, "account_type": "checking", "account_class": "Totally Made Up"}
    )
    assert result.ok, "the exact class string is checked by our catalogue, not by the bank"


def test_open_savings_needs_a_checking_account_fourteen_days_old():
    shared = world()
    refused = shared.call(
        "open_bank_account_4821", {"user_id": MIA, "account_type": "savings", "account_class": "Silver Account"}
    )
    assert not refused.ok, "Mia's checking account is six days old"
    allowed = shared.call(
        "open_bank_account_4821", {"user_id": SAM, "account_type": "savings", "account_class": "Silver Plus Account"}
    )
    assert allowed.ok


def test_a_new_account_is_stored_under_other_field_names_than_every_other_tool_reads():
    shared = world()
    shared.call("open_bank_account_4821", {"user_id": SAM, "account_type": "checking", "account_class": "Purple Account"})
    new = [r for k, r in rows(shared).items() if r.get("account_class") == "Purple Account"]
    assert len(new) == 1
    assert new[0]["account_type"] == "checking"
    assert "class" not in new[0] and "level" not in new[0], "class-based tools cannot see this account"


def test_opening_the_same_type_and_class_twice_is_a_silent_no_op_because_the_id_is_a_hash():
    shared = world()
    args = {"user_id": SAM, "account_type": "checking", "account_class": "Purple Account"}
    assert shared.call("open_bank_account_4821", args).ok
    count = len(rows(shared))
    again = shared.call("open_bank_account_4821", args)
    # The bank words this "Failed to open account", not "Error:", so the wrapper calls it a success.
    assert again.ok and "may already exist" in again.text
    assert len(rows(shared)) == count, "nothing was written: the open read-back must check the account is there"


def test_open_does_not_enforce_the_account_caps_the_documents_set():
    shared = world()
    for number in range(5):  # personal checking is capped at 4 in document 001
        result = shared.call(
            "open_bank_account_4821",
            {"user_id": SAM, "account_type": "checking", "account_class": f"Blue Account {number}"},
        )
        assert result.ok


def test_business_checking_is_blocked_by_any_closed_account_and_does_not_need_five_hundred_dollars():
    shared = world()
    rows(shared)[SAM_CHECKING]["current_holdings"] = "1.00"  # document 003 wants at least $500
    allowed = shared.call(
        "open_bank_account_4821", {"user_id": SAM, "account_type": "business_checking", "account_class": "Navy Blue"}
    )
    assert allowed.ok
    rows(shared)[SAM_SAVINGS]["status"] = "CLOSED"
    blocked = shared.call(
        "open_bank_account_4821", {"user_id": SAM, "account_type": "business_checking", "account_class": "Cobalt Blue"}
    )
    assert not blocked.ok


# ---------------------------------------------------------------- closing

def test_close_checks_the_fee_but_never_takes_it():
    shared = world()
    rows(shared)[MIA_CHECKING]["current_holdings"] = "25"  # a Blue account inside its 60 days owes $25
    result = shared.call("close_bank_account_7392", {"account_id": MIA_CHECKING, "reason": "Customer requested closure"})
    assert result.ok
    assert rows(shared)[MIA_CHECKING]["status"] == "CLOSED"
    assert rows(shared)[MIA_CHECKING]["current_holdings"] == "25", "the fee was never debited"


def test_close_refuses_a_young_checking_account_that_cannot_pay_the_fee():
    shared = world()
    rows(shared)[MIA_CHECKING]["current_holdings"] = "0"
    assert not shared.call("close_bank_account_7392", {"account_id": MIA_CHECKING, "reason": "x"}).ok


def test_the_savings_fee_never_applies_because_the_data_says_saving_not_savings():
    shared = world()
    assert rows(shared)[MIA_SAVINGS]["class"] == "saving"
    rows(shared)[MIA_SAVINGS]["current_holdings"] = "0"  # a Bronze savings account inside its 60 days owes $20
    result = shared.call("close_bank_account_7392", {"account_id": MIA_SAVINGS, "reason": "x"})
    assert result.ok, "with a $0 balance the fee should have blocked the close; it never fires"


def test_waiving_the_fee_needs_no_authority():
    shared = world()
    rows(shared)[MIA_CHECKING]["current_holdings"] = "0"
    result = shared.call(
        "close_bank_account_7392", {"account_id": MIA_CHECKING, "reason": "x", "waive_early_closure_fee": True}
    )
    assert result.ok, "any caller can set the flag; the specialist never does"


def test_close_ignores_a_balance_that_the_documents_say_must_move_first_only_when_it_is_nonzero():
    shared = world()
    refused = shared.call("close_bank_account_7392", {"account_id": SAM_CHECKING, "reason": "x"})
    assert not refused.ok, "a balance of $3,400 must be moved first (funds first is also the bank's own rule)"
