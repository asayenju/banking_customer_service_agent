import pytest

from banking_tools.session import _base_db
from banking_tools.statehash import (
    TOOL_LOG_TABLE,
    VERIFICATION_TABLE,
    compare,
    domain_delta,
    normalise,
)

ACCOUNT = next(iter(_base_db().accounts.data))


def fresh():
    return _base_db().model_copy(deep=True)


def edited(**holdings_and_more):
    db = fresh()
    db.accounts.data[ACCOUNT].update(holdings_and_more)
    return db


@pytest.mark.parametrize(
    "a, b",
    [
        ("$1,234.56", "$1234.56"),
        ("$12.00", "$12"),
        (33, 33.0),
        ("2,156 points", "2156 points"),
        ("11/14/2025", "11/14/2025"),
    ],
)
def test_cosmetic_forms_normalise_to_the_same_value(a, b):
    assert normalise(a) == normalise(b)


def test_a_bare_decimal_string_is_money_only_in_a_money_field():
    assert normalise("100.00", table="accounts", field="current_holdings") == normalise("$100.00")
    assert normalise("100.00", table="users", field="name") == "100.00"
    assert normalise("123456", table="users", field="user_id") == "123456"


def test_different_amounts_do_not_normalise_together():
    assert normalise("$12.00") != normalise("$12.01")
    assert normalise(0.1 + 0.2) == normalise(0.3)


def test_order_debit_cards_bare_balance_equals_the_dollar_form():
    start = fresh()
    dollars, bare = edited(current_holdings="$100.00"), edited(current_holdings="100.00")
    result = compare(start, dollars, bare)
    assert result.semantic_equal and not result.strict_equal and result.strict_only


def test_a_real_difference_is_reported_with_table_row_and_field():
    start = fresh()
    result = compare(start, edited(current_holdings="$100.00"), edited(current_holdings="$101.00"))
    assert not result.semantic_equal and not result.strict_equal
    (m,) = result.mismatches
    assert (m.table, m.pk) == ("accounts", ACCOUNT) and "current_holdings" in m.problem


def test_identical_states_agree_at_both_levels():
    start = fresh()
    one, two = edited(current_holdings="$100.00"), edited(current_holdings="$100.00")
    result = compare(start, one, two)
    assert result.strict_equal and result.semantic_equal and not result.strict_only
    assert result.mismatches == ()


def test_a_missing_and_an_unexpected_change_are_both_found():
    start = fresh()
    changed = edited(current_holdings="$100.00")
    assert [m.problem for m in compare(start, changed, fresh()).mismatches] == ["expected change is missing"]
    assert [m.problem for m in compare(start, fresh(), changed).mismatches] == ["unexpected change"]


def verified():
    db = fresh()
    db.verification_history.data["u_2025"] = {"name": "x", "user_id": "u"}
    return db


def test_the_verification_log_is_ignored_by_default_and_counted_on_request():
    start = fresh()
    assert compare(start, fresh(), verified()).semantic_equal
    strict_on = compare(start, fresh(), verified(), ignore_verification=False)
    assert not strict_on.semantic_equal
    assert strict_on.mismatches[0].table == VERIFICATION_TABLE


def with_tool_row(name):
    db = fresh()
    db.agent_discoverable_tools.data[f"id_{name}"] = {"tool_name": name, "status": "CALLED"}
    return db


def test_read_log_rows_can_be_ignored_by_tool_name_but_mutating_rows_still_count():
    start = fresh()
    read_row, write_row = with_tool_row("get_payment_history_6183"), with_tool_row("freeze_debit_card_3892")
    assert not compare(start, fresh(), read_row).semantic_equal
    assert compare(start, fresh(), read_row, read_log_names={"get_payment_history_6183"}).semantic_equal
    result = compare(start, fresh(), write_row, read_log_names={"get_payment_history_6183"})
    assert not result.semantic_equal and result.mismatches[0].table == TOOL_LOG_TABLE


def test_delta_lists_added_removed_and_changed_rows():
    start = fresh()
    end = fresh()
    end.accounts.data[ACCOUNT]["current_holdings"] = "$1.00"
    end.verification_history.data["k"] = {"name": "x"}
    delta = domain_delta(start, end, ignore_verification=False)
    kinds = {key: change.kind for key, change in delta.items()}
    assert kinds == {("accounts", ACCOUNT): "changed", (VERIFICATION_TABLE, "k"): "added"}
