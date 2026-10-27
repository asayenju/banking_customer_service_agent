import pytest

from banking_tools import get_registry
from orchestrator.taxonomy import TOOL_INTENT, intent_for_tool
from orchestrator.types import IntentKind


def test_every_catalogued_tool_maps_to_an_intent_kind():
    missing = set(get_registry()) - set(TOOL_INTENT)
    assert not missing, f"tools without an intent kind: {sorted(missing)}"


def test_no_mapping_for_a_tool_that_is_not_catalogued():
    stale = set(TOOL_INTENT) - set(get_registry())
    assert not stale, f"mapped tools missing from the catalog: {sorted(stale)}"


def test_every_mapping_value_is_an_intent_kind():
    assert all(isinstance(kind, IntentKind) for kind in TOOL_INTENT.values())


@pytest.mark.parametrize(
    ("tool", "kind"),
    [
        ("freeze_debit_card_3892", IntentKind.FREEZE_CARD),
        ("unfreeze_debit_card_3893", IntentKind.UNFREEZE_CARD),
        ("activate_debit_card_8292", IntentKind.ACTIVATE_CARD),
        ("file_debit_card_transaction_dispute_6281", IntentKind.DISPUTE_TRANSACTION),
        ("transfer_funds_between_bank_accounts_7291", IntentKind.TRANSFER_FUNDS),
        ("apply_statement_credit_8472", IntentKind.FEE_OR_INTEREST_ADJUSTMENT),
        ("set_debit_card_recurring_block_7382", IntentKind.CARD_CONTROLS),
        ("transfer_to_human_agents", IntentKind.REQUEST_HUMAN),
        ("log_verification", IntentKind.OTHER),
    ],
)
def test_known_tools_map_to_their_intent(tool, kind):
    assert intent_for_tool(tool) is kind


def test_an_unknown_tool_maps_to_other():
    assert intent_for_tool("not_a_real_tool") is IntentKind.OTHER
