import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from banking_tools.args import build_args_model, call_kwargs
from banking_tools.catalog import build_catalog

SNAPSHOT = Path(__file__).parent / "snapshots" / "arg_schemas.json"


def test_every_catalogued_tool_has_an_args_model():
    for name, spec in build_catalog().items():
        assert spec.args_model is not None, name


def test_required_and_optional_fields():
    schema = build_args_model("order_replacement_credit_card_7291").model_json_schema()
    assert set(schema["required"]) == {
        "credit_card_account_id",
        "user_id",
        "shipping_address",
        "reason",
    }
    assert "expedited_shipping" in schema["properties"]


def test_missing_argument_is_rejected():
    model = build_args_model("freeze_debit_card_3892")
    with pytest.raises(ValidationError, match="card_id"):
        model()


def test_extra_argument_is_rejected():
    model = build_args_model("freeze_debit_card_3892")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model(card_id="c1", surprise="x")


@pytest.mark.parametrize("bad", ["12.5", True, None])
def test_wrongly_typed_float_is_rejected(bad):
    model = build_args_model("transfer_funds_between_bank_accounts_7291")
    with pytest.raises(ValidationError):
        model(source_account_id="a", destination_account_id="b", amount=bad)


def test_wrongly_typed_bool_is_rejected():
    model = build_args_model("set_debit_card_recurring_block_7382")
    with pytest.raises(ValidationError):
        model(card_id="c1", block_recurring="yes")


def test_literal_is_enforced():
    model = build_args_model("transfer_to_human_agents")
    assert model(summary="s", reason="other").reason == "other"
    with pytest.raises(ValidationError):
        model(summary="s", reason="not_a_reason")


def test_integral_float_is_accepted_for_int_field_but_fractional_is_not():
    model = build_args_model("submit_credit_limit_increase_request_7392")
    base = {"credit_card_account_id": "cc", "user_id": "u"}
    assert model(**base, requested_increase_amount=5000.0).requested_increase_amount == 5000
    with pytest.raises(ValidationError):
        model(**base, requested_increase_amount=5000.5)


def test_numbers_become_floats_like_tau2_dispatcher():
    model = build_args_model("submit_credit_limit_increase_request_7392")
    kwargs = call_kwargs(
        model(credit_card_account_id="cc", user_id="u", requested_increase_amount=5000)
    )
    assert kwargs["requested_increase_amount"] == 5000.0
    assert isinstance(kwargs["requested_increase_amount"], float)


def test_bools_strings_and_none_are_left_alone():
    model = build_args_model("set_debit_card_recurring_block_7382")
    kwargs = call_kwargs(model(card_id="c1", block_recurring=True))
    assert kwargs == {"card_id": "c1", "block_recurring": True}
    assert kwargs["block_recurring"] is True

    dispute = build_args_model("file_credit_card_transaction_dispute_4829")
    assert dispute.model_fields["partial_refund_amount"].default is None


def test_arg_schemas_match_snapshot():
    current = {
        name: spec.args_model.model_json_schema()  # type: ignore[union-attr]
        for name, spec in build_catalog().items()
    }
    rendered = json.dumps(current, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.read_text() == rendered, (
        "a tau2 tool signature changed; review it, then rerun with UPDATE_SNAPSHOTS=1"
    )
