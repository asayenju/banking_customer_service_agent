import asyncio
import json

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, assistant_message, tool_message
from agent.toolbox import (
    KEPT_PRIVATE,
    ScopedToolBox,
    SecretVault,
    ToolExecution,
    is_strict_valid,
    tool_schema,
)
from banking_tools import BankingSession, Fidelity, ToolKind, get_registry
from orchestrator.context import scoped

CARD = "dbc_lj82d4f1a9_bluest"
USER = "lj82d4f1a9"
ACCOUNT = "chk_lj82d4f1a9"
PIN, CVV = "7351", "739"
REGISTRY = get_registry()
ALL = tuple(REGISTRY)


def call(name, **arguments):
    return ToolCall(id="c", name=name, arguments_json=json.dumps(arguments))


def toolbox(allowed=ALL, vault=None, user=USER, session=None):
    shared = session or BankingSession(fidelity=Fidelity.TAU2)
    return ScopedToolBox(scoped(shared)(_request("i")), user_id=user, allowed_tools=allowed, vault=vault), shared


def _request(intent_id):
    from orchestrator.context import SessionRequest

    return SessionRequest(intent_id)


# ------------------------------------------------------------------ schemas

@pytest.mark.parametrize("name", ALL)
def test_every_catalog_tool_yields_a_schema_a_strict_mode_provider_accepts(name):
    schema = tool_schema(REGISTRY[name])
    assert schema.name == name and schema.description
    assert is_strict_valid(schema.parameters), schema.parameters


@pytest.mark.parametrize("name", ALL)
def test_no_schema_lets_the_model_choose_the_customer(name):
    properties = tool_schema(REGISTRY[name]).parameters["properties"]
    assert "user_id" not in properties


def test_pin_and_cvv_fields_take_a_reference_never_a_value():
    secret_fields = [
        (n, k)
        for n, spec in REGISTRY.items()
        for k in tool_schema(spec).parameters["properties"]
        if k in {"pin", "new_pin", "current_pin", "cvv"}
    ]
    assert secret_fields, "the catalog has tools that take a PIN or CVV"
    for name, key in secret_fields:
        prop = tool_schema(REGISTRY[name]).parameters["properties"][key]
        assert prop["type"] == "string" and "{{secret:" in prop["description"]


def test_descriptions_come_from_the_real_docstring_not_the_knowledge_base_names():
    schema = tool_schema(REGISTRY["close_debit_card_4721"])
    assert schema.parameters["properties"]["reason"]["description"].startswith("Reason for closing")
    assert "permanently" in schema.description


def test_optional_fields_become_required_nullable_ones_for_strict_mode():
    schema = tool_schema(REGISTRY["order_debit_card_5739"]).parameters
    assert "excess_replacement_fee" in schema["required"]
    assert {"type": "null"} in schema["properties"]["excess_replacement_fee"]["anyOf"]
    assert "default" not in json.dumps(schema)


def test_the_strictness_check_rejects_what_a_provider_would():
    assert not is_strict_valid({"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False})
    assert not is_strict_valid({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]})
    assert not is_strict_valid({"type": "object", "properties": {"a": {"type": "string", "default": "x"}}, "required": ["a"], "additionalProperties": False})
    assert is_strict_valid({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False})


# ------------------------------------------------------------------ offering and running

def test_only_the_allowed_tools_are_offered_and_nothing_else_can_be_called():
    box, shared = toolbox(allowed={"get_all_user_accounts_by_user_id_3847"})
    assert [s.name for s in box.schemas()] == ["get_all_user_accounts_by_user_id_3847"]
    result = box.execute(call("freeze_debit_card_3892", card_id=CARD))
    assert not result.ok and "not available" in result.text and shared.audit_log == ()


def test_an_unknown_tool_in_the_allow_list_is_a_setup_error_not_a_silent_gap():
    with pytest.raises(ValueError, match="not in the tool catalog"):
        toolbox(allowed={"made_up_tool_9999"})


def test_the_verified_customer_is_filled_in_and_a_different_one_cannot_be_named():
    box, _ = toolbox(allowed={"get_all_user_accounts_by_user_id_3847"})
    mine = box.execute(call("get_all_user_accounts_by_user_id_3847"))
    assert mine.ok and ACCOUNT in mine.text
    other = box.execute(call("get_all_user_accounts_by_user_id_3847", user_id="123"))
    assert other.text == mine.text, "the model's user_id is ignored"


def test_a_tool_that_needs_a_customer_refuses_when_nobody_is_verified():
    box, _ = toolbox(allowed={"get_all_user_accounts_by_user_id_3847"}, user=None)
    result = box.execute(call("get_all_user_accounts_by_user_id_3847"))
    assert not result.ok and "no verified customer" in result.text


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"text"'])
def test_arguments_that_are_not_a_json_object_come_back_as_an_error(raw):
    box, _ = toolbox()
    result = box.execute(ToolCall(id="c", name="freeze_debit_card_3892", arguments_json=raw))
    assert not result.ok and result.text.startswith("Error:")


def test_a_write_runs_and_is_audited_for_the_intent():
    box, shared = toolbox()
    assert box.execute(call("freeze_debit_card_3892", card_id=CARD)).ok
    assert [e.tool for e in shared.audit_for("i")] == ["freeze_debit_card_3892"]


def test_a_long_result_is_trimmed():
    box = ScopedToolBox(
        scoped(BankingSession(fidelity=Fidelity.TAU2))(_request("i")),
        user_id=USER,
        allowed_tools={"get_bank_account_transactions_9173"},
        max_result_chars=200,
    )
    result = box.execute(call("get_bank_account_transactions_9173", account_id=ACCOUNT))
    assert result.ok and len(result.text) <= 200


# ------------------------------------------------------------------ secrets

def test_the_vault_swaps_a_pin_for_a_reference_and_hands_the_value_back_only_to_code():
    vault = SecretVault()
    cleaned = vault.capture("Please set my new pin to 7351 thanks")
    assert PIN not in cleaned and "{{secret:pin}}" in cleaned
    assert vault.resolve("{{secret:pin}}") == PIN and vault.resolve("{{secret:nope}}") is None
    assert vault.resolve("7351") is None, "a bare value is not a reference"


def test_two_different_secrets_of_one_kind_get_two_references():
    vault = SecretVault()
    cleaned = vault.capture("my pin is 7351 and the old pin is 2468")
    assert "7351" not in cleaned and "2468" not in cleaned
    assert {vault.resolve(r) for r in ("{{secret:pin}}", "{{secret:pin_b}}")} == {"7351", "2468"}


def test_the_vault_scrubs_values_even_when_they_appear_without_a_label():
    vault = SecretVault()
    vault.store("new_pin", "7351")
    assert vault.scrub("the code 7351 was rejected") == f"the code {KEPT_PRIVATE} was rejected"
    assert vault.scrub("cvv: 739 and card 4111 1111 1111 1111") .count(KEPT_PRIVATE) == 2


def test_a_secret_field_given_a_real_value_is_refused_and_never_echoed():
    box, shared = toolbox()
    result = box.execute(
        call("reset_debit_card_pin_6284", card_id=CARD, last_4_digits="4821", new_pin=PIN)
    )
    assert not result.ok and "secret reference" in result.text and PIN not in result.text
    assert shared.audit_log == ()


def test_a_reference_to_a_secret_that_does_not_exist_is_refused():
    box, _ = toolbox()
    result = box.execute(
        call("reset_debit_card_pin_6284", card_id=CARD, last_4_digits="4821", new_pin="{{secret:pin}}")
    )
    assert not result.ok and "secret reference" in result.text


def test_a_pin_reference_is_resolved_at_call_time_and_the_audit_keeps_it_redacted():
    vault = SecretVault()
    ref = vault.store("pin", PIN)
    box, shared = toolbox(vault=vault)
    result = box.execute(call("reset_debit_card_pin_6284", card_id=CARD, last_4_digits="4821", new_pin=ref))
    assert result.ok, result.text
    (entry,) = shared.audit_log
    assert entry.arguments["new_pin"] == "***" and PIN not in entry.model_dump_json()
    assert PIN not in result.text


def test_a_validation_error_that_would_echo_the_pin_is_scrubbed():
    vault = SecretVault()
    ref = vault.store("pin", PIN)
    box, _ = toolbox(vault=vault)
    # card_id missing: pydantic's message may quote the input it saw
    result = box.execute(call("reset_debit_card_pin_6284", last_4_digits=PIN, new_pin=ref))
    assert not result.ok and PIN not in result.text


def test_cvv_in_a_real_read_result_never_reaches_the_model():
    box, _ = toolbox(allowed={"get_debit_cards_by_account_id_7823"})
    result = box.execute(call("get_debit_cards_by_account_id_7823", account_id=ACCOUNT))
    assert result.ok and CVV not in result.text and "cvv" in result.text.lower()
    assert "4821" in result.text, "the last four digits are not secret and stay"


def test_sentinel_pin_and_cvv_never_reach_the_model_through_a_whole_exchange():
    """The customer types a PIN, the model asks for the CVV-bearing read, then resets the PIN."""
    vault = SecretVault()
    customer = vault.capture("Reset my pin, the new pin is 7351. Card ends 4821.")
    box, shared = toolbox(allowed={"get_debit_cards_by_account_id_7823", "reset_debit_card_pin_6284"}, vault=vault)
    ref = "{{secret:pin}}"
    script = [
        AssistantTurn(tool_calls=(call("get_debit_cards_by_account_id_7823", account_id=ACCOUNT),)),
        AssistantTurn(tool_calls=(call("reset_debit_card_pin_6284", card_id=CARD, last_4_digits="4821", new_pin=ref),)),
        AssistantTurn(text="Your PIN was reset."),
    ]
    fake = FakeToolLLM(script)
    history = [{"role": "user", "content": customer}]

    async def exchange():
        while True:
            turn = await fake.chat(history, tools=box.schemas())
            history.append(assistant_message(turn))
            if not turn.tool_calls:
                return turn
            for tool_call in turn.tool_calls:
                history.append(tool_message(tool_call.id, box.execute(tool_call).text))

    final = asyncio.run(exchange())
    assert final.text == "Your PIN was reset."
    sent = json.dumps(fake.calls)
    assert PIN not in sent and CVV not in sent, "a secret reached the model"
    assert "{{secret:pin}}" in sent, "the model was told which reference to use"
    assert [e.tool for e in shared.audit_log] == ["reset_debit_card_pin_6284"]
    assert PIN not in json.dumps([e.model_dump(mode="json") for e in shared.audit_log])


def test_the_toolbox_never_raises_whatever_the_model_sends():
    box, _ = toolbox()
    for raw in ("{}", '{"card_id": 5}', '{"unexpected": 1}', "null", ""):
        for name in ("freeze_debit_card_3892", "close_debit_card_4721", "no_such_tool"):
            result = box.execute(ToolCall(id="c", name=name, arguments_json=raw))
            assert isinstance(result, ToolExecution) and isinstance(result.text, str)


def test_the_catalog_has_write_tools_to_protect():
    assert any(spec.kind is ToolKind.WRITE for spec in REGISTRY.values())


# ------------------------------------------------------------------ a PIN typed as bare digits

@pytest.mark.parametrize(
    "text",
    ["Set it to 2589", "2589", "ok, 2589 then", "how about 5739?", "yes 7351!"],
)
def test_when_a_pin_was_just_asked_for_bare_digits_are_the_pin(text):
    vault = SecretVault()
    cleaned = vault.capture(text, expecting_secret=True)
    assert not any(ch.isdigit() for ch in cleaned), cleaned
    assert vault.resolve("{{secret:pin}}") in ("2589", "5739", "7351")


def test_without_having_asked_for_a_pin_a_bare_number_is_left_alone():
    """The cue words are the only evidence then, and "Set it to 2589" has none."""
    for text in ("Set it to 2589", "my card ends 4821", "I paid on 2025"):
        assert SecretVault().capture(text) == text


@pytest.mark.parametrize(
    "text",
    ["I paid $1500 on 11/14/2025", "the fee is 12.50", "call 555-0100", "my balance is 1,250.75", "in 2025", "ref ABC1234", "30% off", "10:45 am"],
)
def test_even_when_a_pin_is_expected_amounts_dates_decimals_and_words_are_not_taken_for_one(text):
    vault = SecretVault()
    cleaned = vault.capture(text, expecting_secret=True)
    if text == "in 2025":
        assert "2025" not in cleaned, "a bare four digits is exactly what a PIN looks like"
    else:
        assert cleaned == text, cleaned


def test_a_cue_based_capture_still_wins_and_two_secrets_get_two_references():
    vault = SecretVault()
    cleaned = vault.capture("my new pin is 7351 and the old one 2468", expecting_secret=True)
    assert "7351" not in cleaned and "2468" not in cleaned
    assert {vault.resolve(r) for r in ("{{secret:pin}}", "{{secret:pin_b}}")} == {"7351", "2468"}


def test_hide_replaces_known_values_wherever_they_appear():
    vault = SecretVault()
    out = vault.hide('Sure, set it to 2589. Then maybe 5678 for the other.', {"pin_a": "2589", "pin_b": "5678"})
    assert "2589" not in out and "5678" not in out
    assert vault.resolve("{{secret:pin_a}}") == "2589" and vault.resolve("{{secret:pin_b}}") == "5678"
    assert "{{secret:pin_a}}" in out and "{{secret:pin_b}}" in out
