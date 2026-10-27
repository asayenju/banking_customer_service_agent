import json
import os
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from orchestrator.types import (
    AskCustomer,
    Conflict,
    Decision,
    Decomposition,
    HandOff,
    Intent,
    IntentKind,
    Proceed,
    Question,
)

SNAPSHOT = Path(__file__).parent / "snapshots" / "orchestrator_schemas.json"


def intent(id="i1", kind=IntentKind.FREEZE_CARD, depends_on=()):
    return Intent(id=id, kind=kind, summary="freeze my card", depends_on=depends_on)


def test_the_closed_list_includes_the_catch_alls():
    assert IntentKind.OTHER.value == "OTHER"
    assert IntentKind.UNCLEAR_OR_GREETING.value == "UNCLEAR_OR_GREETING"
    assert IntentKind.FEE_OR_INTEREST_ADJUSTMENT.value == "FEE_OR_INTEREST_ADJUSTMENT"
    assert IntentKind.CARD_CONTROLS.value == "CARD_CONTROLS"
    assert IntentKind.GENERAL_QUESTION.value == "GENERAL_QUESTION"


def test_intent_defaults_are_empty():
    i = intent()
    assert (i.slots, i.missing_slots, i.depends_on, i.source_span) == ({}, (), (), None)


def test_intent_is_frozen():
    with pytest.raises(ValidationError):
        intent().summary = "changed"


def test_intent_rejects_extra_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Intent(id="i1", kind=IntentKind.OTHER, summary="s", surprise="x")


def test_unknown_kind_is_rejected_from_model_json():
    raw = '{"id": "i1", "kind": "TELEPORT_MONEY", "summary": "s"}'
    with pytest.raises(ValidationError, match="kind"):
        Intent.model_validate_json(raw)


def test_known_kind_is_accepted_from_model_json():
    raw = '{"id": "i1", "kind": "FREEZE_CARD", "summary": "s"}'
    assert Intent.model_validate_json(raw).kind is IntentKind.FREEZE_CARD


def test_kind_must_be_the_enum_in_python_mode():
    with pytest.raises(ValidationError):
        Intent(id="i1", kind="NOT_A_KIND", summary="s")


def test_empty_id_is_rejected():
    with pytest.raises(ValidationError):
        intent(id="")


def test_decomposition_keeps_valid_dependencies():
    d = Decomposition(
        intents=(
            intent("a", IntentKind.CLOSE_CARD),
            intent("b", IntentKind.TRANSFER_FUNDS, depends_on=("a",)),
        )
    )
    assert d.intents[1].depends_on == ("a",)
    assert d.conflicts == () and d.flags == () and d.schema_version == 1


def test_duplicate_intent_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate intent id 'a'"):
        Decomposition(intents=(intent("a"), intent("a")))


def test_dependency_on_unknown_id_is_rejected():
    with pytest.raises(ValidationError, match="'b' depends on unknown intent 'zzz'"):
        Decomposition(intents=(intent("a"), intent("b", depends_on=("zzz",))))


def test_decomposition_is_frozen():
    with pytest.raises(ValidationError):
        Decomposition(intents=()).schema_version = 2


def test_conflict_names_the_intents_involved():
    c = Conflict(intent_ids=("a", "b"), reason="freeze and unfreeze the same card")
    assert c.intent_ids == ("a", "b")


def test_decision_variants_parse_by_their_tag():
    adapter = TypeAdapter(Decision)
    proceed = adapter.validate_json('{"kind": "proceed", "intent_ids": ["a"]}')
    ask = adapter.validate_json(
        '{"kind": "ask_customer",'
        ' "questions": [{"intent_id": "a", "slot": "card_id", "text": "Which card?"}]}'
    )
    hand_off = adapter.validate_json('{"kind": "hand_off", "reason": "asked for a human"}')
    assert isinstance(proceed, Proceed)
    assert isinstance(ask, AskCustomer) and ask.questions[0].slot == "card_id"
    assert isinstance(hand_off, HandOff) and hand_off.intent_ids == ()


def test_decision_with_an_unknown_tag_is_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(Decision).validate_json('{"kind": "yolo"}')


def test_proceed_and_ask_need_at_least_one_target():
    with pytest.raises(ValidationError):
        Proceed(intent_ids=())
    with pytest.raises(ValidationError):
        AskCustomer(questions=())


def test_question_ties_a_slot_to_an_intent():
    q = Question(intent_id="a", slot="amount", text="How much?")
    assert (q.intent_id, q.slot) == ("a", "amount")


def test_schemas_match_snapshot():
    current = {
        "Intent": Intent.model_json_schema(),
        "Decomposition": Decomposition.model_json_schema(),
        "Decision": TypeAdapter(Decision).json_schema(),
    }
    rendered = json.dumps(current, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.read_text() == rendered, (
        "an orchestrator type changed; review it, then rerun with UPDATE_SNAPSHOTS=1"
    )
