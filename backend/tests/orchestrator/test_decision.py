import pytest
from pydantic import ValidationError

from banking_tools import Reversibility, ToolKind, get_registry
from orchestrator.decision import (
    CLARIFY_SLOT,
    CONFIRM_SLOT,
    NO_YES_NEEDED,
    Action,
    IntentDecision,
    decide,
    fingerprint,
    needs_confirmation,
)
from orchestrator.scoring import IntentScore
from orchestrator.taxonomy import TOOL_INTENT
from orchestrator.types import Intent, IntentKind, Question

IRREVERSIBLE_KINDS = [
    IntentKind.CLOSE_CARD,
    IntentKind.TRANSFER_FUNDS,
    IntentKind.DISPUTE_TRANSACTION,
    IntentKind.REPORT_FRAUD_OR_LOSS,
    IntentKind.OPEN_ACCOUNT,
    IntentKind.PAY_CREDIT_CARD,
    IntentKind.CHANGE_OR_RESET_PIN,
]
NO_YES_KINDS = [
    IntentKind.FREEZE_CARD,
    IntentKind.UNFREEZE_CARD,
    IntentKind.UPDATE_PROFILE,
    IntentKind.ACCOUNT_BALANCE_OR_HISTORY,
    IntentKind.LOOKUP_INFO,
    IntentKind.GENERAL_QUESTION,
    IntentKind.UNCLEAR_OR_GREETING,
]


def intent(kind=IntentKind.CLOSE_CARD, slots=None, missing=(), id="a"):
    return Intent(
        id=id, kind=kind, summary="s", slots=slots or {"card_id": "c1"}, missing_slots=missing
    )


def run(i, confidence=0.99, confirmed=None):
    return decide(i, IntentScore.flat(i.id, confidence), confirmed or {})


@pytest.mark.parametrize("kind", IRREVERSIBLE_KINDS)
def test_an_irreversible_intent_without_a_yes_asks_even_at_99_percent(kind):
    d = run(intent(kind))
    assert d.action is Action.ASK
    assert [(q.intent_id, q.slot) for q in d.questions] == [("a", CONFIRM_SLOT)]
    assert d.fingerprint == fingerprint(intent(kind))


@pytest.mark.parametrize("kind", IRREVERSIBLE_KINDS)
def test_an_irreversible_intent_runs_once_the_exact_thing_was_confirmed(kind):
    i = intent(kind)
    d = run(i, confirmed={"a": fingerprint(i)})
    assert d.action is Action.RUN and d.questions == ()


def test_a_yes_goes_stale_when_a_detail_changes():
    yes = fingerprint(intent(slots={"card_id": "c1", "amount": "500"}))
    changed = intent(slots={"card_id": "c1", "amount": "900"})
    d = run(changed, confirmed={"a": yes})
    assert d.action is Action.ASK
    assert d.reason == "confirmation is stale: the details changed"
    assert d.fingerprint == fingerprint(changed) != yes


def test_a_yes_for_another_intent_does_not_count():
    other = intent(id="b")
    d = run(intent(id="a"), confirmed={"b": fingerprint(other)})
    assert d.action is Action.ASK and d.reason == "awaiting the customer's yes"


def test_a_yes_for_the_same_details_under_another_kind_does_not_count():
    close_card = intent(IntentKind.CLOSE_CARD)
    close_account = intent(IntentKind.CLOSE_ACCOUNT)
    assert fingerprint(close_card) != fingerprint(close_account)
    assert run(close_account, confirmed={"a": fingerprint(close_card)}).action is Action.ASK


def test_a_yes_never_rescues_low_confidence():
    i = intent()
    yes = {"a": fingerprint(i)}
    assert run(i, 0.7, yes).action is Action.ASK
    assert run(i, 0.5, yes).action is Action.ESCALATE


READ_KINDS = [k for k in NO_YES_KINDS if k not in (IntentKind.FREEZE_CARD, IntentKind.UNFREEZE_CARD, IntentKind.UPDATE_PROFILE)]
REVERSIBLE_WRITE_KINDS = [k for k in NO_YES_KINDS if k not in READ_KINDS]


@pytest.mark.parametrize("kind", READ_KINDS)
def test_reads_run_at_85_percent_with_no_yes(kind):
    d = run(intent(kind), confidence=0.85)
    assert d.action is Action.RUN and d.questions == ()


@pytest.mark.parametrize("kind", REVERSIBLE_WRITE_KINDS)
def test_reversible_writes_run_with_no_yes_once_the_kind_is_85_percent_sure(kind):
    d = run(intent(kind), confidence=0.85)
    assert d.action is Action.RUN and d.questions == ()


@pytest.mark.parametrize("kind", REVERSIBLE_WRITE_KINDS + IRREVERSIBLE_KINDS)
def test_a_write_whose_kind_is_under_the_proceed_line_asks_instead_of_running(kind):
    d = run(intent(kind), confidence=0.8499)
    assert d.action is Action.ASK
    assert [q.slot for q in d.questions] == [CLARIFY_SLOT]


def test_only_the_kind_answer_gates_a_write_not_the_other_two():
    i = intent(IntentKind.FREEZE_CARD)
    sure_of_kind = IntentScore(
        intent_id=i.id, kind_confidence=0.97, slots_confidence=0.9, scope_confidence=0.9
    )
    unsure_of_kind = IntentScore(
        intent_id=i.id, kind_confidence=0.8, slots_confidence=1.0, scope_confidence=1.0
    )
    assert decide(i, sure_of_kind, {}).action is Action.RUN
    assert decide(i, unsure_of_kind, {}).action is Action.ASK


@pytest.mark.parametrize(
    ("confidence", "action"),
    [
        (0.99, Action.RUN),
        (0.85, Action.RUN),
        (0.8499, Action.ASK),
        (0.65, Action.ASK),
        (0.6499, Action.ESCALATE),
        (0.0, Action.ESCALATE),
    ],
)
def test_bands_decide_for_a_read(confidence, action):
    assert run(intent(IntentKind.LOOKUP_INFO), confidence).action is action


@pytest.mark.parametrize("kind", [IntentKind.FREEZE_CARD, IntentKind.CLOSE_CARD])
def test_missing_slots_are_asked_for_and_nothing_runs(kind):
    d = run(intent(kind, slots={}, missing=("card_id", "reason")), confidence=0.99)
    assert d.action is Action.ASK
    assert [q.slot for q in d.questions] == ["card_id", "reason"]
    assert d.fingerprint is None


def test_a_middling_score_with_no_named_gap_asks_a_general_question():
    d = run(intent(IntentKind.LOOKUP_INFO), confidence=0.7)
    assert [(q.intent_id, q.slot) for q in d.questions] == [("a", CLARIFY_SLOT)]


def test_a_middling_score_asks_for_the_missing_slots_when_it_knows_them():
    d = run(intent(IntentKind.FREEZE_CARD, slots={}, missing=("card_id",)), confidence=0.7)
    assert [q.slot for q in d.questions] == ["card_id"]


def test_an_unscored_intent_escalates_and_says_why():
    i = intent(IntentKind.LOOKUP_INFO)
    d = decide(i, IntentScore.unscored("a", "scorer timed out after 5.0s"), {})
    assert d.action is Action.ESCALATE
    assert d.reason == "could not be scored: scorer timed out after 5.0s"


def test_low_confidence_escalates_with_a_reason():
    assert run(intent(), 0.5).reason == "confidence below the escalation threshold"


@pytest.mark.parametrize("kind", [IntentKind.REQUEST_HUMAN, IntentKind.OTHER])
def test_kinds_that_hand_off_escalate_whatever_the_confidence(kind):
    d = run(intent(kind), confidence=1.0)
    assert d.action is Action.ESCALATE and d.reason == "this kind hands off to a human"


def test_fingerprint_ignores_slot_order_and_changes_with_the_details():
    a = intent(slots={"x": "1", "y": "2"})
    b = intent(slots={"y": "2", "x": "1"})
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(intent(slots={"x": "1", "y": "3"}))
    assert fingerprint(a) != fingerprint(intent(slots={"x": "1"}))


def test_fingerprint_does_not_carry_the_raw_details():
    secret = "4321-pin-value"
    f = fingerprint(intent(IntentKind.CHANGE_OR_RESET_PIN, slots={"new_pin": secret}))
    assert secret not in f and len(f) == 64


def test_the_no_yes_list_agrees_with_the_tool_catalog():
    registry = get_registry()
    irreversible_kinds = {
        TOOL_INTENT[name]
        for name, spec in registry.items()
        if spec.kind is ToolKind.WRITE and spec.reversibility is Reversibility.IRREVERSIBLE
    }
    assert not (NO_YES_NEEDED & irreversible_kinds), "a kind that can do something irreversible skips the yes"


def test_needs_confirmation_is_the_opposite_of_the_no_yes_list():
    assert all(not needs_confirmation(k) for k in NO_YES_NEEDED)
    assert all(needs_confirmation(k) for k in IRREVERSIBLE_KINDS)
    assert needs_confirmation(IntentKind.REPORT_FRAUD_OR_LOSS)


def test_decision_shape_is_enforced():
    q = (Question(intent_id="a", slot="s"),)
    with pytest.raises(ValidationError):
        IntentDecision(intent_id="a", action=Action.ASK, reason="r")
    with pytest.raises(ValidationError):
        IntentDecision(intent_id="a", action=Action.RUN, reason="r", questions=q)
    with pytest.raises(ValidationError):
        IntentDecision(intent_id="a", action=Action.ESCALATE, reason="")


def test_decision_is_frozen():
    d = run(intent(IntentKind.LOOKUP_INFO))
    with pytest.raises(ValidationError):
        d.action = Action.ESCALATE


@pytest.mark.parametrize("kind", [IntentKind.FREEZE_CARD, IntentKind.CLOSE_CARD])
def test_a_request_the_scorer_understands_asks_for_missing_details_instead_of_escalating(kind):
    i = intent(kind, slots={}, missing=("card_id",))
    # What Jev returns for "freeze my card" with no card named: sure of the request,
    # slots incomplete (scored 0 by code).
    score = IntentScore(intent_id=i.id, kind_confidence=0.99, slots_confidence=0.0, scope_confidence=0.99)
    d = decide(i, score, {})
    assert d.action is Action.ASK
    assert [q.slot for q in d.questions] == ["card_id"]


def test_missing_details_do_not_rescue_a_request_the_scorer_does_not_understand():
    i = intent(IntentKind.FREEZE_CARD, slots={}, missing=("card_id",))
    unsure_kind = IntentScore(intent_id=i.id, kind_confidence=0.5, slots_confidence=0.0, scope_confidence=0.99)
    unsure_scope = IntentScore(intent_id=i.id, kind_confidence=0.99, slots_confidence=0.0, scope_confidence=0.4)
    assert decide(i, unsure_kind, {}).action is Action.ESCALATE
    assert decide(i, unsure_scope, {}).action is Action.ESCALATE


def test_the_write_kind_bar_is_below_the_proceed_line_so_it_no_longer_adds_a_gate_of_its_own():
    """Lowered from 0.95 to 0.80 after a hedged lost-wallet report scored 0.92 and was asked about in a loop."""
    from orchestrator.bands import PROCEED_AT, WRITE_KIND_ASK_AT

    assert WRITE_KIND_ASK_AT == 0.80 < PROCEED_AT
    d = run(intent(IntentKind.REPORT_FRAUD_OR_LOSS), confidence=0.92)
    assert d.action is not Action.ASK or d.questions[0].slot != CLARIFY_SLOT
