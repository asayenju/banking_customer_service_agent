"""A button next to a line means what the server decided when it offered it: no model reads the press."""

from agent.recipes.direct import DirectRunner  # noqa: F401  (the recipe that lists charges)
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.mocks import ScriptedRoute
from orchestrator.repair import RepairLadder
from orchestrator.types import Choice, Intent, IntentKind as K, Question
from scenario_kit import GOOD_CLAIMS, I, S, World, interp

LINE = "10/21/2025  $278.94  JetBlue Airways (completed)"
OFFER = Choice(
    id="dispute:txn_1",
    label=LINE,
    action="Dispute",
    kind=K.DISPUTE_TRANSACTION,
    summary="dispute the $278.94 JetBlue Airways charge on 10/21/2025",
    slots={"transaction_id": "txn_1", "merchant": "JetBlue Airways", "amount": "$278.94", "date": "10/21/2025"},
)


class Lister:
    """A read route that lists a charge with a Dispute button, like the transaction history."""

    def __init__(self):
        self.seen = []

    async def run(self, intent, ctx):
        self.seen.append((intent.kind, dict(intent.slots)))
        if intent.kind is K.DISPUTE_TRANSACTION:
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="ok")
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text=f"Recent charges:\n- {LINE}", choices=(OFFER,))


def listing_world():
    lister = Lister()
    scores = [[S("a", 0.99)]] * 4
    w = World([interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY, {}), claims=GOOD_CLAIMS), interp()], scores, direct=lister, sub=lister)
    return w, lister


def test_a_list_offers_a_button_per_line_and_the_reply_carries_it():
    w, _ = listing_world()
    result = w.turn("show me my recent transactions")
    assert [(c.id, c.action, c.label) for c in result.reply.choices] == [("dispute:txn_1", "Dispute", LINE)]
    assert LINE in result.reply.text, "the line is in the text, so the page can put the button beside it"


def test_pressing_a_button_makes_that_request_without_asking_the_decomposer_or_the_scorer():
    w, lister = listing_world()
    w.turn("show me my recent transactions")
    asked_before = len(w.scorer.asked)
    decomposer_turns = len(w.decomposer.histories)
    result = w.turn("Dispute: " + LINE, pick="dispute:txn_1")
    assert len(w.decomposer.histories) == decomposer_turns, "nothing read the press: no decomposer call"
    assert len(w.scorer.asked) == asked_before, "nothing scored it: the customer chose it"
    made = w.orch._active["p1-i1"]
    assert made.kind is K.DISPUTE_TRANSACTION and made.slots["transaction_id"] == "txn_1" and made.slots["amount"] == "$278.94"
    assert w.orch._scores["p1-i1"].confidence == 1.0
    # A dispute is a write: the press is a clear choice, but the customer's yes to what will be done is still asked.
    assert [d.action.value for d in result.decisions] == ["ask"] and result.reply.questions[0].slot == "confirmation"


def test_a_press_can_only_mean_an_offer_the_conversation_made():
    w, lister = listing_world()
    w.turn("show me my recent transactions")
    w.turn("Dispute: something else", pick="dispute:txn_not_offered")  # an id nobody offered is just words
    assert all(kind is not K.DISPUTE_TRANSACTION for kind, _ in lister.seen)


def test_a_pressed_request_still_has_to_pass_the_identity_check():
    lister = Lister()
    w = World([interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY, {}))], [[S("a", 0.99)]], direct=lister, sub=lister)
    w.orch._offers[OFFER.id] = OFFER  # offered, but this caller was never verified
    result = w.turn("Dispute: " + LINE, pick=OFFER.id)
    assert all(kind is not K.DISPUTE_TRANSACTION for kind, _ in lister.seen)
    assert result.reply.questions and result.reply.questions[0].slot == "identity_verification"


# ------------------------------------------------------------ the other fixes from the same conversation

def test_a_reply_that_becomes_a_new_request_closes_the_weak_reading_it_was_clarifying():
    first = I("a", K.ACCOUNT_BALANCE_OR_HISTORY, {"amount": "$278.94", "merchant": "JetBlue"})
    dispute = I("d", K.DISPUTE_TRANSACTION, {"merchant": "JetBlue"})
    w = World([interp(first, claims=GOOD_CLAIMS), interp(dispute)], [[S("a", 0.6)], [S("d", 0.95)]], direct=ScriptedRoute(), sub=ScriptedRoute())
    w.turn("JetBlue $278.94")
    assert "a" in w.orch.active_ids
    w.turn("it's not higher, I never made that purchase")
    assert "a" not in w.orch.active_ids, "the clarification about the weak reading is closed"
    assert "t2-d" in w.orch.active_ids or "d" in w.orch.active_ids


def test_a_clarification_about_a_write_stays_open_beside_a_new_request():
    first = I("c", K.CLOSE_CARD, {})
    other = I("f", K.FREEZE_CARD, {"card_id": "c1"})
    w = World([interp(first, claims=GOOD_CLAIMS), interp(other)], [[S("c", 0.6)], [S("f", 0.95)]], direct=ScriptedRoute(), sub=ScriptedRoute())
    w.turn("close it")
    w.turn("also freeze my card c1")
    assert "c" in w.orch.active_ids


def test_a_clarification_never_quotes_a_models_note_about_the_customer():
    from orchestrator.reply import wording

    note = "The customer is referencing a charge of $278.94 from JetBlue Airways, likely in relation to their balance inquiry"
    intent = Intent(id="a", kind=K.ACCOUNT_BALANCE_OR_HISTORY, summary=note)
    said = wording({"a": intent}, Question(intent_id="a", slot="clarification"))
    assert "The customer" not in said and "likely" not in said and "see your balances or recent transactions" in said
    short = Intent(id="b", kind=K.FEE_OR_INTEREST_ADJUSTMENT, summary="sort out my fees")
    assert "sort out my fees" in wording({"b": short}, Question(intent_id="b", slot="clarification"))


def test_a_repeated_clarification_is_reworded_as_a_question_not_as_which_you_meant():
    from orchestrator.decision import Action, IntentDecision

    intent = Intent(id="a", kind=K.ACCOUNT_BALANCE_OR_HISTORY, summary="s")
    ladder = RepairLadder()
    texts = []
    for _ in range(3):
        decision = IntentDecision(intent_id="a", action=Action.ASK, reason="r", questions=(Question(intent_id="a", slot="clarification", text="To be sure: x?"),))
        (fixed,) = ladder.apply({"a": intent}, [decision], lambda i, q: "unused")
        texts.append(fixed.questions[0].text)
    assert texts[0] == "To be sure: x?"
    assert "didn't catch which you meant" not in texts[1] and "own words" in texts[1]
    assert "person" in texts[2]
