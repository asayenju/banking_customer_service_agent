import pytest
from pydantic import ValidationError
from scenario_kit import GOOD_CLAIMS, I, S, World, interp, trace_kinds

from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.orchestrator import MAX_ASKS
from orchestrator.trace import TraceKind as T
from orchestrator.types import IntentKind as K
from orchestrator.types import Question

QUESTION = "what are your fees"


def ask(intent, *slots):
    return RouteOutcome(
        intent_id=intent.id,
        status=OutcomeStatus.NEEDS_INFO,
        questions=tuple(Question(intent_id=intent.id, slot=s) for s in slots),
    )


def done(intent):
    return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text=f"answered {intent.id}")


class NeedsTopic:
    """Asks for a topic until the intent has one."""

    def __init__(self):
        self.calls = []

    async def run(self, intent, ctx):
        self.calls.append(dict(intent.slots))
        return done(intent) if intent.slots.get("topic") else ask(intent, "topic")


class AsksOnlyForA:
    """Intent a needs a topic; anything else is answered at once."""

    def __init__(self):
        self.calls = []

    async def run(self, intent, ctx):
        self.calls.append(intent.id)
        if intent.id == "a" and not intent.slots.get("topic"):
            return ask(intent, "topic")
        return done(intent)


class AlwaysAsks:
    def __init__(self, slot="topic"):
        self.slot, self.calls = slot, 0

    async def run(self, intent, ctx):
        self.calls += 1
        return ask(intent, self.slot)


def general(id="a", **slots):
    return I(id, K.GENERAL_QUESTION, slots)


def test_a_needs_info_outcome_must_carry_questions_and_no_error():
    with pytest.raises(ValidationError):
        RouteOutcome(intent_id="a", status=OutcomeStatus.NEEDS_INFO)
    with pytest.raises(ValidationError):
        RouteOutcome(intent_id="a", status=OutcomeStatus.NEEDS_INFO, error="x", questions=(Question(intent_id="a", slot="s"),))
    with pytest.raises(ValidationError):
        RouteOutcome(intent_id="a", status=OutcomeStatus.DONE, questions=(Question(intent_id="a", slot="s"),))
    with pytest.raises(ValidationError):  # a route may only ask about its own intent
        RouteOutcome(intent_id="a", status=OutcomeStatus.NEEDS_INFO, questions=(Question(intent_id="other", slot="s"),))


def test_a_route_that_needs_more_asks_the_customer_and_keeps_the_intent_open():
    route = NeedsTopic()
    w = World([interp(general())], [[S("a", 0.99)]], sub=route)
    r = w.turn(QUESTION)
    assert [d.action.value for d in r.decisions] == ["ask"]
    assert [q.slot for q in r.reply.questions] == ["topic"]
    assert "What is the topic" in r.reply.text
    assert "problem" not in r.reply.text and r.reply.failed_intent_ids == ()
    assert w.orch.active_ids == ("a",) and w.orch.runner_ask_counts == {"a": 1}
    asked = [e for e in r.trace.events if e.kind is T.ASKED and e.intent_id == "a"]
    assert asked and asked[0].detail == "topic"


def test_the_answer_runs_the_route_again_with_the_new_details_and_finishes():
    route = NeedsTopic()
    w = World(
        [interp(general()), interp(general(topic="fees"))],
        [[S("a", 0.99)], [S("a", 0.99)]],
        sub=route,
    )
    w.turn(QUESTION)
    r = w.turn("fees please")
    assert route.calls == [{}, {"topic": "fees"}]
    assert [o.status for o in r.outcomes] == [OutcomeStatus.DONE]
    assert w.orch.active_ids == () and w.orch.done_ids == ("a",) and w.orch.runner_ask_counts == {}
    assert w.scorer.asked == [["a"], ["a"]], "the answered intent is scored again"


def test_the_fourth_question_from_a_route_escalates_even_when_each_answer_changes_the_details():
    route = AlwaysAsks()
    answers = [interp(general())] + [interp(general(topic=f"try {n}")) for n in range(1, MAX_ASKS + 1)]
    w = World(answers, [[S("a", 0.99)]] * (MAX_ASKS + 1), sub=route)
    results = [w.turn(QUESTION)] + [w.turn(f"my answer {n}") for n in range(1, MAX_ASKS + 1)]
    assert [r.decisions[0].action.value for r in results] == ["ask"] * MAX_ASKS + ["escalate"]
    assert route.calls == MAX_ASKS + 1
    assert "human agent" in results[-1].reply.text
    assert w.orch.active_ids == () and w.orch.runner_ask_counts == {}
    assert T.ESCALATED in trace_kinds(results[-1], "a")


def test_a_reply_to_the_one_question_on_screen_is_taken_as_its_answer_even_if_the_decomposer_found_nothing():
    route = NeedsTopic()
    w = World([interp(general()), interp()], [[S("a", 0.99)]], sub=route)
    w.turn(QUESTION)
    r = w.turn("the Blue Account fees")
    assert route.calls[0] == {} and route.calls[1]["topic"] == "the Blue Account fees", "the words became the detail asked for"
    assert "Customer: the Blue Account fees" in route.calls[1]["conversation_so_far"], "and the exchange is kept in words"
    assert [o.status for o in r.outcomes] == [OutcomeStatus.DONE]


def test_an_answer_the_decomposer_mistook_for_a_new_read_is_not_made_into_a_request():
    route = NeedsTopic()
    spurious = I("t2", K.ACCOUNT_BALANCE_OR_HISTORY, summary="noticed on the 10th")
    w = World([interp(general()), interp(spurious)], [[S("a", 0.99)]], sub=route)
    w.turn(QUESTION)
    r = w.turn("November 6th, noticed on the 10th")
    assert route.calls[-1]["topic"] == "November 6th, noticed on the 10th"
    assert w.orch.active_ids == () and [o.intent_id for o in r.outcomes] == ["a"], "no second intent was made"


def test_a_real_new_request_in_the_reply_is_kept_and_the_question_is_not_answered_with_it():
    route = NeedsTopic()
    freeze = I("t2", K.FREEZE_CARD, {"card_id": "c1"})
    w = World([interp(general()), interp(freeze)], [[S("a", 0.99)], [S("t2", 0.99)]], sub=route)
    w.turn(QUESTION)
    w.turn("actually freeze card c1 now")
    assert all("topic" not in call for call in route.calls[1:]), "the reply was a new request, not the topic"


def test_a_repeated_non_answer_still_runs_out_of_questions_and_escalates():
    route = AlwaysAsks()
    w = World([interp(general())] + [interp()] * MAX_ASKS, [[S("a", 0.99)]] * (MAX_ASKS + 1), sub=route)
    results = [w.turn(QUESTION)] + [w.turn(f"hmm {n}") for n in range(MAX_ASKS)]
    assert [r.decisions[0].action.value for r in results] == ["ask"] * MAX_ASKS + ["escalate"]


def test_answering_one_intent_leaves_the_other_intents_counters_alone():
    route = NeedsTopic()
    w = World(
        [interp(general("a"), general("b")), interp(general("a", topic="fees"))],
        [[S("a", 0.99), S("b", 0.99)], [S("a", 0.99)]],
        sub=route,
    )
    first = w.turn(QUESTION)
    assert w.orch.runner_ask_counts == {"a": 1, "b": 1}
    second = w.turn("fees")
    assert [o.intent_id for o in second.outcomes if o.status is OutcomeStatus.DONE] == ["a"]
    assert w.orch.runner_ask_counts == {"b": 1}, "b's count survives, a's was cleared by finishing"
    assert w.orch.ask_counts == {"b": 1}
    assert {d.intent_id: d.action.value for d in second.decisions} == {"a": "run", "b": "ask"}


def test_a_double_submitted_message_does_not_ask_or_count_twice():
    route = AlwaysAsks()
    w = World([interp(general())], [[S("a", 0.99)]], sub=route)
    first = w.turn(QUESTION, message_id="same")
    again = w.turn(QUESTION, message_id="same")
    assert again is first
    assert route.calls == 1 and w.orch.runner_ask_counts == {"a": 1}


def test_what_the_customer_typed_in_an_answer_never_reaches_the_trace_or_the_question_store():
    route = AlwaysAsks(slot="new_pin")
    w = World(
        [interp(general("a")), interp(general("a", new_pin="4821"))],
        [[S("a", 0.99)], [S("a", 0.99)]],
        sub=route,
    )
    w.turn("I want a new pin")
    second = w.turn("it is 4821")
    for visible in (repr(second.trace), repr(second.reply), repr(w.orch._pending_route_questions), repr(w.orch.runner_ask_counts)):
        assert "4821" not in visible


def after_a():
    return I("c", K.GENERAL_QUESTION, depends_on=("a",))


def test_something_that_depends_on_an_intent_that_is_asking_waits_instead_of_failing():
    route = AsksOnlyForA()
    w = World(
        [interp(general("a"), after_a()), interp(general("a", topic="fees"))],
        [[S("a", 0.99), S("c", 0.99)], [S("a", 0.99)]],
        sub=route,
    )
    first = w.turn(QUESTION)
    assert route.calls == ["a"], "the dependent did not run"
    assert "Still to do" in first.reply.text and first.reply.failed_intent_ids == ()
    assert set(w.orch.active_ids) == {"a", "c"}
    second = w.turn("fees")
    assert route.calls == ["a", "a", "c"], "once a finished, c ran"
    assert {o.intent_id for o in second.outcomes if o.status is OutcomeStatus.DONE} == {"a", "c"}


def test_when_the_asking_intent_is_escalated_the_ones_waiting_on_it_are_not_left_open():
    route = AlwaysAsks()
    w = World(
        [interp(general("a"), after_a())] + [interp(general("a", topic=f"t{n}")) for n in range(MAX_ASKS)],
        [[S("a", 0.99), S("c", 0.99)]] + [[S("a", 0.99)]] * MAX_ASKS,
        sub=route,
    )
    w.turn(QUESTION)
    for n in range(MAX_ASKS):
        w.turn(f"answer {n}")
    assert w.orch.active_ids == (), "neither the asker nor its dependent waits forever"


class NeedsThreeThings:
    """Wants a reason, then a date, then an amount, all missing at first (the route knows the whole list)."""

    ORDER = ("reason", "date", "amount", "merchant", "card")

    def __init__(self):
        self.calls = []

    async def run(self, intent, ctx):
        self.calls.append(dict(intent.slots))
        missing = [s for s in self.ORDER if not intent.slots.get(s)]
        return done(intent) if not missing else ask(intent, *missing)


def test_a_route_that_needs_several_details_is_asked_for_one_at_a_time_and_never_escalates_for_that():
    route = NeedsThreeThings()
    script = [interp(general())]
    given: dict[str, str] = {}
    for slot in NeedsThreeThings.ORDER:
        given = {**given, slot: "x"}
        script.append(interp(general(**given)))
    w = World(script, [[S("a", 0.99)]] * (len(script)), sub=route)
    first = w.turn(QUESTION)
    assert [q.slot for q in first.reply.questions] == ["reason"], "not all five at once"
    asked = ["reason"]
    for number, slot in enumerate(NeedsThreeThings.ORDER, start=1):
        result = w.turn(f"answer {number}")
        if slot != NeedsThreeThings.ORDER[-1]:
            assert result.decisions[0].action.value == "ask" and [q.slot for q in result.reply.questions] == [NeedsThreeThings.ORDER[number]]
            asked.append(NeedsThreeThings.ORDER[number])
    assert asked == list(NeedsThreeThings.ORDER) and w.orch.done_ids == ("a",), "five questions in turn, then the answer"


def test_a_no_to_the_one_question_on_screen_is_its_answer_not_a_withdrawal():
    route = NeedsTopic()
    w = World([interp(general()), interp(declined=["a"])], [[S("a", 0.99)]], sub=route)
    w.turn(QUESTION)
    r = w.turn("No I haven't contacted them")
    assert route.calls[-1]["topic"] == "no" and "Customer: No I haven't contacted them" in route.calls[-1]["conversation_so_far"]
    assert [o.status for o in r.outcomes] == [OutcomeStatus.DONE] and "withdrawn" not in " ".join(e.reason or "" for e in r.trace.events)


def test_never_mind_still_withdraws_the_request_even_with_a_question_on_screen():
    route = AlwaysAsks()
    w = World([interp(general()), interp(declined=["a"])], [[S("a", 0.99)]], sub=route)
    w.turn(QUESTION)
    w.turn("never mind, forget it")
    assert w.orch.active_ids == ()


def test_the_same_request_said_again_with_more_detail_is_the_answer_not_a_second_request():
    route = NeedsTopic()
    again = general("b", date="November 6th")
    w = World([interp(general()), interp(again)], [[S("a", 0.99)]], sub=route)
    w.turn(QUESTION)
    r = w.turn("on November 6th")
    assert w.orch.active_ids == () and [o.intent_id for o in r.outcomes] == ["a"], "no second request was made"
    assert route.calls[-1]["date"] == "November 6th" and route.calls[-1]["topic"] == "on November 6th"


# ---------------------------------------------------------------- the lost wallet: "was your credit card in it too?" answered in words

class AsksAboutTheCreditCard:
    """The wallet workflow's question, then done once it has an answer."""

    proposes_plans = True

    def __init__(self):
        self.calls = []

    async def run(self, intent, ctx):
        self.calls.append(dict(intent.slots))
        if intent.slots.get("credit_cards_in_the_wallet"):
            return done(intent)
        return ask(intent, "credit_cards_in_the_wallet")


def wallet(id="w"):
    return I(id, K.REPORT_FRAUD_OR_LOSS, {"card_ids": "Bluest, Light Green"})


def wallet_world(second):
    route = AsksAboutTheCreditCard()
    w = World([interp(wallet(), claims=GOOD_CLAIMS), interp(*second)], [[S("w", 0.99)], [S("w", 0.99), S("n2", 0.99)], [S("n2", 0.99)]], sub=route)
    w.turn("I think I lost my wallet with my two debit cards")
    return w, route


def test_a_yes_in_words_that_the_decomposer_made_a_replacement_request_of_is_the_answer():
    replacement = I("n1", K.ORDER_REPLACEMENT_CARD, summary="order a replacement for the Silver Rewards Card")
    w, route = wallet_world([replacement])
    r = w.turn("yes the silver rewards card as well")
    assert "human agent" not in r.reply.text and w.orch.active_ids == () and w.orch.done_ids == ("w",)
    assert route.calls[-1]["credit_cards_in_the_wallet"] == "yes the silver rewards card as well"


@pytest.mark.parametrize("what_the_decomposer_did", ["relisted", "nothing"])
def test_a_plain_yes_is_the_answer_whatever_the_decomposer_made_of_it(what_the_decomposer_did):
    relisted = [I("n9", K.REPORT_FRAUD_OR_LOSS, summary="the customer thinks they lost their wallet")] if what_the_decomposer_did == "relisted" else []
    w, route = wallet_world(relisted)
    r = w.turn("yes")
    assert "human agent" not in r.reply.text and w.orch.done_ids == ("w",)
    assert route.calls[-1]["credit_cards_in_the_wallet"] == "yes"


def test_a_real_unrelated_request_in_the_middle_of_the_wallet_report_is_still_its_own_request():
    w, _ = wallet_world([I("n2", K.TRANSFER_FUNDS, {"from_account_id": "a", "to_account_id": "b", "amount": "5"})])
    w.turn("also move 5 dollars from a to b")
    assert "n2" in w.orch.active_ids or "n2" in w.orch.done_ids


# ---------------------------------------------------------------- the decomposer was told the question and says it was answered

def test_the_decomposer_is_told_the_question_on_screen_and_its_answer_flag_is_enough():
    from orchestrator.decomposer import Interpretation

    route = AsksAboutTheCreditCard()
    odd = I("n1", K.ORDER_REPLACEMENT_CARD, summary="order a replacement for the silver card")
    answered = Interpretation(decomposition=interp(odd).decomposition, answers=("w",))
    w = World([interp(wallet(), claims=GOOD_CLAIMS), answered], [[S("w", 0.99)], [S("w", 0.99)]], sub=route)
    w.turn("I think I lost my wallet with my two debit cards")
    r = w.turn("yes it was in there too, the silver one " + "and I am sure " * 40)  # long and not a plain yes
    assert w.decomposer.pendings[0] == () and w.decomposer.pendings[1][0][0] == "w"
    assert "credit cards in the wallet" in w.decomposer.pendings[1][0][1].lower()
    assert route.calls[-1]["credit_cards_in_the_wallet"].startswith("yes it was in there too")
    assert w.orch.active_ids == () and r.reply.text and "human agent" not in r.reply.text, "no request was made from the answer"


def test_a_real_new_request_the_decomposer_found_beside_an_answer_stays_its_own_request():
    from orchestrator.decomposer import Interpretation

    route = AsksAboutTheCreditCard()
    freeze = I("n2", K.TRANSFER_FUNDS, {"from_account_id": "a", "to_account_id": "b", "amount": "5"})
    answered = Interpretation(decomposition=interp(freeze).decomposition, answers=("w",))
    w = World([interp(wallet(), claims=GOOD_CLAIMS), answered], [[S("w", 0.99)], [S("w", 0.99), S("n2", 0.99)], [S("n2", 0.99)]], sub=route)
    w.turn("I think I lost my wallet")
    w.turn("yes the silver card was in it, and also move 5 dollars from a to b")
    assert any(c.get("credit_cards_in_the_wallet", "").startswith("yes the silver card") for c in route.calls)
    assert "n2" in w.orch.active_ids or "n2" in w.orch.done_ids
