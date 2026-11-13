"""The simulated customer answers the specialist's questions from the persona, and its answers come back as words."""

import asyncio

import pytest

from agent.customer import SimulatedCustomer
from agent.llm_tools import AssistantTurn, FakeToolLLM, Usage
from agent.tier1 import MAX_CUSTOMER_ROUNDS, SegmentError, plan_and_execute
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, Question
from orchestrator.types import IntentKind as K

TASK = {t["id"]: t for t in load_tasks()}["task_082"]


def say(text):
    return AssistantTurn(text=text, usage=Usage(input_tokens=1))


def test_the_customer_is_given_only_its_persona_and_the_question_asked():
    llm = FakeToolLLM([say("Yes, close the Blue card.")])
    customer = SimulatedCustomer(TASK, llm)
    reply = asyncio.run(customer.answer("Do you want me to close it?"))
    assert reply == "Yes, close the Blue card."
    messages = llm.calls[0]["messages"]
    assert "Marcus Chen" in messages[0]["content"] and "Answer only what the bank agent just asked" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "The bank agent says: Do you want me to close it?"}
    assert llm.calls[0]["tools"] == []


def test_the_conversation_is_remembered_between_questions():
    llm = FakeToolLLM([say("Yes."), say("No.")])
    customer = SimulatedCustomer(TASK, llm)
    asyncio.run(customer.answer("First?"))
    asyncio.run(customer.answer("Second?"))
    assert [m["content"] for m in llm.calls[1]["messages"][1:]] == [
        "The bank agent says: First?", "Yes.", "The bank agent says: Second?"
    ]


def test_nothing_left_means_no_next_request():
    customer = SimulatedCustomer(TASK, FakeToolLLM([say("NOTHING LEFT"), say("Now my ATM dispute.")]))
    assert asyncio.run(customer.next_request(["card fraud"])) is None
    assert asyncio.run(customer.next_request(["card fraud"])) == "Now my ATM dispute."


class Asks:
    """A specialist that asks one question, then fails, so the loop's hand-over can be seen."""

    def __init__(self):
        self.seen = []

    async def run(self, intent, ctx):
        self.seen.append(intent)
        if len(self.seen) == 1:
            question = Question(intent_id=intent.id, slot="card_closure", text="Close it or keep it?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(question,))
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.FAILED, error="stop here")


def test_an_answer_goes_back_to_the_specialist_in_the_slot_and_in_the_summary():
    specialist = Asks()
    session = BankingSession.from_task(TASK)
    intent = Intent(id="i", kind=K.DISPUTE_TRANSACTION, summary="four disputes", slots={})
    with pytest.raises(SegmentError):
        plan_and_execute(specialist, intent, session, "mc47a2b9e1", label="seg", ask=lambda q: "Keep the Green one.")
    second = specialist.seen[1]
    assert second.slots["card_closure"] == "Keep the Green one."
    assert "The agent asked: Close it or keep it?" in second.summary and "four disputes" in second.summary


def test_without_a_customer_a_question_still_stops_the_segment():
    specialist = Asks()
    with pytest.raises(SegmentError, match="needs_info"):
        plan_and_execute(specialist, Intent(id="i", kind=K.DISPUTE_TRANSACTION, summary="x", slots={}), BankingSession.from_task(TASK), "mc47a2b9e1", label="seg")
    assert len(specialist.seen) == 1


def test_a_customer_cannot_keep_a_segment_asking_forever():
    class Endless:
        calls = 0

        async def run(self, intent, ctx):
            Endless.calls += 1
            q = Question(intent_id=intent.id, slot="x", text="again?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(q,))

    with pytest.raises(SegmentError):
        plan_and_execute(Endless(), Intent(id="i", kind=K.DISPUTE_TRANSACTION, summary="x", slots={}), BankingSession.from_task(TASK), "mc47a2b9e1", label="seg", ask=lambda q: "ok")
    assert Endless.calls == MAX_CUSTOMER_ROUNDS + 1
