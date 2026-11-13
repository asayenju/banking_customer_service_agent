"""The model behind the controller: it may call Jev, must end with a decision, and is told how to think about low scores."""

import asyncio
import json

from agent.controller import ASK_JEV, DECIDE, MAX_STEPS, SYSTEM, LLMController
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from orchestrator.controller import Choice, ControlCase


def case(jev=None):
    async def consult():
        consulted.append(1)
        return "Jev says: the kind is right with confidence 0.53."

    consulted = []
    c = ControlCase(
        request_kind="ACCOUNT_BALANCE_OR_HISTORY", request_summary="see my silver reward card", details={}, missing=(),
        message="I want to see my silver reward card", recent=(("customer", "hi"), ("agent", "How can I help?")), on_screen="",
        already_asked=("Which card?",), asks_so_far=1, what_it_can_do="Read only.", safety="read only", products="credit cards: Silver Rewards Card",
        consult_jev=jev or consult,
    )
    return c, consulted


def turn(name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(id=f"c-{name}", name=name, arguments_json=json.dumps(arguments)),), usage=Usage(input_tokens=5))


def decide(action="run", question="", reason="clear"):
    return turn("decide", action=action, question=question, reason=reason)


def run(script, the_case=None):
    llm = FakeToolLLM(script)
    return asyncio.run(LLMController(llm).decide((the_case or case()[0]))), llm


def test_it_can_decide_straight_away():
    choice, llm = run([decide("run", reason="the conversation makes it clear")])
    assert choice.action is Choice.RUN and choice.reason == "the conversation makes it clear"
    assert {t for t in llm.calls[0]["tools"]} == {ASK_JEV.name, DECIDE.name}


def test_it_can_ask_jev_first_and_then_decide_and_jevs_words_reach_it():
    the_case, consulted = case()
    choice, llm = run([turn("ask_jev"), decide("ask", "Do you want the balance or the transactions?")], the_case)
    assert choice.action is Choice.ASK and choice.question == "Do you want the balance or the transactions?" and consulted == [1]
    last = llm.calls[1]["messages"][-1]
    assert last["role"] == "tool" and "0.53" in last["content"]


def test_what_it_is_shown_is_the_whole_situation_with_the_questions_already_asked():
    _, llm = run([decide()])
    shown = json.loads(llm.calls[0]["messages"][1]["content"])
    assert shown["questions_already_asked_about_this_request"] == ["Which card?"] and shown["times_asked_so_far"] == 1
    assert shown["recent_conversation"] == ["customer: hi", "agent: How can I help?"]
    assert shown["how_risky"] == "read only" and shown["products_the_customer_holds"] == "credit cards: Silver Rewards Card"


def test_on_the_last_step_it_must_decide_and_a_run_of_jev_calls_ends_without_a_choice():
    choice, llm = run([turn("ask_jev")] * MAX_STEPS)
    assert choice is None
    assert [t for t in llm.calls[-1]["tools"]] == [DECIDE.name]


def test_a_garbled_or_unknown_choice_is_no_choice_so_the_gate_takes_over():
    assert run([turn("decide", action="explode", question="", reason="")])[0] is None
    assert run([AssistantTurn(text="I think run", usage=Usage(input_tokens=1))])[0] is None


def test_the_instructions_say_a_low_jev_number_alone_is_never_a_reason_for_a_person_and_a_question_is_never_repeated():
    assert "never a reason to hand the customer to a person" in SYSTEM
    assert "Never repeat a question" in SYSTEM and "read the recent turns" in SYSTEM
