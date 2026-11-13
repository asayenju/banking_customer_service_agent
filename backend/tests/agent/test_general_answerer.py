import asyncio
import json

import pytest

from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.guard import allowed_writes
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist, system_prompt
from agent.specialists.general import GENERAL_ANSWERER
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from knowledge.index import Document, KnowledgeIndex, get_index
from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.executor import OutcomeStatus
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

BLUE = "doc_checking_accounts_blue_account_001"
GOLD = "doc_credit_cards_gold_card_001"
INTERNAL = "doc_bank_accounts_close_internal_001"
INDEX = KnowledgeIndex(
    {
        BLUE: Document(BLUE, "Blue Account", "The Blue Account has a $10 monthly fee, waived with a $1,500 balance."),
        GOLD: Document(GOLD, "Gold Card", "The Gold Card has a $95 annual fee and 3% cash back on travel."),
        INTERNAL: Document(INTERNAL, "Internal: Closing Accounts", "Internal: staff close accounts with close_bank_account_7392 after verifying."),
    }
)


def call(name, **arguments):
    return ToolCall(id=f"c-{name}-{len(json.dumps(arguments))}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=40))


def answerer(script, shared=None):
    shared = shared or BankingSession(fidelity=Fidelity.TAU2)
    llm = FakeToolLLM(script)
    return Specialist(GENERAL_ANSWERER, llm=llm, index=INDEX, database=shared.database, vault=SecretVault()), llm, shared


def ask(spec, shared, question="What does the Blue Account cost?", user=None):
    intent = Intent(id="q", kind=K.GENERAL_QUESTION, summary=question)
    ctx = RouteContext(intent_id="q", verified_user_id=user, session=scoped(shared)(SessionRequest("q")), mode="plan")
    return asyncio.run(spec.run(intent, ctx))


def search(query):
    return call("search_policy", query=query)


# ------------------------------------------------------------------ nothing it could change anything with

def test_the_general_kind_may_make_no_writes():
    assert allowed_writes(K.GENERAL_QUESTION) == frozenset()


def test_the_model_is_offered_policy_search_and_the_three_ways_to_end_and_nothing_else():
    spec, llm, shared = answerer([turn(call("escalate", reason="x"))])
    ask(spec, shared)
    assert set(llm.calls[0]["tools"]) == {"search_policy", "finish", "ask_customer", "escalate"}


def test_no_account_read_and_no_write_tool_is_ever_in_the_toolbox():
    from banking_tools import ToolKind, get_registry

    spec, llm, shared = answerer([turn(call("escalate", reason="x"))])
    ask(spec, shared)
    offered = set(llm.calls[0]["tools"])
    catalog = set(get_registry())
    assert not (offered & catalog), "not one tool from the banking catalog is offered"
    assert not any(t.kind is ToolKind.WRITE and n in offered for n, t in get_registry().items())


def test_it_cannot_propose_a_plan():
    spec, llm, shared = answerer([turn(call("propose_plan", steps=[]))])
    outcome = ask(spec, shared)
    assert "propose_plan" not in llm.calls[0]["tools"] and outcome.status is OutcomeStatus.FAILED


def test_the_prompt_says_it_changes_nothing_and_only_reads_the_verified_customers_own_records():
    prompt = system_prompt(GENERAL_ANSWERER, K.GENERAL_QUESTION)
    assert "never changes anything" in prompt and "look_up_customer_records" in prompt and "holds nothing" in prompt
    assert "propose_plan" not in prompt


def test_an_attempt_to_call_a_banking_tool_anyway_is_just_an_unavailable_tool():
    spec, llm, shared = answerer([turn(call("freeze_debit_card_3892", card_id="dbc_lj82d4f1a9_bluest")), turn(call("escalate", reason="x"))])
    outcome = ask(spec, shared)
    assert shared.audit_log == () and shared.database.debit_cards.data["dbc_lj82d4f1a9_bluest"]["status"] == "ACTIVE"
    assert outcome.status is OutcomeStatus.FAILED


# ------------------------------------------------------------------ no identity needed

def test_an_unverified_caller_can_ask():
    spec, _, shared = answerer([turn(search("blue account monthly fee")), turn(call("finish", message=f"It is $10 a month [{BLUE}]."))])
    outcome = ask(spec, shared, user=None)
    assert outcome.status is OutcomeStatus.DONE and BLUE in outcome.text


# ------------------------------------------------------------------ answers rest on cited, retrieved documents

def test_an_answer_that_cites_a_document_it_found_is_accepted():
    spec, llm, shared = answerer(
        [turn(search("blue account fee")), turn(call("finish", message=f"The Blue Account costs $10 a month, waived above $1,500 [{BLUE}]."))]
    )
    outcome = ask(spec, shared)
    assert outcome.status is OutcomeStatus.DONE and f"[{BLUE}]" in outcome.text
    assert BLUE in llm.calls[1]["messages"][-1]["content"], "the model was shown the document it cited"


def test_an_answer_with_no_citation_is_refused_and_a_person_is_offered():
    spec, _, shared = answerer([turn(search("blue account fee")), turn(call("finish", message="It costs $10 a month."))])
    outcome = ask(spec, shared)
    assert outcome.status is OutcomeStatus.FAILED and "documents" in outcome.error


def test_an_answer_citing_a_document_that_was_never_retrieved_is_refused():
    spec, llm, shared = answerer([turn(search("blue waived")), turn(call("finish", message=f"The Gold Card has a $95 fee [{GOLD}]."))])
    outcome = ask(spec, shared)
    assert GOLD not in llm.calls[1]["messages"][-1]["content"], "the search never returned the Gold Card document"
    assert outcome.status is OutcomeStatus.FAILED


def test_citing_one_real_and_one_invented_source_is_refused():
    spec, _, shared = answerer(
        [turn(search("blue account fee")), turn(call("finish", message=f"$10 [{BLUE}] and also [doc_credit_cards_made_up_999]."))]
    )
    assert ask(spec, shared).status is OutcomeStatus.FAILED


def test_answering_without_searching_at_all_is_refused():
    spec, _, shared = answerer([turn(call("finish", message=f"Trust me [{BLUE}]."))])
    assert ask(spec, shared).status is OutcomeStatus.FAILED


def test_two_sources_both_retrieved_are_both_fine():
    spec, _, shared = answerer(
        [turn(search("fee")), turn(call("finish", message=f"Blue is $10 [{BLUE}]; the Gold Card is $95 [{GOLD}]."))]
    )
    assert ask(spec, shared).status is OutcomeStatus.DONE


# ------------------------------------------------------------------ out of scope

def test_when_the_notes_do_not_cover_it_the_model_hands_over_and_the_customer_is_offered_a_person():
    spec, llm, shared = answerer([turn(search("mortgage rates in mars")), turn(call("escalate", reason="not covered"))])
    outcome = ask(spec, shared, "What are mortgage rates on Mars?")
    assert outcome.status is OutcomeStatus.FAILED and "human agent" in outcome.error
    assert "No matching documents" in llm.calls[1]["messages"][-1]["content"]


def test_a_question_about_the_customers_own_account_is_handed_over_not_answered():
    spec, _, shared = answerer([turn(call("escalate", reason="needs account data"))])
    assert ask(spec, shared, "What is my balance?").status is OutcomeStatus.FAILED


def test_it_can_ask_what_is_meant():
    spec, _, shared = answerer([turn(call("ask_customer", details=["which_account"]))])
    outcome = ask(spec, shared, "How much is it?")
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].slot == "which_account"


# ------------------------------------------------------------------ what it can see

def test_staff_procedures_are_never_shown_to_the_general_answerer():
    spec, llm, shared = answerer([turn(search("close account verify staff")), turn(call("escalate", reason="x"))])
    ask(spec, shared)
    seen = llm.calls[1]["messages"][-1]["content"]
    assert INTERNAL not in seen and "close_bank_account_7392" not in seen and "Internal:" not in seen


def test_a_document_cannot_make_the_answerer_cite_a_staff_document_it_was_never_shown():
    spec, _, shared = answerer([turn(search("close account")), turn(call("finish", message=f"Staff close accounts [{INTERNAL}]."))])
    assert ask(spec, shared).status is OutcomeStatus.FAILED


def test_over_the_real_documents_a_product_question_finds_product_documents_only():
    index = get_index()
    spec = Specialist(GENERAL_ANSWERER, llm=FakeToolLLM([turn(search("foreign transaction fee credit card")), turn(call("escalate", reason="x"))]), index=index, database=BankingSession().database, vault=SecretVault())
    shared = BankingSession(fidelity=Fidelity.TAU2)
    spec._db = shared.database
    ask(spec, shared)
    notes = spec._llm.calls[1]["messages"][-1]["content"]
    ids = [line[1 : line.index("]")] for line in notes.splitlines() if line.startswith("[doc_")]
    assert ids and not any(index.get(i).title.lower().startswith("internal") for i in ids)


# ------------------------------------------------------------------ through the dispatcher and the orchestrator

def dispatcher(shared, script):
    spec, llm, _ = answerer(script, shared)
    return SubAgentDispatcher({Name.GENERAL: spec}, RecordKinds(shared.database)), llm


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def test_an_unverified_caller_gets_a_cited_answer_through_the_real_orchestrator():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    d, llm = dispatcher(shared, [turn(search("blue account fee")), turn(call("finish", message=f"It is $10 a month [{BLUE}]."))])
    intent = Intent(id="q", kind=K.GENERAL_QUESTION, summary="what does the blue account cost")
    orch = Orchestrator(
        decomposer=ScriptedDecomposer([Interpretation(decomposition=Decomposition(intents=(intent,)))]),
        scorer=Scores(),
        directory=ScriptedDirectory(),
        runners={Route.DIRECT: ScriptedRoute(), Route.SUB_AGENT: d},
        session_for=scoped(shared),
    )
    result = asyncio.run(orch.handle_turn("m1", "what does the blue account cost?"))
    assert orch.verified is False, "nobody was verified"
    assert [o.status for o in result.outcomes] == [OutcomeStatus.DONE]
    assert f"[{BLUE}]" in result.reply.text and shared.audit_log == ()


# ------------------------------------------------------------------ the capped live spot check, rehearsed offline

def test_the_spot_check_has_four_answers_and_one_hand_off_and_is_cheap_to_estimate():
    from agent import spot

    assert [q.expect for q in spot.QUESTIONS] == ["answer"] * 4 + ["hand_off"]
    assert spot.estimate_dollars(2.5, 10.0) < spot.DEFAULT_CAP, "the default cap covers the estimate on the stronger model"


class Rehearsal:
    """Answers each spot question properly: searches, then cites what it found (or hands over)."""

    def __init__(self, index_doc):
        self.doc, self.conversation, self.calls = index_doc, -1, 0

    async def chat(self, messages, *, tools, tool_choice="auto", temperature=0.0):
        from agent import spot

        if len(messages) <= 2:
            self.conversation += 1
            self.step = 0
        self.step += 1
        self.calls += 1
        usage = Usage(input_tokens=900, output_tokens=60)
        if self.step == 1:
            return AssistantTurn(tool_calls=(call("search_policy", query="blue account monthly fee waived balance gold card"),), usage=usage)
        if spot.QUESTIONS[self.conversation].expect == "hand_off":
            return AssistantTurn(tool_calls=(call("escalate", reason="needs account"),), usage=usage)
        return AssistantTurn(tool_calls=(call("finish", message=f"See the notes [{self.doc}]."),), usage=usage)


def test_the_spot_check_runs_records_and_then_replays_for_free(tmp_path):
    from agent import spot

    model = Rehearsal(BLUE)
    first = spot.run_spot(lambda m: model, INDEX, cassettes=tmp_path, model="openai/gpt-4o", price_in=2.5, price_out=10.0, cap=0.30)
    assert all(r.ok for r in first), spot.render(first)
    assert first[0].cited == (BLUE,) and first[-1].outcome == "failed"
    assert sum(r.dollars for r in first) > 0 and model.calls == 10

    def forbidden(m):
        raise AssertionError("a replay must not build a live client")

    again = spot.run_spot(forbidden, INDEX, cassettes=tmp_path, model="openai/gpt-4o", price_in=2.5, price_out=10.0, cap=0.30)
    assert all(r.ok for r in again) and sum(r.dollars for r in again) == 0


def test_the_spot_check_stops_at_its_cap(tmp_path):
    from agent import spot

    model = Rehearsal(BLUE)
    results = spot.run_spot(lambda m: model, INDEX, cassettes=tmp_path, model="openai/gpt-4o", price_in=500.0, price_out=500.0, cap=0.10)
    assert not all(r.ok for r in results) and sum(r.dollars for r in results) < 0.10 + 0.5
    assert any(r.outcome in ("stopped", "skipped") for r in results)


def test_an_answer_that_should_have_been_a_hand_off_is_reported_as_not_as_expected(tmp_path):
    from agent import spot

    class Answers(Rehearsal):
        async def chat(self, messages, **kw):
            turn = await super().chat(messages, **kw)
            if turn.tool_calls[0].name == "escalate":
                return AssistantTurn(tool_calls=(call("finish", message=f"Your balance is high [{BLUE}]."),), usage=turn.usage)
            return turn

    results = spot.run_spot(lambda m: Answers(BLUE), INDEX, cassettes=tmp_path, model="m", price_in=1.0, price_out=1.0, cap=5.0)
    assert not results[-1].ok and "expected a hand-off" in results[-1].note


def test_a_verified_customer_gets_the_records_tool_and_an_unverified_one_does_not():
    spec, llm, shared = answerer([turn(call("escalate", reason="x")), turn(call("escalate", reason="x"))])
    ask(spec, shared)
    ask(spec, shared, user="lj82d4f1a9")
    assert "look_up_customer_records" not in llm.calls[0]["tools"]
    assert "look_up_customer_records" in llm.calls[1]["tools"]
    assert not (set(llm.calls[1]["tools"]) & set(__import__("banking_tools").get_registry())), "still no catalog tool"
