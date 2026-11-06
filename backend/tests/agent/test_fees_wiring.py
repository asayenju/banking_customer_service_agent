"""Slice 5 of the fees specialist: it is wired in. The dispatcher hands the kind to it, the live harness can
grade its tasks, the orchestrator notices a dropped fee request, a hand-off note can describe its changes, and
it cannot plan a tool that belongs to another specialist."""

import asyncio
from datetime import datetime

from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher, unmapped_kinds
from agent.guard import KIND_WRITE_TOOLS
from agent.handoff import describe
from agent.live import gradeable_clusters
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.fees import FEES_SPECIALIST, KINDS
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.types import AuditEntry, Reversibility
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Decomposition, Intent, IntentKind as K
from orchestrator.validate import validate

INDEX = KnowledgeIndex({"doc_fees_x": Document("doc_fees_x", "Internal: x", "fee procedure")})
MIA = "123"


def test_every_kind_still_has_a_specialist_and_the_fee_kind_is_the_fees_ones():
    assert unmapped_kinds() == frozenset()
    assert KINDS == frozenset({K.FEE_OR_INTEREST_ADJUSTMENT}) and FEES_SPECIALIST.kinds == KINDS


def test_the_dispatcher_hands_the_fee_kind_to_the_specialist_and_a_missing_one_still_fails_plainly():
    shared = BankingSession.from_task({"initial_state": None})  # type: ignore[arg-type]
    llm = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c1", name="finish", arguments_json='{"message": "ok"}'),), usage=Usage(input_tokens=1))])
    specialist = Specialist(FEES_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    it = Intent(id="i1", kind=K.FEE_OR_INTEREST_ADJUSTMENT, summary="s", slots={})

    def ctx() -> RouteContext:
        session = scoped(shared)(SessionRequest(it.id, mode="plan", approved=None))
        return RouteContext(intent_id=it.id, verified_user_id=MIA, session=session, mode="plan", approved=None)

    wired = SubAgentDispatcher({Name.FEES: specialist}, RecordKinds(shared.database))
    assert asyncio.run(wired.run(it, ctx())).status is OutcomeStatus.DONE
    unwired = SubAgentDispatcher({}, RecordKinds(shared.database))
    assert asyncio.run(unwired.run(it, ctx())).status is OutcomeStatus.FAILED


def test_the_live_harness_grades_the_ten_fee_tasks():
    clusters = gradeable_clusters({"fees": FEES_SPECIALIST})
    assert {t for t, c in clusters.items() if c == "fees"} == {
        "task_026", "task_028", "task_072", "task_073", "task_074", *(f"task_{n:03d}" for n in range(93, 98))
    }


def test_it_plans_only_tools_that_are_its_own():
    """The guard lists the statement credit under this kind too, but it belongs to the credit specialist: this
    specialist has no validator for it, so it cannot be planned here."""
    writable = set(FEES_SPECIALIST.validators)
    assert writable == {
        "apply_checking_account_credit_5829", "apply_savings_account_credit_6831",
        "submit_interest_discrepancy_report_7294", "update_transaction_rewards_3847",
    }
    assert writable <= KIND_WRITE_TOOLS[K.FEE_OR_INTEREST_ADJUSTMENT]
    assert "apply_statement_credit_8472" not in writable and "apply_credit_card_account_flag_6147" not in writable


def test_a_dropped_fee_request_is_flagged_and_a_plain_question_is_not():
    def flagged(message: str, kind: K = K.FREEZE_CARD) -> bool:
        decomposition = validate(message, Decomposition(intents=(Intent(id="i1", kind=kind, summary="s", slots={"card_id": "c"}),)))
        return bool(decomposition.flags)

    assert flagged("The ATM fees on my checking account are wrong, please refund them")
    assert flagged("I was paid too little interest on my savings")
    assert flagged("My rewards points are incorrect")
    assert not flagged("What is the balance of my checking account?", K.ACCOUNT_BALANCE_OR_HISTORY)
    assert not flagged("Why was I charged this ATM fee?", K.LOOKUP_INFO)
    assert not flagged("The fee is wrong, please refund it", K.FEE_OR_INTEREST_ADJUSTMENT)


def test_a_handoff_note_can_describe_each_of_its_changes():
    db = BankingSession.from_task({"initial_state": None}).database  # type: ignore[arg-type]
    def entry(tool: str, **args) -> AuditEntry:
        return AuditEntry(audit_id="a", tool=tool, reversibility=Reversibility.IRREVERSIBLE, arguments=args, ok=True, timestamp=datetime(2025, 11, 14))

    assert "$14.00 ATM fee credit" in describe(db, entry("apply_checking_account_credit_5829", amount=14.0, credit_type="fee_refund"))
    assert "$140.00 interest correction" in describe(db, entry("apply_savings_account_credit_6831", amount=140.0))
    assert "not a credit" in describe(db, entry("submit_interest_discrepancy_report_7294"))
    assert "corrected to 95 points" in describe(db, entry("update_transaction_rewards_3847", transaction_id="x", new_rewards_earned="95 points"))
