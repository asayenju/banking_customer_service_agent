"""Slice 5 of the fraud composer: a request for a person while cards are being protected is counted and kept,
and the note a person receives names what was done without an id or a card number."""

import asyncio
from datetime import datetime
from typing import get_args

from agent.handoff import SECURITY_REASON, describe, done_events, handoff_summary
from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.containment import WalletContainment
from agent.llm_tools import FakeToolLLM
from agent.recipes.direct import DirectRunner
from agent.specialist import Specialist
from agent.specialists.fraud import FRAUD_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.types import AuditEntry, Reversibility
from orchestrator.auth import ScriptedDirectory
from orchestrator.context import scoped
from orchestrator.decision import HUMAN_AFTER_ASKS, Action
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.types import Decomposition, Intent, IntentKind as K
from tau2.domains.banking_knowledge.tools import TransferReasonLiteral

from tests.agent.test_fraud_containment import (
    CLAIMS, INDEX, LB, NAMED, RECORD, TASK, Scores, form_turn, report, status, wallet_form,
)

CONFIRM = Interpretation(decomposition=Decomposition(intents=()), confirmed=("w",))


def human(n):
    return Intent(id=f"h{n}", kind=K.REQUEST_HUMAN, summary="a person please")


def turn_with(*intents, confirmed=()):
    return Interpretation(decomposition=Decomposition(intents=tuple(intents)), confirmed=tuple(confirmed))


def world(later, first):
    shared = BankingSession.from_task(TASK)
    spec = Specialist(FRAUD_SPECIALIST, llm=FakeToolLLM([form_turn(wallet_form([LB]))]), index=INDEX, database=shared.database, vault=SecretVault())
    dispatcher = SubAgentDispatcher({Name.FRAUD: spec}, RecordKinds(shared.database))
    decomposer = ScriptedDecomposer([Interpretation(decomposition=Decomposition(intents=tuple(first)), claims=CLAIMS), *later])
    orch = Orchestrator(
        decomposer=decomposer, scorer=Scores(), directory=ScriptedDirectory(RECORD),
        runners={Route.DIRECT: DirectRunner(shared.database), Route.SUB_AGENT: dispatcher, Route.HAND_OFF: ScriptedRoute()},
        session_for=scoped(shared), containment=WalletContainment(shared.database),
    )
    return orch, shared


def actions(result):
    return {d.intent_id: d.action for d in result.decisions}


def test_a_request_for_a_person_waits_while_the_report_is_open_then_is_honoured_when_the_cards_are_safe():
    orch, shared = world([turn_with(human(2), confirmed=("w",))], [report({"card_ids": "Light Blue"}), human(1)])
    first = asyncio.run(orch.handle_turn("m1", "my wallet was stolen, I want a person"))
    assert actions(first)["h1"] is Action.DEFER and "Your cards come first" in first.reply.text
    assert status(shared, LB) == "FROZEN", "the work went ahead"
    second = asyncio.run(orch.handle_turn("m2", "yes, and I still want a person"))
    assert status(shared, LB) == "CLOSED"
    assert actions(second)["h2"] is Action.ESCALATE, "the work is done in this turn, so the person is owed now"
    assert Action.ESCALATE in actions(second).values() and "handing this to a human agent" in second.reply.text


def test_the_fourth_ask_escalates_even_though_the_work_is_still_waiting_for_a_yes():
    asks = [turn_with(human(n)) for n in range(2, HUMAN_AFTER_ASKS + 1)]
    orch, shared = world(asks, [report({"card_ids": "Light Blue"}), human(1)])
    results = [asyncio.run(orch.handle_turn("m1", "stolen"))]
    for n in range(2, HUMAN_AFTER_ASKS + 1):
        results.append(asyncio.run(orch.handle_turn(f"m{n}", "a person now")))
    assert [actions(r)[f"h{n}"] for n, r in enumerate(results, start=1)] == [Action.DEFER] * (HUMAN_AFTER_ASKS - 1) + [Action.ESCALATE]
    assert status(shared, LB) == "FROZEN", "nothing irreversible ran without the yes"


def test_with_no_report_open_a_request_for_a_person_escalates_at_once():
    orch, _ = world([], [human(1)])
    first = asyncio.run(orch.handle_turn("m1", "I want a person"))
    assert actions(first) == {"h1": Action.ESCALATE}


def test_the_security_reason_is_one_the_banks_transfer_tool_accepts():
    assert SECURITY_REASON in get_args(TransferReasonLiteral)


def test_the_note_names_what_was_done_by_account_never_by_id_or_card_number():
    shared = BankingSession.from_task(TASK)
    db = shared.database
    def entry(tool, **args):
        return AuditEntry(audit_id="a", tool=tool, reversibility=Reversibility.IRREVERSIBLE, arguments=args, ok=True, timestamp=datetime(2025, 11, 14))

    audit = [
        entry("freeze_debit_card_3892", card_id=LB),
        entry("close_debit_card_4721", card_id=LB, reason="lost"),
        entry("order_debit_card_5739", account_id="chk_mc78a5b9d2_1", delivery_option="STANDARD"),
    ]
    events = done_events(db, audit)
    note = handoff_summary("Mei-Lin Chen", events, ["The Green card stays frozen: purchases are pending."], "the customer asked for a person")
    assert "debit card on the Light Blue Account was closed (lost)" in note
    for forbidden in (LB, "chk_mc78a5b9d2_1", "4582", "dbc_"):
        assert forbidden not in note
    assert describe(db, entry("activate_debit_card_8292", card_id=LB)).endswith("was activated.")
