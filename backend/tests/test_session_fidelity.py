import json

import pytest
from tau2.domains.banking_knowledge.tools import KnowledgeTools

from banking_tools import BankingSession, Fidelity, Reversibility
from banking_tools.session import REDACTED, _base_db, tau2_now

CARD = "dbc_lj82d4f1a9_bluest"
CREDIT_CARD = {"credit_card_account_id": "cc_lj82d4f1a9_silver", "user_id": "lj82d4f1a9"}
WRITES = [
    ("freeze_debit_card_3892", {"card_id": CARD}),
    ("set_debit_card_recurring_block_7382", {"card_id": CARD, "block_recurring": True}),
    ("submit_credit_limit_increase_request_7392", {**CREDIT_CARD, "requested_increase_amount": 5000}),
]
READ = ("get_all_user_accounts_by_user_id_3847", {"user_id": "lj82d4f1a9"})


def tau2_run(calls, allowlist=()):
    """The same calls made the way a tau2 agent makes them. Returns (db, [outputs])."""
    db = _base_db().model_copy(deep=True)
    toolkit = KnowledgeTools(db)
    toolkit.set_read_log_allowlist(set(allowlist))
    outputs = []
    for name, arguments in calls:
        toolkit.unlock_discoverable_agent_tool(name)
        outputs.append(toolkit.call_discoverable_agent_tool(name, json.dumps(arguments)))
    return db, outputs


def ours(calls, allowlist=()):
    session = BankingSession(fidelity=Fidelity.TAU2, read_log_allowlist=allowlist)
    return session, [session.call(name, arguments) for name, arguments in calls]


@pytest.mark.parametrize("name, arguments", WRITES)
def test_a_write_leaves_exactly_the_database_tau2_leaves_audit_row_included(name, arguments):
    tau2_db, _ = tau2_run([(name, arguments)])
    session, (result,) = ours([(name, arguments)])
    assert result.ok, result.text
    assert session.db_hash() == tau2_db.get_hash()
    assert len(session._db.agent_discoverable_tools.data) == 1


def test_a_discoverable_read_adds_no_row_unless_the_gold_names_it():
    tau2_db, _ = tau2_run([READ])
    session, (result,) = ours([READ])
    assert result.ok
    assert session.db_hash() == tau2_db.get_hash() == BankingSession().db_hash()
    assert session._db.agent_discoverable_tools.data == {}

    tau2_db, _ = tau2_run([READ], allowlist={READ[0]})
    session, _ = ours([READ], allowlist={READ[0]})
    assert session.db_hash() == tau2_db.get_hash() != BankingSession().db_hash()
    assert len(session._db.agent_discoverable_tools.data) == 1


def test_a_repeated_identical_write_is_rejected_the_way_tau2_rejects_it():
    call = WRITES[2]
    tau2_db, tau2_out = tau2_run([call, call])
    session, results = ours([call, call])
    assert results[0].ok and not results[1].ok
    assert results[1].text == tau2_out[1]
    assert session.db_hash() == tau2_db.get_hash()


def test_the_unlock_gate_is_tau2s_and_unknown_tools_still_fail():
    session = BankingSession(fidelity=Fidelity.TAU2)
    start = session.db_hash()
    assert not session.call("no_such_tool_0000", {}).ok
    assert session.db_hash() == start


def test_pins_never_reach_the_audit_log_or_the_dispatcher_rows():
    session = BankingSession(fidelity=Fidelity.TAU2)
    session.call(
        "change_debit_card_pin_6285",
        {"card_id": CARD, "current_pin": "1234", "new_pin": "9876"},
    )
    (entry,) = session.audit_log
    assert entry.arguments["new_pin"] == entry.arguments["current_pin"] == REDACTED
    rows = json.dumps(session._db.agent_discoverable_tools.data)
    assert "9876" not in rows and "1234" not in rows and "9876" not in entry.model_dump_json()


def test_record_verification_matches_a_tau2_log_verification_call():
    fields = dict(
        name="Kenji Tanaka",
        user_id="6680a37184",
        address="12 Pine St",
        email="kenji@example.com",
        phone_number="206-555-0293",
        date_of_birth="03/14/1990",
    )
    db = _base_db().model_copy(deep=True)
    KnowledgeTools(db).log_verification(**fields, time_verified="2025-11-14 03:40:00 EST")
    session = BankingSession(fidelity=Fidelity.TAU2)
    result = session.record_verification(**fields)
    assert result.ok, result.text
    assert session.db_hash() == db.get_hash()
    (entry,) = session.audit_log
    assert entry.tool == "log_verification" and entry.timestamp == tau2_now()


def test_direct_mode_is_still_the_default_and_adds_no_row():
    session = BankingSession()
    assert session.call(*WRITES[0]).ok
    assert session._db.agent_discoverable_tools.data == {}


def test_from_task_defaults_to_the_faithful_mode():
    session = BankingSession.from_task({"id": "t"})
    assert session.call(*WRITES[0]).ok
    assert len(session._db.agent_discoverable_tools.data) == 1


def test_the_audit_label_is_unchanged_in_both_modes():
    for fidelity in Fidelity:
        session = BankingSession(fidelity=fidelity)
        session.call(*WRITES[0])
        assert session.audit_log[0].reversibility is Reversibility.REVERSIBLE
