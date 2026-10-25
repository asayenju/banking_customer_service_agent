import json
import threading

import pytest
from tau2.domains.banking_knowledge.tools import KnowledgeTools

from banking_tools import BankingSession, Reversibility
from banking_tools.session import REDACTED, _base_db

CARD = "dbc_lj82d4f1a9_bluest"
USER = "123"
CREDIT_CARD = {
    "credit_card_account_id": "cc_lj82d4f1a9_silver",
    "user_id": "lj82d4f1a9",
}


def fresh_db():
    return _base_db().model_copy(deep=True)


def test_sessions_do_not_share_state():
    first, second = BankingSession(), BankingSession()
    start = second.db_hash()
    assert first.call("change_user_email", {"user_id": USER, "new_email": "a@b.co"}).ok
    assert first.db_hash() != start
    assert second.db_hash() == start
    assert BankingSession().db_hash() == start, "the pristine database was modified"
    assert "a@b.co" not in second.call("get_user_information_by_id", {"user_id": USER}).text


def test_concurrent_calls_match_sequential_calls():
    users = ["123", "125"] + [
        key for key in list(_base_db().users.data)[2:8]
    ]
    calls = [(u, f"user{i}@example.com") for i, u in enumerate(users)]

    threaded = BankingSession()
    results = []
    threads = [
        threading.Thread(
            target=lambda u=u, e=e: results.append(
                threaded.call("change_user_email", {"user_id": u, "new_email": e})
            )
        )
        for u, e in calls
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sequential = BankingSession()
    for u, e in calls:
        sequential.call("change_user_email", {"user_id": u, "new_email": e})

    assert all(r.ok for r in results)
    assert len(threaded.audit_log) == len(calls)
    assert len({entry.audit_id for entry in threaded.audit_log}) == len(calls)
    assert threaded.db_hash() == sequential.db_hash()


def test_error_string_becomes_failure():
    result = BankingSession().call("freeze_debit_card_3892", {"card_id": "no_such_card"})
    assert not result.ok
    assert result.text.startswith("Error:")
    assert result.error and not result.error.startswith("Error:")


def test_invalid_arguments_never_reach_tau2_and_are_not_audited():
    session = BankingSession()
    start = session.db_hash()
    for bad in ({}, {"card_id": "c", "extra": 1}, {"card_id": 5}):
        result = session.call("freeze_debit_card_3892", bad)
        assert not result.ok and result.error.startswith("Invalid arguments")
    assert session.db_hash() == start
    assert session.audit_log == ()


def test_unknown_tool_is_a_failure():
    result = BankingSession().call("unlock_discoverable_agent_tool", {})
    assert not result.ok and "Unknown tool" in result.error


def test_reads_are_not_audited_writes_are():
    session = BankingSession()
    read = session.call("get_user_information_by_id", {"user_id": USER})
    assert read.ok and read.audit_id is None and session.audit_log == ()

    write = session.call("freeze_debit_card_3892", {"card_id": CARD})
    assert write.ok, write.text
    (entry,) = session.audit_log
    assert entry.audit_id == write.audit_id
    assert entry.tool == "freeze_debit_card_3892"
    assert entry.reversibility is Reversibility.REVERSIBLE
    assert entry.ok and entry.arguments == {"card_id": CARD}


def test_failed_write_is_still_audited():
    session = BankingSession()
    result = session.call("freeze_debit_card_3892", {"card_id": "no_such_card"})
    assert not result.ok
    (entry,) = session.audit_log
    assert entry.ok is False and entry.audit_id == result.audit_id


def test_pins_and_cvv_are_redacted_in_audit():
    session = BankingSession()
    session.call(
        "change_debit_card_pin_6285",
        {"card_id": CARD, "current_pin": "1234", "new_pin": "9876"},
    )
    (entry,) = session.audit_log
    assert entry.arguments["current_pin"] == REDACTED
    assert entry.arguments["new_pin"] == REDACTED
    assert "9876" not in entry.model_dump_json() and "1234" not in entry.model_dump_json()


def dispatcher_hash(name, arguments):
    """Hash after calling through tau2's own unlock + JSON dispatcher."""
    db = fresh_db()
    toolkit = KnowledgeTools(db)
    toolkit.unlock_discoverable_agent_tool(name)
    out = toolkit.call_discoverable_agent_tool(name, json.dumps(arguments))
    assert not out.startswith("Error:"), out
    return without_dispatcher_row(db).get_hash()


def without_dispatcher_row(db):
    # The dispatcher records a row in this table; the wrapper keeps an audit log instead.
    copy = db.model_copy(deep=True)
    copy.agent_discoverable_tools = type(copy.agent_discoverable_tools)()
    return copy


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("freeze_debit_card_3892", {"card_id": CARD}),
        ("set_debit_card_recurring_block_7382", {"card_id": CARD, "block_recurring": True}),
        # int in the call: the dispatcher turns it into a float, and so must the wrapper.
        (
            "submit_credit_limit_increase_request_7392",
            {**CREDIT_CARD, "requested_increase_amount": 5000},
        ),
    ],
)
def test_wrapper_matches_tau2_dispatcher(name, arguments):
    db = fresh_db()
    result = BankingSession(db).call(name, arguments)
    assert result.ok, result.text
    assert without_dispatcher_row(db).get_hash() == dispatcher_hash(name, arguments)
