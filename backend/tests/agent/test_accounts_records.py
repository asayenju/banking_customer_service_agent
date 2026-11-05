"""Slice 1 of the accounts specialist: which account a request means, read from either of the bank's
two record layouts, settled in code. Another customer's account is never listed and never resolves,
and reads like an account that does not exist."""

import asyncio
import re
from datetime import date

from agent import bank_accounts as accounts
from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher, unmapped_kinds
from agent.live import gradeable_clusters
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.accounts import ACCOUNTS_SPECIALIST, KINDS, prepare
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K

MIA = "123"  # checking 01 (Blue Account, 11/08/2025) and savings 02 (Bronze Account)
SAM = "125"  # checking 03 (Green Account, 09/02/2023) and savings 04 (Silver Account)
SOLO = "224959b99e"  # one checking account, 05 (Green Account)
NOBODY = "no-such-customer"
TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def intent(kind=K.CLOSE_ACCOUNT, **slots) -> Intent:
    return Intent(id="i1", kind=kind, summary="s", slots=slots)


def settled(result) -> Intent:
    assert isinstance(result, Intent), result
    return result


def question(result) -> tuple[str, str]:
    assert isinstance(result, RouteOutcome) and result.status is OutcomeStatus.NEEDS_INFO, result
    assert len(result.questions) == 1
    return result.questions[0].slot, result.questions[0].text


# ---------------------------------------------------------------- one reading of both layouts

def test_the_banks_starting_layout_is_read_with_saving_as_savings():
    shared = world()
    mine = accounts.accounts_of(shared.database, MIA)
    assert [(a.account_id, a.product, a.name) for a in mine] == [
        ("01", accounts.Product.CHECKING, "Blue Account"),
        ("02", accounts.Product.SAVINGS, "Bronze Account"),
    ]
    assert mine[0].balance == 0.0 and mine[1].balance == 129.02
    assert mine[0].age_days(TODAY) == 6


def test_an_account_opened_by_the_bank_tool_is_seen_although_it_has_other_field_names():
    shared = world()
    assert shared.call("open_bank_account_4821", {"user_id": SAM, "account_type": "checking", "account_class": "Purple Account"}).ok
    new = [a for a in accounts.accounts_of(shared.database, SAM) if a.name == "Purple Account"]
    assert len(new) == 1
    assert new[0].product is accounts.Product.CHECKING and new[0].is_open and new[0].opened == TODAY and new[0].balance == 0.0


def test_both_balance_formats_parse():
    assert accounts.read({"account_id": "a", "user_id": "u", "class": "checking", "level": "X Account", "status": "OPEN", "current_holdings": "$1,234.50"}).balance == 1234.5
    assert accounts.read({"account_id": "a", "user_id": "u", "class": "checking", "level": "X Account", "status": "OPEN", "current_holdings": "1234.50"}).balance == 1234.5
    assert accounts.read({"account_id": "a", "user_id": "u", "class": "checking", "level": "X Account", "status": "OPEN"}).balance is None


def test_closed_accounts_and_other_customers_are_left_out_unless_asked_for():
    shared = world()
    shared.database.accounts.data["02"]["status"] = "CLOSED"
    assert [a.account_id for a in accounts.accounts_of(shared.database, MIA)] == ["01"]
    assert [a.account_id for a in accounts.accounts_of(shared.database, MIA, include_closed=True)] == ["01", "02"]
    assert accounts.get(shared.database, MIA, "03") is None, "Sam's account is not Mia's"
    assert accounts.get(shared.database, MIA, "02").status == "CLOSED"


# ---------------------------------------------------------------- resolving a name

def test_a_name_an_id_or_a_number_picks_one_account():
    mine = accounts.accounts_of(world().database, MIA)
    assert accounts.resolve("01", mine).account_id == "01"
    assert accounts.resolve("Bronze savings", mine).account_id == "02"
    assert accounts.resolve("the Blue Account", mine).account_id == "01"
    assert accounts.resolve("my checking", mine).account_id == "01"
    assert accounts.resolve("2", mine).account_id == "02"


def test_a_name_that_fits_none_or_several_resolves_to_nothing():
    shared = world()
    mine = accounts.accounts_of(shared.database, MIA)
    assert accounts.resolve("Platinum", mine) is None
    assert accounts.resolve("", mine) is None
    assert accounts.resolve("9", mine) is None
    twin = {**shared.database.accounts.data["01"], "account_id": "01b"}
    shared.database.accounts.data["01b"] = twin
    twins = accounts.accounts_of(shared.database, MIA)
    assert accounts.resolve("Blue Account", twins) is None, "two Blue accounts: the customer chooses"


def test_another_customers_id_does_not_resolve_and_reads_like_an_unknown_one():
    shared = world()
    mine = accounts.accounts_of(shared.database, MIA)
    assert accounts.resolve("03", mine) is None
    assert accounts.resolve("no-such-id", mine) is None


def test_a_zero_padded_id_is_never_read_as_a_place_in_the_list():
    """Found by the third-party test below: Mia's id "01" was taken as 'the first one' in Sam's list."""
    sams = accounts.accounts_of(world().database, SAM)
    assert accounts.resolve("01", sams) is None
    assert accounts.resolve("1", sams).account_id == "03"
    assert accounts.resolve("123", sams) is None


# ---------------------------------------------------------------- prepare

def test_a_close_that_names_the_account_gets_its_id():
    shared = world()
    out = settled(prepare(intent(account_id="Bronze savings"), MIA, shared.database))
    assert out.slots["account_id"] == "02"


def test_a_customer_with_one_account_is_not_asked_which_to_close():
    shared = world()
    out = settled(prepare(intent(), SOLO, shared.database))
    assert out.slots["account_id"] == "05"


def test_a_close_with_several_accounts_and_no_name_is_a_question_without_ids():
    shared = world()
    slot, text = question(prepare(intent(), MIA, shared.database))
    assert slot == "account_id"
    assert "1. your Blue Account (checking), opened 11/08/2025" in text
    assert "2. your Bronze Account (savings), opened 11/08/2025" in text
    assert not re.search(r"\b0[12]\b", text), "no id in the question"


def test_a_transfer_resolves_both_ends_and_asks_for_the_one_that_is_missing():
    shared = world()
    both = settled(prepare(intent(K.TRANSFER_FUNDS, from_account_id="Green checking", to_account_id="Silver savings", amount="100"), SAM, shared.database))
    assert (both.slots["from_account_id"], both.slots["to_account_id"]) == ("03", "04")
    slot, text = question(prepare(intent(K.TRANSFER_FUNDS, from_account_id="03", amount="100"), SAM, shared.database))
    assert slot == "to_account_id"
    assert "Green Account" not in text, "the account the money comes from is not offered as the destination"
    assert "1. your Silver Account (savings)" in text


def test_a_transfer_to_a_third_partys_account_is_asked_again_with_only_the_customers_own():
    shared = world()
    for third_party in ("01", "someone elses account", "9999"):
        slot, text = question(prepare(intent(K.TRANSFER_FUNDS, from_account_id="03", to_account_id=third_party, amount="100"), SAM, shared.database))
        assert slot == "to_account_id"
        assert "Bronze" not in text and "Blue" not in text, "Mia's accounts never appear in Sam's list"


def test_the_answer_to_a_list_question_is_taken_by_its_number():
    shared = world()
    asked = intent(K.TRANSFER_FUNDS, from_account_id="03", to_account_id="1", amount="100")
    assert settled(prepare(asked, SAM, shared.database)).slots["to_account_id"] == "04"


def test_a_customer_with_no_open_account_is_told_so_and_nothing_is_asked():
    out = prepare(intent(), NOBODY, world().database)
    assert isinstance(out, RouteOutcome) and out.status is OutcomeStatus.DONE and "open bank account" in out.text


def test_opening_an_account_needs_no_account_named():
    out = prepare(intent(K.OPEN_ACCOUNT), MIA, world().database)
    assert isinstance(out, Intent)


# ---------------------------------------------------------------- wiring

def test_every_accounts_kind_has_this_specialist_and_nothing_is_left_unmapped():
    assert unmapped_kinds() == frozenset()
    assert KINDS == {K.TRANSFER_FUNDS, K.PAY_CREDIT_CARD, K.OPEN_ACCOUNT, K.CLOSE_ACCOUNT}
    assert ACCOUNTS_SPECIALIST.name == Name.ACCOUNTS.value
    from agent.dispatcher import FIXED

    assert {k for k, n in FIXED.items() if n is Name.ACCOUNTS} == KINDS


def test_the_read_tools_are_real_and_none_looks_up_other_customers():
    from banking_tools.catalog import get_registry

    assert ACCOUNTS_SPECIALIST.reads is not None
    assert ACCOUNTS_SPECIALIST.reads <= set(get_registry())
    assert not {"get_user_information_by_name", "get_user_information_by_email"} & ACCOUNTS_SPECIALIST.reads


def test_the_dispatcher_hands_an_accounts_kind_to_the_specialist_and_a_missing_one_still_fails_plainly():
    shared = world()
    llm = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c1", name="finish", arguments_json='{"message": "ok"}'),), usage=Usage(input_tokens=1))])
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    it = intent(K.CLOSE_ACCOUNT)

    def ctx() -> RouteContext:
        session = scoped(shared)(SessionRequest(it.id, mode="plan", approved=None))
        return RouteContext(intent_id=it.id, verified_user_id=MIA, session=session, mode="plan", approved=None)

    wired = SubAgentDispatcher({Name.ACCOUNTS: specialist}, RecordKinds(shared.database))
    answer = asyncio.run(wired.run(it, ctx()))
    assert answer.status is OutcomeStatus.NEEDS_INFO and answer.questions[0].slot == "account_id", "prepare ran before any model"
    unwired = SubAgentDispatcher({}, RecordKinds(shared.database))
    assert asyncio.run(unwired.run(it, ctx())).status is OutcomeStatus.FAILED


def test_the_live_harness_knows_the_accounts_specialist():
    clusters = gradeable_clusters({"accounts": ACCOUNTS_SPECIALIST})
    assert set(clusters.values()) <= {"accounts"}
    assert {"task_060", "task_061", "task_075"} <= set(clusters)
