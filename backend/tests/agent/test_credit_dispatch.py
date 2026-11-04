"""Credit versus debit: the record the customer names decides; a customer who names nothing is asked
(with a list written in code) unless there is nothing to choose between; another customer's
credit account is never revealed or used."""

import asyncio

import pytest

from agent.cards import owned_credit_accounts, resolve_credit
from agent.dispatcher import CHOSEN, Name, RecordKinds, SubAgentDispatcher, unmapped_kinds
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.credit import CREDIT_SPECIALIST, prepare
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from knowledge.index import Document, KnowledgeIndex
from knowledge.scopes import CREDIT_CARDS
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K

BOTH = "lj82d4f1a9"  # a debit card, a checking account and one credit card (silver)
BOTH_CREDIT = "cc_lj82d4f1a9_silver"
CREDIT_ONLY = "76ad9cc60e"  # one credit card (gold), no bank account
CREDIT_ONLY_ACCOUNT = "cc_76ad9cc60e_gold"
MANY_CREDIT = "890389b165"  # four credit cards, no bank account
DEBIT_ONLY = "kj93a7b2e1"
INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: x", "credit card closure procedure")})


def intent(kind=K.CLOSE_CARD, **slots):
    return Intent(id="i1", kind=kind, summary="s", slots=slots)


def dispatcher(shared, **specialists):
    return SubAgentDispatcher(specialists, RecordKinds(shared.database))


def ctx(shared, it, user):
    session = scoped(shared)(SessionRequest(it.id, mode="plan", approved=None))
    return RouteContext(intent_id=it.id, verified_user_id=user, session=session, mode="plan", approved=None)


def test_the_credit_specialist_is_configured_for_the_kinds_the_dispatcher_sends_it():
    from agent.dispatcher import BY_RECORD

    assert CREDIT_SPECIALIST.kinds <= BY_RECORD
    assert CREDIT_SPECIALIST.kinds == {K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.CREDIT_LIMIT_REQUEST, K.DISPUTE_TRANSACTION}
    assert CREDIT_SPECIALIST.scope is CREDIT_CARDS
    assert unmapped_kinds() == frozenset()


def test_a_credit_id_goes_to_the_credit_specialist_and_runs_it():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    llm = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c", name="finish", arguments_json='{"message": "routed"}'),), usage=Usage(input_tokens=5))])
    spec = Specialist(CREDIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    it = intent(credit_card_account_id=BOTH_CREDIT)
    d = SubAgentDispatcher({Name.CREDIT: spec}, RecordKinds(shared.database))
    outcome = asyncio.run(d.run(it, ctx(shared, it, BOTH)))
    assert outcome.status is OutcomeStatus.DONE and outcome.text == "routed"


def test_both_products_and_nothing_named_gets_a_list_written_in_code_without_ids():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    d = dispatcher(shared)
    it = intent()
    assert d.specialist_name(it, BOTH) == [CHOSEN]
    outcome = asyncio.run(d.run(it, ctx(shared, it, BOTH)))
    assert outcome.status is OutcomeStatus.NEEDS_INFO
    text = outcome.questions[0].text
    assert outcome.questions[0].slot == CHOSEN
    assert "Silver Rewards Card" in text and "credit card" in text and "card ending" in text
    for hidden in (BOTH_CREDIT, "dbc_", "chk_", BOTH):
        assert hidden not in text


@pytest.mark.parametrize("user, expected", [(CREDIT_ONLY, Name.CREDIT), (DEBIT_ONLY, Name.DEBIT)])
def test_a_customer_with_one_product_and_nothing_named_is_not_asked_which(user, expected):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    assert dispatcher(shared).specialist_name(intent(), user) is expected


def test_naming_someone_elses_credit_account_is_still_a_question_even_for_a_one_product_customer():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    d = dispatcher(shared)
    for slots in ({"credit_card_account_id": BOTH_CREDIT}, {"card_id": "dbc_kj93a7b2e1_blue"}, {CHOSEN: "dbc_made_up"}):
        assert d.specialist_name(intent(**slots), CREDIT_ONLY) == [CHOSEN]


@pytest.mark.parametrize(
    "answer, user, expected",
    [
        ("Silver", BOTH, Name.CREDIT),
        ("the Silver Rewards Card", BOTH, Name.CREDIT),
        (BOTH_CREDIT, BOTH, Name.CREDIT),
        ("dbc_lj82d4f1a9_bluest", BOTH, Name.DEBIT),
    ],
)
def test_the_answer_to_which_card_is_understood_by_name_id_or_last_four(answer, user, expected):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    assert dispatcher(shared).specialist_name(intent(**{CHOSEN: answer}), user) is expected


def test_a_name_that_fits_two_cards_or_none_is_asked_again():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    accounts = owned_credit_accounts(shared.database, MANY_CREDIT)
    assert len(accounts) == 4
    types = [a["card_type"] for a in accounts]
    assert resolve_credit("rewards", accounts) is None, types  # every card is a "... Rewards ..." card or none: never guessed
    assert resolve_credit("unicorn", accounts) is None
    assert resolve_credit("", accounts) is None
    one = accounts[0]
    assert resolve_credit(str(one["account_id"]), accounts) == one


def test_credit_account_listing_never_includes_another_customer_or_a_closed_account():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    db = shared.database
    assert {a["user_id"] for a in owned_credit_accounts(db, MANY_CREDIT)} == {MANY_CREDIT}
    db.credit_card_accounts.data[CREDIT_ONLY_ACCOUNT]["status"] = "CLOSED"
    assert owned_credit_accounts(db, CREDIT_ONLY) == []


def test_prepare_fills_the_only_account_and_lists_several_and_leaves_disputes_alone():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    db = shared.database
    one = prepare(intent(), CREDIT_ONLY, db)
    assert isinstance(one, Intent) and one.slots["credit_card_account_id"] == CREDIT_ONLY_ACCOUNT
    many = prepare(intent(), MANY_CREDIT, db)
    assert isinstance(many, RouteOutcome) and many.status is OutcomeStatus.NEEDS_INFO
    assert "cc_" not in many.questions[0].text and many.questions[0].text.count("- ") == 4
    other = prepare(intent(credit_card_account_id=CREDIT_ONLY_ACCOUNT), MANY_CREDIT, db)
    assert isinstance(other, RouteOutcome), "someone else's account is a question, not a use"
    dispute = intent(K.DISPUTE_TRANSACTION)
    assert prepare(dispute, MANY_CREDIT, db) is dispute


def test_the_existing_ambiguity_for_a_customer_with_both_products_is_unchanged():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    d = dispatcher(shared)
    assert d.specialist_name(intent(card_id="dbc_lj82d4f1a9_bluest", credit_card_account_id=BOTH_CREDIT), BOTH) == [CHOSEN]
