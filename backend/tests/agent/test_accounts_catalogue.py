"""Slice 3 of the accounts specialist: which accounts a customer can be offered, settled in code. Eligibility
is the validator's own, promotions count only between their dates, every candidate cites a real page, and a
class that is not a candidate cannot be chosen."""

from datetime import date

import pytest

from agent import bank_accounts as accounts
from agent.bank_accounts import Product
from agent.specialist import ValidationContext
from agent.specialists import accounts_rules as rules
from agent.workflows import account_catalogue as catalogue
from agent.workflows import account_products as products
from banking_tools import BankingSession, Fidelity

SAM = "125"  # checking 03 (Green Account, 2023, $3,400) and savings 04 (Silver Account, $900)
MIA = "123"  # checking 01 (Blue Account, six days old) and savings 02 (Bronze Account)
TODAY = date(2025, 11, 14)


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def names(found: catalogue.Candidates) -> list[str]:
    return [c.name for c in found.offered]


# ---------------------------------------------------------------- the data is consistent and real

def test_every_class_has_exactly_one_page_and_every_page_is_a_class():
    classes = {(p, n) for p, ns in products.CLASSES.items() for n in ns}
    assert set(products.PAGE) == classes


def test_every_page_and_promotion_document_exists_in_the_knowledge_base():
    from knowledge.index import get_index

    ids = set(get_index().ids())
    for key, page in products.PAGE.items():
        assert any(i == page or i.startswith(page + "_") for i in ids), f"{key}: no document starts with {page}"
    for promo in products.PROMOTIONS:
        assert promo.document in ids


def test_every_promotion_names_real_classes_of_its_product():
    for promo in products.PROMOTIONS:
        assert set(promo.priority) <= set(products.CLASSES[promo.product]), promo.label


def test_every_apy_pairing_is_a_checking_class_and_a_savings_class():
    assert len(products.APY_PAIRINGS) == 18
    for checking, savings in products.APY_PAIRINGS:
        assert checking in products.CLASSES[Product.CHECKING] and savings in products.CLASSES[Product.SAVINGS]


# ---------------------------------------------------------------- promotions by date

@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("10/11/2025", None),  # before either window
        ("10/12/2025", "October"),  # first day of the October window
        ("10/31/2025", "October"),
        ("11/01/2025", "November"),  # both windows run until 11/12: the one that started later is current
        ("11/12/2025", "November"),
        ("11/13/2025", "November"),  # October is over
        ("11/14/2025", "November"),  # the date in every task
        ("11/30/2025", "November"),  # last day
        ("12/01/2025", None),
    ],
)
def test_the_promotion_in_force_on_a_date(day, expected):
    m, d, y = (int(x) for x in day.split("/"))
    found = catalogue.running_promotion(Product.BUSINESS_CHECKING, date(y, m, d))
    assert (found.label.split()[0] if found else None) == expected


def test_an_expired_promotion_is_never_attached_to_a_candidate():
    shared = world()
    found = catalogue.candidates(shared.database, SAM, TODAY, Product.BUSINESS_CHECKING)
    by_name = {c.name: c for c in found.offered}
    assert by_name["Sky Blue"].promotion_rank == 1 and by_name["Lime Green"].promotion_rank == 2
    assert by_name["Hunter Green"].promotion_rank is None, "Hunter Green was second in the October promotion, which is over"
    assert "November" in by_name["Sky Blue"].promotion and "(general)_013" in by_name["Sky Blue"].promotion
    assert catalogue.candidates(shared.database, SAM, date(2025, 10, 20), Product.BUSINESS_CHECKING).offered[0].name == "Beige"
    october = {c.name: c.promotion_rank for c in catalogue.candidates(shared.database, SAM, date(2025, 10, 20), Product.BUSINESS_CHECKING).offered}
    assert october["Lime Green"] == 1 and october["Hunter Green"] == 2 and october["Sky Blue"] is None


def test_business_savings_promotion_names_the_current_pair():
    shared = world()
    shared.database.accounts.data["bc"] = {
        "account_id": "bc", "user_id": SAM, "class": "business_checking", "level": "Navy Blue",
        "date_opened": "01/01/2025", "status": "OPEN", "current_holdings": "9000",
    }
    ranks = {c.name: c.promotion_rank for c in catalogue.candidates(shared.database, SAM, TODAY, Product.BUSINESS_SAVINGS).offered}
    assert ranks["Gold Saver Account"] == 1 and ranks["Silver Plus Saver Account"] == 2
    assert ranks["Gold Plus Saver Account"] is None


# ---------------------------------------------------------------- eligibility is the validator's

def test_a_customer_with_a_fresh_checking_account_is_offered_no_savings_and_is_told_why():
    found = catalogue.candidates(world().database, MIA, TODAY, Product.SAVINGS)
    assert found.offered == ()
    assert len(found.not_offered) == len(products.CLASSES[Product.SAVINGS])
    assert all("open at least 14 days" in why for _, why in found.not_offered)


def test_an_established_customer_is_offered_every_savings_class_with_the_pairings_they_hold():
    found = catalogue.candidates(world().database, SAM, TODAY, Product.SAVINGS)
    assert names(found) == list(products.CLASSES[Product.SAVINGS]) and found.not_offered == ()
    by_name = {c.name: c for c in found.offered}
    assert by_name["Gold Account"].boosts_with == ("Green Account",), "Green checking + Gold savings is pairing 1"
    assert by_name["Silver Account"].boosts_with == ("Green Account",)
    assert by_name["Green Account"].boosts_with == ()


def test_a_class_the_validator_would_refuse_is_not_offered():
    shared = world()
    shared.database.accounts.data["03"]["current_holdings"] = "100"  # business checking wants $500
    found = catalogue.candidates(shared.database, SAM, TODAY, Product.BUSINESS_CHECKING)
    assert found.offered == () and all("$500.00" in why for _, why in found.not_offered)


def test_a_class_already_opened_through_the_bank_tool_is_not_offered_again():
    shared = world()
    assert shared.call("open_bank_account_4821", {"user_id": SAM, "account_type": "checking", "account_class": "Purple Account"}).ok
    found = catalogue.candidates(shared.database, SAM, TODAY, Product.CHECKING)
    assert "Purple Account" not in names(found)
    assert dict(found.not_offered)["Purple Account"].endswith("cannot be opened")


def test_product_page_age_limits():
    shared = world()
    shared.database.users.data[SAM]["date_of_birth"] = "01/01/1990"  # 35
    out = dict(catalogue.candidates(shared.database, SAM, TODAY, Product.CHECKING).not_offered)
    assert out["Dark Green Account"] == "a Dark Green Account is for customers 26 or younger"
    assert "Gold Years Account" in out and "62 or older" in out["Gold Years Account"]
    shared.database.users.data[SAM]["date_of_birth"] = "01/01/2005"  # 20
    found = catalogue.candidates(shared.database, SAM, TODAY, Product.CHECKING)
    assert "Dark Green Account" in names(found) and "Gold Years Account" not in names(found)
    shared.database.users.data[SAM]["date_of_birth"] = "11/14/1963"  # 62 today
    assert "Gold Years Account" in names(catalogue.candidates(shared.database, SAM, TODAY, Product.CHECKING))
    shared.database.users.data[SAM]["date_of_birth"] = "11/15/1963"
    assert "Gold Years Account" not in names(catalogue.candidates(shared.database, SAM, TODAY, Product.CHECKING))


# ---------------------------------------------------------------- citing and choosing

def test_candidates_cite_the_documents_of_their_page():
    ids = ["doc_savings_accounts_gold_account_001", "doc_savings_accounts_gold_account_002", "doc_savings_accounts_gold_plus_account_001", "unrelated"]
    found = catalogue.candidates(world().database, SAM, TODAY, Product.SAVINGS, doc_ids=ids)
    by_name = {c.name: c for c in found.offered}
    assert by_name["Gold Account"].documents == ("doc_savings_accounts_gold_account_001", "doc_savings_accounts_gold_account_002")
    assert by_name["Gold Plus Account"].documents == ("doc_savings_accounts_gold_plus_account_001",), "a prefix of another class is not its page"
    assert by_name["Bronze Account"].documents == ()


def test_only_a_candidate_can_be_chosen_and_the_reason_is_given_otherwise():
    shared = world()
    chosen = catalogue.choose(shared.database, SAM, TODAY, Product.SAVINGS, "Silver Plus Account")
    assert isinstance(chosen, catalogue.Candidate) and chosen.name == "Silver Plus Account"
    assert "open at least 14 days" in catalogue.choose(shared.database, MIA, TODAY, Product.SAVINGS, "Silver Plus Account")
    for wrong in ("silver plus account", "Silver Plus", "Platinum Savings", ""):
        reason = catalogue.choose(shared.database, SAM, TODAY, Product.SAVINGS, wrong)
        assert isinstance(reason, str) and "not the exact name" in reason


def test_the_text_for_a_model_lists_offers_promotions_and_reasons():
    shared = world()
    text = catalogue.render(catalogue.candidates(shared.database, SAM, TODAY, Product.BUSINESS_CHECKING), TODAY)
    assert text.startswith("Accounts this customer can open today (11/14/2025):")
    assert "- Sky Blue (November 2025 business checking promotion puts it number 1" in text
    none = catalogue.render(catalogue.candidates(shared.database, MIA, TODAY, Product.SAVINGS), TODAY)
    assert "- none" in none and "Not available to this customer:" in none and "Silver Plus Account: a savings account needs" in none


# ---------------------------------------------------------------- the reference answers

OPENING_TASKS = [f"task_{n:03d}" for n in [*range(55, 72), 75, 76]]


@pytest.mark.parametrize("task_id", OPENING_TASKS)
def test_every_class_the_reference_opens_is_a_candidate_when_it_is_opened(task_id):
    from tau2.domains.banking_knowledge.tools import KnowledgeUserTools

    from agent.tier1 import gold_steps
    from banking_tools.manifest import load_tasks

    task = next(t for t in load_tasks() if str(t["id"]) == task_id)
    session = BankingSession.from_task(task, fidelity=Fidelity.TAU2)
    customer = KnowledgeUserTools(session.database)
    steps = gold_steps(task)
    user_id = next(str(s.args["user_id"]) for s in steps if s.kind == "verify")
    opened = 0
    for step in steps:
        if step.kind == "user":
            getattr(customer, step.name)(**step.args)
        elif step.kind == "other":
            getattr(session.toolkit, step.name)(**step.args)
        elif step.name == "open_bank_account_4821":
            product = {v: k for k, v in catalogue._TOOL_TYPE.items()}[step.args["account_type"]]
            chosen = catalogue.choose(session.database, user_id, TODAY, product, step.args["account_class"])
            assert isinstance(chosen, catalogue.Candidate), f"{task_id}: {chosen}"
            opened += 1
            assert session.call(step.name, step.args).ok
        elif step.kind in ("read", "write"):
            session.call(step.name, step.args)
    assert opened > 0


def test_the_reference_answers_agree_with_the_running_promotion():
    """Tasks 070 and 071 are the promotion tasks: the reference opens the promotion's first choice."""
    from banking_tools.manifest import load_tasks
    from agent.tier1 import gold_steps

    tasks = {str(t["id"]): t for t in load_tasks()}
    for task_id, product, expected_rank in (("task_070", "business_checking", 1), ("task_071", "business_savings", 1)):
        opens = [s for s in gold_steps(tasks[task_id]) if s.name == "open_bank_account_4821" and s.args["account_type"] == product]
        assert len(opens) == 1
        promo = catalogue.running_promotion(Product(product.replace("_", " ")), TODAY)
        assert promo is not None and promo.priority.index(opens[0].args["account_class"]) + 1 == expected_rank
