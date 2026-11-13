import ast
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent import recipes
from agent.recipes.direct import DirectRunner
from banking_tools import BankingSession, Fidelity, Reversibility, get_registry
from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.executor import OutcomeStatus
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

ME, MY_CARD, OTHER_CARD = "lj82d4f1a9", "dbc_lj82d4f1a9_bluest", "dbc_kj93a7b2e1_blue"
SECOND = "dbc_538bfb9cba"
MANY = "kj93a7b2e1"
MY_ACCOUNT, OTHER_ACCOUNT = "chk_lj82d4f1a9", "chk_kj93a7b2e1_1"


def fresh():
    return BankingSession(fidelity=Fidelity.TAU2)


def run(kind, shared, slots=None, user=ME, summary="please", intent_id="d"):
    intent = Intent(id=intent_id, kind=kind, summary=summary, slots=slots or {})
    ctx = RouteContext(intent_id=intent_id, verified_user_id=user, session=scoped(shared)(SessionRequest(intent_id)))
    return asyncio.run(DirectRunner(shared.database).run(intent, ctx))


def one_active_card():
    shared = fresh()
    shared.database.debit_cards.data[SECOND]["status"] = "CLOSED"
    return shared


# ------------------------------------------------------------------ no model, ever

def test_the_module_imports_no_llm_client_directly_or_through_anything_it_imports():
    path = Path(recipes.__file__).parent / "direct.py"
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not {m for m in imported if m.startswith(("litellm", "agent.llm_tools", "agent.loop", "agent.specialist", "orchestrator.llm"))}
    root = Path(__file__).resolve().parents[2]
    code = "import sys; import agent.recipes.direct; bad=[m for m in sys.modules if m in ('agent.llm_tools','agent.loop','agent.specialist','orchestrator.llm','orchestrator.llm_decomposer')]; print('clean' if not bad else bad)"
    env = {"PYTHONPATH": f"{root / 'src'}:{root.parent / 'tau2-bench/src'}", "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "clean" in done.stdout, done.stdout + done.stderr


# ------------------------------------------------------------------ freeze

def test_one_active_card_is_frozen_without_a_question():
    shared = one_active_card()
    outcome = run(K.FREEZE_CARD, shared)
    assert outcome.status is OutcomeStatus.DONE and "is now frozen" in outcome.text and "4821" in outcome.text
    assert shared.database.debit_cards.data[MY_CARD]["status"] == "FROZEN"
    assert [e.tool for e in shared.audit_log] == ["freeze_debit_card_3892"]


def test_several_active_cards_means_a_list_written_by_code_and_nothing_is_frozen():
    shared = fresh()
    outcome = run(K.FREEZE_CARD, shared, user=MANY)
    assert outcome.status is OutcomeStatus.NEEDS_INFO
    text = outcome.questions[0].text
    assert text.startswith("Which card do you mean?") and text.count("the card ending") == 3
    assert not any(c in text for c in ("dbc_kj93a7b2e1_blue", "dbc_kj93a7b2e1_green", "dbc_kj93a7b2e1_lightgreen"))
    assert shared.audit_log == ()


def test_the_customer_may_answer_with_the_last_four_digits():
    shared = fresh()
    last4 = shared.database.debit_cards.data["dbc_kj93a7b2e1_green"]["last_4_digits"]
    outcome = run(K.FREEZE_CARD, shared, {"card_id": last4}, user=MANY)
    assert outcome.status is OutcomeStatus.DONE
    assert shared.database.debit_cards.data["dbc_kj93a7b2e1_green"]["status"] == "FROZEN"
    assert shared.database.debit_cards.data["dbc_kj93a7b2e1_blue"]["status"] == "ACTIVE"


def test_naming_a_card_by_its_id_works():
    shared = fresh()
    assert run(K.FREEZE_CARD, shared, {"card_id": SECOND}).status is OutcomeStatus.DONE
    assert shared.database.debit_cards.data[SECOND]["status"] == "FROZEN"


def test_someone_elses_card_is_not_found_and_not_listed_and_never_frozen():
    shared = fresh()
    outcome = run(K.FREEZE_CARD, shared, {"card_id": OTHER_CARD}, user=ME)
    assert outcome.status is OutcomeStatus.NEEDS_INFO
    assert OTHER_CARD not in outcome.questions[0].text and outcome.questions[0].text.count("the card ending") == 2
    assert shared.database.debit_cards.data[OTHER_CARD]["status"] == "ACTIVE" and shared.audit_log == ()


@pytest.mark.parametrize("status, part", [("FROZEN", "already frozen"), ("CLOSED", None)])
def test_a_card_that_cannot_be_frozen_is_explained_not_attempted(status, part):
    shared = fresh()
    shared.database.debit_cards.data[MY_CARD]["status"] = status
    outcome = run(K.FREEZE_CARD, shared, {"card_id": MY_CARD})
    assert outcome.status in (OutcomeStatus.DONE, OutcomeStatus.NEEDS_INFO) and shared.audit_log == ()
    if part:
        assert part in outcome.text


def test_a_customer_with_no_active_card_is_told_so():
    shared = fresh()
    for c in shared.database.debit_cards.data.values():
        if c["user_id"] == ME:
            c["status"] = "CLOSED"
    outcome = run(K.FREEZE_CARD, shared)
    assert outcome.status is OutcomeStatus.DONE and "no card that can be frozen" in outcome.text


# ------------------------------------------------------------------ unfreeze

def test_unfreezing_the_only_frozen_card_needs_no_question():
    shared = fresh()
    shared.database.debit_cards.data[MY_CARD]["status"] = "FROZEN"
    outcome = run(K.UNFREEZE_CARD, shared)
    assert outcome.status is OutcomeStatus.DONE and "is now unfrozen" in outcome.text
    assert shared.database.debit_cards.data[MY_CARD]["status"] == "ACTIVE"


def test_several_frozen_cards_means_a_question():
    shared = fresh()
    for c in ("dbc_kj93a7b2e1_blue", "dbc_kj93a7b2e1_green"):
        shared.database.debit_cards.data[c]["status"] = "FROZEN"
    assert run(K.UNFREEZE_CARD, shared, user=MANY).status is OutcomeStatus.NEEDS_INFO


def test_an_active_card_is_not_unfrozen_and_a_frozen_account_blocks_it():
    shared = fresh()
    assert "no card that can be unfrozen" in run(K.UNFREEZE_CARD, shared).text
    shared.database.debit_cards.data[MY_CARD]["status"] = "FROZEN"
    shared.database.accounts.data[MY_ACCOUNT]["status"] = "CLOSED"
    outcome = run(K.UNFREEZE_CARD, shared)
    assert outcome.status is OutcomeStatus.FAILED and shared.database.debit_cards.data[MY_CARD]["status"] == "FROZEN"


# ------------------------------------------------------------------ the email on file

def test_the_email_is_changed_for_the_verified_customer_even_if_a_slot_names_someone_else():
    shared = fresh()
    outcome = run(K.UPDATE_PROFILE, shared, {"new_email": "new.me@example.com", "user_id": "125"})
    assert outcome.status is OutcomeStatus.DONE and "new.me@example.com" in outcome.text
    assert shared.database.users.data[ME]["email"] == "new.me@example.com"
    assert shared.database.users.data["125"]["email"] != "new.me@example.com"


@pytest.mark.parametrize("bad", ["", "not an email", "a@b", "a b@c.com", "@x.com"])
def test_a_missing_or_malformed_email_is_asked_for_and_nothing_changes(bad):
    shared = fresh()
    outcome = run(K.UPDATE_PROFILE, shared, {"new_email": bad} if bad else {})
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].slot == "new_email"
    assert shared.audit_log == ()


def test_changing_the_email_keeps_its_reversible_label_in_the_catalog_and_the_audit():
    assert get_registry()["change_user_email"].reversibility is Reversibility.REVERSIBLE
    shared = fresh()
    run(K.UPDATE_PROFILE, shared, {"new_email": "x@y.co"})
    assert shared.audit_log[0].reversibility is Reversibility.REVERSIBLE


# ------------------------------------------------------------------ balances and history

def test_balances_list_only_the_verified_customers_accounts():
    shared = fresh()
    outcome = run(K.ACCOUNT_BALANCE_OR_HISTORY, shared)
    assert outcome.status is OutcomeStatus.DONE
    assert "Bluest Account (checking): $127,450.00" in outcome.text and "Light Green Account" in outcome.text
    assert "Silver Rewards Card: balance $842.15" in outcome.text
    other = {a["level"] for a in shared.database.accounts.data.values() if a["user_id"] == MANY}
    assert not any(f"{name} (checking): $5,200" in outcome.text for name in other), "another customer's balance must not appear"
    assert "5,200.00" not in outcome.text and "12,750.00" not in outcome.text


def test_history_of_a_named_account_is_newest_first_and_capped():
    shared = fresh()
    account = next(k for k, v in shared.database.accounts.data.items() if v["user_id"] == ME)
    rows = [t for t in shared.database.bank_account_transaction_history.data.values() if t["account_id"] == account]
    outcome = run(K.ACCOUNT_BALANCE_OR_HISTORY, shared, {"account_id": account})
    assert outcome.status is OutcomeStatus.DONE
    lines = [l for l in outcome.text.splitlines() if l.startswith("- ")]
    assert 0 < len(lines) <= 10 and (len(rows) <= 10 or len(lines) == 10)
    dates = [l[2:12] for l in lines]
    key = lambda d: (d[6:10], d[0:2], d[3:5])
    assert dates == sorted(dates, key=key, reverse=True)


def test_history_of_someone_elses_account_is_refused_and_leaks_nothing():
    shared = fresh()
    secret_rows = [t for t in shared.database.bank_account_transaction_history.data.values() if t["account_id"] == OTHER_ACCOUNT]
    outcome = run(K.ACCOUNT_BALANCE_OR_HISTORY, shared, {"account_id": OTHER_ACCOUNT})
    assert outcome.status is OutcomeStatus.DONE and "can't find that account among yours" in outcome.text
    for row in secret_rows:
        assert str(row["description"]) not in outcome.text
    assert OTHER_ACCOUNT not in outcome.text and "Blue Account" not in outcome.text.split("Your accounts:")[0]


def test_a_customer_with_no_accounts_is_told_so():
    shared = fresh()
    assert "don't see any accounts" in run(K.ACCOUNT_BALANCE_OR_HISTORY, shared, user="no_such_user").text


# ------------------------------------------------------------------ lookups

@pytest.mark.parametrize(
    "summary, part",
    [
        ("what time is it", "2025-11-14"),
        ("what are my referrals", "referrals"),
        ("which cards do I have", "Your cards"),
        ("what is on my profile", "On your profile"),
    ],
)
def test_a_lookup_answers_from_the_matching_record(summary, part):
    shared = fresh()
    user = "76ad9cc60e" if "referrals" in summary else ME
    outcome = run(K.LOOKUP_INFO, shared, summary=summary, user=user)
    assert outcome.status is OutcomeStatus.DONE and part in outcome.text


def test_a_profile_lookup_shows_only_the_customers_own_details():
    shared = fresh()
    text = run(K.LOOKUP_INFO, shared, summary="what is my email").text
    assert "jinhai.liang@techventures.com" in text
    assert "date_of_birth" not in text and "04/12/2002" not in text


def test_no_verified_customer_means_no_account_lookup_but_the_time_is_fine():
    shared = fresh()
    assert run(K.LOOKUP_INFO, shared, summary="what is my profile", user=None).status is OutcomeStatus.FAILED
    assert run(K.LOOKUP_INFO, shared, summary="what time is it", user=None).status is OutcomeStatus.DONE
    assert run(K.FREEZE_CARD, shared, user=None).status is OutcomeStatus.FAILED


def test_a_greeting_gets_the_menu_and_a_kind_that_is_not_direct_is_refused():
    shared = fresh()
    assert "I can help with" in run(K.UNCLEAR_OR_GREETING, shared).text
    assert run(K.CLOSE_CARD, shared).status is OutcomeStatus.FAILED


# ------------------------------------------------------------------ through the orchestrator

def real_customer():
    user = BankingSession().database.users.data[ME]
    return CustomerRecord(user_id=ME, date_of_birth=user["date_of_birth"], email=user["email"], phone_number=user["phone_number"], address=user["address"])


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def orchestrator(shared, intents):
    record = real_customer()
    return Orchestrator(
        decomposer=ScriptedDecomposer([Interpretation(decomposition=Decomposition(intents=tuple(intents)), claims={"email": record.email, "phone_number": record.phone_number})]),
        scorer=Scores(),
        directory=ScriptedDirectory(record),
        runners={Route.DIRECT: DirectRunner(shared.database), Route.SUB_AGENT: ScriptedRoute()},
        session_for=scoped(shared),
    )


def test_a_freeze_and_a_balance_question_run_together_through_the_real_orchestrator_with_no_yes():
    shared = one_active_card()
    freeze = Intent(id="f", kind=K.FREEZE_CARD, summary="freeze my card", slots={})
    balance = Intent(id="b", kind=K.ACCOUNT_BALANCE_OR_HISTORY, summary="my balance", slots={})
    result = asyncio.run(orchestrator(shared, [freeze, balance]).handle_turn("m1", "freeze my card and what is my balance"))
    assert {o.intent_id: o.status for o in result.outcomes} == {"f": OutcomeStatus.DONE, "b": OutcomeStatus.DONE}
    assert "is now frozen" in result.reply.text and "Your accounts" in result.reply.text
    assert shared.database.debit_cards.data[MY_CARD]["status"] == "FROZEN"


def test_an_ambiguous_freeze_asks_through_the_orchestrator_instead_of_guessing():
    shared = fresh()
    intent = Intent(id="f", kind=K.FREEZE_CARD, summary="freeze my card", slots={})
    result = asyncio.run(orchestrator(shared, [intent]).handle_turn("m1", "freeze my card"))
    assert "Which card do you mean?" in result.reply.text and result.reply.text.count("the card ending") == 2
    assert shared.audit_log == ()


# ------------------------------------------------------------------ the transactions on one card, named in the customer's words

def world(task):
    from banking_tools.manifest import load_tasks

    return BankingSession.from_task({t["id"]: t for t in load_tasks()}["task_031" if task == "fatima" else "task_078"], fidelity=Fidelity.TAU2)


def history(shared, said, user, **slots):
    return run(K.ACCOUNT_BALANCE_OR_HISTORY, shared, slots=slots, user=user, summary=said)


FATIMA, MEI = "890389b165", "mc78a5b9d2"


def test_the_charges_on_one_credit_card_are_listed_for_that_card_only():
    out = history(world("fatima"), "show me the transactions on my Silver card", FATIMA)
    assert out.status is OutcomeStatus.DONE and out.text.startswith("Recent charges on your Silver Rewards Card:")
    assert "Marriott Hotels" in out.text and "$167.34" in out.text and "11/07/2025" in out.text
    assert "EcoCard" not in out.text and len(out.text.splitlines()) <= 11


def test_a_more_specific_card_name_wins_over_a_part_of_another():
    shared = world("fatima")
    assert "Business Platinum Rewards Card" in history(shared, "what is on my business platinum card", FATIMA).text
    assert history(shared, "charges on the crypto card", FATIMA).text.startswith("Recent charges on your Crypto-Cash Back:")


def test_a_card_with_no_charges_says_so():
    shared = world("fatima")
    for t in list(shared.database.credit_card_transaction_history.data.values()):
        if t["user_id"] == FATIMA and t["credit_card_type"] == "EcoCard":
            del shared.database.credit_card_transaction_history.data[t["transaction_id"]]
    assert "no charges on your EcoCard" in history(shared, "transactions on my eco card", FATIMA).text


def test_one_debit_account_is_listed_and_not_the_light_green_one_when_green_is_named():
    shared = world("mei")
    green = history(shared, "show the transactions on my Green Account card", MEI)
    assert "Recent activity on your Green Account:" in green.text and "CHEESECAKE" in green.text and "AMAZON" not in green.text
    light = history(shared, "what happened on my light green card", MEI)
    assert "Recent activity on your Light Green Account:" in light.text and "AMAZON" in light.text


def test_a_debit_card_named_by_its_last_four_digits():
    out = history(world("mei"), "the card ending 7391", MEI)
    assert "Recent activity on your Green Account:" in out.text


def test_credit_or_debit_in_the_words_picks_the_product():
    shared = world("fatima")
    assert "Recent charges on your Silver Rewards Card" in history(shared, "my silver credit card transactions", FATIMA).text


def test_a_name_that_fits_two_products_is_asked_about_and_another_customers_card_is_never_shown():
    shared = world("mei")
    shared.database.credit_card_accounts.data["cc_mei_x"] = {"account_id": "cc_mei_x", "user_id": MEI, "card_type": "Green Rewards Card", "status": "ACTIVE", "current_balance": "$1.00"}
    ask = history(shared, "transactions on my green card", MEI)
    assert ask.status is OutcomeStatus.NEEDS_INFO and "Which card do you mean?" in ask.questions[0].text
    assert "Silver" not in history(world("mei"), "transactions on my silver card", MEI).text


def test_no_card_named_still_gives_the_account_overview():
    out = history(world("fatima"), "what are my balances", FATIMA)
    assert out.text.startswith("Your accounts:")
