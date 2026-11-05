"""The accounts and money movement specialist: opening and closing bank accounts, transfers between
the customer's own accounts, and paying a credit card from checking.

It is a configuration of ``Specialist``, like the credit and debit ones. This module holds what every
slice shares: the kinds, the instructions, the document scope, the read tools and ``prepare``, which
settles in code which bank account is meant before a model gets involved. The validators and the
workflows are added by the later slices (see ``docs/prd-accounts-money-specialist.md``); every
decision in them is written down, with its evidence, in ``docs/accounts-policy-decisions.md``.
"""

from collections.abc import Mapping
from typing import Any

from agent import bank_accounts as accounts
from agent.cards import describe_credit, owned_credit_accounts, resolve_credit
from agent.specialist import SpecialistConfig, Validator
from agent.specialists import accounts_rules as rules
from agent.workflows.account_close import CloseAccount
from agent.workflows.account_move import PayCard, TransferFunds
from agent.workflows.account_open import OpenAccount
from knowledge.scopes import ACCOUNTS
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K, Question

KINDS = frozenset({K.TRANSFER_FUNDS, K.PAY_CREDIT_CARD, K.OPEN_ACCOUNT, K.CLOSE_ACCOUNT})

INSTRUCTIONS = """You handle bank account requests: opening an account, closing an account, moving money
between the customer's own accounts, and paying a credit card from a checking account.
How to work:
1. Search the policy notes for the procedure for this request before you act. The internal
   procedures are for you; do not quote them to the customer.
2. Look the customer's accounts up with your read tools. Believe the records, not the customer: if
   they say an account is old enough, funded or empty, check. If the records disagree, say what they show.
3. The bank's tools check very little themselves. Check the rules in the policy notes yourself, and
   expect your plan to be refused if it breaks one.
4. Money only moves between accounts the customer owns. Never offer or accept another person's account.
5. Opening has a workflow: end with start_open_account. Report only what the customer said: the type and
   the exact class name of each account, and whether they want the opening deposit of a savings account
   moved. If you do not know the exact names yet, call it with an empty account_class and you are given the
   list of what this customer can open today, with the promotions in force; recommend from that list only,
   and keep to what the customer asked for. If the customer has not said what they need, ask. What can be
   opened is worked out from the records, never from what the customer says about their own accounts.
6. Closing has a workflow: end with start_close_account. Report the account (use its id from the request's
   details), where the customer wants the money to go if they said, and why they are closing it. Which
   money moves, the early-closure fee and the order of the steps are worked out for you: an account closes
   with a zero balance, or with exactly the fee in it, and the fee is never moved or waived. If the
   customer asks for the fee to be waived, say so in the form. A new account must be opened before an old
   one is closed when the new one needs the old one (a savings account needs a checking account of 14 days).
7. Moving money has workflows: end with start_transfer_funds for a transfer between the customer's own
   accounts and start_pay_card for a credit card payment from checking. Report the accounts as the request's
   details give them and the exact amount the customer said; the rules are worked out for you. Money never
   goes to anyone else's account.
8. Wires, ACH, bill pay, overdraft, holds and check deposits have no tool. Answer questions from the
   notes; for a request, say what the customer can do themselves or hand over."""

# The reads this specialist needs. Looking up other customers by name or email is never offered.
READS = frozenset(
    {
        "get_all_user_accounts_by_user_id_3847",
        "get_bank_account_transactions_9173",
        "get_credit_card_accounts_by_user",
        "get_current_time",
        "get_user_information_by_id",
    }
)

# Slots that name a bank account, per kind: (the slot we settle, the names a decomposer may have used).
_ROLES: Mapping[K, tuple[tuple[str, tuple[str, ...]], ...]] = {
    K.CLOSE_ACCOUNT: (("account_id", ("account_id", "from_account_id")),),
    K.TRANSFER_FUNDS: (
        ("from_account_id", ("from_account_id", "source_account_id")),
        ("to_account_id", ("to_account_id", "destination_account_id")),
    ),
}
_PROMPT: Mapping[str, str] = {
    "account_id": "Which account do you mean?",
    "checking_account_id": "Which checking account should the payment come from?",
    "from_account_id": "Which account should the money come from?",
    "to_account_id": "Which account should the money go to?",
}


def _which(intent: Intent, slot: str, choices: list[accounts.BankAccount]) -> RouteOutcome:
    listing = "\n".join(f"{n}. {accounts.describe(a)}" for n, a in enumerate(choices, start=1))
    return RouteOutcome(
        intent_id=intent.id,
        status=OutcomeStatus.NEEDS_INFO,
        questions=(Question(intent_id=intent.id, slot=slot, text=f"{_PROMPT[slot]}\n{listing}"),),
    )


def _first(intent: Intent, names: tuple[str, ...]) -> str:
    return next(((intent.slots.get(n) or "").strip() for n in names if (intent.slots.get(n) or "").strip()), "")


def _prepare_payment(intent: Intent, user_id: str, db: Any) -> Intent | RouteOutcome:
    """Which card is paid and from which checking account. A decomposer may have put either in ``account_id``;
    whichever of the customer's records it names decides. One card, or one checking account, needs no question."""
    cards = owned_credit_accounts(db, user_id)
    checking = [a for a in accounts.accounts_of(db, user_id) if a.is_open and a.product is accounts.Product.CHECKING]
    card_name = _first(intent, ("credit_card_account_id", "card_account_id"))
    checking_name = _first(intent, ("checking_account_id", "from_account_id", "source_account_id"))
    loose = _first(intent, ("account_id",))
    if loose and not card_name and resolve_credit(loose, cards) is not None:
        card_name = loose
    elif loose and not checking_name:
        checking_name = loose
    if not cards:
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="I do not see a credit card on your profile to pay.")
    if not checking:
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="I do not see an open checking account on your profile to pay from.")
    card = resolve_credit(card_name, cards) if card_name else (cards[0] if len(cards) == 1 else None)
    if card is None:
        listing = "\n".join(f"- {describe_credit(c)}" for c in cards)
        return RouteOutcome(
            intent_id=intent.id,
            status=OutcomeStatus.NEEDS_INFO,
            questions=(Question(intent_id=intent.id, slot="credit_card_account_id", text=f"Which card would you like to pay?\n{listing}"),),
        )
    paying = accounts.resolve(checking_name, checking) if checking_name else (checking[0] if len(checking) == 1 else None)
    if paying is None:
        return _which(intent, "checking_account_id", checking)
    return intent.model_copy(
        update={"slots": {**intent.slots, "credit_card_account_id": str(card["account_id"]), "checking_account_id": paying.account_id}}
    )


def _settle_what_is_named(
    intent: Intent, roles: tuple[tuple[str, tuple[str, ...]], ...], owned: list[accounts.BankAccount]
) -> Intent:
    """For a transfer: put the id of each account the customer named and that is theirs into the slots, and
    ask nothing. What is missing or does not resolve is left for the workflow, which asks with the same list
    once it knows the request is a transfer between the customer's own accounts at all (a wire, a bill payment
    and a check deposit are not, and a question about which account would be the wrong thing to ask)."""
    settled: dict[str, str] = {}
    for slot, names in roles:
        named = _first(intent, names)
        chosen = accounts.resolve(named, [a for a in owned if a.account_id not in settled.values()]) if named else None
        if chosen is not None:
            settled[slot] = chosen.account_id
    return intent.model_copy(update={"slots": {**intent.slots, **settled}})


def prepare(intent: Intent, user_id: str, db: Any) -> Intent | RouteOutcome:
    """Settle which of the customer's own accounts a request names. Never another customer's.

    A name that matches none or several, a missing account, or a number outside the list all give a
    question with the choices written here. A customer with a single account to choose from is not
    asked. An id that belongs to someone else reads exactly like an unknown one."""
    if intent.kind is K.PAY_CREDIT_CARD:
        return _prepare_payment(intent, user_id, db)
    roles = _ROLES.get(intent.kind)
    if roles is None:
        return intent
    owned = accounts.accounts_of(db, user_id)
    if not owned:
        return RouteOutcome(
            intent_id=intent.id,
            status=OutcomeStatus.DONE,
            text="I do not see an open bank account on your profile, so there is nothing to change or move.",
        )
    if intent.kind is K.TRANSFER_FUNDS:
        return _settle_what_is_named(intent, roles, owned)
    settled: dict[str, str] = {}
    for slot, names in roles:
        named = next(((intent.slots.get(n) or "").strip() for n in names if (intent.slots.get(n) or "").strip()), "")
        choices = [a for a in owned if a.account_id not in settled.values()]
        if named:
            chosen = accounts.resolve(named, choices)
        else:
            chosen = choices[0] if len(choices) == 1 and intent.kind is K.CLOSE_ACCOUNT else None
        if chosen is None:
            return _which(intent, slot, choices)
        settled[slot] = chosen.account_id
    return intent.model_copy(update={"slots": {**intent.slots, **settled}})


VALIDATORS: dict[str, Validator] = {
    "open_bank_account_4821": rules.open_account,
    "close_bank_account_7392": rules.close_account,
    "transfer_funds_between_bank_accounts_7291": rules.transfer,
    "pay_credit_card_from_checking_9182": rules.pay_card,
}
# Checks after a write that the change is really there. Opening can report success and write nothing.
READBACKS: dict[str, Validator] = {
    "open_bank_account_4821": rules.opened_readback,
    "close_bank_account_7392": rules.closed_readback,
    "transfer_funds_between_bank_accounts_7291": rules.transfer_readback,
}

ACCOUNTS_SPECIALIST = SpecialistConfig(
    name="accounts",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=ACCOUNTS,
    validators=VALIDATORS,
    readbacks=READBACKS,
    prepare=prepare,
    reads=READS,
    workflows=(OpenAccount(), CloseAccount(), TransferFunds(), PayCard()),
)
