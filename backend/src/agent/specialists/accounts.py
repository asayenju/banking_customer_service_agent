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
from agent.specialist import SpecialistConfig, Validator
from agent.specialists import accounts_rules as rules
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
6. Closing: an account closes with a zero balance, or with exactly the early-closure fee in it when
   one applies. Move the money first, to an account the customer chooses from their own. Never move
   the fee-sized balance, and never waive a fee. A new account must be opened before an old one is
   closed when the new one needs the old one (a savings account needs a checking account of 14 days).
7. Wires, ACH, bill pay, overdraft, holds and check deposits have no tool. Answer questions from the
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


def prepare(intent: Intent, user_id: str, db: Any) -> Intent | RouteOutcome:
    """Settle which of the customer's own accounts a request names. Never another customer's.

    A name that matches none or several, a missing account, or a number outside the list all give a
    question with the choices written here. A customer with a single account to choose from is not
    asked. An id that belongs to someone else reads exactly like an unknown one."""
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
    workflows=(OpenAccount(),),
)
