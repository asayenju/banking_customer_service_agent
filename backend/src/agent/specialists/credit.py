"""The credit-card specialist: closing a card (with retention), a limit increase, a replacement
and a credit transaction dispute.

It is a configuration of ``Specialist``, like the debit one. This module holds what every
slice shares: the kinds, the instructions, the document scope and ``prepare``, which settles
in code which credit account is meant before a model gets involved. The validators and the
workflows are added by the later slices (see ``docs/prd-credit-card-specialist.md``).

Two decisions are fixed by ``docs/credit-policy-decisions.md`` and shown to the model here:
the bank's credit tools check almost nothing themselves, and a card type with no tier gets no
automatic offer.
"""

from typing import Any

from agent.cards import describe_credit, owned_credit_accounts, resolve_credit
from agent.specialist import SpecialistConfig, Validator
from knowledge.scopes import CREDIT_CARDS
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K, Question

KINDS = frozenset({K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.CREDIT_LIMIT_REQUEST, K.DISPUTE_TRANSACTION})

INSTRUCTIONS = """You handle credit card requests: closing a card, replacing a card, credit limit increases and
disputes of credit card transactions.
How to work:
1. Search the policy notes for the procedure for this request before you act. The internal
   procedures are for you; do not quote them to the customer.
2. Find the customer's credit card accounts with your read tools. If the customer did not say
   which card and more than one could be meant, ask. Do not choose.
3. The bank's tools check almost no rules themselves. Check the rules in the policy notes
   yourself, and expect your plan to be refused if it breaks one.
4. Believe the records, not the customer. If the customer says a card arrived, a payment was
   made or they own another card, look it up; if the records disagree, say what the records show.
5. A card closes only with a zero balance, no open dispute, no pending replacement and an
   account at least 60 days old. Before closing, log the reason and make the retention offer
   that the notes give for the card's tier. Never offer a downgrade or an upgrade; you cannot
   apply for a card for the customer.
6. For a limit increase, check the requested amount against the tier's cap before you submit.
   An open dispute or a pending replacement blocks an approval: do the limit request first when
   the customer wants several things.
7. A card type with no tier in the notes gets no automatic offer or limit rule. Hand over."""

# Kinds where "which credit account?" must be answered before anything else.
# A dispute is not here: its workflow finds the account from the transaction.
ACCOUNT_KINDS = frozenset({K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.CREDIT_LIMIT_REQUEST})
_NAMING_SLOTS = ("credit_card_account_id", "card_or_account_id", "account_id")


def _which(intent: Intent, accounts: list[Any]) -> RouteOutcome:
    listing = "\n".join(f"- {describe_credit(a)}" for a in accounts)
    return RouteOutcome(
        intent_id=intent.id,
        status=OutcomeStatus.NEEDS_INFO,
        questions=(
            Question(intent_id=intent.id, slot="credit_card_account_id", text=f"Which card do you mean?\n{listing}"),
        ),
    )


def prepare(intent: Intent, user_id: str, db: Any) -> Intent | RouteOutcome:
    """Settle which credit account is meant, in code. Never another customer's."""
    if intent.kind not in ACCOUNT_KINDS:
        return intent
    accounts = owned_credit_accounts(db, user_id)
    if not accounts:
        return intent
    named = next(((intent.slots.get(k) or "").strip() for k in _NAMING_SLOTS if (intent.slots.get(k) or "").strip()), "")
    if named:
        chosen = resolve_credit(named, accounts)
        if chosen is None:
            return _which(intent, accounts)
    elif len(accounts) == 1:
        chosen = accounts[0]
    else:
        return _which(intent, accounts)
    return intent.model_copy(update={"slots": {**intent.slots, "credit_card_account_id": str(chosen["account_id"])}})


# Filled in by the validator slice.
VALIDATORS: dict[str, Validator] = {}

CREDIT_SPECIALIST = SpecialistConfig(
    name="credit",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=CREDIT_CARDS,
    validators=VALIDATORS,
    prepare=prepare,
)
