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

from agent.credit_lookup import NAMES as LOOKUP_NAMES, credit_lookups
from agent.entities import ordinal_picks
from agent.cards import describe_credit, owned_credit_accounts, resolve_credit
from agent.specialist import SpecialistConfig, Validator
from agent.specialists import credit_rules as rules
from agent.workflows.credit_closure import CreditClosure
from agent.workflows.credit_dispute import CreditDispute
from agent.workflows.credit_limit import CreditLimit
from agent.workflows.credit_replacement import CreditReplacement
from knowledge.scopes import CREDIT_CARDS
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K, Question

KINDS = frozenset({K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.CREDIT_LIMIT_REQUEST, K.DISPUTE_TRANSACTION})

INSTRUCTIONS = """You handle credit card requests: closing a card, replacing a card, credit limit increases and
disputes of credit card transactions.
How to work:
1. Search the policy notes for the procedure for this request before you act. The internal
   procedures are for you; do not quote them to the customer.
2. LOOK THINGS UP BEFORE YOU ASK. You have search tools over the customer's own records:
   find_credit_transactions (merchant, rough amount, date, or a transaction id), list_credit_cards and
   credit_dispute_history, and look_up_customer_records. For a charge the customer describes, call
   find_credit_transactions first with whatever they said (the merchant alone is enough), and again with fewer filters if
   it finds nothing (dates may be written any way; a transaction id the customer gives is enough). You are
   not allowed to ask the customer anything until you have used one of them. Ask only if the search finds
   nothing or several, and then ask about one thing: which of the matches. Never ask for a transaction id,
   a card id, a card's last four digits or the date of a charge that a search can find; the customer cannot know
   them. If the customer did not say which card and more than one could be meant, ask. Do not choose.
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
7. A card type with no tier in the notes gets no automatic offer or limit rule. Hand over.
8. Closing a card, disputes, replacements and limit increases have a workflow: end with
   start_credit_closure, start_credit_dispute, start_credit_replacement or start_credit_limit. For a
   closure report the card, the reason once the customer gives it, and their answers to the offers;
   which offers are made, and the order, are worked out for you. For a limit increase report only the card and the
   dollar increase the customer asks for; the rules, the denial reason and the new limit are worked out.
   Report only what the customer said (which charge and why, whether they tried the merchant, when
   they noticed, the card's last four digits, which card, why, delivery speed and address); the
   provisional credit, the contact details and the limits are worked out for you. If the workflow
   refuses, it says why: fix the form or ask the customer."""

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
        if chosen is None:  # "the second one", "2": counted along the list they were shown
            picks = ordinal_picks(named.lower(), accounts)
            chosen = picks[0] if len(picks) == 1 else None
        if chosen is None:
            return _which(intent, accounts)
    elif len(accounts) == 1:
        chosen = accounts[0]
    else:
        return _which(intent, accounts)
    return intent.model_copy(update={"slots": {**intent.slots, "credit_card_account_id": str(chosen["account_id"])}})


VALIDATORS: dict[str, Validator] = {
    "file_credit_card_transaction_dispute_4829": rules.dispute,
    "order_replacement_credit_card_7291": rules.replacement,
    "log_credit_card_closure_reason_4521": rules.log_reason,
    "close_credit_card_account_7834": rules.close,
    "apply_statement_credit_8472": rules.statement_credit,
    "apply_credit_card_account_flag_6147": rules.flag,
    "pay_credit_card_from_checking_9182": rules.pay,
    "submit_credit_limit_increase_request_7392": rules.submit,
    "approve_credit_limit_increase_5847": rules.approve,
    "deny_credit_limit_increase_5848": rules.deny,
}
# Checks after a write that the change is really there.
READBACKS: dict[str, Validator] = {
    "close_credit_card_account_7834": rules.close_readback,
    "approve_credit_limit_increase_5847": rules.approve_readback,
}

CREDIT_SPECIALIST = SpecialistConfig(
    name="credit",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=CREDIT_CARDS,
    validators=VALIDATORS,
    readbacks=READBACKS,
    prepare=prepare,
    workflows=(CreditClosure(), CreditDispute(), CreditReplacement(), CreditLimit()),
    lookups=credit_lookups,
    lookup_first=frozenset({K.DISPUTE_TRANSACTION}),
    lookup_names=LOOKUP_NAMES,
)
