"""The fraud and loss composer: a configuration of ``Specialist`` with no tools of its own.

It handles ``REPORT_FRAUD_OR_LOSS``. The one workflow, ``WalletLoss``, joins the debit and credit step
builders into a single plan (``docs/prd-fraud-loss-composer.md``). Every step is still checked by the code
guard (the kind's allow-list and the customer's ownership) and by the validator of the specialist the step
comes from, which is why the validators of both are merged here. A freeze is made before this runs
(``agent/containment.py``), so a card in the wallet is already frozen when the plan is made.
"""

import re
from collections.abc import Mapping
from typing import Any, Optional

from agent.specialist import SpecialistConfig, ValidationContext, Validator
from agent.specialists import credit, debit
from agent.workflows.fraud_wallet import WalletLoss
from knowledge.scopes import FRAUD
from orchestrator.types import IntentKind as K

KINDS = frozenset({K.REPORT_FRAUD_OR_LOSS})

INSTRUCTIONS = """You handle reports of lost or stolen cards and wallets and suspected fraud on cards.
How to work:
1. Search the policy notes for the lost and stolen card protocol before you act. The internal
   procedures are for you; do not quote them to the customer.
2. A lost wallet usually holds several cards. Find the customer's debit cards and credit cards with your read
   tools. Never guess which cards were in the wallet: if the customer did not say, end with ask_customer.
3. The detail "said" is what the customer wrote when they reported it: the debit cards they name there were in the
   wallet. List every one of them, unless the customer later said they found it. A card the customer found is not
   part of the report.
4. Report only what the customer said. loss_is_definite is true when the customer says the cards are gone, lost
   or stolen for certain ("definitely gone", "I'm sure", "it was stolen", "cancel them", "I've searched everywhere"),
   and false when they only think so ("I think I lost it", "might be lost", "can't find it yet"). The fees, the waiting periods, which cards can be closed and
   the order of the steps are worked out for you. If the workflow refuses, it says why: fix the form or ask.
5. You have no tool for account takeover, identity theft, a lost phone or ID, a suspicious login, or a card
   with a security hold. For those, search the notes, say what can be done and what cannot, and ask whether
   the customer wants a person; end with finish. Never say a card is flagged for fraud and never mention a
   fraud score.
6. End with start_wallet_loss as soon as you know which debit cards were in the wallet (the customer's own words
   and their cards tell you). It asks the customer anything else that is needed, in its own words: whether a credit
   card was in the wallet, an excess fee, a pending refund. Do not use ask_customer for those, and never invent a
   detail name. Use ask_customer only when you cannot tell which cards were in the wallet.
7. credit_answer: none_were_in_the_wallet if the customer has no credit cards or said none were in the wallet;
   some_were_in_the_wallet if they named one; otherwise not_asked_yet. Leave credit_cards empty unless they named one."""

# What a plain answer about fraud may not do (docs/fraud-policy-decisions.md, section 3).
_DISCLOSURE = re.compile(r"\bflag(?:ged|s)?\b|\b(?:fraud|risk|security)\s+scor(?:e|es|ing)\b|\bscore of\b", re.IGNORECASE)
_INVENTED_ACTION = re.compile(
    r"\b(?:I|we)(?:'ve|\s+have|'ll|\s+will|\s+am going to|\s+are going to)\s+(?:reported|report|filed|file|notified|"
    r"notify|alerted|alert|lock(?:ed)?|block(?:ed)?|freez(?:e|ing)|froze|(?:changed|change|reset) your password|"
    r"put a hold|placed a hold)\b",
    re.IGNORECASE,
)
_OFFERS_A_PERSON = re.compile(r"\b(?:person|human|agent|specialist|security team|representative)\b", re.IGNORECASE)


def answer_check(answer: str) -> Optional[str]:
    """The three rules for a fraud answer with no action behind it: never say a card is flagged or give a
    score (037, 041), never claim an action there is no tool for, and ask whether the customer wants a person."""
    if _DISCLOSURE.search(answer):
        return "never say a card or account is flagged and never mention a fraud or risk score; say only what the customer can do"
    if _INVENTED_ACTION.search(answer):
        return "you have no tool to report, alert, lock or block anything here: say what can be done and what cannot"
    if "?" not in answer or not _OFFERS_A_PERSON.search(answer):
        return "end by asking whether the customer wants to be connected with a person"
    return None


def close_card(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """The debit rule, and the one the bank's tool never checks (doc 025): no pending purchases. The workflow
    checks it when it plans; this checks it again just before the step runs, since a purchase can appear between
    the preview and the yes. A pending refund is not re-checked: a refund to a closed card is credited to the
    checking account anyway (025), so closing never loses it."""
    problem = debit.close(v, args)
    if problem is not None:
        return problem
    card = v.database.debit_cards.data.get(str(args.get("card_id"))) or {}
    if any(
        t.get("account_id") == card.get("account_id")
        and str(t.get("status")).lower() == "pending"
        and t.get("type") != "refund"
        for t in v.database.bank_account_transaction_history.data.values()
    ):
        return "this card's account has purchases still pending, so the card cannot be closed yet"
    return None


# A step comes from the debit or the credit specialist: its own validator checks it.
VALIDATORS: dict[str, Validator] = {
    **debit.VALIDATORS,
    **credit.VALIDATORS,
    "close_debit_card_4721": close_card,
}

FRAUD_SPECIALIST = SpecialistConfig(
    name="fraud",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=FRAUD,
    validators=VALIDATORS,
    readbacks=credit.READBACKS,
    # It never hands over by itself: a person is offered, and only the customer's yes (a request for a
    # person, which the orchestrator routes) transfers.
    terminals=("finish", "ask_customer"),
    workflows=(WalletLoss(),),
    answer_check=answer_check,
)
