"""The fraud and loss composer: a configuration of ``Specialist`` with no tools of its own.

It handles ``REPORT_FRAUD_OR_LOSS``. The one workflow, ``WalletLoss``, joins the debit and credit step
builders into a single plan (``docs/prd-fraud-loss-composer.md``). Every step is still checked by the code
guard (the kind's allow-list and the customer's ownership) and by the validator of the specialist the step
comes from, which is why the validators of both are merged here. A freeze is made before this runs
(``agent/containment.py``), so a card in the wallet is already frozen when the plan is made.
"""

from agent.specialist import SpecialistConfig, Validator
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
3. A card the customer found is not part of the report. Do not list it.
4. Report only what the customer said. Say the loss is definite only if the customer said the cards are gone
   for sure; "I think I lost it" is not definite. The fees, the waiting periods, which cards can be closed and
   the order of the steps are worked out for you. If the workflow refuses, it says why: fix the form or ask.
5. You have no tool for account takeover, identity theft, a lost phone or ID, a suspicious login, or a card
   with a security hold. For those, search the notes, say what can be done and what cannot, and ask whether
   the customer wants a person; end with finish. Never say a card is flagged for fraud and never mention a
   fraud score.
6. End with start_wallet_loss once you know which cards were in the wallet."""

# A step comes from the debit or the credit specialist: its own validator checks it.
VALIDATORS: dict[str, Validator] = {**debit.VALIDATORS, **credit.VALIDATORS}

FRAUD_SPECIALIST = SpecialistConfig(
    name="fraud",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=FRAUD,
    validators=VALIDATORS,
    readbacks=credit.READBACKS,
    workflows=(WalletLoss(),),
)
