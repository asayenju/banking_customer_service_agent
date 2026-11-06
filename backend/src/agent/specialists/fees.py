"""The fees, interest and rewards specialist: ATM fee credits on checking accounts, interest corrections on
savings accounts, and corrected rewards on card purchases.

It is a configuration of ``Specialist``, like the credit and accounts ones. The amounts and the eligibility
are computed here from the records and the policy tables (``agent/workflows/fee_products.py``), never taken
from the customer or a model; the bank's own tools check almost nothing. Every decision, with its
evidence, is in ``docs/fees-policy-decisions.md``; the plan is in ``docs/prd-fees-interest-rewards-specialist.md``.
"""

from agent.specialist import SpecialistConfig, Validator
from agent.specialists import fees_rules as rules
from agent.workflows.fee_checking import CheckingFeeCredit
from agent.workflows.fee_interest import SavingsInterestCorrection
from agent.workflows.fee_rewards import RewardsCorrection
from knowledge.scopes import FEES
from orchestrator.types import IntentKind as K

KINDS = frozenset({K.FEE_OR_INTEREST_ADJUSTMENT})

INSTRUCTIONS = """You handle requests about fees, interest and rewards on the customer's accounts.
How to work:
1. Search the policy notes for the procedure before you act. The internal procedures are for you; do not
   quote them to the customer.
2. Believe the records, not the customer. If they say a fee or an interest payment is wrong, the amount of
   any credit is worked out from the records and the published fee schedule; never from their figure.
3. A question or a complaint with no request for a change is answered from the notes. Do not propose a
   change unless the customer asked for one.
4. ATM fees on a checking account have a workflow: end with start_checking_fee_credit. Report only which
   checking accounts the customer named (none if they did not). Whether anything is owed, how much and what
   type of credit is worked out for you.
5. Savings interest paid at the wrong rate has a workflow: end with start_savings_interest_correction. Report
   only which savings accounts the customer named (none if they did not). The rate the account should earn,
   the amount owed and the report that follows are worked out for you; never use a figure the customer gives.
6. Wrong rewards on a card purchase has a workflow: end with start_rewards_correction. Report only which
   purchases the customer named (none if they did not). Rewards are corrected only after the customer's cash back
   dispute for the purchase was approved; the right number of points is worked out for you, never taken from the
   customer or the dispute record.
7. A credit card's annual fee, late fee or interest charge is not handled here; say a person will review it."""

READS = frozenset(
    {
        "get_all_user_accounts_by_user_id_3847",
        "get_bank_account_transactions_9173",
        "get_credit_card_accounts_by_user",
        "get_current_time",
        "get_user_information_by_id",
    }
)

VALIDATORS: dict[str, Validator] = {
    rules.CHECKING_CREDIT_TOOL: rules.checking_credit,
    rules.SAVINGS_CREDIT_TOOL: rules.savings_credit,
    rules.REPORT_TOOL: rules.interest_report,
    rules.REWARDS_TOOL: rules.rewards_update,
}
READBACKS: dict[str, Validator] = {
    rules.CHECKING_CREDIT_TOOL: rules.checking_credit_readback,
    rules.SAVINGS_CREDIT_TOOL: rules.savings_credit_readback,
    rules.REPORT_TOOL: rules.interest_report_readback,
    rules.REWARDS_TOOL: rules.rewards_readback,
}

FEES_SPECIALIST = SpecialistConfig(
    name="fees",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=FEES,
    validators=VALIDATORS,
    readbacks=READBACKS,
    reads=READS,
    workflows=(CheckingFeeCredit(), SavingsInterestCorrection(), RewardsCorrection()),
)
