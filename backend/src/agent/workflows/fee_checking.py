"""Credit for ATM fees charged above an account's fee schedule.

The model reports which checking accounts the customer asked about (none named means all of them).
Code reviews every named account against its fee schedule (``agent/fee_review.py``), nets over-charges
against under-charges, and proposes one credit per account, never the customer's own figure. An account
with no schedule on file, a withdrawal whose place cannot be told, or a credit already given in the last
14 days is left out and said so in the plan; nothing is guessed.

The plan is one step per account with something owed; the customer approves the exact amounts shown.
"""

from pydantic import BaseModel, ConfigDict, Field

from agent import bank_accounts as accounts
from agent.bank_accounts import BankAccount, Product
from agent.fee_review import FeeReview, recent_credit, review_atm_fees
from agent.specialist import ValidationContext
from agent.specialists import fees_rules as rules
from agent.specialists.accounts_rules import dollars
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Reply, Step, WorkflowContext
from orchestrator.types import IntentKind

CHECKING_CREDIT_TOOL = rules.CHECKING_CREDIT_TOOL


class CheckingFeeFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accounts: list[str] = Field(
        description=(
            "The checking accounts the customer asked about, each as the request's details or the customer name it "
            "('Bluest', 'the Light Green one'). Leave empty if the customer did not name any: all are reviewed."
        )
    )


def _listing(options: list[BankAccount]) -> str:
    return "\n".join(f"{n}. {accounts.describe(a)}" for n, a in enumerate(options, start=1))


class CheckingFeeCredit:
    name = "checking_fee_credit"
    kinds = frozenset({IntentKind.FEE_OR_INTEREST_ADJUSTMENT})
    summary = "The customer thinks ATM fees on a checking account were charged wrongly and wants them reviewed or refunded."
    facts = CheckingFeeFacts

    def decide(self, ctx: WorkflowContext, facts: CheckingFeeFacts) -> Decision:
        options = [a for a in accounts.accounts_of(ctx.database, ctx.user_id) if a.is_open and a.product is Product.CHECKING]
        if not options:
            return Reply("I do not see an open personal checking account on your profile, so there are no ATM fees to review.")
        chosen: list[BankAccount] = []
        for named in facts.accounts:
            found = accounts.resolve(named, options)
            if found is None:
                return NeedInfo((Ask("accounts", f"Which checking account do you mean?\n{_listing(options)}"),))
            if found not in chosen:
                chosen.append(found)
        chosen = chosen or options

        steps: list[Step] = []
        left_out: list[str] = []
        reviewed_ok = 0
        for account in chosen:
            label = accounts.label(account)
            if (last := recent_credit(ctx.database, account, ctx.today)) is not None:
                left_out.append(f"{label}: a fee credit was already applied on {last:%m/%d/%Y}, and the policy allows one every 14 days")
                continue
            review: FeeReview = review_atm_fees(ctx.database, account)
            if review.cannot_decide is not None:
                left_out.append(f"{label}: I could not confirm its fees ({review.cannot_decide})")
                continue
            reviewed_ok += 1
            if review.credit <= 0:
                continue
            args = {"account_id": account.account_id, "amount": review.credit, "credit_type": review.credit_type}
            if (why := rules.checking_credit(ValidationContext(ctx.database, ctx.user_id, ctx.today), args)) is not None:
                left_out.append(f"{label}: {why}")
                continue
            steps.append(Step(CHECKING_CREDIT_TOOL, args, note=f"Credit {dollars(review.credit)} to {label} for ATM fees charged above its fee schedule"))
        if not steps:
            if reviewed_ok == 0 and left_out:
                return HandOff("; ".join(left_out))
            text = "I reviewed the ATM fees on " + ("that account" if len(chosen) == 1 else "those accounts") + " against their fee schedules and found nothing charged above them, so there is nothing to credit."
            if left_out:
                text += " Not reviewed: " + "; ".join(left_out) + "."
            return Reply(text)
        if left_out:
            first = steps[0]
            steps[0] = Step(first.tool, first.args, note=f"{first.note} (not credited: {'; '.join(left_out)})")
        return Plan(tuple(steps))
