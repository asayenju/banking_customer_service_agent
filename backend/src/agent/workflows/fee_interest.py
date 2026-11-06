"""A correction for savings interest paid at the wrong rate, and the report to the backend team that follows it.

The model reports which savings accounts the customer named (none named means all of them). Code works out
what the latest month's interest should have been (``agent/interest_review.py``) and proposes, per account,
an interest correction for the difference and then the discrepancy report, in that order. The customer's own
figures never enter: they are often wrong (task 094's believes $72, the records say $140).

The report changes nothing the bank can show, so it is never filed alone and is never described as a credit.
An account that already has a correction for that payment, a level without a table, or a balance that moved
after the interest was paid is left out and said so; nothing is guessed. When the customer disputes a rate
and nothing is owed, the answer is the rate's parts, in words, and no write.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from agent import bank_accounts as accounts
from agent.bank_accounts import BankAccount, Product
from agent.interest_review import InterestReview, review_interest
from agent.specialist import ValidationContext
from agent.specialists import fees_rules as rules
from agent.specialists.accounts_rules import dollars
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Reply, Step, WorkflowContext
from orchestrator.types import IntentKind


class InterestFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accounts: list[str] = Field(
        description=(
            "The savings accounts the customer says were paid too little interest, each as the request's details or "
            "the customer name it ('Silver', 'the Gold one'). Leave empty if the customer did not name any."
        )
    )


def _listing(options: list[BankAccount]) -> str:
    return "\n".join(f"{n}. {accounts.describe(a)}" for n, a in enumerate(options, start=1))


def _parts(review: InterestReview) -> str:
    bits = [f"a base rate of {review.base:g}%"]
    if review.checking_boost:
        bits.append(f"+{review.checking_boost:g}% from a linked checking account")
    if review.card_bonus:
        bits.append(f"+{review.card_bonus:g}% from a credit card")
    if review.relationship_bonus:
        bits.append(f"+{review.relationship_bonus:g}% relationship bonus")
    return ", ".join(bits)


class SavingsInterestCorrection:
    name = "savings_interest_correction"
    kinds = frozenset({IntentKind.FEE_OR_INTEREST_ADJUSTMENT})
    summary = "The customer thinks a savings account was paid too little interest and wants it corrected."
    facts = InterestFacts

    def decide(self, ctx: WorkflowContext, facts: InterestFacts) -> Decision:
        options = [a for a in accounts.accounts_of(ctx.database, ctx.user_id) if a.is_open and a.product is Product.SAVINGS]
        if not options:
            return Reply("I do not see an open personal savings account on your profile, so there is no interest to review.")
        chosen: list[BankAccount] = []
        for named in facts.accounts:
            found = accounts.resolve(named, options)
            if found is None:
                return NeedInfo((Ask("accounts", f"Which savings account do you mean?\n{_listing(options)}"),))
            if found not in chosen:
                chosen.append(found)
        chosen = chosen or options

        steps: list[Step] = []
        left_out: list[str] = []
        fine: list[str] = []
        decided = 0
        for account in chosen:
            label = accounts.label(account)
            review = review_interest(ctx.database, account, ctx.user_id)
            if review.cannot_decide is not None:
                left_out.append(f"{label}: I could not confirm its interest ({review.cannot_decide})")
                continue
            decided += 1
            if review.corrected:
                left_out.append(f"{label}: an interest correction was already applied for the latest payment")
                continue
            if review.credit <= 0:
                fine.append(f"{label} should earn {review.expected_apy:g}% ({_parts(review)}), which is what it was paid")
                continue
            credit = {"account_id": account.account_id, "amount": review.credit, "credit_type": "interest_correction"}
            report = {
                "account_id": account.account_id, "user_id": ctx.user_id, "expected_apy": review.expected_apy,
                "actual_apy": review.actual_apy, "amount_difference": review.credit,
            }
            if (why := rules.savings_credit(ValidationContext(ctx.database, ctx.user_id, ctx.today), credit)) is not None:
                left_out.append(f"{label}: {why}")
                continue
            note = f"Credit {dollars(review.credit)} to {label}: {review.working()}"
            steps.append(Step(rules.SAVINGS_CREDIT_TOOL, credit, note=note))
            steps.append(Step(rules.REPORT_TOOL, report, note=f"Send a report to the bank's interest team about {label} (a report for review, not a credit)"))
        if not steps:
            if decided == 0 and left_out:
                return HandOff("; ".join(left_out))
            text = "I checked the latest interest payment against the rate each account should earn."
            if fine:
                text += " " + "; ".join(fine) + "."
            if left_out:
                text += " Nothing to change for: " + "; ".join(left_out) + "."
            return Reply(text)
        if left_out or fine:
            first = steps[0]
            extra = "; ".join(left_out + fine)
            steps[0] = Step(first.tool, first.args, note=f"{first.note} (other accounts, no change: {extra})")
        return Plan(tuple(steps))

    def after(self, ctx: WorkflowContext, steps: Sequence[Step]) -> str:
        if any(s.tool == rules.REPORT_TOOL for s in steps):
            return "The report has been submitted to the bank's team for review; it is a report, not a credit, and the credit above is the only money that moved."
        return ""
