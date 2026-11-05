"""Moving money: a transfer between the customer's own accounts, and a credit card payment from checking.

There is no gold task for either on its own (the 19 account tasks are openings and closings; 043, 045, 048
and 051 pay a card inside a credit flow, with the credit specialist's own validator), so the rules are tested on
synthetic data. The model reports which accounts and how much; code decides everything else.

  * Both accounts are the customer's own open accounts, found in their records. A name that matches none or
    several, and anyone else's account, are asked again with a list of the customer's own; nothing else is ever
    offered or accepted (a third party's account reads like one that does not exist).
  * The amount in the form must be the amount the customer gave. The orchestrator binds the customer's yes to
    the request as it holds it, so a form with another amount is refused instead of shown for approval.
  * What the bank's tools do not check (same owner for a transfer, a card payment from a checking account that
    is not open or that the bank's payment tool cannot see) is checked by the rules in ``accounts_rules``, and a
    refusal is told to the customer in those words.
  * A repeated run never moves the money twice: the session skips a write it already made (tests).

The plan is a single step; the customer approves it with the exact accounts and amount shown.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent import bank_accounts as accounts
from agent.bank_accounts import BankAccount, Product
from agent.cards import describe_credit, owned_credit_accounts, resolve_credit
from agent.specialist import ValidationContext
from agent.specialists import accounts_rules as rules
from agent.specialists.credit_rules import money
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from orchestrator.types import IntentKind

TRANSFER_TOOL = "transfer_funds_between_bank_accounts_7291"
PAY_TOOL = "pay_credit_card_from_checking_9182"


Where = Literal["my_own_account", "wire", "ach", "bill_pay", "another_person_or_bank", "check_deposit"]

# Document 011: a check is deposited by the customer in the mobile app; the agent cannot do it.
CHECK_DEPOSIT_STEPS = (
    "I can't deposit a check for you; you do that yourself in the Rho-Bank mobile app. Sign in, pick the account you want "
    "to deposit into, choose Mobile Check Deposit, enter the amount exactly as printed, and photograph the front and the "
    "endorsed back of the check. Standard deposits are usually available within 1 to 2 business days."
)


class TransferFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    destination: Where = Field(
        default="my_own_account",
        description=(
            "Where the money is going: 'my_own_account' for a move between the customer's own Rho-Bank accounts; "
            "'wire', 'ach' or 'bill_pay' for those; 'another_person_or_bank' for sending money to someone else or "
            "another bank; 'check_deposit' if the customer wants to deposit a check. The bank can only do the first."
        ),
    )

    from_account_id: str = Field(description="The account the money comes from, as the request's details or the customer name it.")
    to_account_id: str = Field(description="The account the money goes to, as the request's details or the customer name it.")
    amount: float = Field(description="The dollar amount, exactly as the customer gave it.")


class PayFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_card_account_id: str = Field(description="The credit card to pay, as the request's details or the customer name it.")
    checking_account_id: str = Field(description="The checking account to pay from, as the request's details or the customer name it.")
    amount: float = Field(description="The dollar amount, exactly as the customer gave it.")


def _amount_differs(ctx: WorkflowContext, amount: float) -> Optional[str]:
    said = money(ctx.slots.get("amount")) if ctx.slots.get("amount") else None
    if said is not None and rules.cents(said) != rules.cents(amount):
        return f"the customer gave the amount {rules.dollars(said)}; fill in that exact amount"
    return None


def _listing(options: list[BankAccount]) -> str:
    return "\n".join(f"{n}. {accounts.describe(a)}" for n, a in enumerate(options, start=1))


class TransferFunds:
    name = "transfer_funds"
    kinds = frozenset({IntentKind.TRANSFER_FUNDS})
    summary = "The customer wants to move money between two of their own bank accounts."
    facts = TransferFacts

    def decide(self, ctx: WorkflowContext, facts: TransferFacts) -> Decision:
        if facts.destination == "check_deposit":
            return Reply(CHECK_DEPOSIT_STEPS)
        if facts.destination != "my_own_account":
            means = facts.destination.replace("_", " ")
            return HandOff(f"the customer wants to send money by {means} to someone outside their own accounts, which the bank has no tool for")
        if rules.cents(facts.amount) <= 0:
            return Refuse("a transfer must be a dollar amount above zero")
        if (problem := _amount_differs(ctx, facts.amount)) is not None:
            return Refuse(problem)
        owned = [a for a in accounts.accounts_of(ctx.database, ctx.user_id) if a.is_open]
        source = accounts.resolve(facts.from_account_id, owned)
        questions: list[Ask] = []
        if source is None:
            questions.append(Ask("from_account_id", f"Which account should the money come from?\n{_listing(owned)}"))
        target_options = [a for a in owned if source is None or a.account_id != source.account_id]
        target = accounts.resolve(facts.to_account_id, target_options)
        if target is None:
            questions.append(Ask("to_account_id", f"Which account should the money go to?\n{_listing(target_options)}"))
        if questions or source is None or target is None:
            return NeedInfo(tuple(questions))
        args = {"source_account_id": source.account_id, "destination_account_id": target.account_id, "amount": float(facts.amount)}
        if (why := rules.transfer(ValidationContext(ctx.database, ctx.user_id, ctx.today), args)) is not None:
            return Reply(f"I can't move that money: {why}.")
        return Plan((Step(TRANSFER_TOOL, args, note=f"Move {rules.dollars(float(facts.amount))} from {accounts.label(source)} to {accounts.label(target)}"),))


class PayCard:
    name = "pay_card"
    kinds = frozenset({IntentKind.PAY_CREDIT_CARD})
    summary = "The customer wants to pay a credit card from one of their checking accounts."
    facts = PayFacts

    def decide(self, ctx: WorkflowContext, facts: PayFacts) -> Decision:
        if rules.cents(facts.amount) <= 0:
            return Refuse("a payment must be a dollar amount above zero")
        if (problem := _amount_differs(ctx, facts.amount)) is not None:
            return Refuse(problem)
        cards = owned_credit_accounts(ctx.database, ctx.user_id)
        card = resolve_credit(facts.credit_card_account_id, cards)
        checking_options = [a for a in accounts.accounts_of(ctx.database, ctx.user_id) if a.is_open and a.product is Product.CHECKING]
        checking = accounts.resolve(facts.checking_account_id, checking_options)
        questions: list[Ask] = []
        if card is None:
            questions.append(Ask("credit_card_account_id", "Which card would you like to pay?\n" + "\n".join(f"- {describe_credit(c)}" for c in cards)))
        if checking is None:
            questions.append(Ask("checking_account_id", f"Which checking account should the payment come from?\n{_listing(checking_options)}"))
        if questions or card is None or checking is None:
            return NeedInfo(tuple(questions))
        args = {
            "user_id": ctx.user_id,
            "checking_account_id": checking.account_id,
            "credit_card_account_id": str(card["account_id"]),
            "amount": float(facts.amount),
        }
        if (why := rules.pay_card(ValidationContext(ctx.database, ctx.user_id, ctx.today), args)) is not None:
            return Reply(f"I can't make that payment: {why}.")
        note = f"Pay {rules.dollars(float(facts.amount))} to {describe_credit(card)} from {accounts.label(checking)}"
        return Plan((Step(PAY_TOOL, args, note=note),))
