"""Closing accounts, funds first: the model reports which account and where the money goes, code decides.

Source: documents 005 to 008 and ``docs/accounts-policy-decisions.md`` (sections 1 and 3). An account closes
with a zero balance, or, inside its early-closure window, with exactly the fee in it. The bank's tool checks
that sum but never takes the fee, so the fee-sized balance stays where it is and nothing here ever moves it.

  * Outside the window the whole balance moves out first; inside it, everything above the fee does.
  * The money goes to an account the customer chooses from their own open accounts, never to one they are
    closing in the same request. Nothing else is offered and nothing else is accepted.
  * Inside the window with less than the fee, the account cannot close. The customer is told how much to add;
    adding it (a transfer from another account, or a check they deposit themselves) is their step, and the
    close is planned on a later turn from the balance then found (task 061).
  * A tier that needs a manager's or supervisor's approval, an account with no closing rules on file, and a
    request to waive the fee are handed to a person. The agent never sets the waiver.

The plan is the transfer, then the closing, for each account in the order given.
"""

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from agent import bank_accounts as accounts
from agent.bank_accounts import BankAccount
from agent.specialists.accounts_rules import cents, dollars, fee_owed
from agent.workflows import account_products as products
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from orchestrator.types import IntentKind

CLOSE_TOOL = "close_bank_account_7392"
TRANSFER_TOOL = "transfer_funds_between_bank_accounts_7291"
DEFAULT_REASON = "Customer requested closure"


class Move(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    to_account: str = Field(description="An account of the customer's own that should receive money, as they named it.")
    amount: Optional[float] = Field(
        description="The dollar amount the customer wants moved there, or null if they want all of it to go to this account."
    )


class Closure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: str = Field(description="The id of the account to close, as given in the request's details.")
    moves: list[Move] = Field(
        description=(
            "Where the customer wants the money in this account to go before it closes: one entry per account "
            "that should receive some. Empty if they have not said. Do not choose for them."
        )
    )
    reason: str = Field(description="Why the customer is closing it, in a few words; 'Customer requested closure' if they did not say.")
    asks_to_waive_fee: bool = Field(description="True only if the customer asks for the early-closure fee to be waived.")


class CloseFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    closures: list[Closure] = Field(min_length=1, max_length=4)


class CloseAccount:
    name = "close_account"
    kinds = frozenset({IntentKind.CLOSE_ACCOUNT})
    summary = "The customer wants to close one or more of their bank accounts."
    facts = CloseFacts

    def decide(self, ctx: WorkflowContext, facts: CloseFacts) -> Decision:
        db, user, today = ctx.database, ctx.user_id, ctx.today
        closing = {c.account_id for c in facts.closures}
        steps: list[Step] = []
        questions: list[Ask] = []
        for wanted in facts.closures:
            account = accounts.get(db, user, wanted.account_id)
            if account is None:
                return Refuse("that account is not one of this customer's accounts; use the id from the request's details")
            if account.status == "CLOSED":
                return Reply(f"Your {account.name} is already closed.")
            if wanted.asks_to_waive_fee:
                return HandOff("the customer asks for the early-closure fee to be waived, which only a person can decide")
            rules = products.closing_rules(account.product, account.name)
            if rules is None:
                return HandOff(f"there are no closing rules on file for a {account.name}")
            if rules.approval:
                return HandOff(f"closing a {account.name} needs {rules.approval} approval")
            if account.balance is None:
                return HandOff(f"the balance of the {account.name} is not on file")
            owed = fee_owed(account, today)
            if owed is None:
                return HandOff(f"the date the {account.name} was opened is not on file, so its fee cannot be worked out")
            if cents(account.balance) < owed * 100:
                return Reply(_short_of_fee(account, owed))
            to_move = round(account.balance - owed, 2)
            if cents(to_move) > 0:
                moves, ask = self._moves(ctx, account, wanted, closing, to_move)
                if moves is None:
                    questions.extend(ask)
                    continue
                steps.extend(
                    Step(TRANSFER_TOOL, {"source_account_id": account.account_id, "destination_account_id": to.account_id, "amount": amount})
                    for to, amount in moves
                )
            steps.append(Step(CLOSE_TOOL, {"account_id": account.account_id, "reason": wanted.reason.strip() or DEFAULT_REASON}))
        if questions:
            return NeedInfo(tuple(questions))
        return Plan(tuple(steps))

    def _moves(
        self, ctx: WorkflowContext, account: BankAccount, wanted: Closure, closing: set[str], total: float
    ) -> tuple[Optional[list[tuple[BankAccount, float]]], list[Ask]]:
        """Where the money goes: every destination one of the customer's own open accounts that is not being
        closed, and the amounts adding up to exactly what has to leave. Otherwise the question to ask."""
        options = [a for a in accounts.accounts_of(ctx.database, ctx.user_id) if a.is_open and a.account_id != account.account_id and a.account_id not in closing]
        have = dollars(account.balance or 0.0)
        if not options:
            return None, [Ask("move_to_account", f"Your {account.name} has {have}, and you have no other open account to move {dollars(total)} to before it closes.")]
        listing = "\n".join(f"{n}. {accounts.describe(a)}" for n, a in enumerate(options, start=1))
        resolved = [(accounts.resolve(m.to_account, options), m.amount) for m in wanted.moves]
        targets = [(to, amount) for to, amount in resolved if to is not None]
        if not resolved or len(targets) != len(resolved):
            return None, [Ask("move_to_account", f"Your {account.name} has {have}. Where should {dollars(total)} go before it closes?\n{listing}")]
        if len(targets) == 1 and targets[0][1] is None:
            return [(targets[0][0], total)], []
        given = [(to, amount) for to, amount in targets if amount is not None and amount > 0]
        if len(given) != len(targets) or sum(cents(amount) for _, amount in given) != cents(total):
            return None, [Ask("move_amounts", f"{dollars(total)} has to leave your {account.name}. How much should go to each account?\n{listing}")]
        return [(to, float(amount)) for to, amount in given], []

    def after(self, ctx: WorkflowContext, steps: Sequence[Step]) -> str:
        """What to tell the customer once the plan has run: a fee that was inside the account stays there."""
        notes = []
        for s in steps:
            if s.tool != CLOSE_TOOL:
                continue
            account = accounts.get(ctx.database, ctx.user_id, s.args["account_id"])
            if account is None:
                continue
            owed = fee_owed(account, ctx.today) if account.balance else 0
            if owed and cents(account.balance or 0.0) == owed * 100:
                notes.append(f"The {dollars(float(owed))} early-closure fee on your {account.name} was taken from the balance left in it.")
        return " ".join(notes)


def _short_of_fee(account: BankAccount, owed: int) -> str:
    have = account.balance or 0.0
    return (
        f"Your {account.name} was opened less than {_window(account)} days ago, so closing it now costs an early-closure fee of "
        f"{dollars(float(owed))}, taken from its balance. It has {dollars(have)}, so it needs {dollars(owed - have)} more before "
        "it can close. You can add that by moving money in from another of your accounts, or by depositing a check yourself; "
        "then ask me to close it again."
    )


def _window(account: BankAccount) -> int:
    rules = products.closing_rules(account.product, account.name)
    return rules.window_days if rules is not None else 0
