"""Opening accounts: the model reports what the customer wants, code decides what can be opened.

Source: documents 001 to 004 (opening), the product pages, and ``docs/accounts-policy-decisions.md``.
The model names the type and the exact class of each account the customer wants and says whether the
customer wants the opening deposit moved now. Code does the rest:

  * the class must be one the customer can be offered today (``account_catalogue``: the validator's own
    rules, promotions only between their dates). A class that is not an exact name gets the list back;
  * an account the customer cannot have is never put in a plan. If none can be opened, the customer is
    told why in words written here, from the records, whatever they claimed (task 057: "it has been 14
    days"). If only some can, the customer is asked whether to open those;
  * after a savings or business savings account the customer is asked about the opening deposit (document
    002 and 004): "yes" needs a source account of their own and an amount, "no" is told they have 30 days;
  * the plan is the opening, then the deposit transfer into the new account. The new account's id is a hash
    of user, type and class, so code writes it into the transfer instead of waiting for the opening's result.

The customer sees the exact class names in the plan and approves them with one yes, so nothing is opened
under a name the customer has not seen.
"""

from collections.abc import Sequence
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from tau2.domains.banking_knowledge.utils import _deterministic_id

from agent import bank_accounts as accounts
from agent.bank_accounts import Product
from agent.workflows import account_catalogue as catalogue
from agent.workflows import account_products as products
from agent.workflows.base import Ask, Decision, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from orchestrator.types import IntentKind

_TYPE: dict[str, Product] = {
    "checking": Product.CHECKING,
    "savings": Product.SAVINGS,
    "business_checking": Product.BUSINESS_CHECKING,
    "business_savings": Product.BUSINESS_SAVINGS,
}
_FUNDED = (Product.SAVINGS, Product.BUSINESS_SAVINGS)
OPEN_TOOL = "open_bank_account_4821"
TRANSFER_TOOL = "transfer_funds_between_bank_accounts_7291"
FUNDING_DAYS = 30


class NewAccount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account_type: Literal["checking", "savings", "business_checking", "business_savings"]
    account_class: str = Field(
        description=(
            "The exact class name of the account, as the list of accounts this customer can open gives it, "
            "for example 'Silver Plus Account' or 'Navy Blue'. If you do not have that list yet, use an empty "
            "string and the list is returned to you."
        )
    )
    fund_now: Literal["yes", "no", "not_asked"] = Field(
        description=(
            "For a savings or business savings account only: 'yes' if the customer wants its opening deposit "
            "moved from one of their accounts now, 'no' if they do not, 'not_asked' if they have not said. "
            "Use 'no' for checking accounts."
        )
    )
    fund_from_account: Optional[str] = Field(
        description="When fund_now is 'yes': the account the money comes from, as the customer named it. Otherwise null."
    )
    fund_amount: Optional[float] = Field(
        description="When fund_now is 'yes': the dollar amount the customer wants to move. Otherwise null."
    )


class OpenFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accounts: list[NewAccount] = Field(min_length=1, max_length=4)


def new_account_id(user_id: str, account_type: str, account_class: str) -> str:
    """The id the bank gives the account: a hash of user, type and class (open_bank_account_4821)."""
    return _deterministic_id(f"account:{user_id}:{account_type}:{account_class}")


class OpenAccount:
    name = "open_account"
    kinds = frozenset({IntentKind.OPEN_ACCOUNT})
    summary = "The customer wants to open one or more bank accounts: checking, savings, business checking or business savings."
    facts = OpenFacts

    def decide(self, ctx: WorkflowContext, facts: OpenFacts) -> Decision:
        db, user, today = ctx.database, ctx.user_id, ctx.today
        openable: list[NewAccount] = []
        refused: list[tuple[str, str]] = []
        for wanted in facts.accounts:
            product = _TYPE[wanted.account_type]
            if not products.is_known_class(product, wanted.account_class):
                listing = catalogue.render(catalogue.candidates(db, user, today, product), today)
                return Refuse(
                    f"{wanted.account_class!r} is not the exact name of a {product.value} account. Use one of "
                    f"these exactly, and tell the customer about the ones that suit what they asked for.\n{listing}"
                )
            chosen = catalogue.choose(db, user, today, product, wanted.account_class)
            if isinstance(chosen, str):
                refused.append((wanted.account_class, chosen))
            else:
                openable.append(wanted)
        if refused and not openable:
            return Reply(_cannot_open(refused))
        if refused:
            names = _join([f"a {a.account_class}" for a in openable])
            return NeedInfo((Ask("account_class", f"I can open {names} now, but not everything you asked for: {_cannot_open(refused)} Shall I open just {names}?"),))

        questions: list[Ask] = []
        steps: list[Step] = []
        for wanted in openable:
            product = _TYPE[wanted.account_type]
            steps.append(Step(OPEN_TOOL, {"account_type": wanted.account_type, "account_class": wanted.account_class}))
            if product not in _FUNDED:
                continue
            if wanted.fund_now == "not_asked":
                questions.append(
                    Ask(
                        "fund_now",
                        f"Would you like me to move the opening deposit for your {wanted.account_class} from one of your "
                        f"accounts right after it is opened? If not, you have {FUNDING_DAYS} days to fund it or it will be closed.",
                    )
                )
            elif wanted.fund_now == "yes":
                move, ask = self._funding(ctx, wanted)
                questions.extend(ask)
                if move is not None:
                    steps.append(move)
        if questions:
            return NeedInfo(tuple(questions))
        return Plan(tuple(steps))

    def _funding(self, ctx: WorkflowContext, wanted: NewAccount) -> tuple[Optional[Step], list[Ask]]:
        owned = [a for a in accounts.accounts_of(ctx.database, ctx.user_id) if a.is_open]
        source = accounts.resolve(wanted.fund_from_account or "", owned) if wanted.fund_from_account else None
        asks: list[Ask] = []
        if source is None:
            listing = "\n".join(f"{n}. {accounts.describe(a)}" for n, a in enumerate(owned, start=1))
            asks.append(Ask("fund_from_account", f"Which account should the opening deposit for your {wanted.account_class} come from?\n{listing}"))
        if wanted.fund_amount is None or wanted.fund_amount <= 0:
            asks.append(Ask("fund_amount", f"How much would you like to move into your {wanted.account_class}?"))
        if asks or source is None or wanted.fund_amount is None:
            return None, asks
        destination = new_account_id(ctx.user_id, wanted.account_type, wanted.account_class)
        return Step(TRANSFER_TOOL, {"source_account_id": source.account_id, "destination_account_id": destination, "amount": float(wanted.fund_amount)}), []

    def after(self, ctx: WorkflowContext, steps: Sequence[Step]) -> str:
        """What to tell the customer once the plan has run: a savings account left unfunded is closed in 30 days."""
        funded = {s.args["destination_account_id"] for s in steps if s.tool == TRANSFER_TOOL}
        notes = []
        for s in steps:
            if s.tool != OPEN_TOOL or _TYPE[str(s.args["account_type"])] not in _FUNDED:
                continue
            if new_account_id(ctx.user_id, str(s.args["account_type"]), str(s.args["account_class"])) not in funded:
                notes.append(
                    f"Your {s.args['account_class']} needs its opening deposit within {FUNDING_DAYS} days, by a transfer "
                    "from one of your accounts or a deposit from outside the bank, or it will be closed."
                )
        return " ".join(notes)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _cannot_open(refused: list[tuple[str, str]]) -> str:
    reasons = "; ".join(f"a {name} cannot be opened now: {why}" for name, why in refused)
    return f"{reasons[0].upper()}{reasons[1:]}."
