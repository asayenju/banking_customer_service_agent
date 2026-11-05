"""The bank-account write tools' rules, as code.

The bank's account tools check very little: opening accepts any class name and enforces about half
of the eligibility rules, closing checks the early-closure fee but never takes it, and a transfer
does not check that both accounts have the same owner (docs/accounts-policy-decisions.md, section 2,
with a test for each statement). These validators are the policy documents written as code. Each is
checked at plan time, on the database as the earlier steps of the plan would leave it, and again just
before the step runs. A validator returns None to accept, or a short reason written here.

Another customer's account, a made-up id and a closed account all read the same to the customer:
"that account does not exist" (or is already closed), so nothing about anyone else leaks.

The tools' own failures are almost all loud ("Error: ..."). Two are not: opening an account whose id
already exists reports success and writes nothing, so the read-back confirms the account is there;
and the early-closure fee is checked but never debited, so nothing here ever moves a fee-sized balance.
"""

from collections.abc import Mapping
from datetime import date
from typing import Any, Optional

from tau2.domains.banking_knowledge.utils import _deterministic_id

from agent import bank_accounts as accounts
from agent.bank_accounts import BankAccount, Product
from agent.specialist import ValidationContext
from agent.specialists.credit_rules import day, money
from agent.workflows import account_products as products

# What the tool calls each type.
_TYPES: Mapping[str, Product] = {
    "checking": Product.CHECKING,
    "savings": Product.SAVINGS,
    "business_checking": Product.BUSINESS_CHECKING,
    "business_savings": Product.BUSINESS_SAVINGS,
}
NOT_FOUND = "that account does not exist"


def cents(value: float) -> int:
    return round(value * 100)


def dollars(value: float) -> str:
    return f"${value:,.2f}"


def _years_old(born: date, today: date) -> int:
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _mine(v: ValidationContext, account_id: Any) -> Optional[BankAccount]:
    return accounts.get(v.database, v.user_id, account_id)


def _live(held: list[BankAccount]) -> list[BankAccount]:
    return [a for a in held if a.status != "CLOSED"]


# ---------------------------------------------------------------- opening

def open_account(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    if str(args.get("user_id")) != v.user_id:
        return "an account can only be opened for the customer who is verified"
    product = _TYPES.get(str(args.get("account_type")))
    if product is None:
        return "that is not a type of account that can be opened"
    name = args.get("account_class")
    if not isinstance(name, str) or not products.is_known_class(product, name):
        return f"that is not the exact name of a {product.value} account; the bank's list gives the full official name"
    seed = f"account:{v.user_id}:{args.get('account_type')}:{name}"
    if _deterministic_id(seed) in v.database.accounts.data:
        # The bank would report success and write nothing.
        return f"you already have, or had, a {name}; a second one with the same name cannot be opened"

    held = accounts.accounts_of(v.database, v.user_id, include_closed=True)
    live = _live(held)
    if (too_young_or_old := _page_age(v, product, name)) is not None:
        return too_young_or_old
    if len([a for a in live if a.product is product]) >= products.CAP[product]:
        label = "personal checking" if product is Product.CHECKING else product.value
        return f"you already hold the most {label} accounts allowed ({products.CAP[product]})"
    if product is Product.CHECKING:
        return _may_open_checking(v)
    if product is Product.SAVINGS:
        return _may_open_savings(v, live)
    if product is Product.BUSINESS_CHECKING:
        return _may_open_business_checking(held, live)
    return _may_open_business_savings(v, live)


def _page_age(v: ValidationContext, product: Product, name: str) -> Optional[str]:
    """The holder age limits a product page states (Gold Years: 62 or older; Dark Green: 17 through 26)."""
    limits = products.AGE_LIMITS.get((product, name))
    if limits is None:
        return None
    user = v.database.users.data.get(v.user_id)
    born = day(user.get("date_of_birth")) if isinstance(user, Mapping) else None
    if born is None:
        return "your date of birth is not on file, so a person has to check that you can have this account"
    age = _years_old(born, v.today)
    youngest, oldest = limits
    if youngest is not None and age < youngest:
        return f"a {name} is for customers {youngest} or older"
    if oldest is not None and age > oldest:
        return f"a {name} is for customers {oldest} or younger"
    return None


def _may_open_checking(v: ValidationContext) -> Optional[str]:
    user = v.database.users.data.get(v.user_id)
    born = day(user.get("date_of_birth")) if isinstance(user, Mapping) else None
    if born is None:
        return "your date of birth is not on file, so a person has to check that you are 18 or older"
    if _years_old(born, v.today) < products.MIN_AGE_FOR_PERSONAL_CHECKING:
        return "a personal checking account is for customers 18 or older"
    return None


def _negative(live: list[BankAccount]) -> bool:
    return any(a.balance is not None and a.balance < 0 for a in live)


def _may_open_savings(v: ValidationContext, live: list[BankAccount]) -> Optional[str]:
    if _negative(live):
        return "an account with a negative balance has to be put right before a savings account can be opened"
    checking = [a for a in live if a.product is Product.CHECKING and a.is_open]
    if not checking:
        return "a savings account needs an open personal checking account"
    needed = products.MIN_CHECKING_AGE_FOR_SAVINGS_DAYS
    ages = [a.age_days(v.today) for a in checking]
    if not any(age is not None and age >= needed for age in ages):
        newest = max((age for age in ages if age is not None), default=None)
        held = f"your checking account has been open {newest} days" if newest is not None else "the age of your checking account is not on file"
        return f"a savings account needs a checking account that has been open at least {needed} days; {held}"
    return None


def _may_open_business_checking(held: list[BankAccount], live: list[BankAccount]) -> Optional[str]:
    if any(a.status == "CLOSED" for a in held):
        return "a business checking account cannot be opened when any of your accounts has been closed"
    personal = [a for a in live if a.product is Product.CHECKING and a.is_open]
    if not personal:
        return "a business checking account needs an open personal checking account"
    need = products.MIN_PERSONAL_CHECKING_BALANCE_FOR_BUSINESS
    if not any((a.balance or 0.0) >= need for a in personal):
        return f"a business checking account needs {dollars(need)} in your personal checking account"
    return None


def _may_open_business_savings(v: ValidationContext, live: list[BankAccount]) -> Optional[str]:
    if _negative(live):
        return "an account with a negative balance has to be put right before a business savings account can be opened"
    business = [a for a in live if a.product is Product.BUSINESS_CHECKING and a.is_open]
    if not business:
        return "a business savings account needs an open business checking account"
    days, need = products.MIN_BUSINESS_CHECKING_AGE_DAYS, products.MIN_BUSINESS_CHECKING_BALANCE_FOR_SAVINGS
    for a in business:
        age = a.age_days(v.today)
        if age is not None and age >= days and (a.balance or 0.0) >= need:
            return None
    return f"a business savings account needs a business checking account that has been open at least {days} days with {dollars(need)} in it"


def opened_readback(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """None if the account is really there, open, and of the type and name asked for."""
    seed = f"account:{v.user_id}:{args.get('account_type')}:{args.get('account_class')}"
    row = v.database.accounts.data.get(_deterministic_id(seed))
    account = accounts.read(row) if isinstance(row, Mapping) else None
    if account is None or account.user_id != v.user_id or not account.is_open or account.name != args.get("account_class"):
        return "the account is not there"
    return None


# ---------------------------------------------------------------- closing

def fee_owed(account: BankAccount, today: date) -> Optional[int]:
    """The early-closure fee in dollars that closing today would owe: 0 outside the window or when no fee is
    listed, None when it cannot be told (no opening date)."""
    rules = products.closing_rules(account.product, account.name)
    if rules is None:
        return 0
    age = account.age_days(today)
    if age is None:
        return None
    return rules.fee if age < rules.window_days else 0


def close_account(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = _mine(v, args.get("account_id"))
    if account is None:
        return NOT_FOUND
    if account.status == "CLOSED":
        return "that account is already closed"
    if args.get("waive_early_closure_fee"):
        return "the early-closure fee cannot be waived here; a person has to decide that"
    rules = products.closing_rules(account.product, account.name)
    if rules is None:
        return f"there are no closing rules on file for a {account.name}, so a person has to close it"
    if rules.approval:
        return f"closing a {account.name} needs {rules.approval} approval, so a person has to handle it"
    if any(
        str(t.get("account_id")) == account.account_id and str(t.get("status") or "").lower() != "posted"
        for t in v.database.bank_account_transaction_history.data.values()
        if isinstance(t, Mapping)
    ):
        return "the account has a transaction that has not posted yet; it can be closed when that clears"
    live = _live(accounts.accounts_of(v.database, v.user_id, include_closed=True))
    others = [a for a in live if a.account_id != account.account_id]
    if account.product is Product.BUSINESS_CHECKING and any(a.product is Product.BUSINESS_SAVINGS for a in others):
        return "a business checking account cannot be closed while a business savings account is open; close that first"
    if account.product is Product.BUSINESS_SAVINGS and not any(a.product is Product.BUSINESS_CHECKING and a.is_open for a in others):
        return "a business savings account needs an open business checking account to stay linked to"
    if account.balance is None:
        return "the balance of this account is not on file"
    owed = fee_owed(account, v.today)
    if owed is None:
        return "the date this account was opened is not on file, so the early-closure fee cannot be worked out"
    have = cents(account.balance)
    if owed > 0:
        if have < owed * 100:
            return (
                f"this account is inside its early-closure window, so it needs {dollars(owed)} in it to close; "
                f"it has {dollars(account.balance)}"
            )
        if have > owed * 100:
            return (
                f"move {dollars(account.balance - owed)} out first; only the {dollars(owed)} early-closure fee "
                "stays in the account when it closes"
            )
        return None
    if have != 0:
        return f"the account has {dollars(account.balance)}; move it out before closing"
    return None


def closed_readback(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = _mine(v, args.get("account_id"))
    return None if account is not None and account.status == "CLOSED" else "the account is not closed"


# ---------------------------------------------------------------- transfers

def transfer(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    source = _mine(v, args.get("source_account_id"))
    if source is None:
        return "the account the money comes from does not exist"
    target = _mine(v, args.get("destination_account_id"))
    if target is None:
        return "the account the money goes to does not exist"
    if source.account_id == target.account_id:
        return "the money has to move between two different accounts"
    for role, account in (("comes from", source), ("goes to", target)):
        if not account.is_open:
            return f"the account the money {role} is not open"
    amount = money(args.get("amount"))
    if amount is None or cents(amount) <= 0:
        return "a transfer must be more than zero"
    if source.balance is None:
        return "the balance of the account the money comes from is not on file"
    if cents(amount) > cents(source.balance):
        return f"the account the money comes from has {dollars(source.balance)}, which is not enough for {dollars(amount)}"
    return None


def transfer_readback(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """A transfer's failures are all loud, so this only checks what must still hold afterwards: both
    accounts are there and open, and the source did not go below zero."""
    source, target = _mine(v, args.get("source_account_id")), _mine(v, args.get("destination_account_id"))
    if source is None or target is None or not source.is_open or not target.is_open:
        return "an account involved is no longer there or open"
    if source.balance is not None and source.balance < 0:
        return "the account the money came from went below zero"
    return None


# ---------------------------------------------------------------- paying a credit card

def pay_card(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    if str(args.get("user_id")) != v.user_id:
        return "a payment can only be made for the customer who is verified"
    row = v.database.accounts.data.get(str(args.get("checking_account_id")))
    checking = _mine(v, args.get("checking_account_id"))
    if checking is None or not isinstance(row, Mapping):
        return "that checking account does not exist"
    if checking.product is not Product.CHECKING or not checking.is_open:
        return "the payment must come from an open personal checking account"
    if row.get("class") != "checking":
        # The bank's payment tool reads the older field names and cannot see an account opened in this visit.
        return "this checking account is too new for the bank's payment system to use; it can pay a card once the bank has processed it"
    card = v.database.credit_card_accounts.data.get(str(args.get("credit_card_account_id")))
    if not isinstance(card, Mapping) or card.get("user_id") != v.user_id:
        return "that credit card account does not exist"
    if str(card.get("status") or "").upper() == "CLOSED" or str(card.get("account_status") or "").upper() == "CLOSED":
        return "that credit card account is closed"
    amount = money(args.get("amount"))
    if amount is None or cents(amount) <= 0:
        return "a payment must be more than zero"
    owed = money(card.get("current_balance")) or 0.0
    if cents(amount) > cents(owed):
        return f"the card's balance is {dollars(owed)}, which is less than {dollars(amount)}"
    if checking.balance is None or cents(amount) > cents(checking.balance):
        return f"the checking account has {dollars(checking.balance or 0.0)}, which is not enough"
    return None
