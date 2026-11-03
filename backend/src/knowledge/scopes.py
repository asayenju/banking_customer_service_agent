"""Which documents each specialist may search.

A scope is by topic: the product families a specialist's work touches. It is deliberately
not the union of the tasks' ``required_documents``: that would fit the 97 tasks and fail
the first customer who asks about anything else. How well a scope covers the tasks is
measured and reported (``knowledge.report``), not assumed.

Internal procedures ("Internal: ...") are written for bank staff. Specialists may read
them; the answerer for general questions may not, so internal policy is never quoted back
to a customer.

A scope also names the tools it leaves visible in document text. A document that tells
the reader to use some other tool has that name hidden, so reference text cannot point a
model at a tool it was not given.
"""

from dataclasses import dataclass

# Document id prefixes, one per product family.
BANK = "doc_bank_accounts"
CHECKING = "doc_checking_accounts"
SAVINGS = "doc_savings_accounts"
BUSINESS_CHECKING = "doc_business_checking_accounts"
BUSINESS_SAVINGS = "doc_business_savings"
CREDIT = "doc_credit_cards"
BUSINESS_CREDIT = "doc_business_credit_cards"
EVERYONE_PAY = "doc_everyone_pay"
BNPL = "doc_buy_now_pay_later"
MISC = ("doc_personal", "doc_customer")

ALL_FAMILIES = (
    BANK, CHECKING, SAVINGS, BUSINESS_CHECKING, BUSINESS_SAVINGS,
    CREDIT, BUSINESS_CREDIT, EVERYONE_PAY, BNPL, *MISC,
)


@dataclass(frozen=True)
class DocScope:
    name: str
    prefixes: tuple[str, ...]
    include_internal: bool = True
    # Catalog tool names that stay visible in document text; every other tool name is hidden.
    tools: frozenset[str] = frozenset()

    def allows(self, doc_id: str, title: str) -> bool:
        if not doc_id.startswith(self.prefixes):
            return False
        return self.include_internal or not title.lower().startswith("internal")


DEBIT = DocScope("debit", (BANK, CHECKING, BUSINESS_CHECKING))
CREDIT_CARDS = DocScope("credit", (BANK, CREDIT, BUSINESS_CREDIT, BNPL))
ACCOUNTS = DocScope("accounts", (BANK, CHECKING, SAVINGS, BUSINESS_CHECKING, BUSINESS_SAVINGS))
FEES = DocScope(
    "fees", (BANK, CHECKING, SAVINGS, BUSINESS_CHECKING, BUSINESS_SAVINGS, CREDIT, BUSINESS_CREDIT)
)
FRAUD = DocScope("fraud", (BANK, CHECKING, CREDIT, BUSINESS_CREDIT, EVERYONE_PAY))
# Product, fee and policy questions for a customer: everything except staff procedures.
GENERAL = DocScope("general", ALL_FAMILIES, include_internal=False)

SCOPES: dict[str, DocScope] = {
    s.name: s for s in (DEBIT, CREDIT_CARDS, ACCOUNTS, FEES, FRAUD, GENERAL)
}


def with_tools(scope: DocScope, tools: frozenset[str]) -> DocScope:
    """The same scope, leaving these tool names visible in document text."""
    return DocScope(scope.name, scope.prefixes, scope.include_internal, tools)
