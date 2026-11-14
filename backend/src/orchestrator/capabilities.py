"""What the assistant can actually do for each kind of request, in plain words, for Jev.

Jev scores how sure we should be that a request is what the decomposer says it is. It used to see only the message, the
kind's name and a summary, with no idea what the assistant can do for that kind, so a vague "see my card" scored 0.53 and
went to a person. Each sentence here describes what the code behind the kind does today (a test checks every kind has
one), so Jev can judge a request against what is really on offer. Nothing here is a detail value.
"""

from orchestrator.types import IntentKind as K

CAPABILITY: dict[K, str] = {
    K.FREEZE_CARD: "Freeze a debit card the customer names so nothing can be spent on it; they can unfreeze it later.",
    K.UNFREEZE_CARD: "Unfreeze a debit card that was frozen.",
    K.ACTIVATE_CARD: "Activate a new debit card once the customer has it in hand and gives the details printed on it.",
    K.ORDER_REPLACEMENT_CARD: "Order a replacement debit card (closing the old one first when it is still open), with a delivery speed and design.",
    K.CHANGE_OR_RESET_PIN: "Reset or change a debit card PIN, or look into a card whose PIN is locked.",
    K.CLOSE_CARD: "Close a debit or credit card for good, after checking pending items and, for credit, offering alternatives.",
    K.CARD_CONTROLS: "Change a debit card's controls: a temporary limit increase, a recurring-payment block, or a security alert.",
    K.DISPUTE_TRANSACTION: "Find a charge in the customer's own records and file a dispute for it after the customer agrees.",
    K.CREDIT_LIMIT_REQUEST: "Submit a credit card limit increase request and give the decision.",
    K.FEE_OR_INTEREST_ADJUSTMENT: "Review an ATM fee, a savings interest payment or credit card rewards and credit back what was wrongly charged or paid.",
    K.TRANSFER_FUNDS: "Move money between the customer's own accounts.",
    K.PAY_CREDIT_CARD: "Pay a credit card from one of the customer's checking accounts.",
    K.OPEN_ACCOUNT: "Open a new checking or savings account of an offered type, with optional first funding.",
    K.CLOSE_ACCOUNT: "Close an account after moving its money out.",
    K.ACCOUNT_BALANCE_OR_HISTORY: (
        "Read only. Show the balance of each account and card, and list the recent transactions on one named card or account."
    ),
    K.LOOKUP_INFO: "Read only. Look up the current date or time, the customer's referrals, their list of cards, or the details on their profile.",
    K.GENERAL_QUESTION: "Answer a question about the bank's products, fees and policies from its policy notes, or explain something about the customer's own records.",
    K.UPDATE_PROFILE: "Change the email address on the customer's profile.",
    K.REPORT_FRAUD_OR_LOSS: "Protect cards after a loss or theft: freeze them, close and replace them, replace credit cards and dispute fraudulent charges.",
    K.REQUEST_HUMAN: "Connect the customer with a human agent.",
    K.UNCLEAR_OR_GREETING: "Offer a menu of what the assistant can help with.",
    K.OTHER: "Nothing: the request fits no kind the assistant handles, so a person takes over.",
}


def capability(kind: K) -> str:
    return CAPABILITY[kind]


# A short phrase for "do you want me to ...?", used to ask between the two readings Jev finds most likely.
LABEL: dict[K, str] = {
    K.FREEZE_CARD: "freeze a card",
    K.UNFREEZE_CARD: "unfreeze a card",
    K.ACTIVATE_CARD: "activate a new card",
    K.ORDER_REPLACEMENT_CARD: "order a replacement card",
    K.CHANGE_OR_RESET_PIN: "change or reset a PIN",
    K.CLOSE_CARD: "close a card",
    K.CARD_CONTROLS: "change a card's limits or alerts",
    K.DISPUTE_TRANSACTION: "dispute a charge",
    K.CREDIT_LIMIT_REQUEST: "ask for a credit limit increase",
    K.FEE_OR_INTEREST_ADJUSTMENT: "review a fee, interest or rewards",
    K.TRANSFER_FUNDS: "move money between your accounts",
    K.PAY_CREDIT_CARD: "pay a credit card",
    K.OPEN_ACCOUNT: "open an account",
    K.CLOSE_ACCOUNT: "close an account",
    K.ACCOUNT_BALANCE_OR_HISTORY: "see your balances or recent transactions",
    K.LOOKUP_INFO: "look up your cards or profile details",
    K.GENERAL_QUESTION: "answer a question about the bank's products or policies",
    K.UPDATE_PROFILE: "change your email",
    K.REPORT_FRAUD_OR_LOSS: "protect your cards after a loss or theft",
    K.REQUEST_HUMAN: "connect you with a person",
    K.UNCLEAR_OR_GREETING: "show you what I can help with",
    K.OTHER: "pass this to a person",
}
