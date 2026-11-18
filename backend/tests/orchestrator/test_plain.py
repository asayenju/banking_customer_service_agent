"""What the customer reads is plain text: no asterisks, no headings, no internal document ids."""

from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.plain import plain_text
from orchestrator.reply import compose_reply
from orchestrator.types import Intent, IntentKind as K

ECO = (
    "You can close your EcoCard as long as:\n\n1. **Zero Balance**: The balance must be $0.00.\n"
    "2. **No Pending Disputes**: None open.\n\nYou have 45 days to redeem rewards [doc_credit_cards_account_logistics_001].\n\n"
    "Your EcoCard has **$0.00**. Would you like to proceed with closing it?"
)


def test_bold_headings_backticks_and_document_ids_are_removed_and_the_lists_stay():
    out = plain_text(ECO)
    assert "*" not in out and "`" not in out
    assert "1. Zero Balance: The balance must be $0.00." in out and "Your EcoCard has $0.00." in out
    assert "[doc_credit_cards_account_logistics_001]" in out, "citations are a separate decision and are left alone"
    assert plain_text("### Fees\n* one\n* two\n`code`") == "Fees\n- one\n- two\ncode"
    assert plain_text("__bold__ and *italic* words") == "bold and italic words"


def test_arithmetic_and_ordinary_text_are_left_alone():
    assert plain_text("3 * 4 = 12, and 2*5 is 10") == "3 * 4 = 12, and 2*5 is 10"
    assert plain_text("Your card ends in 1234 (credit card).") == "Your card ends in 1234 (credit card)."


def test_an_answer_from_a_route_reaches_the_customer_as_plain_text():
    intent = Intent(id="q", kind=K.GENERAL_QUESTION, summary="can I close the ecocard")
    outcome = RouteOutcome(intent_id="q", status=OutcomeStatus.DONE, text=ECO)
    reply = compose_reply({"q": intent}, outcomes=[outcome])
    assert "**" not in reply.text and "Zero Balance" in reply.text
