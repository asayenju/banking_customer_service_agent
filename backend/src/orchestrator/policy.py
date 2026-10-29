"""What counts as an action request, in one place.

The decomposer prompt and the evaluation's labeling prompt both embed this text,
so the model being graded and the answer key follow the same rule. Change it here
and re-run the evaluation.
"""

ACTION_POLICY = """<action_policy>
An intent is an ACTION when the customer asks the bank to do something: freeze, unfreeze,
activate, close, open or order a card or account, dispute a card charge, adjust, fix or
credit a fee, interest or reward, raise a limit, pay, transfer, or reset a PIN.
Two actions need the customer's own words, not your inference:
- REQUEST_HUMAN only when they ask for a person ("agent", "human", "representative",
  "manager", "speak to someone"). Being upset, confused or in a hurry is not a request.
- UPDATE_PROFILE only when they ask to change their email, phone or address on file.

These are NOT actions, because the customer wants information, not a change:
- checking, investigating, explaining or "figuring out what is going on";
- balances, transaction history, statements;
- how-to and policy questions;
- describing a symptom (cards declined or locked, a deposit missing, a score dropped)
  without asking for a specific step.
Use ACCOUNT_BALANCE_OR_HISTORY when answering needs the customer's own account data,
GENERAL_QUESTION for product, fee and policy questions, and OTHER when the problem fits
no kind.

Two edge rules:
- A conditional request counts: "if any are wrong, fix them" is an action request.
- "Fix it" counts only when what to fix maps to exactly one kind in the list.
- Wanting to open or close something while asking which option to choose is still an
  action request; "discuss" or "help me sort this out" is not a request for a person.
- Reporting fraud on a charge and asking to dispute it is one DISPUTE_TRANSACTION.
  Disputing interest or a fee is FEE_OR_INTEREST_ADJUSTMENT, not DISPUTE_TRANSACTION.
- Preferences about how an account should work (paper statements, early deposit, no
  overdraft fees) are details of the request, not separate actions.
A lost or stolen card with no requested step is REPORT_FRAUD_OR_LOSS.
</action_policy>"""
