# Debit card policy decisions

Every rule names its document (`doc_bank_accounts_bank_accounts_(general)_NNN`) or the task that shows it. Statements about
the bank's tools are tests in `backend/tests/agent/test_debit_policy.py`. Plan: `~/.claude/plans/` "debit card policy and
workflows" (3 phases). Rules marked *inferred* come from the reference answers, not the documents.

## 1. The bank's tools (tested)
| Tool | Does not check |
|---|---|
| `close_debit_card_4721` | pending purchases or refunds, the 14 day card age |
| `order_debit_card_5739` | account age (3 business days) and a balance under the 25 dollar minimum (both tested); the code also checks the cardholder's age, a card already on order and a US address, which were not run against the tool |
| `request_temporary_debit_card_limit_increase_8374` | account age (60 days), overdraft in the last 30 days |

## 2. Rules enforced in code (done)
- **Pending purchase holds a card; a pending refund needs the accept** (025): `DebitReplacement`, so every close follows it.
- **Closing for a non-urgent reason needs a card 14 days old** (025 req 6); lost, stolen, fraud_suspected bypass it.
- **An order is left out, with the reason, unless** the account has been open 3 business days, holds 25 dollars, the
  cardholder is 18, no card is on order, and the address on file is a US address (023). A shipping address the customer
  gives is shown in the plan and confirmed by them, so it is not judged.
- **Excess replacement fee**: yes, no or not asked; only "not asked" asks. "No" closes the card and orders nothing (078).
- **Entry 48 hour wait** (029): for an Entry account past its limit whose card is closed in the same plan (*inferred* from
  079 and 091). The wallet composer applies it to every Entry card it closes.
- **A partial category never disputes the whole charge** (031/033): `atm_cash_discrepancy`, `incorrect_amount`,
  `atm_deposit_not_credited` must be below the charge; every reference is (task 084).
- **Closing after a dispute is the customer's choice** (031 note: the dispute records metadata only; task 082).
- **Provisional credit** (032): required categories as documented; discretionary credit only for a cancelled subscription
  the customer already took to the merchant (*inferred*, task 082; 084 and 086 do not for other categories).
- **A plain answer may not claim a card was or will be closed, ordered or issued** (task 082's false claim).
- **Temporary limit increase** (040): account OPEN, 60 days old, no overdraft fee in 30 days (task 089: Blue is refused).
- **PIN lock fraud score** (041): `workflows/debit_pin_lock.py` computes the flags A1 to E3 and the level from the records
  and clears flag groups the customer's answers clear. Built and tested against tasks 090 to 092; not yet wired in as a
  workflow (see section 4).

## 3. Contradictions
| What | Document | Reference | Ruling |
|---|---|---|---|
| Entry 48 hour wait | 029: after every closure | 091 orders an Entry card the same day; 079 waits | wait only when past the limit (inferred) |
| Provisional credit, recurring charge | 032: not required | 082: eligible when the merchant was contacted | eligible only for that category (inferred) |
| Card action after a card-not-present fraud dispute | 031: close and reissue | 082: Green card kept | the customer chooses |

## 4. Not done yet (from the plan)
- Workflows for PIN lock decisions (090 to 092, 087 questions), PIN reset (last four digits from the customer), fraud alert
  (039), activation (024) and the recurring block offer (034).
- The temporary limit amount: the card record has no ATM limit field (`daily_atm_limit` is empty in task 089); the limit
  comes from the account's documents, so the 150 percent cap is not enforced and the amount still comes from the model.
- A simulated customer for the live harness (087, 088, 091, 092 fail only because nobody can answer).
