# Credit-card policy decisions (Issue #49)

Result of the policy spike in `docs/prd-credit-card-specialist.md`. Where the bank's tools, its documents and the reference answers ("gold") disagree, the documented policy wins (PRD decision 3). Each point below says what was found, what we do, and which tasks it touches.

## 1. Where the tier comes from
**The tier is the tier of the card type, not of the customer.** The documents name entry, mid and premium tiers without a table, except in the replacement-fee documents (`doc_credit_cards_credit_card_replacements_001` and `_002`), which list the card types of each tier. That table is `TIER_OF_CARD_TYPE` in `agent/workflows/credit_tiers.py`:

| Tier | Card types |
|---|---|
| Entry | Bronze Rewards, EcoCard, Business Bronze Rewards |
| Mid | Silver Rewards, Business Silver Rewards, Green Rewards, Silver Zoom |
| Premium and above | Gold Rewards, Business Gold Rewards, Platinum Rewards, Business Platinum Rewards, Diamond Elite |

A test checks the table against the document and against every card type in the bank's data.

Checked against the reference answers:
- Retention credit: Silver Zoom/Platinum (047) $50 is the premium amount for Platinum; Business Silver (045) $20 is mid. Both fit the card-type tier.
- Limit caps: Gold 5,000 to 7,500 (050, 054: 50 percent) is premium; Silver 15,000 to 22,500 (053: 50 percent) is mid; Bronze (051, 052: 25 percent, 70 percent utilization, 120-day cooldown) is entry.

## 2. Crypto-Cash Back is entry tier (one document, one gold departure)
The replacement table does not list it. The provisional-credit document (`doc_credit_cards_credit_cards_(general)_015`) lists it with the entry tier ("Entry Tier (Bronze Rewards Card, EcoCard, Business Bronze Rewards Card, Crypto-Cash Back Card)"), and nothing else places it. So it is entry tier. (The first version of this note missed that document and called the card unmapped; the test now checks the document.)
- Task 049 offers its owner the entry-tier $5 retention credit: consistent with policy.
- Task 048 offers the same card type the mid-tier $20 credit: **a departure from policy.** Its own notes call the card "mid-tier"; no document does.
- **Exclusion to record in `evals/parity_exclusions.json` (Slice 7):** the Crypto-Cash Back statement credit in 048 ($20, policy says $5). The remaining writes in 048 are still compared.
- The provisional-credit document uses finer tiers for its caps (Entry $2,500; Mid $5,000; Premium, Gold cards only, $10,000; Elite, Platinum cards, $15,000; Invitation, Diamond Elite, $25,000). Slice 3 keeps that table by card type; the three-tier mapping above is for retention, limit rules and replacements.

## 3. Cooldown
Documents 005 and 007 disagree. 005 (customer-facing): the cooldown applies only if the most recent request was **approved**, counted from the submit date; denied requests do not start one. 007 (staff): deny if any request exists inside the period.
- **Decision:** 005 wins. Only approved requests count. A pending row never counts, in particular the row created by the step that submits the request being decided, and a denied row never counts.
- Checks against the reference answers: 052 (approved request on 2025-09-15, today 2025-11-14 = 60 days, entry cooldown 120, so deny with `cooldown_period_active`); 051 (first request denied for utilization, then the resubmission is allowed and approved, which 007's reading would have refused).

## 4. Order of work in a request
- The amount is checked against the tier's cap **before** submitting (007 step 0). An over-cap amount is a question that names the cap (051: 30 percent, then 25; 054: $4,000, then $2,500).
- Submit comes before the eligibility reads (007 step 1), so the history seen by the reads contains the new pending row (see 3).
- A dispute or a replacement for the same customer blocks a later approval; the limit goes first and the customer is told why (053, 054).

## 5. Where the bank's tools disagree with the documents
The code follows the documents; each point is a validator or a read-back in Slice 2.
- `approve_credit_limit_increase` refuses on any open dispute of the **user** (not the card), on orders in PENDING, PROCESSING or SHIPPED (a replacement ordered with `order_replacement_credit_card_7291` is stored as ORDERED, so it does not block), and on `account_status`. The documents also name a past-due balance; the tool reads `account_status`, not `past_due_amount`. The code checks all of it and reads back the new limit.
- `close_credit_card_account`, `apply_statement_credit`, `apply_credit_card_account_flag`, `order_replacement_credit_card` and `file_credit_card_transaction_dispute` check no eligibility. The log of a closure reason reports success for a duplicate.
- The document that describes a downgrade names a tool that does not exist, and an upgrade is a customer-side tool. The specialist does not offer either.
- The dispute document lists `reversal_of_charge` as a resolution; the tool rejects it. Only `full_refund` and `partial_refund` are used.

## 6. Where the gold departs from policy or cannot be reproduced
- The Crypto-Cash Back $20 credit in 048 (section 2).
- Bare checking ids `"05"` and `"07"` in 043, 045, 048: the code resolves the checking account from the customer's records; the id is not taken from the customer's words.
- 049's hand-off summary is free text; the code writes its own and the comparison is on the database only.
- 053 has no overlay and relies on the base data (Silver Rewards, 15,000 limit).
- Tasks 077, 080 and 081 are not part of this PRD.

## 7. Open after this spike
- Whether retention applies to a card whose reason is `negative_experience` beyond a "modest goodwill credit" (no amount is given); the specialist hands over for that reason.
