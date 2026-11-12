# Fraud and loss composer: policy decisions (Slice 0, Issue #78)

Every statement names its document or database evidence. Statements about the bank's tools are tests in
`backend/tests/agent/test_fraud_policy.py`, so this note cannot drift from the tools. Documents are
`doc_bank_accounts_bank_accounts_(general)_NNN` unless stated.

## 1. The bank's tools (tested)

| Tool | What it checks | What it does not check |
|---|---|---|
| `freeze_debit_card_3892` | the card exists and is ACTIVE | owner |
| `unfreeze_debit_card_3893` | the card is FROZEN | |
| `close_debit_card_4721` | status is ACTIVE or PENDING (a FROZEN card is refused) | pending transactions, pending refunds, the 14 day age, owner |
| `order_debit_card_5739` | checking class, OPEN, balance, no pending order, no ACTIVE card | the tier limit, the 48 hour wait, that the fees match the tier (a RUSH CUSTOM card with fees 0 goes through) |
| `order_replacement_credit_card_7291` | the account exists; refuses an identical repeat order | owner, another pending order with different details, any limit; it closes the account at once (even an already closed one) and charges nothing |
| `file_credit_card_transaction_dispute_4829` | enum values | eligibility, ownership |

There is no credit card freeze, no "report fraud" tool and no password change tool. A fraud report can
therefore only be acted on by freezing, closing, ordering, replacing and disputing; everything else is guidance.

## 2. The wallet protocol (docs 030, 025, 026, 029)

1. **Freeze first.** A card that may be lost is frozen (026: "misplaced and looking for it"). Freezing needs
   no yes: the catalog marks it reversible and `decision.NO_YES_NEEDED` lists `FREEZE_CARD`. Only ACTIVE
   cards can be frozen.
2. **A close needs a definite loss.** 026: close "when the card is confirmed lost or stolen". Task 077 opens
   with "I think I might have lost my wallet", so a possible loss is a freeze and nothing more. The customer's
   later "it is definitely gone" (078, 079, 080) unlocks the close.
3. **Credit cards are asked about, always.** 030: after a debit loss, read the credit cards, ask whether one
   was in the wallet, and offer a replacement. Nothing is replaced unasked (077: the customer only remembers
   when asked). A declined offer is still the end of the protocol.
4. **One yes.** The closes, the reorders, the credit replacements and the credit disputes are one preview.
5. **Order of steps.** A FROZEN card is unfrozen before it is closed (tool fact, section 1), the old card is
   closed before a new one is ordered (the order tool refuses an account with an ACTIVE card), a credit
   dispute follows the credit replacement (tasks 080, 081 gold: replacement, then dispute history, then disputes).

## 3. Rulings

**Pending transactions block a close (025, requirement 4).** Task 078: Light Green has two `pending`
purchases, gold freezes it and leaves it frozen. A pending purchase on the card's account means that card is not
closed in this plan; it stays frozen and the customer is told why. Evidence: 078 gold has no close for Light Green.

**A pending refund needs the customer's accept (025, requirement 5).** Task 079: Green has a `pending` refund
(`type: refund`) and gold closes it after the customer agrees the refund goes to checking. Rule: a pending
refund blocks the close until the customer accepts, then the close is planned.

**The Entry-tier 48 hour wait is enforced when the card is closed in the same plan (029).** The tool never
checks it. Entry tier is `Light Blue Account` in `workflows/tiers.py`. Gold 078 and 079 do not reorder Light
Blue. A reorder for an Entry account whose card this plan closes is replaced by a plain message with the
earliest date. A level with no tier in the table (Light Green, Green Fee-Free: gold never reorders them) keeps
today's rule: a reorder for it is a hand-off, never a guess.

**Where each rule lives.** The pending-purchase hold and the pending-refund accept are in `DebitReplacement`, so every
debit close follows them, not only a lost wallet (the live debit run of 078 and 079 had no such rule). The Entry wait
is enforced by the composer for every Entry card it closes, and by `DebitReplacement` only for an Entry account that is
past its replacement limit and whose card is closed in the same plan. That narrower rule fits all three golds: 079
(over the limit, waits), 078 (declines the fee) and 091 (no prior replacements, ordered the same day). It is inferred
from the golds, not written in doc 029, which says only "48-hour waiting period after card closure".

**The excess replacement fee is the customer's choice (029).** The answer is yes, no or not asked: only "not asked" asks. 078: the customer declines $25 and Light Blue is
closed with no reorder. The fee is computed by code from the tier table (`debit_replacement.py`), never
taken from the customer or the model.

**Credit replacement reason is the loss type.** `lost`, `stolen` or `fraud_suspected` (the credit reasons
include all three). Expedited delivery is the customer's choice; gold uses expedited in 080, and the fee comes
from `credit_tiers.EXPEDITED_FEE`.

**Disputes after a replacement.** The dispute workflow (`credit_dispute.py`) is reused unchanged, including its
provisional-credit rule. The caps for tasks 039 to 041 are inferred from gold, not documented
(`credit-policy-decisions.md`); the composer inherits that and says so here.

**Pending purchases are checked twice.** The workflow checks them when it plans; the fraud close validator checks
them again just before the step runs. A pending refund is not checked again: closing never loses it, because a refund
to a closed card is credited to the linked checking account (025, Important Notes).

**No double close.** A fraud dispute implies `close_and_reissue` (doc 031) but the composer plans the closes
itself and files only credit disputes, so a card is closed once.

**Human transfer.** `policy_header.md`: transfer only after the customer has asked four times. Task 081 asks
twice at the start and twice at the end, and gold does all the work first. Rule: while a card-protection plan is
open, human requests are counted and acknowledged; the transfer happens when the work is done and the customer
asks again, at the fourth request, or at once when the knowledge base offers no action. Outside an open
protection plan nothing changes (`REQUEST_HUMAN` escalates at once, `docs/prd-orchestrator-routing.md`).

**No-action reports.** Account takeover, identity theft, a lost phone or ID, a suspicious login, a code 43
decline, a `security_hold` (041) and a bank-initiated alert (039) have no tool. The composer gives the
knowledge base's guidance, says plainly what it cannot do, asks whether the customer wants a person, and
transfers (reason `fraud_or_security_concern`, 042) only on a yes. It never says a card is flagged (037) and
never states a fraud score (041).

## 4. Contradictions between documents and gold

| What | Document | Gold | Ruling |
|---|---|---|---|
| Freeze or close a lost card | 026: close | freeze, unfreeze, close | freeze first; a close needs a definite loss |
| Say a card is flagged | 037: never | 025, 039, 041: discuss | never say flagged; discuss the activity |
| `security_hold` | 041: offer a transfer | 090: none; 092: transfer | offer a transfer, transfer on a yes |
| Pending refund | 025: wait or acknowledge | 079: acknowledged and closed | the accept unlocks the close |
| Credit replacement window | 004 (credit): 60 days | none | `credit_replacement.py` already enforces it |

## 5. Where the gold departs

- **Task 081** transfers with an empty summary (`transfer_to_human_agents {"summary": ""}`) and no reason. The
  note is written by code (`agent/handoff.py`, no ids), so the call differs from gold's. The transfer is not a
  database change, so the replay still matches; recorded here, not in `policy_departures.json`.
- **Task 083** stays excluded: its gold omits `customer_max_liability_amount`.
- **Order of calls.** Gold interleaves reads and a credit replacement between the debit steps; the composer
  plans debit first, then credit. The database ends the same.

## 6. Status

Decided in this note: the pending-transaction, pending-refund and Entry-wait rules, the transfer rule, 078 and
079 stay in the debit pilot's 12 tasks (the composer's report replays them too, so #38's count does not change).
The 081 summary difference is recorded. Nothing here is pending.
