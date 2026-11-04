# PRD: Credit-card specialist (Issue #41)

Part of `docs/prd-specialist-agents.md`, phase 4. This PRD replaces the three acceptance criteria that Issue #41 started with.

## Why
Issue #41 asked for three things: closure blocked unless the balance is zero, nothing is pending and the account is 60 days old; limit approval blocked by a pending dispute, replacement or past-due balance; offline tests on the 14 wholly-credit tasks minus traps. Reading the bank's tools, the knowledge-base documents and the 21 tasks that touch credit cards shows that the issue rests on assumptions that do not hold:

1. **Closing is mostly retention.** The gold closes a card in 2 of the 21 tasks (048 and 049, the Green card). In 043, 044, 045, 047, 048 and 049 it logs a reason and offers a credit, a fee waiver or an upgrade, and the card stays open. The issue's closure rule is missing the fourth condition in the documents (no pending replacement), the rule that a closure reason logged in the last year skips retention, and the retention offers by tier.
2. **The limit rule in the issue is a small part of the real one.** The documents set age (120, 90 or 60 days by tier), a cooldown, utilization (below 70, 80 or 90 percent), on-time payments (6, 3 or 3) and a cap on the increase (25, 50 or 50 percent). Tasks 050 to 054 turn on those rules.
3. **"Minus traps" is wrong.** Tasks 049 and 052 do have gold writes (049: close one card, log a reason, a $5 credit, a hand-off; 052: submit, then deny). They are cases where only the safe writes may happen, not zero-write cases.
4. **The bank's tools enforce almost nothing.** Closing, statement credit, account flag, replacement and dispute check no eligibility; the dispute tool does not even check that the transaction exists; the closure-reason log reports success on a duplicate; statement credit writes a negative transaction and does not change the balance. Every rule has to live in our code.
5. **The approve tool disagrees with the documents.** It blocks on any open dispute of the user (not of the card), on orders in PENDING, PROCESSING or SHIPPED (a replacement ordered by the same tool is stored as ORDERED and so does not block), and on `account_status` (only 1 of 38 fixtures has one; the past-due amount field is never read). Failures all return the same generic error.
6. **The documents conflict.** The cooldown is described one way in the limit-increase policy and another in the procedure; card types are never mapped to the entry, mid and premium tiers the rules use; one document names a downgrade tool that does not exist; a dispute document lists a resolution the tool rejects. The same card type gets a $20 retention credit in task 048 and $5 in task 049.
7. **The gold has order quirks that matter.** Submit comes before the eligibility reads, so a pending request row must not trip the cooldown check; the replacement is first in tasks 037 and 038 and last in 039; a dispute and a replacement each block a later approval, so a mixed request (053, 054) has to do the limit first.
8. **Fixtures are thin.** Only 2 of 38 credit accounts have a credit limit and none is past due in a way the tool reads. Tests need synthetic fixtures, and a missing limit must hand off.

## What
A credit-card specialist, configured the way the debit specialist is, whose policy decisions are made by code and approved by the customer as a previewed plan.

### Goals
- Four workflows (closure with retention, limit increase, dispute, replacement) where the model reports what the customer said and code computes eligibility, amounts, reasons and the order of steps.
- Validators for every credit write tool, checked at plan time and again before each step runs.
- Credit versus debit resolved from the record the customer names; an unclear record is a question, never a guess.
- Offline evidence on 18 tasks: the 14 wholly-credit tasks (031, 036 to 041, 044, 047, 049, 050, 052, 053, 054) plus 043, 045, 048 and 051 (which pay the card from checking first).
- One small, capped, opt-in live probe of 3 tasks, after Issue #38 passes.

### Non-goals
- Customer-side tools (applying for a card, asking for a human). They are replayed directly in the offline tests and never offered by the agent.
- A downgrade or an upgrade. The tools for them are absent or customer-side, so the specialist neither offers nor promises them.
- Lost-wallet tasks that mix credit and debit (077, 080, 081): Issue #44. Tasks 080 and 081 also use credentials the agent cannot reproduce.
- Editing the bank's tools, or making their weaknesses match the gold.
- Live runs beyond the 3-task probe.

### Definitions
- **Staged workflow**: one workflow, more than one preview. The customer approves each stage; later stages depend on what the customer says after the first.
- **Retention**: the steps between "I want to close" and a close: log the reason, make one offer, close only if the customer still wants to.
- **Tier**: entry, mid or premium as the documents use the word. Where it comes from is Slice 0.
- **Safe-writes-only case**: a task where the right outcome contains writes, but a larger set of tempting writes must not happen (049, 052).

## How

### Shape
`agent/specialists/credit.py` is a `SpecialistConfig` like `specialists/debit.py`: instructions, the `CREDIT_CARDS` scope from `knowledge/scopes.py`, `VALIDATORS` keyed by tool name, a `prepare` that settles which card is meant in code, and the workflows. Workflows follow `agent/workflows/base.py`: `decide` returns Plan, NeedInfo, Refuse, Reply or HandOff and never raises. Reused as they are: `workflows/tiers.py`, `agent/cards.py`, `agent/guard.py` (`KIND_WRITE_TOOLS` already lists the credit tools per kind), `agent/dispatcher.py` (`RecordKinds`, `BY_RECORD`), `agent/tier1.py`, `evals/manifest.json`, `evals/parity_exclusions.json`.

### Closure with retention (`CLOSE_CARD`, credit)
- Eligibility from fresh state: balance is $0.00, no open dispute for the user, no pending replacement for the account, account at least 60 days old. Anything else is a Reply that names the blocker. A balance gives a question about paying it from checking, with the payment step checked against the funds (task 049: $175 owed, $50 available, no payment).
- A closure reason already logged for the account in the last year skips retention.
- Stage 1 (one yes): `log_credit_card_closure_reason_4521`. Code then writes the offer for the tier: a statement credit (amounts by tier), an annual-fee waiver flag (customer for 2 or more years, reason `loyalty_benefit`, one-year expiry), or a reminder of the card's benefits, depending on the reason.
- Stage 2 (a second yes): the credit, the flag, or `close_credit_card_account_7834`, chosen by what the customer says. The close is never in a plan the customer has not seen.
- Never promised: a downgrade, a product upgrade, rewards payout. Unredeemed rewards are forfeited 45 days after closure and the fee refund window is 37 days; both are stated by code.

### Limit increase (`CREDIT_LIMIT_REQUEST`)
- Code computes the tier, account age, the cooldown, utilization (needs the credit limit; without it the specialist hands off), the on-time streak (the history tool counts rows, not months) and the cap.
- The cooldown counts approved requests, from their submit date. The pending row that the submit step creates never counts against the request it belongs to.
- An amount over the cap is a question that names the cap (task 051: 30 percent becomes 25; task 054: $4,000 becomes $2,500).
- Plan: submit, then approve or deny. The deny reason comes from the rule that failed (task 052: cooldown; task 051: utilization, then a payment and a second request). An open dispute or pending replacement denies, because the tool's own refusal is a silent generic error. A read-back confirms the new limit.
- In a request that mixes a limit with a dispute or replacement, the limit goes first, and the customer is told why.

### Disputes and replacements
- Same pattern as the debit workflows. The customer says which transaction and why. Code fills the 14 tool arguments from the transaction record, checks that the transaction exists, belongs to the verified customer and matches the card's last four digits (asked once; the tool never checks), and computes `eligible_for_provisional_credit`: account at least 60 days, an allowed reason, at least $25 and under the tier cap, no more than 2 disputes in 12 months counting earlier disputes in the same plan, the merchant contacted for non-fraud reasons, a purchase older than 30 days for goods not received. This rule matches the gold in tasks 039 to 041, where only the first two of many disputes get provisional credit.
- A partial refund needs an amount below the transaction. A duplicate is detected against the dispute history before a call.
- A replacement closes the old card at once, so code refuses one when an order is already pending or the tier's limit in 60 days is used, and never applies a fee the tool cannot charge. A single message may carry up to 16 disputes, shown in one preview.

### Records and routing
Credit routing by `credit_card_account_id`, credit transaction id or credit account id already exists in `RecordKinds`. New: a customer who owns both products and names nothing gets a code-written list; another customer's credit id resolves to nothing.

### Validators
One per write tool, each with an accept test and a reject test: dispute, replacement, closure reason, close, statement credit (amount above zero, reason in the 10, retention amount equal to the tier table), account flag (type and reason enums, expiry one year out), pay, limit submit, approve, deny. The approve validator pre-checks the three silent failures and the read-back confirms the new limit.

### Measurement
Tier-1 compares the final database with the one tau2 gets from the gold (`agent/tier1.py`). Safe-writes-only cases additionally assert the writes that must not happen. Gold actions that depart from documented policy (Slice 0 lists them) go to `evals/parity_exclusions.json` with a reason, as tasks 082 and 083 did. Order rules (limit before dispute, close after pay) are asserted separately from the state comparison.

## Constraints and risks
Ranked; each is an acceptance criterion in a slice.
1. **The tier source is not stated anywhere.** Retention amounts and caps depend on it. Slice 0 settles it against tasks 043 to 054; if it cannot be derived, the tier-dependent amounts are marked policy-ambiguous and those tasks are excluded with a reason, not fitted.
2. **A customer who lies** (049): a replacement that "arrived", a payment "just made", a card they do not own. The specialist believes the database, not the customer. A fake model scripted to obey the lies cannot cause an unapproved write.
3. **Silent no-ops and misleading successes**: approve with a hidden block, a duplicate closure log that reports success, statement credit that leaves the balance unchanged. Pre-checks plus read-back.
4. **Eligibility drift inside a plan**: a dispute filed earlier in the plan blocks the later approval. The plan is ordered by code and re-validated at execute time.
5. **Timeout after a committed write** (a filed dispute): PARTIAL, resume by step id, no second dispute.
6. **Wrong customer or card**: ownership of every credit id, the last four digits, two cards of one type.
7. **Missing data**: no credit limit, no closure history, a closed account. A hand-off with a reason, never a guess.
8. **Hand-off text**: the summary for a human is written by code from the audit and contains no private data.
9. **Document conflicts**: the documented policy wins over the tool and the gold; each conflict is listed in Slice 0.
10. **Cost**: nothing live until Issue #38 passes; then 3 tasks, capped, with recorded cassettes.

Known limits: tier-1 proves the plumbing, not the model; the interest and fee side of credit cards belongs to Issue #43; staged retention is measured on the gold's path only, not on every branch a customer may take.

## Issues and acceptance criteria
"Live" criteria need API keys, are opt-in and capped. Everything else runs offline. GitHub numbers are in the table at the end.

**Slice 0: Policy spike: tier source and conflicts.** Documentation and a small test only.
- A committed note fixes where the tier comes from (customer tier from the account level, or the card type), shown against the retention amounts in 043, 045, 047, 048 and 049 and the caps in 050 to 054.
- The note settles the cooldown reading and lists every place where gold departs from documented policy.
- If the tier cannot be derived, the affected tasks are listed as policy-ambiguous. Blocks Slices 3 to 6.

**Slice 1: Credit records and dispatcher.**
- Credit versus debit is resolved from the id; an id owned by another customer resolves to nothing.
- A customer with both products who names nothing gets a code-written list (NEEDS_INFO).
- The credit specialist is wired in, and the test for unmapped sub-agent kinds still passes. Depends on: none.

**Slice 2: Validators for the credit write tools.**
- Every credit write has accept and reject tests on synthetic fixtures: missing credit limit, past due, closed account, another customer's account, unknown transaction.
- The three silent failures of approve are caught before the call and confirmed by read-back. Depends on: none.

**Slice 3: Credit dispute and replacement workflows.**
- Provisional credit follows the rule above, with the 2-in-12-months count across history and plan.
- The disputed amount equals the transaction unless the reason allows a part; a duplicate dispute and a second pending replacement are refused.
- Tier-1 passes 031, 036, 037, 038, 039, 040, 041. Depends on: Slices 0, 1, 2.

**Slice 4: Credit limit workflow.**
- Each documented rule has a test that denies with its own reason; an amount over the cap is a question naming the cap.
- The request's own pending row does not trip the cooldown; a dispute or pending replacement denies with a read-back.
- Tier-1 passes 050, 051, 052 (deny), 053, 054. Depends on: Slices 0, 1, 2.

**Slice 5: Credit closure with staged retention.**
- Closure is blocked unless the balance is zero, nothing is pending, no replacement is pending and the account is at least 60 days old.
- A closure reason logged in the last year skips retention; no stage-2 step runs without its own yes; no downgrade or upgrade is promised.
- Tier-1 passes 043, 044, 045, 047, 048. Depends on: Slices 0, 1, 2.

**Slice 6: Traps and adversarial cases.**
- Task 049: the Eco card is blocked (pending replacement), the Gold card is not paid (funds short), the Crypto card gets $5 and not $50, and the hand-off text is written by code.
- A fake model scripted to obey the customer's false claims cannot cause an unapproved write.
- Tier-1 passes 049 as a safe-writes-only case. Depends on: Slices 3, 4, 5.

**Slice 7: Evidence and report.**
- A per-task report (semantic and strict pass, steps) for the 18 tasks; manifest counts and exclusions updated and its test passing.
- Zero guard bypasses; a default `pytest` run builds no live client. Depends on: Slices 3 to 6.

**Slice 8: Live probe (blocked on Issue #38).**
- Opt-in and capped, with an estimate that needs `--yes` and cassettes: tasks 043 (closure with a balance and a waiver), 054 (limit, dispute and replacement order), 038 (dispute plus reissue).
- The result is reported as a measurement, not a gate. Depends on: Slice 7, #38.

## Decisions
Decided by the project owner for this PRD:
1. Scope: the 14 wholly-credit tasks plus 043, 045, 048 and 051; 077, 080 and 081 stay with Issue #44.
2. Retention is a staged workflow with a yes at each stage.
3. When the tool, the documents and the gold disagree, the documented policy wins and gold departures go to the exclusion file.
4. Offline slices now; the live probe stays blocked on Issue #38.

Adopted from the design review (open to veto):
- Eligibility is recomputed in code from fresh state because the bank's tools check almost none of it.
- The specialist never offers a downgrade or an upgrade.
- A mixed request puts the limit first, and says why.

## GitHub issues
| Slice | GitHub | Title |
|---|---|---|
| parent | #41 | Credit cards |
| 0 | #49 | Policy spike on tier source and conflicts |
| 1 | #50 | Records and dispatcher |
| 2 | #51 | Validators for the credit write tools |
| 3 | #52 | Dispute and replacement workflows |
| 4 | #53 | Limit increase workflow |
| 5 | #54 | Closure with staged retention |
| 6 | #55 | Traps and adversarial cases |
| 7 | #56 | Evidence and report |
| 8 | #57 | Live probe (blocked on #38) |

All issues carry the `enhancement` and `specialist-agents` labels. Issue titles are `[4.1.N]` for Slice N.
