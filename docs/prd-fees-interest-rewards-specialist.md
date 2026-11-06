# PRD: Fees, interest and rewards specialist (Issue #43)

Part of `docs/prd-specialist-agents.md`, phase 4 (Issue 4.3). This PRD replaces the three acceptance criteria that Issue #43 started with. Status: draft, three open questions under Decisions. No code before Slice 0 passes.

## Why
Issue #43 asked for three things: an interest credit equal to balance x APY gap / 12 recomputed in code, a discrepancy-report reply that never claims a credit, and tier-1 evidence on 10 tasks. Reading the bank's tools, the knowledge-base documents and the tasks shows that this is right but not enough:

1. **`FEE_OR_INTEREST_ADJUSTMENT` has a route and no specialist.** `dispatcher.py` maps it to `Name.FEES`, and `FEES` has a document scope, but no specialist is registered, so a turn returns FAILED "not implemented".
2. **The bank's tools check almost nothing.**
   - `apply_checking_account_credit_5829` and `apply_savings_account_credit_6831` check a positive amount, class and an open status. They do not check ownership, a cap, a cooldown or once-per-account. A repeat call with identical arguments rewrites the same transaction row but adds the balance again.
   - The two credit tools write the balance in different formats (`"$x.xx"` against `"x.xx"`). The savings tool accepts class `saving` and `savings`.
   - `update_transaction_rewards_3847` accepts any string for any transaction.
   - `submit_interest_discrepancy_report_7294` stores no report: `interest_discrepancy_reports` is not a table in tau2's database, `add_to_db` returns False and the tool ignores it, yet it prints "Submitted... PENDING_REVIEW". Only the tool's `CALLED` marker remains, so it cannot be verified beyond "it ran".
3. **The documents carry the rules, and some are wrong or inconsistent.** One credit per checking account with a 14-day cooldown (doc 017), goodwill over $25 needs a supervisor (doc 043), credit before report (doc 044), non-stacking boost rules (docs 045, 046), floored whole points (docs 004, 007). Gold 013 says 5.5 + 0.025 = 6.0; Silver 002 has an unfilled `[[linked_checking_apy_bonus]]`; Light Green's daily limit is 300 in one document and 250 in another; rebate caps differ.
4. **Customers say untrue things.** In 094 the customer believes the shortfall is $72 and the real figure is $140. In 097 the customer quotes $583, $554, $1,050 and $112 against real figures far lower. The amount has to come from the records.
5. **The formula matches gold almost everywhere.** Balance x gap / 100 / 12 reproduces the gold credit for 093, 094, 095, 096 (both accounts) and 097 (platinum, diamond, silver plus). A balance x gap / 12 formula is one cent off on task 097 silver (220.83 against 220.84), but the expected month's interest minus the interest paid matches all nine gold credits (found while building Slice 3); the credit is computed that way. Rewards have one real quirk: task 026's e411 gives 1499 by the floor policy and gold has 1500.
6. **The checking tasks have no formula.** Tasks 072 to 074 need a fee schedule per checking class applied to each transaction (free withdrawals, tiers, caps, minimums). This is the largest unknown and is settled in Slice 0.
7. **Some wiring is missing or inconsistent.**
   - `tier1._preference` has no entry for the kind, so a replay would fall through to a direct call.
   - `validate._CUES` has no fee, interest or rewards cue.
   - `handoff.describe` has no wording for these four tools.
   - `taxonomy.py` maps `apply_credit_card_account_flag_6147` to this kind, but `guard.py` omits it from the kind's allow-list. `apply_statement_credit_8472` is in the guard lists of both the close-card and fee kinds. Both belong to the credit cluster.
8. **The orchestrator's recurring error is an extra fee action** from a symptom or a question ("why is my fee so high?"). The specialist must answer such questions from documents and write nothing.

## What
A fees specialist, configured the way the credit and accounts specialists are: the model reports what the customer said, code decides the amount and the eligibility, and the customer approves one previewed plan.

### Goals
- Four write tools covered by validators checked at plan time and again before each step: checking credit, savings credit, discrepancy report, rewards update. Read-backs where the bank's tool is silent or misleading.
- Every amount recomputed in code from the records and hand-curated policy tables. A customer's claimed amount is compared and shown, never used.
- Three workflows: checking fee credit, savings interest correction with report, rewards correction.
- Policy the tools do not enforce is enforced here: one credit per account per turn, a 14-day cooldown read from transaction history, a supervisor limit on goodwill credit, credit before report, own accounts only.
- Questions and symptoms with no proven discrepancy are answered from documents with no write.
- Tier-1 evidence on the 10 tasks, with any departure recorded.

### Non-goals
- Credit card annual-fee, late-fee or interest-charge reversals (statement credit, account flag) stay in the credit specialist. Outside a closure they are handed off with a reason written by code (see Decision 5).
- The customer-side dispute tool (`submit_cash_back_dispute_0589`) and any tool the customer runs.
- Fee or interest changes that need a person (rate changes, product changes).
- Live evaluation beyond an optional capped 3-task probe, blocked on the pilot gate (#38).

### Definitions
- **Fee schedule**: per checking class, the documented fee rules (free count, tiers, caps, minimums), with the document id for each value.
- **Expected APY**: base APY for the savings level and balance tier, plus the highest applicable checking boost, plus the highest applicable card bonus, plus the relationship bonus where the documents grant it. Boosts and card bonuses do not stack within their own group.
- **Net correction**: charged fee minus documented fee, summed per account, exact, no rounding.

## How

### Shape
`agent/specialists/fees.py` (config), `fees_rules.py` (validator per write tool, read-backs), `workflows/fee_products.py` (data tables, each value with a document id), `fee_checking.py`, `fee_interest.py`, `fee_rewards.py`. `fees_report.py` replays the 10 tasks offline, copying `accounts_report.py`. Template: the accounts specialist.

### Records
Reads both savings-account layouts (`class`/`level`, `account_type`/`account_class`) and both balance formats. Resolves an account only among the customer's own. A level not in the tables is a hand-off, never a guess.

### Validators and read-backs
- **Checking credit**: own account, class checking, open, positive amount with two decimals at most, `credit_type` valid, amount equals the recomputed net correction, no credit of type `fee_refund` or `rebate_credit` in the past 14 days, one call per account in the plan, transaction id not already present (a repeat would add the balance twice). Read-back: balance rose by exactly the amount.
- **Savings credit**: own account, class `saving` or `savings`, open, `interest_correction` amount equals the expected month's interest (balance x expected APY / 12) minus the interest paid, goodwill over $25 refused (hand-off), fee refund no larger than the fee found. Read-back: balance and transaction row.
- **Discrepancy report**: only after a completed savings credit for the same account in the same plan, expected APY equals the recomputed value, own account and user. Read-back sees only the tool's `CALLED` marker; the reply says "submitted for review", never "credited".
- **Rewards update**: own card transaction, value is `"N points"` with N a whole number equal to the floored recomputation (EcoCard: 5 points per dollar on green purchases, 1 otherwise; every other card 1 point = $0.01), differs from the stored value, a resolved cash-back dispute exists (Decision 2). `expected_rewards` in a dispute record is never trusted.

### Workflows
- **Checking credit**: read accounts and transactions, apply each account's schedule, combine corrections per account, credit type is the majority one, one step per account.
- **Savings correction**: per account, credit then report, in that order; two accounts means two ordered pairs under one preview.
- **Rewards**: per transaction, only the rows whose recomputed value differs.
- All end in one preview and one yes bound to the plan fingerprint. Balance or level changed between preview and yes: recompute at execute, refuse the step if the amount changed.

### Orchestrator changes
`tier1._preference`, `gold_facts` branch, `validate._CUES` (fee, interest, APY, rewards, points, cash back, refund), `handoff.describe` for the four tools, `live.py` configs, dispatcher registration, and a decision on the taxonomy/guard mapping for the card flag and statement credit. `plan.py` gets an ordering rule only if Slice 5 finds one is needed.

### Measurement
`uv run python -m agent.fees_report`: 10 tasks, strict pass, with 097 a recorded departure. Optional live probe of 072, 094 and 026 after #38, capped.

## Constraints and risks
- **Fee schedules are the biggest risk.** Nine checking classes with different rules and some contradictory documents. Slice 0 must recompute all gold amounts for 072 to 074 from `db.json`. If more than 2 of the 10 tasks cannot be derived, stop and re-plan.
- **The relationship bonus needs a ruling.** 093 includes 0.025 with no card held, 094 and 095 do not. Working reading: Silver's 0.025 is a stacking relationship bonus, Gold's 0.025 is a card bonus that loses to EcoCard's +0.6. Confirm against the document text in Slice 0.
- **Interest uses the current balance**, as gold does. Real interest accrues daily; this ignores balance history. State it in the decisions document.
- **Hand-curated tables can drift** from the documents. Each value has a test that checks the document text.
- **A wrong table is wrong money.** Credits are irreversible; the preview shows the working (balance, gap, result) and the customer's yes binds to the exact amount.
- **Gold quirks.** Task 026 e411 is one point above the documented floor. The code writes the documented value and the task is a recorded departure.
- Tier-1 proves plumbing, not model quality.

## Edge cases
Another customer's account or transaction id; closed, wrong-class or business account; `saving` against `savings`; amount zero, negative, NaN, infinite or three decimals; refund larger than the fee; no discrepancy found (no write, optional hand-off); goodwill over $25 (hand-off); repeat after timeout; two accounts in one turn; four accounts and 8 writes in one preview with a partial failure; unknown savings level; level or balance changed between preview and yes; rewards on a non-card transaction, floor against round ($99.99 x 2.5% = 249), unchanged value, no resolved dispute; free-text rewards ("6300", "6,300 pts"); injection in transaction descriptions, `expected_rewards` or `intent.summary`; unverified caller; a mixed fee and card-dispute turn; a symptom or question with no request.

## Issues and acceptance criteria
Sub-issues of #43, label `specialist-agents`.

**Slice 0: policy spike (blocks all).**
- `docs/fees-policy-decisions.md` exists with, for each statement, the document id and database evidence.
- Fee schedule per checking class used by 072 to 074; each gold amount recomputed from `db.json`.
- APY derivation for all nine savings credits, the relationship-bonus rule, and the 097 rounding.
- Rewards recomputation for every transaction in 026 and 028.
- Every document contradiction listed with a ruling. Any gold value not derivable is a departure candidate.
- Statements about the bank's tools are tests in `tests/agent/test_fees_policy.py`.

**Slice 1: records and data.**
- Tables in `fee_products.py`, a test per value against the document text.
- Both balance formats and both savings layouts parse. Another customer's ids never resolve. An unknown level is a hand-off.

**Slice 2: checking fee credit.**
- 072 to 074 gold writes reproduced offline.
- A second credit within 14 days is refused. An over-refund is refused. One call per account.

**Slice 3: savings interest correction and report.**
- 093 to 096 pass tier-1; 097 passes with the recorded departure.
- A claimed amount never changes a write. Goodwill over $25 is handed off. The report is never filed alone and the reply never calls it a credit.

**Slice 4: rewards update.**
- 026 and 028 pass; only the wrong rows are written; `expected_rewards` is ignored.

**Slice 5: wiring and dispatch.**
- `fees_report` replays 10 tasks with no direct fallthrough. The dispatcher registers the specialist and `unmapped_kinds()` stays empty. `_CUES` and the handoff wording are tested.

**Slice 6: safety.**
- A model scripted to obey injection (transactions, `intent.summary`, documents), a false claim, an unverified caller, preview-to-execute drift, a timeout repeat or a partial failure cannot make a write outside an approved plan. Mirrors Issue #67.

**Slice 7: evidence and docs.**
- `agent.fees_report` committed output, CLAUDE.md section, "what changed while building" below, optional capped live probe.

## Decisions
1. Task 026 e411 (1499 by the documented floor, gold 1500): follow policy and record a departure in `evals/policy_departures.json`. Task 097 is not a departure (the interest method matches). Tier-1 total: 9 of 10 strict plus 1 departure.
2. Rewards update only when a resolved cash-back dispute exists. **Pending.** The alternative is to fix any wrong row the agent finds. The user-side dispute tool is out of scope, so this needs a fixture decision.
3. Hand-curated policy tables in code, not parsed from documents at run time and not supplied by the model.
4. Checking credits enforce one credit per account per turn and a 14-day cooldown from transaction history. A discrepancy report is filed only after a completed credit.
5. Card annual-fee, late-fee and interest-charge reversals outside a closure are handed off in v1. **Pending.** The alternative is to extend this specialist to statement credits, which needs policy not in the documents.
6. Whether 097 counts toward the tier-1 pass total as a departure or is excluded. **Pending.**

## What changed from this PRD while building it
Status: built and committed as Slices 0 to 6 (Issues #70 to #76); Slice 7 (#77) has its offline evidence and docs, and its optional live probe is still blocked on #38.

1. **The credit is the expected month's interest minus the interest paid, not balance x gap / 12.** The PRD's formula is a cent off on 097 silver; the interest method matches all nine gold credits, so 097 is not a departure (only task 026 e411 is). Found building Slice 3.
2. **The discrepancy report is not invisible.** It stores no report, but the tool's `CALLED` marker is in the state hash; the read-back checks that marker.
3. **The relationship bonus was settled by inference** (Silver/Silver Plus with a linked checking account and no card bonus), because the documents give no criteria and tau2 has no separate policy. It is one table flag, `RELATIONSHIP_LEVELS`.
4. **One credit type rule** (document 017) became `fee_part` / `rebate_part` in `fee_review`; every gold credit is a `fee_refund`.
5. **Savings credits of any type other than `interest_correction` are never planned** (fee refund and goodwill need a person), so the $25 goodwill limit is enforced by absence, not by a threshold check.
6. **The replay needed one change outside this specialist:** `tier1.workflow_turns` stopped at the first workflow of a kind whose gold facts did not match; with three workflows on one kind it now tries the next.
7. **Green checking's foreign fee** is in document 012, not 001 or 010 (found by the value-by-value document tests).
8. **Not built:** the card annual-fee/late-fee/interest-charge reversal (Decision 5: handed off), Gold Years and Green Fee-Free ATM schedules (no document values found, so a hand-off), and any live measurement.
