# Fees, interest and rewards: policy decisions with evidence

Slice 0 of `docs/prd-fees-interest-rewards-specialist.md` (Issue #70). Status: **partial**. Sections marked TODO are not yet derived. Every statement names its document or database evidence; statements about the bank's tools become tests in `tests/agent/test_fees_policy.py`.

## 1. The bank's tools (read from `tau2-bench/src/tau2/domains/banking_knowledge/tools.py`)
- `submit_interest_discrepancy_report_7294` stores no report: `interest_discrepancy_reports` is not a table in the data model, `add_to_db` returns False, the tool ignores it and prints "Submitted... PENDING_REVIEW". The one trace is the tool's own `CALLED` marker in `agent_discoverable_tools` (that is what changes the state hash), so a read-back can confirm the report was submitted, never what it said. It also checks neither the figures nor that the user owns the account. (An earlier version of this note said the hash is unaffected; running it showed the marker.)
- `apply_checking_account_credit_5829` writes `current_holdings` as `"$x.xx"`; `apply_savings_account_credit_6831` writes `"x.xx"`. Neither checks ownership, a cap, a cooldown or once-per-account. Both write a transaction row with a deterministic id (account, type, amount, date), so a repeat call rewrites the row but adds the balance again.
- `update_transaction_rewards_3847` accepts any string for any transaction id.

## 2. Savings interest correction (tasks 093 to 097): credit = expected month's interest minus the interest paid
Balances are `current_holdings` in each task's `initial_state`; gap is expected minus actual APY from the gold report call.

| Task | Account | Balance | Gap | Computed | Gold | Match |
|---|---|---|---|---|---|---|
| 093 | silver | 144,000 | 4.275 - 4.0 | 33.00 | 33.00 | yes |
| 094 | gold | 96,000 | 6.85 - 5.10 | 140.00 | 140.00 | yes |
| 095 | gold | 96,000 | 6.85 - 5.625 | 98.00 | 98.00 | yes |
| 096 | bronze | 30,000 | 0.70 | 17.50 | 17.50 | yes |
| 096 | gold plus | 60,000 | 0.30 | 15.00 | 15.00 | yes |
| 097 | silver | 100,000 | 6.65 - 4.0 | 220.8333 by the gap, **220.84** by interest | 220.84 | yes, by the interest method |
| 097 | platinum | 70,000 | 7.65 - 6.5 | 67.0833 | 67.08 | yes |
| 097 | diamond | 120,000 | 8.2 - 7.5 | 70.00 | 70.00 | yes |
| 097 | silver plus | 18,000 | 5.3 - 4.5 | 12.00 | 12.00 | yes |

**Correction (found while building Slice 3):** the credit is not balance x gap / 12 but the expected month's interest (balance x expected APY / 12) minus the interest actually paid. The 'gap' column above is a cross-check only. By the interest method all nine accounts match gold, 097 silver included (554.1667 - 333.33 = 220.8367 = 220.84), so **097 is not a departure** and no entry is added to `evals/policy_departures.json` for it. The actual APY is the interest paid x 12 / balance, to 3 decimals (097 silver: 3.99996 shown as 4.0). The earlier 'rounding departure' was an artefact of using the gap formula; the PRD's formula was the less faithful one.

**Relationship bonus (revised).** The earlier hypothesis (it applies only with a matching rewards card) is wrong: task 093 holds no credit card and gold still includes 0.025. A reading that fits all three tasks: the Silver 0.025 relationship bonus is its own bonus and stacks with everything (Silver documents 003 and 006); Gold's 0.025 is the same figure as the Gold card bonus in `gold_account_014` ("Gold +0.025"), so it competes with other card bonuses and only the highest applies. In 094 and 095 the user holds a Gold card and an EcoCard, EcoCard's +0.6 wins, and 5.5 + 0.75 (Green checking with Gold savings) + 0.6 = 6.85. Document 045 confirms the Gold half: it lists "Gold Rewards Card: +0.025% APY bonus" among the card bonuses that do not stack, and says relationship bonuses are a separate kind that does stack. Still open (moved to Slice 1 table tests, because it needs the per-tier checking-boost numbers): derive 096 (Bronze 3.25, Gold Plus 6.85, user holds Diamond Elite and Crypto cards and three checking accounts) and 097 (four accounts, five cards) the same way.

**Relationship bonus ruling (checked against tau2's own material).** tau2 has no separate policy file for this: the only sources are the product documents and the task notes. The documents give no criteria for Silver and Silver Plus ("may qualify", "when you qualify under the relationship criteria"), tie the same 0.025 to a Platinum card in Silver 009 and to a Gold card in Gold 013, and document 045 lists the Gold card's +0.025 among the card bonuses that do not stack. The task notes name it only in 093 ("relationship bonus for having multiple Rho-Bank products") and never in 094 to 097, whose card holders get their card bonus instead. Ruling: grant 0.025 only to a Silver or Silver Plus holder with a linked checking account and no card bonus; otherwise never. This reproduces every gold value (093 = 33.00) but is inferred from the data, not written in the documents, so it is a table flag (`RELATIONSHIP_LEVELS`) a person can change. Not decided by the user yet: raised in the Slice 1 summary.

## 3. Checking fee credit (tasks 072 to 074): net correction per account

### Task 072 (derived)
Bluest account `chk_lj82d4f1a9` (docs: Bluest foreign ATM fee $0; $50 monthly rebate cap on out-of-network ATM fees):
- Foreign ATM fees charged: 8.00 (11/08, Barclays) and 3.50 (11/12, Mizuho); documented fee 0, so refund 11.50.
- Non-Rho ATM fees charged six times (2.00, 2.00, 2.00, 2.00, 2.00, 2.50). Rebates posted on five of them: 2.00 each, and 11/20 charged 2.50 with a 2.00 rebate. Unrebated: the 11/14 fee (2.00) and 0.50 of the 11/20 fee, so 2.50. The $50 cap is not reached.
- Net correction 11.50 + 2.50 = **14.00**. Gold 14.00, `fee_refund`.

Light Green account `chk_538bfb9cba` (docs: 4 free out-of-network ATM withdrawals per month then $1.50; foreign tiers $2.00 up to and including $100, $3.50 up to and including $300, $5.00 above):
- Foreign: 11/13 (80) charged 5.00, correct 2.00, over 3.00. 11/15 (200) charged 4.00 (a 2% minimum $3 rule from another class), correct 3.50, over 0.50. 11/17 (350) 5.00 correct. 11/19 (100) 2.00 correct (exactly 100 is tier 1).
- Domestic: six withdrawals, so the 5th and 6th cost $1.50 each (3.00). Fees were charged on the 3rd and the 6th (3.00). The total is right but the fee fell on the wrong withdrawal; the net is zero.
- Net correction 3.00 + 0.50 = **3.50**. Gold 3.50, `fee_refund`.

**Design consequence:** the engine nets over-charges against under-charges per account and must count free withdrawals by date. It must not credit a mis-timed fee on its own. Foreign withdrawals do not count toward the four free out-of-network ones (assumption matched by gold; confirm against the Light Green document wording).

### Task 073 (derived; matches gold exactly)
Rules read from the documents: Blue non-Rho ATM 1% (max $3.00), foreign max(3%, $5.00); Green non-Rho $3.00, foreign max(3%, $5.00); Light Green 4 free non-Rho withdrawals a month then $1.50, foreign tiers $2.00/$3.50/$5.00; Rho-Bank ATMs never carry a fee. (Green's foreign-fee formula is in `green_account_(checking)_012`, not 001 or 010; the schedule table cites 012.)
- Blue `_1`: foreign 500 charged 20.00, correct 15.00 (over 5.00); 300 domestic charged 4.50, correct 3.00 (over 1.50); a Rho-Bank ATM withdrawal charged 3.00 (over 3.00). Net **9.50**, gold 9.50.
- Green `_2`: a Rho-Bank ATM fee 3.00; the 11/15 fee posted twice (extra 3.00); foreign 400 charged 15.00, correct 12.00 (over 3.00). Net **9.00**, gold 9.00.
- Light Green `_3`: a fee on the 2nd withdrawal (over 1.50); a foreign 150 charged 5.00, correct 3.50 (over 1.50); the 7th withdrawal carried no fee (under 1.50). Net 3.00 - 1.50 = **1.50**, gold 1.50.

### Task 074 (derived; matches gold exactly)
- Purple `_1` (foreign fee $0; domestic out-of-network fees rebated up to $30 a month): foreign fees 8.00 + 10.50 = 18.50 are all wrong; domestic fees 2.50, 3.50 and a duplicated 2.50 carry no rebate, 8.50, and the cap is not reached. Net **27.00**, gold 27.00.
- Light Blue `_2` (2 free out-of-network then $2.50; 2 free foreign then $4.00): a Rho-Bank ATM fee, and the 1st and 2nd out-of-network withdrawals charged (7.50 over); the 4th withdrawal was not charged (2.50 under); foreign fees on the 1st and 2nd (8.00 over), the 4th foreign charged 5.50 against 4.00 (1.50 over). Net 7.50 - 2.50 + 9.50 = **14.50**, gold 14.50.
- Dark Green `_3` (out-of-network 1%, min $1.50; foreign 2.5%, max $6.00): over: a Rho-Bank ATM fee 1.50, a "2%" fee on 200 (2.00 over), a foreign 400 charged 10.00 against the 6.00 cap (4.00 over); under: 50 charged 0.50 against the 1.50 minimum (1.00), and the 175 withdrawal with no fee (1.75). Net 7.50 - 2.75 = **4.75**, gold 4.75.
- Evergreen `_4` (out-of-network 1%, max $2.50; foreign 2%, min $3.00): over: Rho-Bank ATM fee 1.50, 400 charged 4.00 against the 2.50 cap (1.50), foreign 350 charged 10.50 against 7.00 (3.50); under: 180 withdrawal with no fee (1.80), foreign 100 charged 2.00 against the 3.00 minimum (1.00). Net 6.50 - 2.80 = **3.70**, gold 3.70.

**Design consequences, all three checking tasks:**
1. The credit is the net of over-charges and under-charges per account, never the sum of over-charges. An under-charge is netted but never itself billed.
2. The engine needs per account class: free-withdrawal counts by calendar month in date order, in-network detection (`RHO-BANK` in the description), foreign detection (non-US country suffix), a fee formula (percent, minimum, maximum, tiers), and a rebate cap.
3. All gold credits use `fee_refund`, including accounts where rebates are also missing, so the majority rule in the PRD stands but is unexercised by the gold tasks.

## 4. Rewards (tasks 026 and 028)
Rule (documents 004, 006, 007): points = floor(amount x rate x 100) at the rate for the card type, category and any promotion active on the purchase date; EcoCard is 5 per dollar on green purchases, otherwise 1.

| Task | Transaction | Card and category | Stored | Computed | Gold |
|---|---|---|---|---|---|
| 026 | e403 $315 | Business Silver, travel, 03/22/2025 | 3150 | 10% x 2 = 6300 | 6300 |
| 026 | f506 $255 | Silver, software | 2550 | 4% = 1020 | 1020 |
| 026 | e410 $380 | Business Silver, travel, 08/25/2025 | 1520 | 10% = 3800 | 3800 |
| 026 | e411 $149.99 | Business Silver, software, 09/15/2025 | 600 | 10% = 1499.9, floor **1499** | **1500** |
| 028 | 57ecc $47.83 | Crypto | 47 | 2% = 95.66, floor 95 | 95 |
| 028 | d80ae $347.62 | Business Platinum, travel | 521 | 4% = 1390.48 | 1390 |
| 028 | 896ac $128.47 | EcoCard, green | 128 | 5/$ = 642.35 | 642 |
| 028 | adea6 $167.34 | Silver, travel | 167 | 4% = 669.36 | 669 |
| 028 | 0be1c $512.47 | Business Platinum, media | 768 | 4% = 2049.88 | 2049 |
| 028 | e647e $187.56 | Business Platinum, media | 1875 | 4% = 750.24 | 750 |

Findings:
- The promotion is conditional on dates: Business Silver's 2x applies for the first six months after the account's opening (`date_of_account_open` 02/13/2025), so e403 (03/22) doubles and e410 (08/25) does not. Document 012 also states a fixed promo period (2024-11-14 to 2025-11-14); the account must have opened within it.
- **A second gold quirk: 026 e411.** Correct floor gives 1499; gold has 1500 (rounding up). Task 026 therefore fails strict DB comparison if code follows policy. Candidate for the departures file, like 097 silver.
- The stored rewards are wrong in every row, but gold rewrites every row it lists. Rows with a correct stored value would not be written (none occur here).
- **Confirmed (Slice 4):** both 026 and 028 set `dispute_settings.auto_resolve_disputes`, so the customer's own dispute tool leaves a `RESOLVED` / `APPROVED` row, and the rewards update is allowed only when such a row exists for that purchase and customer (PRD Decision 2, now confirmed workable). The earning rules per card (rates, merchant exclusions, the Business Silver doubling inside six months of an opening within its promo window) are the `REWARDS` table, tested against 026 and 028: nine of ten transactions match gold, and the tenth (026 e411) is the documented-floor departure.
- `expected_rewards` does not exist in this data (`cash_back_disputes` is empty in `db.json`); disputes are created by the customer's own tool. That tool writes status `RESOLVED` with resolution `APPROVED` only when the task's `dispute_settings.auto_resolve_disputes` is set, otherwise status `SUBMITTED`. So "a resolved dispute exists" (PRD Decision 2) is a data condition the replay must create first; confirm in Slice 4 that 026 and 028 set auto-resolve.

## 5. Document contradictions and rulings
| Contradiction | Ruling | Evidence |
|---|---|---|
| Gold 013: 5.5 + 0.025 = "6.0%" | Compute 5.525; the stated total is an arithmetic error and is never used | arithmetic |
| Silver 002: unfilled `[[linked_checking_apy_bonus]]` | Take the boost from the pairing table (doc 012) and the tier's own document, never from this placeholder | 093 gold 4.275 |
| Light Green daily purchase limit 300 (001) vs 250 (002) | Not used by any fee or interest rule; out of scope for this specialist | no task uses it |
| Rebate caps: Bluest $50, Purple $30, Gold Savings $30, Silver Plus "25" | Per product document, each cap a table value with its document id; "25" has no unit and is treated as unknown, so a rebate on Silver Plus is a hand-off | 072, 074 |
| Dark Green and Evergreen: domestic and foreign ATM fees in separate documents (001 vs 002/008) | Choose by transaction type: a foreign withdrawal (non-US country code) uses the foreign formula, anything else the out-of-network one | 074 |
| Doc 017: one checking credit per account per interaction and a 14-day cooldown; the tool enforces neither | Enforce in the validator (Decision 4) | tool source |
| Doc 043: goodwill over $25 needs a supervisor; the tool has no limit | Enforce: over $25, hand off | tool source |
| Stored rewards vs documented rate | Recompute; never trust stored value or `expected_rewards` | 026, 028, doc 004 |
| Gold 220.84 (097) and 1500 (026 e411) vs the floor/round rules | Policy wins; both recorded as departures | sections 2, 4 |
| Business Silver 2x promo: fixed promo period (doc 012) and "first six months from opening" | Both must hold: account opened inside the promo period, purchase inside six months of opening | 026 e403/e410 |

## 6. Status
Derived and matching gold: 072, 073, 074, 093, 094, 095 (up to the checking-boost numbers), 096 Bronze, 028 and four of the 026 rows. Two departures (097 silver, 026 e411). Remaining items (the boost numbers for 096 Gold Plus and 097, and whether 026/028 set auto-resolve) are built into the Slice 1 table tests and the Slice 4 acceptance criteria rather than a separate spike, because each needs the table or fixture to exist first.
