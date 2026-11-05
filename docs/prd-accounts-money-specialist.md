# PRD: Accounts and money movement specialist (Issue #42)

Part of `docs/prd-specialist-agents.md`, phase 4. This PRD replaces the four acceptance criteria that Issue #42 started with.

## Why
Issue #42 asked for four things: savings opening checks the 14-day rule and exact class strings; closing requires funds moved first with later steps using `StepRef` ids; a third-party destination is blocked (task 016); tier-1 on the 19 tasks. Reading the bank's tools, the knowledge-base documents and the tasks shows that some of this holds and some does not:

1. **Task 016 is not a transfer.** It is a credit referral (`submit_transaction` on a friend's card). The "third-party destination" rule has no gold task. It needs synthetic fixtures.
2. **The 19 tasks are almost all opening and closing.** Tasks 055 to 071, 075 and 076 open accounts, sometimes close one, and sometimes move money as part of it (056, 062). None is a standalone transfer or a card payment. Fourteen of them are recommendation tasks ("which account should I open?"), where the right class depends on balance, existing cards, dated promotions and fee/APY arithmetic. The issue says nothing about who chooses the class.
3. **The bank's tools enforce much less than the documents.**
   - Transfer (`transfer_funds_between_bank_accounts_7291`) checks amount, different accounts, existence, status and funds. It does not check that both accounts have the same owner, writes no transaction row, has no duplicate guard, and writes balances as `"$x.xx"` where other tools write `"x.xx"`.
   - Open (`open_bank_account_4821`) checks only savings (a checking account at least 14 days old), business checking (no CLOSED account, an open personal checking) and business savings (no negative balance, a business checking at least 30 days old). The documents also require age 18, account caps of 4, 5, 6 and 4, a $500 balance for business checking, $2,500 for business savings, no collections, and the exact class name. The tool accepts any class string.
   - Close (`close_bank_account_7392`) checks that balance minus the fee is zero, but never debits the fee. Its savings fee branch tests `class == "savings"` while the data says `"saving"` (9 rows), so it never fires. It has no notice period, no pending-transaction check, no approval tier and no business fee. `waive_early_closure_fee=True` skips the fee check with no authority check.
4. **A schema mismatch hides new accounts.** Open writes `account_type` and `account_class`; the other tools read `class` and `level`. A newly opened account is invisible to the checking credit tool and to the early-closure fee logic. Its id is a hash of user, type and class, so opening the same type and class twice fails with "may already exist".
5. **Order decides success.** In 060, 062 and 065 to 069 the new account has to be opened before the old one is closed: closing the only qualifying checking account first blocks the savings opening, and any CLOSED account blocks business checking. Nothing in the planner encodes this. OPEN and CLOSE have no required id slots, so `plan.py` serializes them only when no ids are named.
6. **Customers say untrue things.** In 057 the customer claims a 14-day-old account. The clock is injected (11/14/2025), so the answer comes from the records.
7. **Some gold steps are the customer's.** Tasks 055, 057 and 061 end with the customer calling `deposit_check_3847`. The agent only hands the tool over. In 061 the $25 is deposited into the account that is then closed; the fee is "paid" only through the balance check, so the specialist must not move that $25 elsewhere.
8. **The orchestrator has gaps for these kinds.** `validate._CUES` has dropped-intent cues for transfer, close and balance, but none for open or pay. A hidden transfer (056, 062) is planned by the specialist under one plan preview, not by `decision.fingerprint`. Customers name accounts by nickname ("Bronze savings"), which the id-based resources do not match.
9. **No bank tool exists for most money movement.** Wires, ACH, bill pay, overdraft, holds and check deposits are only policy text. The kinds the orchestrator knows are TRANSFER_FUNDS, PAY_CREDIT_CARD, OPEN_ACCOUNT, CLOSE_ACCOUNT (to the specialist) and ACCOUNT_BALANCE_OR_HISTORY (read-only, already handled by DIRECT).
10. **Standalone transfer and card payment have almost no evidence.** One eval message each (`mp_31` for transfers) and no gold task. They are verified with synthetic fixtures, like the credit validators.

## What
An accounts specialist, configured the way the credit and debit specialists are, whose policy decisions are made by code and approved by the customer as a previewed plan.

### Goals
- Four write tools covered by validators checked at plan time and again before each step: open, close, transfer, pay a credit card from checking. Each has read-backs where the bank's tool is silent or misleading.
- A hybrid class choice: code builds the eligible candidates, the model ranks within them with document citations, the customer confirms the exact class string before the yes.
- Workflows for opening, closing with funds first, and standalone transfer and card payment, where the model reports what the customer said and code decides.
- An ordering rule in the planner (open before close) plus an execute-time check in the specialist.
- Requests with no bank tool are answered from the documents or handed off with a note written by code.
- Offline evidence on the 19 tasks (055 to 071, 075, 076) and the transfer-only hand-off tasks as negatives.
- One small, capped, opt-in live probe of 3 tasks, after Issue #38 passes.

### Non-goals
- New IntentKinds (no WIRE_OR_ACH, DEPOSIT). The 22-kind schema snapshot and the eval labels stay as they are.
- Fees, interest and rewards credits and the interest report: tasks 072 to 074 and 093 to 097 belong to Issue #43.
- Tasks 043, 045, 048 and 051 (a card paid from checking inside a credit flow): the credit specialist owns them and calls the pay tool itself.
- Debit-card and lost-wallet tasks (Issue #44 and the debit specialist).
- The customer-side `deposit_check_3847`. The agent hands it over and explains; it never calls it.
- Editing the bank's tools, or making their weaknesses match the gold.
- Live runs beyond the 3-task probe.

### Definitions
- **Candidate set**: the account classes a customer is eligible for today, after code applies the rules and drops expired promotions. The model only ranks inside it.
- **Funds first**: before a close, the balance is moved to an account the customer owns, as a transfer step in the same plan.
- **StepRef**: an id taken from an earlier step's result (a newly opened account has a hashed id that is unknown until the step runs).
- **Hidden transfer**: a transfer the plan needs that the customer did not literally ask for (056, 062).
- **Safe-writes-only case**: a task where the right outcome contains writes, but a larger set of tempting writes must not happen.

## How

### Shape
`agent/specialists/accounts.py` is a `SpecialistConfig` like `specialists/credit.py`: instructions, the `ACCOUNTS` scope in `knowledge/scopes.py`, `VALIDATORS` keyed by tool name, `READBACKS`, a `prepare` that settles which account is meant in code, and the workflows. Workflows follow `agent/workflows/base.py`: `decide` returns Plan, NeedInfo, Refuse, Reply or HandOff and never raises. Reused as they are: `agent/cards.py`-style record helpers, `credit_rules.py` helpers (`money()`, `day()`), `agent/guard.py` (`KIND_WRITE_TOOLS` already lists the four tools), `agent/dispatcher.py` (`Name.ACCOUNTS` already routes the four kinds; it only needs the specialist registered), `agent/handoff.py`, `agent/tier1.py`, `evals/manifest.json`, `evals/policy_departures.json`.

### Records
- An account resolver takes an id, a nickname or the last four digits. If more than one account matches, or none, it returns NEEDS_INFO with a list written by code (class, last four, status). Another customer's id resolves to nothing, and the reply is the same as for an unknown id.
- A normalised read model reads `account_type` and `account_class` with a fallback to `class` and `level`, so newly opened accounts are seen. Balances in `"$x.xx"` and `"x.xx"` both parse.

### Validators and read-backs
One per write tool, each with an accept test and a reject test per rule, run at plan time and again before the step.
- **Open**: savings needs a checking account at least 14 days old (the clock decides, not the customer); the exact class string comes from the catalogue; account caps; business minimum balances; age 18; no collections; reopening the same type and class is refused with a clear reason. Read-back: the new account exists as OPEN.
- **Close**: the balance is zero, or exactly the early-closure fee inside its window (the fee stays in place; the specialist never moves it); no pending transactions; notice period; accounts that need a manager or supervisor are handed off; `waive_early_closure_fee` is never set by the agent. Read-back: status CLOSED.
- **Transfer**: both accounts belong to the verified customer, are open, differ, and the source covers the amount. A repeat of the same step after a timeout does not move money twice. Read-back: both balances, because the tool writes no transaction row.
- **Pay card from checking**: the source is an open checking account of the customer, funds cover it, and the amount does not exceed the card balance.

### Class catalogue and candidate builder (hybrid)
A data table from the accounts documents: product, type, exact class string, eligibility rules, fees, rates and boosts, card pairings, and dated promotions. Code filters it with the injected clock (expired October promotions are a trap pair), drops anything ineligible, and passes the candidates with their document ids to the model. The model ranks and explains; code checks that the chosen class is in the candidate set before any plan is proposed. The customer confirms the exact string.

### Open workflow (`OPEN_ACCOUNT`)
Needs (type, balance, cards held, features wanted) → candidates → customer confirms an exact class → open. After a savings or business savings opening, the specialist asks about the opening deposit, because the documents give 30 days to fund it. The customer's claims about account age are checked against the records.

### Close workflow (`CLOSE_ACCOUNT`), funds first
Read balance and fee window → if the balance is not zero and not exactly the fee, ask where it goes (the choices are the customer's own open accounts, rendered by code) → transfer with a `StepRef` → close. A destination outside the customer's accounts is not offered and is refused if named.

### Standalone transfer and card payment
Slots from the orchestrator (`from_account_id`, `to_account_id`, `amount`; `account_id`, `amount`) are resolved to the customer's own records. The amount is in the plan the customer approves, so a changed amount makes the yes stale. Payment boundary with credit: standalone "pay my card" comes here; the credit workflows call the tool themselves and are not rerouted.

### Orchestrator changes
- `orchestrator/plan.py`: OPEN_ACCOUNT before CLOSE_ACCOUNT (like credit-limit-before-dispute, no cycle with existing rules).
- `orchestrator/validate.py`: dropped-intent cues for open and pay.
- Specialist: an execute-time check that refuses a close that would break the eligibility of an open still in the plan.
- `tests/orchestrator/test_scenarios.py`: a `@covers` scenario for the 062 shape (close, open, close, open).

### Requests with no bank tool
Wire, ACH, bill pay, overdraft, holds and check deposits: a question is answered from the accounts documents with citations. A request to perform one gets the customer-side steps if a customer tool exists (`deposit_check`), otherwise a hand-off with a note written by code from the audit. The eight transfer-only ACTION tasks (004, 008, 012, 014, 032, 033, 034, 035) are negative assertions: a hand-off, the right transfer tool, no domain write.

### Measurement
Tier-1 compares the final database with the one tau2 gets from the gold (`agent/tier1.py`). Order rules (open before close, funds before close) are asserted separately from the state comparison. Gold departures from documented policy go to `evals/policy_departures.json` with a reason. Whether tier-1 replays `requestor=user` gold actions (the check deposits) is settled in Slice 0.

## Constraints and risks
Ranked; each is an acceptance criterion in a slice.
1. **Wrong class opened.** An eligible but wrong class is an irreversible write. The candidate filter, the exact string check and the customer's confirmation guard it. Live runs measure ranking quality.
2. **Closing before opening** blocks a savings or business account. Fixed in the planner and in the specialist.
3. **Money moved to the wrong place**: a third-party or another customer's destination, a nickname that matches two accounts, a changed amount after the yes. Ownership check, code-rendered choices, stale-fingerprint rule.
4. **Duplicate transfer** after a timeout. PARTIAL, resume by step id, no second move.
5. **Fee semantics.** The tool checks but never debits the fee; the specialist must neither move the fee-sized balance nor set the waiver flag.
6. **Schema mismatch.** New accounts are invisible to class-based tools; the read model covers both layouts.
7. **Hash collision** on reopening the same type and class.
8. **Customer lies** (057) and **expired promotions** presented as current. The records and the injected clock decide.
9. **Injection** in account nicknames, transaction text or documents. The guard is code; a fake model scripted to obey cannot make an unapproved write.
10. **Document conflicts.** The documented policy wins over the tool and the gold; every departure is listed in Slice 0 and in the exclusion file.
11. **Customer-side steps** (the check deposit) must not be treated as agent writes or silently skipped.
12. **Cost.** Nothing live until Issue #38 passes; then 3 tasks, capped, with cassettes.

Known limits: tier-1 proves the plumbing, not the model; ranking quality is measured live on only 3 tasks; standalone transfer and payment are tested on synthetic fixtures, not on gold; the interest and fee side belongs to Issue #43.

## Edge cases
Each maps to a slice (S).
- Funds-first close (S5); fee equal to the balance, so the account closes with the fee left in place and the specialist moves nothing (S5, task 061).
- Insufficient funds; same source and destination; transfer to another customer's account (S2, S7, S9).
- A nickname that matches two accounts; an account already closed (S1).
- Business checking blocked by any CLOSED account; savings blocked by closing the only qualifying checking account first (S2, S6).
- Reopening the same type and class (hash collision); a new account invisible to class-based tools (S1, S2).
- A duplicate transfer on retry; a timeout after a committed write; a balance that changes between preview and yes; an amount changed after the yes (S2, S7, S9).
- A card payment above the card balance; payment from a non-checking or closed account (S2, S7).
- A false "14 days" claim; an expired promotion offered as current (S3, S9).
- Hidden transfers that are not in the customer's words (S6).
- A fee-waiver request with no authority (S2).
- Wire, ACH, hold and overdraft requests with no tool; the customer-side check deposit (S8).
- Injection in nicknames or transaction text (S9).

## Issues and acceptance criteria
"Live" criteria need API keys, are opt-in and capped. Everything else runs offline. GitHub numbers are in the table at the end.

**Slice 0: Policy spike.** Documentation and a small test only.
- A committed note (`docs/accounts-policy-decisions.md`) holds the rule table per product (age, caps, balances, collections, notice, fee windows, approval tiers) beside what each bank tool enforces, and fixes the meaning of the early-closure fee.
- The note settles: how tier-1 treats `requestor=user` gold actions (the check deposits in 055, 057, 061); where the exact class strings come from; the boundary with Issue #41 (pay card) and Issue #43 (credits); transfer ownership; and lists every place where gold departs from policy.
- Tasks that cannot be derived are listed as policy-ambiguous, not fitted. Blocks Slices 1 to 9.

**Slice 1: Records and dispatcher.**
- An account is resolved from an id, nickname or last four digits; more than one match or none gives a code-written NEEDS_INFO list; another customer's id resolves to nothing, with the same reply as an unknown id.
- The normalised read model sees newly opened accounts and parses both balance formats.
- The accounts specialist is registered under `Name.ACCOUNTS`, with its entry in `live.py`, and the test for unmapped sub-agent kinds still passes. Depends on: Slice 0.

**Slice 2: Validators and read-backs for the four write tools.**
- Every rule has an accept test and a reject test on synthetic fixtures, each with its own reason (open: 14-day, caps, balances, age, collections, class string, hash collision; close: funds first, fee window, notice, pending, approval tier, no waiver; transfer: same owner, active, funds, duplicate; pay: open checking, funds, at most the card balance).
- Read-backs confirm the new account, the closed status, and both balances after a transfer. Depends on: Slice 0.

**Slice 3: Class catalogue and candidate builder.**
- The catalogue is built from the documents, every candidate carries a document id, and an expired promotion is never a candidate (a test pairs the October and November promotions on the fixed date).
- A chosen class that is not in the candidate set cannot become a plan. Depends on: Slice 0.

**Slice 4: Open workflow.**
- The customer must confirm the exact class string before the plan is shown; a false "14 days" claim is answered from the records and nothing is opened (057).
- A funding-deposit question follows a savings or business savings opening.
- Tier-1 passes 055, 057, 058, 059, 063, 064, 070, 071, 075, 076. Depends on: Slices 0 to 3.

**Slice 5: Close workflow, funds first.**
- A non-zero balance becomes a question with the customer's own accounts as choices; a third-party destination is not offered and is refused if named.
- The balance equal to the fee closes without a move and without a waiver (061).
- Tier-1 passes 060 and 061. Depends on: Slices 0 to 2.

**Slice 6: Orchestrator ordering and mixed requests.**
- OPEN_ACCOUNT is planned before CLOSE_ACCOUNT, without a cycle with the existing rules; open and pay have dropped-intent cues; a close that would break an open still in the plan is refused at execute time.
- A `@covers` scenario runs the 062 shape (close, open, close, open) and the hidden transfers are previewed in one plan.
- Tier-1 passes 056, 062 and 065 to 069. Depends on: Slices 4, 5.

**Slice 7: Standalone transfer and card payment.**
- On synthetic fixtures: a transfer to another customer's account is refused; a changed amount makes the yes stale; a repeat after a timeout does not move money twice.
- A standalone card payment goes to this specialist, and the credit workflows (043, 045, 048, 051) still pass. Depends on: Slices 1, 2.

**Slice 8: Requests with no bank tool.**
- Wire, ACH, bill pay, overdraft, hold and deposit questions are answered with document citations; a request to perform one gets the customer-side steps or a hand-off written by code, and no write happens.
- The eight transfer-only tasks (004, 008, 012, 014, 032, 033, 034, 035) pass as hand-off negatives. Depends on: Slice 1.

**Slice 9: Traps and adversarial cases.**
- A fake model scripted to obey the customer's lies, an injected nickname, the supervisor bypass code, a closed account blocking business checking, and preview-to-execute drift cannot cause an unapproved write.
- The third-party destination is tested on synthetic data (task 016 is not a transfer). Depends on: Slices 4 to 8.

**Slice 10: Evidence and report.**
- `agent/accounts_report.py` prints a per-task report (semantic and strict pass, steps) for the 19 tasks; manifest counts, exclusions and policy departures are updated and their tests pass.
- Zero guard bypasses; a default `pytest` run builds no live client; Issue #42's body becomes the slice checklist. Depends on: Slices 4 to 9.

**Slice 11: Live probe (blocked on Issue #38).**
- Opt-in and capped, with an estimate that needs `--yes` and cassettes: tasks 057 (false claim, class choice), 060 (open before close), 075 (class ranking).
- The result is reported as a measurement, not a gate. Depends on: Slice 10, #38.

## Decisions
Decided by the project owner for this PRD:
1. The class is chosen by a hybrid: code builds the eligible candidates, the model ranks within them, the customer confirms the exact string.
2. Requests with no bank tool are answered from the documents or handed off; no new IntentKinds.
3. When the tool, the documents and the gold disagree, the documented policy wins and gold departures go to the exclusion file.
4. Open-before-close is enforced in the planner and again in the specialist at execute time.

Adopted from the design review (open to veto):
- Standalone transfer and card payment are validated with synthetic fixtures, because no gold task exercises them.
- The agent never sets the fee-waiver flag and never moves a fee-sized balance.
- Task 016 is dropped from this issue.
- Tasks 065 to 069 are tier-1 in Slice 6 (open and close in one request), not in Slice 5.
- Offline slices now; the live probe stays blocked on Issue #38.

## GitHub issues
| Slice | GitHub | Title |
|---|---|---|
| parent | #42 | Accounts and money movement |
| 0 | #58 | Policy spike |
| 1 | #59 | Records and dispatcher |
| 2 | #60 | Validators and read-backs for the four write tools |
| 3 | #61 | Class catalogue and candidate builder |
| 4 | #62 | Open workflow |
| 5 | #63 | Close workflow, funds first |
| 6 | #64 | Orchestrator ordering and mixed requests |
| 7 | #65 | Standalone transfer and card payment |
| 8 | #66 | Requests with no bank tool |
| 9 | #67 | Traps and adversarial cases |
| 10 | #68 | Evidence and report |
| 11 | #69 | Live probe (blocked on #38) |

All issues carry the `enhancement` and `specialist-agents` labels. Issue titles are `[4.2.N] Accounts: ...` for Slice N.

## What changed from this PRD while building it
- **Slice 0:** tier-1 already replays customer-side gold steps (`requestor=user`) directly, so nothing changes in the specialist for the check deposits. A duplicate open is a silent no-op the wrapper reports as success (the bank words it "Failed to open account"), so the open read-back is required, not optional. See `docs/accounts-policy-decisions.md`.
- **Slice 1:** a customer's answer can be a number from the list we showed. `"01"`, which is another customer's account id in the data, was first read as "the first one" and resolved to the customer's own account; a number is now accepted only when plain (no leading zero, two digits at most). Found by the third-party test.
- **Slice 2:** the transfer read-back is an invariant check (both accounts still open, the source not below zero), not a before-and-after balance comparison, because every failure of the transfer tool is loud and a read-back sees only the state after. A checking account opened in the same visit cannot pay a card: the bank's pay tool reads `class`, which a new account lacks; the validator says so. All 19 reference trajectories pass the validators, so no account departure is recorded.
- **Slice 3:** a promotion's dates are inclusive and two windows overlap (10/12 to 11/12 and 11/01 to 11/30), so the one that started later is current; the reference answers for 070 and 071 agree. The candidate set is the open validator's own verdict per class, so the validator and the offer cannot drift apart. Product pages state a holder age for only two products (Gold Years, Dark Green), and those limits now gate the open validator too.
- **Slice 4:** the customer's confirmation of the exact class is the plan preview itself (it names each class, and one yes approves it), not a separate question. A deposit transfer into a new account uses the account's id computed by code (the bank hashes user, type and class), so no `{{step_N.field}}` reference is needed, and the open read-back proves the id is right. Asking for the list of what can be opened is done by calling the workflow with an empty class, which hands the list back, because a workflow cannot offer an extra read tool. Tier-1 now treats the accounts kinds as segment kinds (`tier1._preference`).
- **Slice 5:** one destination per closing was too narrow: task 062 splits an account's balance between two new accounts, so a closing carries a list of moves whose amounts must add up exactly to what has to leave. When the balance is short of the fee, the customer is told how much to add and the close is planned on a later turn (a check they deposit is theirs to do).
- **Slice 6, a layer that could not be built as written:** the PRD asked for "an execute-time check in the specialist that refuses a close that would break the eligibility of an open still in the plan". A specialist sees only its own intent, never a sibling's, so it cannot know that an opening is pending. What is built instead, and tested: (1) the planner puts every opening in an earlier wave than every closing, unless the customer said the opposite (`plan.py`); (2) the orchestrator writes that order into each closing's `depends_on`, so a closing is skipped when its opening needs an answer, is refused or fails, and the old account is never closed without the new one (`closes_wait_for_openings`); (3) if the order is ever wrong anyway, the open validator refuses at execute time with the reason from the records (10 days, or "any of your accounts has been closed"). A row for this was added to the routing PRD's edge-case table with a scenario. A refusal that is a normal answer (a Reply that the account cannot be opened) still counts as complete, so the closing goes ahead; that is the customer's explicit second request.
- **Slice 6, tier-1:** in a mixed request the transfer out of an account that is closed later belongs to the closing, not to the opening before it (both kinds may transfer, so the longest-run rule had put it in the opening); `tier1._transfers_belong_to_the_closing` moves it.
- **Slice 7:** the plan the customer approves printed raw ids ("source account id: 03"). Workflows can now write each step's description in the customer's words from their own records ("Move $100.00 from your Green Account (checking) to your Silver Account (savings)"); it is shown in the plan but is not part of what the yes is bound to (the tool and arguments are), and a model's own wording is never used. Credit and debit steps are unchanged. A workflow also sees the request's slots, so a form whose amount differs from the amount the customer gave is refused, not shown for approval. A repeat after a timeout does not move money twice because the session skips a write it has already made in the same order (tested for transfers).
- **Slice 8:** the "eight transfer-only tasks" (004, 008, 012, 014, 032 to 035) are transfers to a human agent (the escalation and speedbump protocol), not money transfers, and none of them verifies a customer, so tier-1 cannot replay them; they are checked on their reference (no domain write, only hand-off tools). The service-without-a-tool requests are handled in the transfer form: it now says where the money goes (`my_own_account`, `wire`, `ach`, `bill_pay`, `another_person_or_bank`, `check_deposit`); anything but the first is a hand-off with a reason written here, except a check deposit, which gets the customer's own steps from document 011 (the agent never deposits a check and the reply carries no tool name). Questions are answered by the general answerer, which must cite documents; a test shows the real index finds the documents for wires, ACH, check availability, overdraft, holds and bill pay in both scopes. Found while testing: `prepare` asked a customer who wanted to deposit a check "which account should the money come from?" before the workflow could say it is not a transfer, so for transfers `prepare` now only fills in the accounts the customer named and the workflow asks.
- **Slice 9:** every adversarial case is a model scripted to do exactly what the lie or the instruction asks, then a look at what changed in the bank (nothing, apart from what was approved): a false "14 days", an account that is not the customer's, a "manager approved" waiver, a balance assumed to be there, an instruction hidden in the customer's words and in a transaction description the model reads, a supervisor's override code, a free-form plan for each of the four kinds that have a workflow, tools outside the job, and a plan changed into another transfer when it runs. The guard refuses an account that is not the customer's before any tool runs, and an approved plan cannot be turned into a different transfer. Three drifts between preview and yes (money spent, an account closed, eligibility lost) stop the write at execute time. No slice-9 case needed a code change: the earlier slices already closed them, which is what this slice is there to prove.
- **Slice 10:** the report is built on the credit report's status, row and judging code. All 19 tasks pass strictly (the database equals tau2's own hash, not only semantically), with no departure from policy and no write outside an approved plan, so `evals/policy_departures.json` has no account entry.
- **Slice 11, prepared and not run:** tasks 057, 060 and 075 can be chosen in the capped live harness (`uv run python -m agent.live --tasks task_057 task_060 task_075`), which prices them at about $0.16 for 14 model calls and stops without `--yes`; no key was read and nothing was spent. It is blocked on Issue #38 (the debit pilot gate), which has not passed: the pilot's last recorded runs were 0 of 12 with gpt-4o-mini and 0 of 4 with gpt-4o. What the probe would measure is the part tier-1 cannot: whether a model chooses the right class from the candidates (075), tells a customer the truth about a false claim (057), and files the forms in the right order (060).
