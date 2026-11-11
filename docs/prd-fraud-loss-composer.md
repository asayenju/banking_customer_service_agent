# PRD: fraud and loss composer (Issue #44)

## Context
`REPORT_FRAUD_OR_LOSS` is routed to `Name.FRAUD` in `agent/dispatcher.py`, but no specialist is registered, so the dispatcher returns FAILED "not implemented yet". Issue #44 has three bullets (one combined preview and one yes; a human transfer only after asking and only when the KB offers no action; tier-1 on 077, 080, 081). There is no policy spike for those tasks, and the "two-step human-transfer protocol" is never defined.

The tau2 gold (tasks 077 to 081) shows what the composer really is: a lost or stolen wallet protocol (doc `bank_accounts_(general)_030`) that crosses debit and credit. It is not an agent with its own tools. Format precedent: `prd-fees-interest-rewards-specialist.md`. Slice 0 adds `docs/fraud-policy-decisions.md`; GitHub issues come from the slices below.

Decisions already made with the project owner:
1. **Scope:** the wallet protocol only (077, 080, 081, plus 078 and 079 as the same freeze, close, reorder shape). Debit-only fraud (082, 084 to 092) and credit-only fraud (036 to 041, 053, 054) stay with their specialists.
2. **Freeze:** contain first. Named ACTIVE debit cards are frozen at once, then one combined preview and one yes cover everything else.
3. **Human transfer:** work first. Transfer on the 4th ask, or when the KB offers no action.
4. **Pilot gate:** offline only until #38 passes. The live probe is the last slice, blocked on #38.
5. **Unnamed cards:** ask which, freeze only what is named.
6. **No-action cases:** say what we cannot do, offer a transfer, transfer only on a yes.
7. **Credit side:** always check credit cards after a debit loss and ask. Replacements and the credit disputes the customer names are planned in the same preview.

## What the tau2 data says (evidence)
- **077** (lost, 2 debit cards, forgets a credit card until asked): freeze both, read the credit cards, replace the credit card (`lost`), read transactions, unfreeze, close both (`lost`), reorder Bluest only (Light Green is Entry tier, so it is not reordered), activate the new card.
- **078** (lost, 3 debit): freeze all three. Light Green has pending transactions, so it is not closed and stays frozen. Light Blue is over its replacement limit and the customer declines the $25 fee, so it is closed with no reorder. Green gets an EXPEDITED reorder with fee 15.
- **079** (stolen, 3 debit): close all three. Green has a pending refund, which the customer accepts going to checking. Light Blue is not reordered. Evergreen gets RUSH at $35 and Green gets STANDARD.
- **080** (stolen, 3 debit + 2 credit): freeze three. The customer finds Green, so it is unfrozen and kept. The other two are closed. Blue is reordered. Gold and Eco credit cards get replacements (`stolen`). Dispute history is read, then three fraud disputes are filed (`cancel_and_reissue`, provisional credit). Activation comes last.
- **081**: 080 plus 4 human requests (2 at the start, 2 at the end). Gold does all the work and then calls `transfer_to_human_agents` with an empty summary.
- **Tool facts:**
  - `close_debit_card_4721` accepts only ACTIVE or PENDING, so a frozen card must be unfrozen first.
  - `order_replacement_credit_card_7291` closes the credit account immediately and charges nothing.
  - `order_debit_card_5739` trusts the fees it is given. It does not check the tier limit or the 48-hour wait.
  - There is no credit freeze, no fraud-report tool, and no password-change tool.

## Challenged assumptions
1. **"One combined preview and one yes" vs containment.** A freeze is reversible and already needs no yes (`decision.NO_YES_NEEDED`). Everything irreversible (close, order, credit replacement, dispute) goes in one preview. Freeze is not folded into it.
2. **"I think I might have lost it" is not a loss confirmation.** 077 opens that way. A close needs the customer to say the loss is definite. Freeze alone is the response to a possible loss.
3. **Freeze then unfreeze then close is gold churn.** Containing first is still right, because 078 ends with a card frozen and not closed. The close step contains the unfreeze (`DebitReplacement` already does this).
4. **A fraud dispute also closes a card** (`debit_dispute.CARD_ACTION`). The composer plans the closes itself and must not let a dispute add a second close. `tier1._run_segment` already works around the double close.
5. **`REPORT_FRAUD_OR_LOSS` has no slots,** so `fingerprint(intent)` cannot bind the details. The approval binds the plan's steps. A change between preview and yes (a card found, a refund appearing, a card closed elsewhere) must stop the step. The guard already re-validates against fresh state.
6. **The 48-hour wait for Entry tier is documented but not enforced** by `DebitReplacement` (only the limit is). Slice 0 must settle the rule from docs 029 and gold 077, 079, 080 before code relies on it.
7. **The 4-asks rule applies only while card protection is open.** Changing it globally would break the routing PRD (REQUEST_HUMAN escalates at once). Outside an open fraud or loss plan, nothing changes.
8. **Docs contradict each other** (026 says close rather than freeze a lost card, gold freezes first; 037 says never say a card is flagged, 025/039/041 say discuss suspicious activity; 041 says offer a security transfer but 090's gold has none). Each is a recorded policy decision with evidence, not a silent choice.
9. **Task 083's gold is invalid** (missing `customer_max_liability_amount`) and stays excluded. **081's empty transfer summary** is a documented departure, because the note is written by code with no ids.
10. **Debit-only 078 and 079 may already count in the debit pilot's 12 usable tasks (#38).** Moving them into the composer's cluster changes #38's count. Slice 0 checks `evals/manifest.json` and records the decision.

## PRD

### Why
- Two kinds own no tool: `REPORT_FRAUD_OR_LOSS` and `GENERAL_QUESTION`. Today a lost-wallet report fails with "not implemented".
- A lost wallet is urgent and spans two products. Docs 030 and 025 require a fixed order (freeze, read credit cards, ask, close, reorder).
- The bank's tools check almost nothing here (fees trusted, no tier limit, no ownership on credit orders), so safety must be code.

### What
**Goals**
- A `FRAUD_SPECIALIST` (`Name.FRAUD`) built as configuration over the existing `Specialist` class. It has no tools of its own and composes the debit and credit step builders.
- A wallet protocol that contains first, then makes one combined preview and takes one yes.
- A defined human-transfer protocol and a no-action reply for fraud reports with no KB action.
- Offline tier-1 evidence on 077 to 081.

**Non-goals**
- Debit-only and credit-only fraud (dispute, PIN, alert tasks), fraud scoring, anything with no tool (password change, credit freeze, "report fraud" filing).
- Live evidence before #38.
- A global change to `REQUEST_HUMAN` routing.

### How
**Shape.** `SpecialistConfig` for kind `REPORT_FRAUD_OR_LOSS`, scope `FRAUD` (already in `knowledge/scopes.py`), validators merged from `debit.VALIDATORS` and `credit.VALIDATORS`, and one `WalletLoss` workflow (`agent/workflows/fraud_wallet.py`). The model reports only what the customer said (strict form). Code decides.

**Facts the model reports:** loss type (lost, stolen, fraud_suspected); whether the loss is definite; cards named (by account, product or last four); cards found; credit cards in the wallet; per card the customer's choice (accept excess fee, delivery, design, shipping address); accepted pending-refund handling; fraudulent credit transactions named; whether the customer asks for a human.

**Phases (one conversation)**
1. **Verify** (existing 2-of-4 gate). An unverified caller gets no action. Failed verification is a hand-off with `account_ownership_dispute`.
2. **Contain.** For named, owned, ACTIVE debit cards, the orchestrator expands the report into one `FREEZE_CARD` intent per card in the urgent wave. They run through the existing direct recipe (`agent/recipes/direct.py`, `NO_YES_NEEDED`). The composer intent depends on them. No card named: ask which cards were in the wallet, from the customer's own cards, and freeze nothing.
3. **Read, in code.** Debit cards and transactions per account, pending transactions and refunds, tier, replacement counts, and the credit cards (`get_credit_card_accounts_by_user`, doc 030).
4. **Ask.** Pending questions, never auto-assumed: any credit card in the wallet? Is the loss definite? Excess fee (decline means close and no reorder, as in 078)? Pending refund handling (079)? A found card (080) is unfrozen and left out.
5. **One combined preview and one yes.** Steps in order:
   - (a) unfreeze then close each definite-loss card that has no pending transaction. A card with pending transactions stays frozen (078).
   - (b) order a debit replacement per closed card whose account allows it, with fees computed from tier tables (`DebitReplacement`).
   - (c) replace each credit card the customer says was in the wallet (`CreditReplacement`, reason `lost` or `stolen`).
   - (d) file the credit disputes the customer names (`CreditDispute`), after the replacement, with dispute history read first.
   Each step is checked against its own specialist's allow-list. The whole plan is dry-run on a deep copy of the database.
6. **Execute** with no model. The guard and validator re-run before each step, and a read-back runs after each write.
7. **Activation** of a new card stays with the debit specialist's activate flow. PIN and CVV go only through `SecretRef`.

**Human-transfer protocol.** A counter of human requests is kept in code per conversation. While a card-protection plan is open (urgent freezes done or a combined plan pending or approved and not finished), a human request is acknowledged, counted and not escalated. Transfer happens (a) when the work is done and the customer still asks, or (b) at the 4th request, or (c) at once when the KB offers no action. The constant `HUMAN_AFTER_ASKS = 4` comes from `policy_header.md`. The summary is written by `agent/handoff.py` (code, no ids, no customer claims). The `reason` is a `TransferReasonLiteral` value.

**No-action reports.** Account takeover, identity theft, lost phone or ID, suspicious logins, `security_hold`, bank-initiated alerts, code 43. Reply with KB guidance only, say what cannot be done here, ask whether the customer wants a person, and transfer with `fraud_or_security_concern` only on a yes. Never reveal scores or that a card is flagged.

**Orchestrator changes.** Containment expansion (new, small module). The human-ask counter and the defer rule in `decision.py`. Nothing added to `NO_YES_NEEDED`; `REPORT_FRAUD_OR_LOSS` keeps needing a yes. Fix `guard.py` `REPORT_FRAUD_OR_LOSS` allow-list gaps (the credit replacement tool is present; confirm closures, activation and dispute pairing).

**Reuse (do not rebuild):** `Specialist`, `SpecialistConfig`, `Workflow` and `Plan/NeedInfo/Refuse/Reply/HandOff` (`workflows/base.py`), `DebitReplacement` (`workflows/debit_replacement.py`), `CreditReplacement`, `CreditDispute`, `DebitDispute` card-action table, validators in `specialists/debit.py` and `credit_rules.py`, `agent/cards.py` (`owned_cards`, `resolve`, `describe`), `agent/handoff.py`, `SubAgentDispatcher` (`Name.FRAUD` and `FIXED[REPORT_FRAUD_OR_LOSS]` already exist), `tier1.py`, `credit_report.py` as the report template, `evals/policy_departures.json`.

### Constraints and risks
- Irreversible actions never run without the customer's yes to the exact steps. A freeze is the only action before the yes.
- The model cannot add a step, change a fee, pick another customer's card or skip a validator (obey-the-lies tests).
- Credit provisional-credit caps for 039 to 041 are inferred from gold, not documented. The composer inherits that and flags it.
- Debit pilot is 0/12 live (gpt-4o-mini) and 0/4 (gpt-4o). The composer is proven offline only. Its live probe stays blocked on #38.
- Error messages are recorded by exception type only. Audit never holds PIN, CVV or the new-card secrets.
- A frozen or closed state changed between preview and yes stops that step and says why.

### Edge cases (each gets a test)
Customer names no cards; names a card that is not theirs; unverified caller; card already FROZEN, CLOSED or PENDING; "I think I lost it" (freeze only); loss confirmed later; card found after the freeze (080); pending transactions (078); pending refund accepted or refused (079); replacement limit reached and fee declined (078); Entry-tier wait (077, 079, 080); credit card forgotten until asked (077); a credit replacement already pending; two products in one id list; customer demands a human 4 times (081); human asked once and nothing open; fraud dispute plus close double-close; customer asks to waive a fee or change a fee amount; instruction hidden in a merchant name; customer changes the card list between preview and yes; account takeover with no card step.

### Measurement
`uv run python -m agent.fraud_report` replays 077 to 081 through the real specialist and compares with tau2's database (strict pass or documented departure). Safety counts: writes outside an approved plan must be 0; irreversible steps without a yes must be 0.

### Decisions
- **Decided by the owner (this session):** the seven points above.
- **Adopted from the design review (open to veto):** human-ask rule limited to an open protection plan; freeze is not part of the combined preview; `HUMAN_AFTER_ASKS = 4` as a data constant; 078 and 079 move to the composer if Slice 0 confirms no loss to the debit pilot count.
- **Pending (settle in Slice 0):** Entry-tier 48-hour wait rule; pending refund versus pending transaction on close; 081's empty transfer summary as a departure; 078 and 079 manifest ownership.

## Issues and acceptance criteria (GitHub, label `specialist-agents`)
Each issue body: "Part of `docs/prd-fraud-loss-composer.md`" and the criteria as `- [ ]` boxes. Live criteria are marked blocked on #38.

**Slice 0: Policy spike and decisions** (blocks all)
- `docs/fraud-policy-decisions.md` states every rule with its document or database evidence: freeze, close, reorder tier tables, wait, credit replacement, disputes after replacement, pending transaction or refund, transfer reasons.
- Statements about the bank's tools are tests in `tests/agent/test_fraud_policy.py`.
- A derivation table for 077 to 081 (inputs, computed, gold, match); departures listed in `evals/policy_departures.json` with a reason.
- Pending decisions above settled; manifest ownership of 078 and 079 recorded.

**Slice 1: Records and wiring**
- `FRAUD_SPECIALIST` is registered under `Name.FRAUD`; `unmapped_kinds()` is empty; the guard allow-list for `REPORT_FRAUD_OR_LOSS` is complete and tested (`test_guard.py` still passes).
- A dispatcher with no fraud specialist still returns FAILED; `fraud` is a `manifest` cluster and a `tier1._preference` entry.

**Slice 2: Containment**
- A named, owned, ACTIVE debit card is frozen in the urgent wave before the composer plans; unnamed cards cause one question and no freeze.
- A card not owned, already frozen or closed is skipped with a plain reason; an unverified caller gets nothing.
- "I think I lost it" freezes only; a definite loss is a separate answer that unlocks the close.
- A card found later is unfrozen and removed from closes (080).

**Slice 3: Debit half of the wallet workflow**
- Closes (unfreeze then close) for definite-loss cards; a card with pending transactions stays frozen (078); a pending refund needs the customer's accept (079).
- Reorders per account use tier tables; the limit and fee come from code; a declined excess fee closes with no reorder (078); the Entry wait follows the Slice 0 rule.
- Matches the debit writes of 077 to 079 in tier-1; no card is closed twice.

**Slice 4: Credit half and the combined preview**
- After a debit loss the specialist always reads the credit cards and asks; nothing is replaced unasked (077).
- One preview and one yes cover debit and credit steps in the order (a) to (d); each step is checked against its own specialist's allow-list; a step outside the plan is refused.
- Credit disputes the customer names are filed after the replacement with dispute history read first; the dry run shows the final state (080).
- Tier-1 matches 077 and 080 writes (documented departures only).

**Slice 5: Human-transfer protocol**
- Human requests are counted in code; inside an open protection plan they are acknowledged, not escalated; transfer at completion, at the 4th ask, or at once when the KB offers no action.
- Outside an open plan, `REQUEST_HUMAN` escalates at once, unchanged (routing tests pass).
- The note comes from `agent/handoff.py` with new `describe` cases for freeze, close, order, activate, dispute; no ids or contact details; reason is a valid `TransferReasonLiteral`.
- Task 081 passes tier-1 or departs only on the summary.

**Slice 6: No-action fraud reports**
- Account takeover, identity theft, lost phone or ID, `security_hold`, bank-initiated alert and code 43 get KB guidance, an offer, and a transfer only on a yes.
- Never says a card is flagged and never states a fraud score; no invented action is offered.

**Slice 7: Traps and adversarial cases**
- A model scripted to obey a false fee, a waived fee, another customer's card, a closing before a definite loss, a free-form plan, or an instruction hidden in a merchant name cannot make any write outside an approved plan.
- A card found, a refund appearing, or a card closed between preview and yes stops the step with a reason.
- Audit and handoff contain no PIN, CVV or ids.

**Slice 8: Offline evidence and report**
- `agent/fraud_report.py` (`fraud_task_ids`, `run_fraud_report`, `render`) replays 077 to 081 with strict passes or documented departures only, and zero writes outside an approved plan.
- `evals/policy_departures.json` has a reason and tables per departure; a test fails if an unlisted table differs.

**Slice 9: Docs**
- CLAUDE.md gets a "Fraud and loss composer" section and the PRD gets "What changed while building it".
- `tests/agent/test_fraud_*.py` and `mypy --strict src/agent src/orchestrator` pass.

**Slice 10: Capped live probe** (blocked on #38)
- `uv run python -m agent.live --tasks 077,080,081` with a printed estimate, `--yes`, a dollar cap, and cassettes; reports semantic pass and guard bypasses. Go or no-go follows the #38 rule.

| Slice | Depends on |
|---|---|
| 0 | none |
| 1 | 0 |
| 2 | 1 |
| 3 | 1, 2 |
| 4 | 3 |
| 5 | 1 |
| 6 | 1 |
| 7 | 3, 4, 5, 6 |
| 8 | 3, 4, 5 |
| 9 | 8 |
| 10 | 8, #38 |

## GitHub issues
| Slice | Issue |
|---|---|
| 0 | #78 |
| 1 | #79 |
| 2 | #80 |
| 3 | #81 |
| 4 | #82 |
| 5 | #83 |
| 6 | #84 |
| 7 | #85 |
| 8 | #86 |
| 9 | #87 |
| 10 | #88 |

## Verification
- `cd backend && uv run pytest` (offline), `uv run mypy --strict src/banking_tools src/orchestrator src/agent`, `uv run python -m agent.fraud_report`. No live run before #38.

## What changed from this PRD while building it

- **Containment is not an urgent-wave expansion.** The plan said the orchestrator would expand a report into
  `FREEZE_CARD` intents that the composer depends on. A preview runs before the wave executes, so the composer
  would have planned against ACTIVE cards and then met FROZEN ones at the yes (the close tool refuses a frozen
  card). Instead `Orchestrator._contain` freezes the named cards through the same direct recipe just before the
  preview, so the plan already contains `unfreeze` then `close`. The synthetic freeze intents are reported in the
  reply and the outcomes.
- **Which cards are named is read from the decomposer's details** (`card_id`, `card_ids`, `cards`, `account_name`,
  and similar) in `agent/containment.py`. The decomposer prompt was not changed and no live evaluation was run, so
  how often a real model fills those details is **unmeasured**; when it does not, the composer asks which cards.
- **The composer ends only with `finish`, `ask_customer` or `start_wallet_loss`.** `propose_plan` and `escalate`
  were removed: a free-form plan is not a way to end a run, and a hand-over is the customer's yes, never the
  model's. A plain answer is checked by `fraud.answer_check` (no "flagged", no score, no invented action, a person
  offered).
- **Human requests:** `Action.DEFER` was added and `HUMAN_AFTER_ASKS = 4` lives in `orchestrator/decision.py`.
  The note and the transfer reason are built (`agent/handoff.py`) but no code calls `transfer_to_human_agents`
  yet: an escalation is still a reply, as for every other kind. Task 081's replay does its transfer directly.
- **The guard's allow-list for the kind needed no change.** It already held every tool the composer plans.
- **A fraud-only close validator** re-checks pending purchases just before the step runs. A pending refund is not
  re-checked (a refund to a closed card is credited to checking, doc 025).
- **The Entry wait applies to Light Blue only,** the one level the tier table calls Entry. A reorder for a level
  with no tier is a hand-off, as before.
- **No new manifest cluster and no live harness wiring.** The replay cuts the gold with `tier1.wallet_items`.
  Tasks 078 and 079 stay in the debit pilot and are replayed here as well.
- **Evidence:** `python -m agent.fraud_report` gives 5 strict passes and no departure. The facts the scripted model
  hands the composer are derived from the gold, so this proves the plumbing and the rules, not model behaviour.
- **Not done:** Slice 10, the capped live probe (blocked on #38), and a decomposer prompt change for card details.

## What the first live run changed

One capped live conversation (the real decomposer, Jev scorer and gpt-4o-mini, task 078's customer) found problems no scripted test could, and they were fixed:

- **Containment read nothing.** The model returned the report with empty or partial `card_ids`. The orchestrator now keeps the customer's own words as the stable detail `said` on a report intent, and containment finds the named cards in them (`agent.containment.named_in_text`: account names, longest first, and last four digits). The specialist is told to list every card named there.
- **"Yes" broke the decomposer.** The model confirmed the open request and also re-listed it (as a new intent, with a dependency on the open one), which made an invalid decomposition thrown away three times. A bare yes, a new intent of the kind being confirmed, and a dependency on an open intent are now handled in `llm_decomposer.py`; its prompt says where a report's cards go and that a short reply belongs to an open intent.
- **A yes had nowhere to land.** A route's question ("are you sure?") was answered only by a slot, never by a confirmation. A single pending route question now takes a yes as its answer. The not-definite branch of the workflow asks that question instead of ending the report, and the customer's yes counts in code even if the model reports "not definite".
- **A short reply was scored alone.** Jev rated "yes, I am sure" as an unclear request. A continued request of the same kind keeps the kind and scope confidence it earned (`Orchestrator._continued`); nothing is raised above an earlier score.
- **Result:** the conversation freezes all three named cards, previews the right plan (Light Blue closed with its reorder deferred 48 hours, Green closed and reordered, Light Green held frozen for its pending purchases), and on the yes closes, reorders and holds exactly those, with no write outside the plan.
- **Still true:** one conversation, one customer, one model. How often the model fills the form correctly across many runs is unmeasured, and the debit pilot's live pass rate (#38) is still 0 of 12. The general answerer answered a policy question with a citation and escalated when the notes did not answer.

