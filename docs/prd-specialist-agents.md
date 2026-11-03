# PRD: Specialist agents on a faithful tau2 session

## Why
The orchestrator turns a customer message into intents and decides run, ask or escalate, but nothing executes what it routes. Its two runner slots (`Route.DIRECT`, `Route.SUB_AGENT`) are filled with mocks in every test. Fourteen intent kinds need policy reasoning and have no implementation, six need fixed tool recipes and have none, and two (`REPORT_FRAUD_OR_LOSS`, `GENERAL_QUESTION`) own no tool at all.

Exploration of the repo and of tau2-bench found that the plan "build the specialists on what exists" does not hold:
- **The specialists are not defined.** The repo has a routing table (6 direct, 14 sub-agent, 2 hand-off, `orchestrator/routes.py`) and a tool-to-kind map (`taxonomy.py`), and nothing else. Which agents exist is a design decision, made here.
- **Our tool emulation is not scoring-grade.** `BankingSession` calls tau2's tools directly. It has no unlock gate, writes none of the `agent_discoverable_tools` audit rows tau2 writes, stamps audit entries with wall-clock time, and cannot load the per-task DB overlay that 35 of the 97 tasks use. tau2 decides success by comparing database hashes, so a run on our session could never match tau2's gold.
- **The runner contract cannot carry a real agent.** `RouteRunner.run(intent)` gets no caller identity (`verified_user_id` stays in `AuthState`), cannot ask the customer a follow-up, and a PARTIAL result drops the intent after committed writes. The route timeout is 30 s. `LLMClient` returns strict JSON only, with no tool calling.
- **Nearly every task needs policy knowledge.** All 97 tasks list required documents from a 698-document knowledge base. There is no retrieval, and `rank_bm25` is not a backend dependency.
- **One yes per intent does not fit multi-write workflows.** Closing a card is "log the reason, offer a retention credit, close". A debit dispute is "file, close, reorder". Today a yes is bound to one intent's kind and slots, before the agent knows which branch applies.

## What
Specialist agents behind the orchestrator, running against tau2's own database through a session layer faithful enough to be compared with tau2's gold, with every state-changing step approved once as a previewed plan and enforced by code.

### Goals
- A faithful session: per-task DB overlay, tau2's own unlock and audit behaviour, the fixed clock, verification logging, a state comparison that explains any mismatch.
- A runner contract that carries identity, can ask the customer questions (`NEEDS_INFO`), can ask for approval of a plan (`NEEDS_CONFIRMATION`), and keeps completed work when it stops partway (`PARTIAL`).
- Tool calling in the LLM client, a bounded agent loop, and a code guard that blocks any write outside the approved plan, the kind's tool list, or the verified customer's records.
- Policy knowledge by BM25 search with a per-specialist document scope.
- Five specialists, built as configuration over one `Specialist` class: Debit cards (pilot), Credit cards, Accounts and money movement, Fees/interest/rewards, Fraud and loss.
- The six DIRECT kinds as deterministic recipes with no LLM, and `GENERAL_QUESTION` as a read-only knowledge answerer.
- Evidence a specialist works: offline per-intent DB-state tests, and a small capped live run.

### Non-goals
- A new database, or re-implementing tau2's tools or DB. tau2 is used as is and is never edited.
- Dense (embedding) retrieval, voice, other languages.
- Acting for unverified customers, or any override flow ("supervisor code", task 005).
- Customer-side tools (apply for a card, referral, cash-back dispute, check deposit). The 25 tasks that depend on them (or on reads only) are measured only in the optional staged simulation.
- Running the tau2 banking tasks broadly against a live model. Live runs are small, capped and opt-in (see Cost controls).
- Undoing completed writes after a later failure.

### Definitions
- **Specialist**: configuration over one `Specialist` class: system prompt, per-kind tool allow-list, per-tool validators, `DocScope`, slot schema.
- **Plan preview**: the ordered list of writes (tool, arguments, reversibility) a specialist proposes. Rendered by code, never by the model. The customer's yes binds to its fingerprint.
- **Plan mode / execute mode**: in plan mode reads run and writes are recorded, not executed. In execute mode the approved steps run in code with no LLM.
- **Tier-1 test**: offline; scripted slots and tool calls through the real session, guard and comparator. It proves the plumbing, not the model.
- **Tier-2 run**: opt-in live run with a real model. It tests inference.
- **Semantic pass / strict pass**: the normalised domain delta matches / the full tau2 DB hash matches.

## How

### Specialist division
| Specialist | Kinds | Owns (tools) |
|---|---|---|
| Debit cards (pilot) | ACTIVATE_CARD, CHANGE_OR_RESET_PIN, CARD_CONTROLS, debit CLOSE_CARD, debit ORDER_REPLACEMENT_CARD, debit DISPUTE_TRANSACTION | freeze, unfreeze, close, order, activate x3, PIN change/reset, fraud alert, recurring block, temporary limit, debit dispute |
| Credit cards | credit CLOSE_CARD, credit ORDER_REPLACEMENT_CARD, CREDIT_LIMIT_REQUEST, credit DISPUTE_TRANSACTION | credit dispute, replacement, closure reason, close, statement credit, account flag, limit submit/approve/deny |
| Accounts and money | TRANSFER_FUNDS, PAY_CREDIT_CARD, OPEN_ACCOUNT, CLOSE_ACCOUNT | open, close, transfer, pay card |
| Fees, interest, rewards | FEE_OR_INTEREST_ADJUSTMENT | checking/savings credit, interest discrepancy report, rewards update |
| Fraud and loss | REPORT_FRAUD_OR_LOSS | none of its own: a composer over the debit and credit step builders; the two-step human-transfer protocol |

DISPUTE_TRANSACTION, CLOSE_CARD and ORDER_REPLACEMENT_CARD span debit and credit. The dispatcher picks by the product of the referenced id (a code lookup); an ambiguous id becomes `NEEDS_INFO`. `Route.SUB_AGENT` stays a single slot filled by `SubAgentDispatcher`; an unmapped kind returns FAILED "not implemented" so the rollout is safe phase by phase.

### Session fidelity (`banking_tools/`)
- `BankingSession.from_task(task)` applies `initial_state.initialization_data.agent_data` the way tau2's `update_db` does, on a copy of the cached base DB.
- Discoverable calls go through tau2's own `unlock_discoverable_agent_tool` and `call_discoverable_agent_tool` (JSON-string arguments, `parse_int=float`), so audit rows are tau2's rows. The old direct mode stays for existing tests.
- Injected clock, default tau2's `get_now()` (2025-11-14 03:40 EST). No wall time.
- `record_verification` calls `log_verification` on the first successful 2-of-4, with the fixed time.
- `ScopedSession(intent_id, allowed_tools, user_id)` tags audit entries; per-resource async write leases are keyed by ids in tool arguments (the planner sees only slots).
- Calls stay inline: they are in-memory and take microseconds. No `to_thread`, no `await` while holding the session lock.

### Contract (`orchestrator/`, `agent/context.py`)
- `RouteRunner.run(intent, ctx)`; `RouteContext` carries `intent_id`, `verified_user_id` (from `AuthState` only), `session`, `mode`, `approved`, `progress`, `deadline` and a secret vault.
- `OutcomeStatus` gains `NEEDS_INFO` and `NEEDS_CONFIRMATION`; `RouteOutcome` gains `questions`, `proposal` and `completed`.
- Per-route timeouts replace the single 30 s: DIRECT 10 s, SUB_AGENT 90 s, HAND_OFF 10 s.
- After any non-DONE outcome the executor reconciles with the session audit: a committed write turns FAILED into PARTIAL with text generated from the audit, and the intent is kept so a retry resumes by step id.
- `NEEDS_INFO` uses its own `_runner_asks` counter, capped at `MAX_ASKS`, and is not reset by the fingerprint-change rule (otherwise ask, answer, ask would loop forever). The score is dropped so Jev re-scores. The runner returns slot names, never values; slots only enter from the decomposer reading the customer's next message. A per-kind slot schema keeps decomposer, specialist and guard on one vocabulary.

### Plan preview and consent (`agent/plan_preview.py`, `decision.py`)
- `PlanStep(tool, args | StepRef, reversibility from the catalog)`, `PlanPreview(intent_fp, steps, plan_fp, rendered)`, new `Action.PREVIEW`.
- Flow: PREVIEW (plan mode) -> `NEEDS_CONFIRMATION` -> the customer's yes binds `plan_fp` and `intent_fp` -> RUN (execute mode: approved steps in code, guard re-validates against fresh DB state, no LLM).
- A changed slot makes the proposal stale and re-previews. A plan with only reversible steps at kind confidence >= 0.95 asks nothing. "Needs a yes" becomes a plan-level fact for sub-agent kinds, which fixes `CARD_CONTROLS` mixing a reversible and an irreversible tool. `NO_YES_NEEDED` stays the default for DIRECT kinds.
- Secrets enter a plan only as `SecretRef`; `plan_fp` contains none.
- Rolling previews: a plan whose later writes depend on reads the customer has not seen (task 078) gets another preview round.

### Tool-calling client and agent loop (`agent/llm_tools.py`, `loop.py`)
- A separate `ToolCallingLLMClient` Protocol (the decomposer and Jev are untouched), `LiteLLMClient.chat(tools=..., parallel_tool_calls=False)` inside the pinned litellm range, and `FakeToolLLM`. Schemas come from each tool's pydantic args model, strict-mode post-processed; `user_id` is removed from every schema and injected by code; the real docstring is the description (KB docs use different parameter names than the real signatures). New `AGENT_MODEL` setting, default `openai/gpt-4o`.
- `Budget(max_steps=8, max_tool_calls=12, max_writes, deadline, max_input_tokens)`. No write starts inside a time margin; write steps run under `asyncio.shield`. Terminal pseudo-tools: `propose_plan`, `ask_customer`, `finish`, `escalate`. Failures are recorded by exception type only.

### Guard (`agent/guard.py`): all code, no model
1. Per-kind write allow-list, hand-written (it cannot be derived from `taxonomy`; debit replacement needs `close_debit_card` and `order_debit_card`).
2. Identity: `user_id` forced to the verified user; every account, card, credit-card and transaction id must resolve to a record that user owns; third-party destinations blocked (task 016).
3. Plan membership: in execute mode the tool and canonical arguments must equal an approved step.
4. Per-tool validators recompute eligibility from fresh state: card closure (zero balance, nothing pending, age >= 60 days), savings opening (checking account >= 14 days, exact class string), credit-limit tiers, transfer funds, interest credit = balance x APY gap / 12.
5. Slot equality only for id-like and amount-like keys; the 15-argument debit dispute is grounded in the DB instead (the amount must equal the transaction's).
6. Post-write read-back; the interest discrepancy report cannot be verified and must never be described as a credit.
7. Post-run audit check: an extra write fails the outcome and offers a human. This is detection; enforcement is pre-call.

### Knowledge (`knowledge/`)
`rank-bm25` is added to the backend dependencies. One deterministic BM25 index per process over the 698 documents. `search(query, scope, k=5)` returns document id and capped text, wrapped as untrusted. Scopes are built by topic, not as the union of the tasks' `required_documents`, so they do not overfit the 97 tasks. Sentences in documents that tell the agent to unlock or call a tool are neutralised; when a document and a guard disagree, the code wins.

### Measurement
- Expected state: replay the task's gold agent actions through our session on the per-task overlay. Parity gate: replay the same gold through tau2's `Environment` and require equal `get_hash()`.
- Semantic comparison: row-level `(table, pk) -> changed fields` over domain tables. Money to integer cents (`"$1,234.56"`, `"1234.56"` and float JSON all normalise), rewards to int, dates to `date`, lists as pk-keyed sets, order ignored (ordering rules such as verify-first, approve-before-dispute and move-funds-before-close are asserted separately). Ids compared literally, with a reported role-mapping fallback.
- `verification_history` is asserted separately (exactly one row per verified conversation) and excluded from the delta. For `agent_discoverable_tools`, only the set of mutating tool names is compared, with an empty read allowlist on both sides.
- Reported at two levels: strict (tau2's hash) and semantic. A strict-only mismatch is listed with its table, so a fidelity bug is distinguishable from an agent mistake.

Task classification, measured on `tasks.json` (97 tasks):
| Group | Tasks | Count |
|---|---|---|
| Wholly debit | 078 079 082 083 084 085 086 087 088 089 090 091 092 | 13 |
| Wholly credit | 031 036 037 038 039 040 041 044 047 049 050 052 053 054 | 14 |
| Wholly accounts/money | 055-071, 075, 076 | 19 |
| Wholly fees/interest/rewards | 026 028 072 073 074 093-097 | 10 |
| Profile only | 005 | 1 |
| Spanning accounts + credit | 043 045 048 051 | 4 |
| Spanning credit + debit | 077 080 081 | 3 |
| Transfer-only (ACTION basis) | 004 008 012 014 032 033 034 035 | 8 |
| Customer-side tools or reads only | the rest | 25 |

Gradeable DB-state set: 62 tasks (57 wholly in one cluster plus 7 spanning, minus 083 and 005). Excluded: 005 (anomalous gold: supervisor bypass code); 083 (its gold dispute calls omit `customer_max_liability_amount`, which the bank's tool requires, so no typed call can reproduce them); the 8 ACTION-only transfer tasks (assert a hand-off, the right transfer tool, and no domain write); 083 (ACTION basis: compare the four dispute calls); the 25 customer-side or reads-only tasks. Trap tasks (052 049 008 014 035 099 016) are negative assertions ("no unsafe write").

### Cost controls
- Default `pytest` and CI make zero model calls. The parity gate replays gold with no LLM. Tier-1 tests use `FakeToolLLM`.
- Live runs: explicit `--tasks` (ids or a specialist name; there is no "all"), `--max-tasks` (default 5), `--trials` (default 1), `--max-cost-usd`, and a printed estimate that needs `--yes`.
- Cassette record and replay (keyed by task, model, prompt hash, secrets scrubbed, git-ignored): a failing task is replayed for free.
- The pilot go/no-go is one live run of the 12 usable wholly-debit tasks (13 less task 083), one trial. Each later specialist gets 3 representative live tasks. No scheduled live job. Staged simulation is optional and capped at about 10 tasks.

## Constraints and risks
Edge cases, ranked; each is an acceptance criterion in an issue below.
1. **Timeout or cancel after a committed write** leads to a duplicate dispute or transfer on retry. Fix: audit reconciliation to PARTIAL, step ids, resume.
2. **Wrong-user, third-party or ambiguous resources** (task 016; four debit cards). Ownership guard; NEEDS_INFO with a code-rendered list (last 4 digits and status); the model never picks.
3. **PIN/CVV leaks.** Redacted to `SecretRef` before the decomposer, substituted at call time, scrubbed from results, traces and cassettes; `cvv` and full card numbers are dropped from read results by column allow-list; sentinel-digit tests over every LLM call, trace and outcome text.
4. **Prompt injection** through `intent.summary`, merchant text and KB documents (task 005). The guard is code, so injection can only choose inside the allowed set. Tested with a fake model scripted to obey the injection.
5. **Silent no-ops and misleading successes** (`submit_interest_discrepancy_report_7294` stores nothing; `approve_credit_limit_increase` silently errors on a pending dispute, replacement or past-due balance). Validators pre-check; read-back confirms.
6. **Hashed ids of new accounts across partial plans.** `StepRef` from stored tool results; resume from `ctx.progress`.
7. **Concurrency.** Stateless specialists, per-resource leases, a lint test against `await` under the session lock, execute-time validators for cross-specialist conflicts.
8. **Eligibility drift between preview and yes.** The guard re-validates in execute mode.
9. **Data-dependent plans** (task 078). Rolling previews.
10. **Decomposer and Jev interplay.** Missing slots become NEEDS_INFO in the decomposer's vocabulary; Jev re-scores after the answer.
11. **Fairness of the DB comparison.** tau2's gold is one reference trajectory; read-log rows, repeated freeze/unfreeze, per-tool money formats and user-side rows are handled by the normaliser and flags above.
12. **Cost, latency, determinism.** Per-run agent budgets, fixed clock, temperature 0, recorded model id, capped live runs.

Known limits: a rolling preview can still surprise the customer between rounds; some writes cannot be verified by state; tier-1 passing says nothing about model quality; the post-run audit check detects an extra write but cannot undo it; the 25 customer-side tasks are not covered until the staged simulation exists.

## Issues and acceptance criteria
"Live" criteria need API keys, are opt-in and capped. Everything else runs offline. Issue ids are `Issue N.M`; GitHub numbers are in the table at the end.

### Phase 0: fidelity and harness
**Issue 0.1: Per-task overlay and injected clock.** `BankingSession.from_task`, `task_state.py`, a `clock` parameter defaulting to tau2's `get_now()`.
- Every task with an overlay builds a session whose `db_hash()` equals tau2's `Environment` after `set_state`.
- Audit timestamps equal the fixed tau2 time and contain no wall time.
- `test_session.py` passes unchanged. Depends on: none.

**Issue 0.2: tau2-fidelity dispatch and verification logging.** Discoverable calls go through tau2's unlock and `call_discoverable_agent_tool`; `record_verification`.
- A mutating discoverable call adds exactly one `agent_discoverable_tools` row, identical to tau2's.
- With an empty read allowlist, reads add no row.
- Duplicate identical writes are rejected as in tau2.
- PIN and CVV never appear in audit entries. Depends on: 0.1.

**Issue 0.3: State delta and normaliser.** `statehash.py`.
- Every money format normalises, including no-`$` and float JSON.
- Flags ignore `verification_history` and read-log rows.
- Strict and semantic comparators agree on identical states. Depends on: 0.1.

**Issue 0.4: Gold replay parity gate.**
- Strict hashes are equal for all 62 gradeable tasks, or each mismatch is listed in a committed, justified exclusion file.
- A per-task report names the table of each mismatch. Depends on: 0.2, 0.3.

**Issue 0.5: Task manifest.** `evals/manifest.json`.
- Counts equal the Measurement table (13/14/19/10/1, 7 spanning, 8 transfer-only, 25 customer-side or reads-only).
- Every exclusion carries a reason.
- A test fails if `tasks.json` changes the counts. Depends on: none.

### Phase 1: contract and framework
**Issue 1.1: RouteContext, runner contract, per-route timeouts, audit reconciliation.**
- `verified_user_id` comes from `AuthState` only.
- A runner cancelled after a committed write yields PARTIAL with text from the audit.
- A 40 s fake SUB_AGENT runner is not cut off. Depends on: 0.2.

**Issue 1.2: NEEDS_INFO and orchestrator re-entry.**
- A fourth runner ask escalates, and a slot change does not reset the runner counter.
- Answering one intent does not reset another's counter.
- A double-submitted message id does not re-ask.
- A PIN in an answer never appears in a pending-question store or trace. Depends on: 1.1.

**Issue 1.3: PARTIAL keeps progress; resume.**
- A retry skips committed steps and does not re-write them.
- Dependents are still skipped and the human is offered.
- `test_executor.py` is updated to the new semantics. Depends on: 1.1.

**Issue 1.4: Plan preview consent.**
- A yes runs only the approved steps; a changed slot makes the proposal stale and re-previews.
- A reversible-only plan at kind confidence >= 0.95 asks nothing, and at 0.94 asks.
- `plan_fp` is stable under argument-key order and contains no secret.
- A "no" drops the proposal and leaves no write.
- The `NO_YES_NEEDED` catalog test passes, plus a plan-level check. Depends on: 1.1, 1.2.

**Issue 1.5: Tool-calling client, `FakeToolLLM`, `AGENT_MODEL`.**
- The fake records offered tools and fails on an unscripted call.
- Decomposer and Jev tests pass unchanged.
- Live: one valid tool-call round trip. Depends on: none.

**Issue 1.6: ScopedToolBox, schemas, SecretVault.**
- Every catalog tool yields a strict-valid schema.
- No schema exposes `user_id`.
- PIN and CVV sentinels never reach `FakeToolLLM.calls`. Depends on: 0.2, 1.5.

**Issue 1.7: Agent loop with budgets.**
- Exceeding each budget returns a structured result with a code-only reason.
- No write starts inside the deadline margin.
- A malformed tool call is returned as a counted tool error. Depends on: 1.5, 1.6.

**Issue 1.8: Boundary guard.**
- Every catalog write is in at least one kind's allow-list, and every allow-listed name exists.
- Another user's ids are blocked for all kinds.
- A fake model scripted to obey an injection (task 005) cannot cause an unapproved write.
- An extra audit entry after a run fails the outcome. Depends on: 1.4, 1.6.

**Issue 1.9: Knowledge index and scoped search.**
- The index is deterministic and built once per process.
- Out-of-scope documents are never returned.
- Cluster `required_documents` recall@5 is reported (not asserted).
- Tool-directive text in documents is neutralised. Depends on: none.

**Issue 1.10: Specialist base and dispatcher.**
- Every SUB_AGENT kind has a specialist or returns "not implemented", and a new unmapped kind fails a test.
- Debit versus credit is resolved from the referenced id; an ambiguous id gives NEEDS_INFO. Depends on: 1.1 to 1.9.

### Phase 2: debit-card pilot
**Issue 2.1: Debit lifecycle.** Freeze/unfreeze, close, order, activate x3.
- Validators reject ineligible fixtures and accept eligible ones.
- Unfreeze-before-close/order is asserted.
- Multi-card ambiguity gives NEEDS_INFO. Depends on: 1.10.

**Issue 2.2: PIN, limits, fraud alert, recurring block.**
- Sentinel PIN is absent from all LLM calls, traces and outcome texts.
- `clear_debit_card_fraud_alert` is previewed as irreversible and needs a yes. Depends on: 2.1.

**Issue 2.3: Debit disputes.** The 15-argument dispute.
- The dispute amount must equal the transaction's; a mismatch is blocked.
- Tier-1 tests for the 082, 084, 085, 086 segments pass. Depends on: 2.1.

**Issue 2.4: Pilot gate and report.**
- Tier-1 passes for the 12 usable pilot tasks (078 079 082 084 to 092); 083 is excluded because its own gold omits a required argument.
- The report lists semantic and strict pass, steps, tokens and latency per task.
- Live (opt-in, capped): one trial of the 12 usable tasks with the default model, estimate confirmed first, cassettes recorded. The gpt-4o-mini comparison runs only on failed tasks or on request.
- Go/no-go: all tier-1 green, live semantic pass@1 >= 80%, zero guard bypasses. Depends on: 2.1 to 2.3, 0.4, 0.5, 2.5.

**Issue 2.5: Live-run cost controls.**
- The live harness refuses to run without an explicit `--tasks`, enforces `--max-tasks` and `--max-cost-usd`, prints an estimate and needs `--yes`.
- Cassette record/replay makes a second run of the same task free and deterministic.
- A test asserts that a default `pytest` run constructs no live client. Depends on: 1.5.

### Phase 3: direct recipes and questions
**Issue 3.1: DIRECT recipes (6 kinds).**
- One candidate card runs; several candidates give NEEDS_INFO.
- Balance and history return only the verified user's data.
- `change_user_email` keeps the reversible label.
- The module imports no LLM client. Depends on: 1.1, 1.2.

**Issue 3.2: GENERAL_QUESTION answerer.**
- The toolbox has no write tools (invariant test).
- Answers cite document ids; an out-of-scope question hands off.
- Live: a 5-question spot check, capped. Depends on: 1.9, 1.10.

### Phase 4: remaining specialists (after the pilot gate)
**Issue 4.1: Credit cards.**
- Closure is blocked unless the balance is zero, nothing is pending and the account is at least 60 days old.
- Limit approval is blocked by a pending dispute, replacement or past-due balance, with a read-back confirming the effect.
- Tier-1 (offline) on the 14 wholly-credit tasks minus traps; live on 3 representative tasks only. Depends on: 2.4.

**Issue 4.2: Accounts and money movement.**
- Savings opening checks the 14-day rule and exact class strings.
- Closing requires funds moved first, and later steps use `StepRef` ids.
- A third-party destination is blocked.
- Tier-1 on the 19 tasks. Depends on: 2.4.

**Issue 4.3: Fees, interest, rewards.**
- Interest credit equals balance x APY gap / 12, recomputed in code.
- The discrepancy-report reply never claims a credit.
- Tier-1 on the 10 tasks. Depends on: 2.4.

**Issue 4.4: Fraud and loss composer.**
- One combined preview and one yes; each step is checked against its own specialist's allow-list.
- A human transfer happens only after asking, and only when the KB offers no action.
- Tier-1 on 077, 080, 081. Depends on: 4.1, 2.4.

### Phase 5: integration
**Issue 5.1: Wire dispatcher and DIRECT runner; end-to-end scenarios with fakes.**
- A mixed message (freeze, dispute, balance question) runs urgent first and asks one combined confirmation.
- A timeout after a committed write reports PARTIAL and a retry does not duplicate.
- Double-submit and concurrent waves give a deterministic DB hash. Depends on: 2.4, 3.1, 3.2, 4.1 to 4.4.

**Issue 5.2 (optional, deferred): Staged simulation harness.**
- Runs a chosen sample of at most 10 tasks (never all 25), scored on the final DB, behind the same cost controls.
- Produces a stage report. Depends on: 5.1, 0.4, 2.5.

**Issue 5.3: Cost and latency report, regression CI.**
- Offline CI gate on tier-1 and guard suites only; no live calls, no scheduled live job.
- Per-task tokens and latency are recorded from any live run that happens. Depends on: 5.1.

**Issue 5.4: Docs.**
- README and `CLAUDE.md` updated.
- This PRD lists the exclusions and the open limits. Depends on: 5.1.

## Decisions
Decided by the project owner:
1. Reuse tau2's DB through `BankingSession` with a faithful session layer; no new database.
2. Five domain-cluster specialists.
3. BM25 knowledge search with a per-agent document scope.
4. Per-intent DB-state tests first; staged simulation later.
5. The orchestrator contract may change (context, NEEDS_INFO, PARTIAL keeps progress).
6. Native tool calling in the LiteLLM client and its fake.
7. Hybrid policy decisions: the LLM reasons, code guards money, status, tools and arguments.
8. Plan preview with one yes for the whole plan.
9. Debit cards is the pilot.
10. DIRECT recipes, `GENERAL_QUESTION` and `REPORT_FRAUD_OR_LOSS` are in scope.
11. Customer-side tools are out of scope now.
12. `AGENT_MODEL`, default `openai/gpt-4o`.
13. Pilot go/no-go: tier-1 green and live semantic pass@1 >= 80% on the 12 usable wholly-debit tasks, with zero guard bypasses (one trial).
14. Live evaluation is expensive: never broad, always capped and opt-in.

Adopted from the design review (open to veto):
- Specialists are configuration over one class.
- Fraud and loss is a composer, not an agent with its own tools.
- Rolling previews for data-dependent plans.
- Execute mode runs in code with no LLM.
- Per-route timeouts replace the single 30 s.
- A human transfer is code-triggered: ask first, only when the KB offers no action.

Corrections to earlier statements made while writing this PRD: the debit pilot is 13 tasks (12 usable: task 083's gold is invalid; plus 3 spanning debit and credit), not "about 30"; 35 tasks carry a real DB overlay (53 have an `initialization_data` block, 18 of them empty); and the "34 tasks with no agent-side write" were really 25 customer-side or reads-only tasks plus 8 transfer-only tasks (plus task 005, profile-only), found when the manifest (Issue 0.5) was generated from `tasks.json`.

### Pilot result so far (gpt-4o-mini, not the gate model)
First live run, 12 usable debit tasks, one trial, `openai/gpt-4o-mini`, about $0.10 in all (cap $0.50): **0 of 12**. This is a cheap read, not the gate: the PRD's bar is measured on the configured agent model (`gpt-4o`), which has not been run. What it showed:
- Safety held every time. Every bad plan was refused at plan time (a made-up close reason, a credit-card tool on a debit card, a closed-before-unfrozen order, a PIN written in the open), so no change was ever made outside an approved plan.
- The model's failures were of three kinds: it asked the customer for ids it could have looked up (3), it wrote tool arguments that did not fit the tool or put a PIN in the open (4), it proposed steps the validators refused (3). One plan ran to the end with a final state that differed from the reference in 10 places, and one ran out of steps.
- The run found real defects in our own code, now fixed: the live harness used a new event loop per phase (the client then failed with "Event loop is closed", so half the tasks never reached the model); a PIN typed as bare digits in answer to "what PIN?" was not captured as a secret; and the model was not told each tool's allowed values.

## GitHub issues
| Issue | GitHub | Title |
|---|---|---|
| 0.1 | #19 | Per-task overlay and injected clock |
| 0.2 | #20 | tau2-fidelity dispatch and verification logging |
| 0.3 | #21 | State delta and normaliser |
| 0.4 | #22 | Gold replay parity gate |
| 0.5 | #23 | Task manifest |
| 1.1 | #24 | RouteContext, runner contract, per-route timeouts, audit reconciliation |
| 1.2 | #25 | NEEDS_INFO and orchestrator re-entry |
| 1.3 | #26 | PARTIAL keeps progress; resume |
| 1.4 | #27 | Plan preview consent |
| 1.5 | #28 | Tool-calling client, `FakeToolLLM`, `AGENT_MODEL` |
| 1.6 | #29 | ScopedToolBox, schemas, SecretVault |
| 1.7 | #30 | Agent loop with budgets |
| 1.8 | #31 | Boundary guard |
| 1.9 | #32 | Knowledge index and scoped search |
| 1.10 | #33 | Specialist base and dispatcher |
| 2.1 | #34 | Debit lifecycle |
| 2.2 | #35 | PIN, limits, fraud alert, recurring block |
| 2.3 | #36 | Debit disputes |
| 2.4 | #38 | Pilot gate and report |
| 2.5 | #37 | Live-run cost controls |
| 3.1 | #39 | DIRECT recipes (6 kinds) |
| 3.2 | #40 | GENERAL_QUESTION answerer |
| 4.1 | #41 | Credit cards |
| 4.2 | #42 | Accounts and money movement |
| 4.3 | #43 | Fees, interest, rewards |
| 4.4 | #44 | Fraud and loss composer |
| 5.1 | #45 | Wire dispatcher and DIRECT runner; end-to-end scenarios with fakes |
| 5.2 | #46 | Staged simulation harness (optional, deferred) |
| 5.3 | #47 | Cost and latency report, regression CI |
| 5.4 | #48 | Docs |

All issues carry the `enhancement` and `specialist-agents` labels. Issue 2.5 is created before 2.4 because the pilot gate depends on it.
