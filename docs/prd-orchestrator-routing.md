# PRD: Central agent routing and execution

Builds on `docs/prd-decomposer.md` (decompose, validate, clarification loop). Where the two differ, this document wins; the decomposer PRD has been updated to match.

## Why
The decomposer yields typed intents. Something must now decide, per intent, whether to act, ask or escalate, route each to a direct tool recipe or a sub-agent, run them without conflicts, survive failures, and reply once. Done wrongly, this stage runs irreversible banking actions on guesses, on an unverified caller, or in a race between two writes. The sub-agents do not exist yet, so the logic must be provable with mocks before any are built.

## What
A deterministic orchestrator in `backend/src/orchestrator/` that, for one customer turn:
1. Takes a `Decomposition` (the decomposer is faked in routing tests).
2. Checks caller verification.
3. Scores each intent with a `Scorer` (Jev adapter or scripted fake).
4. Makes a per-intent decision.
5. Plans execution waves.
6. Runs the waves against mock routes.
7. Returns one customer reply plus a `Trace` of every choice.

### Decision bands
Named constants; changing them is a one-line edit.

| Confidence | Decision |
|---|---|
| >= 0.85 | Proceed |
| >= 0.65 and < 0.85 | Ask the customer |
| < 0.65 | Escalate to a human |

Confidence per intent is the minimum of three separate Jev answers: kind, slots complete, in scope. Boundaries are tested exactly: 0.85 proceeds, 0.65 asks, 0.6499 escalates.

### Rules
1. **Irreversible actions always need the customer's explicit "yes"**, whatever the confidence. The yes is bound to the exact action and parameters; if either changes, it is stale and asked again.
2. **Reads and reversible writes** proceed at >= 0.85 with no confirmation. Only six tools are reversible: `change_user_email`, `update_transaction_rewards`, `set_debit_card_recurring_block`, `freeze_debit_card`, `unfreeze_debit_card`, `change_debit_card_pin`.
3. **Unclear messages** (greeting, "help", gibberish, empty, wrong language) are caught by a deterministic pre-check as `UNCLEAR_OR_GREETING`, skip Jev, and run the read-only `offer_help` tool, which returns a fixed menu (not personalised). After 3 menus with no real intent, escalate.
4. **A real intent below 0.65 escalates immediately** with a summary of what was understood.
5. **A verified caller is required** for every account-specific intent, read or write, urgent ones included. General info and the menu stay open. Verification is 2 of 4 fields (date of birth, email, phone, address) matched by code against the record, then `log_verification`. Jev has no say. No claimed code, supervisor or authority bypasses it.
6. **Decisions are per intent, in one reply.** Proceed intents run, ask intents are batched into one question message, escalated intents go to a human with a summary. Anything depending on an intent that is not running waits. Each intent has its own try counter: after 3 ask attempts without reaching 0.85, that intent escalates. Other intents are unaffected, and an answer that resolves one intent does not reset another's counter.
7. **Urgent intents** (`REPORT_FRAUD_OR_LOSS`, `FREEZE_CARD` for a lost or stolen card) run first and are asked about first, after verification. Thresholds still apply. There is no human-alert step because no such feature exists.
8. **Parallel only when safe.** Intents run in parallel if they are independent and do not write to the same card or account (resource keys). Reads are always parallel. Otherwise they run in canonical order.
9. **Failure policy.** A failed intent is skipped and so are its dependents; independent intents still run; completed work is not undone; the final reply says which part hit an error and offers a human.
10. **Routes come from a static table** `IntentKind -> Route` (Direct, SubAgent, HandOff). Direct means a fixed tool sequence with no policy judgment, possibly 2-3 tools (lookup then freeze). SubAgent means policy reasoning or branching (disputes, credit limits, closures, replacements, transfers). A test fails if any `IntentKind` has no route.

### Goals
- Every rule above is provable offline with mocks and trace assertions.
- Irreversible actions never run on unresolved slots, low confidence, an unverified caller, or without an explicit yes.
- No exception escapes; failures are values, as in `banking_tools`.
- Scorer, routes and executors are swappable behind Protocols.

### Non-goals
- Real sub-agents or real banking calls (mocks only; `BankingSession` can back direct recipes later).
- A personalised help menu (needs identity verification first).
- A human-alert or notification feature.
- Undoing completed actions after a later failure.
- HTTP endpoints, UI, voice.
- Calibrating Jev thresholds (fixed constants for now).

## How
New modules beside the decomposer PRD's `types.py`, `validate.py` and `loop.py`; tests in `backend/tests/orchestrator/`.

- `bands.py`: threshold constants and `band(confidence)`.
- `scoring.py`: `IntentScore`, `Scorer` Protocol, scripted fake. One Jev call per message with questions per intent.
- `routes.py`: route table, urgent set, resource keys, confirmation requirement from `banking_tools.get_registry()` reversibility.
- `auth.py`: `VerifiedCaller` state, the 2-of-4 check, 3 failed attempts then hand-off.
- `gate.py`: per-intent decision `Run | Ask | Escalate | Blocked(reason)`.
- `plan.py`: execution waves from `depends_on`, resource conflicts and urgency.
- `executor.py`: async wave runner, failure policy, `Executor` Protocol.
- `trace.py`: ordered `TraceEvent`s (scored, decided, routed, started, finished, skipped, asked, escalated), each failure carrying a reason.
- `reply.py`: one customer message from results, errors and questions (template for tests).
- `orchestrator.py`: turn handler and cross-turn state (pending confirmations, per-intent try counters, idempotency keys).
- `mocks/`: `MockDirectRoute`, `MockSubAgent` and mock tools, each with scripted outcomes: success, slow, timeout, malformed, partial, raises.

Reuse: `banking_tools.get_registry()` for reversibility and `ToolKind`, the redaction in `banking_tools/session.py`, and the decomposer PRD's `Decomposition`, `Intent` and `LLMClient` fake. Style matches `banking_tools`: frozen strict pydantic, `StrEnum`, mypy strict, offline pytest, hard-coded ids.

### Edge cases and expected behavior
| Case | Expected |
|---|---|
| 92/88/40 mixed confidence | first two run, third escalated with a note, one reply |
| Escalated intent that others depend on | dependents wait, then are reported as blocked |
| Freeze card and dispute a charge on it | freeze first, dispute after (same resource, write) |
| Two reads on the same account | parallel |
| Stolen card plus balance | verify, freeze first, then balance |
| Unverified caller, urgent intent | verification asked first; nothing runs |
| Caller claims a supervisor bypass code | ignored; verification still required |
| 2-of-4 fields mismatched 3 times | hand-off |
| Irreversible at 0.99, no yes yet | ask for an explicit yes, not Proceed |
| Yes given, but the amount changed since | stale; ask again |
| Customer changes their mind | earlier pending intent superseded |
| Same message submitted twice | idempotency key; no double execution |
| Reply answers a different intent than the one asked | merged to the right intent, others unchanged |
| Mock times out, malformed or partial | failure policy; no exception |
| Injection text in the message | flagged, never changes route or bands |
| Greeting, gibberish, empty | menu; after 3, escalate |
| Intent still under 0.85 after 3 asks | that intent escalates; others continue |
| Customer answers one intent but not another | only the answered intent is re-scored; the other's counter still advances |
| Customer changes an intent's details mid-asking | supersedes the intent; counter restarts for the new version |
| `REQUEST_HUMAN` | hand-off regardless of confidence |

### Testing approach
Each scenario is one customer message (or a scripted `Decomposition`), scripted Jev scores and scripted mock outcomes, and the assertion is on the full `Trace`: which intents ran, in what order, which ran in parallel, what was asked, what was escalated. Scenarios are hand-written and inspired by a small curated slice of tau2's banking tasks, chosen from a seeded stratified draw (`tau2 tasks subset --dry-run`) so no simulation is paid for. The suite makes no LLM calls.

## Constraints and risks
1. **Jev scores are uncalibrated.** 0.85 and 0.65 are placeholders until shadow-mode data comparing Jev to the rule-based decider exists. Jev is also weak on numbers, dates and adversarial text, so deterministic checks (verification, slots, reversibility) run first and confidence never authorizes an irreversible action.
2. **Resource keys come from slots.** If a card or account slot is missing, treat the intent as conflicting and serialize it.
3. **Mock fidelity.** Mocks can pass while a real sub-agent behaves differently. The `Executor` Protocol and scripted failure modes keep the contract narrow; a contract test per real sub-agent is future work.
4. **Verification is a convention in tau2, not enforced by its tools.** `log_verification` only writes an audit record, so the orchestrator must enforce the 2-of-4 check itself.
5. **Almost every write is irreversible** (only six are reversible), so most "just execute" paths still stop for a yes. This is intended but adds a turn.
6. **Sensitive data.** PIN, CVV and card numbers are redacted before logging or any outbound call, as in the decomposer PRD.
7. **Cost.** The routing suite costs nothing. Live tau2 runs are opt-in; levers are `--num-trials 1`, `--num-tasks` or `--task-ids`, `--dry-run` first, `--max-steps` and `--max-errors`, low reasoning effort, and an offline retrieval config such as `bm25` or `golden_retrieval`. A cheaper user simulator can shift scores (one study measured 15.2 points), so keep it fixed across compared runs. tau2's LLM cache is a hard-coded constant in the submodule, which is never edited; caching belongs in our own `LLMClient`. Banking cost per run and the subset size needed to preserve model rankings were not found; measure with a dry run and a 2-task smoke run.
8. **AgentBench is not a fit** for validating this layer. It measures raw multi-turn model reasoning, not policy-following, verification or escalation.

## Issues and acceptance criteria
Prerequisite: the decomposer PRD's types issue (its issue 2). Routing tests feed scripted `Decomposition`s, so they do not wait for the decomposer itself. Issue text will be created in GitHub from this document; the criteria are repeated here.

1. **Bands, route table and trace types**
   - 0.85 proceeds, 0.65 asks, 0.6499 escalates, tested at the boundaries.
   - Every `IntentKind` has a route, checked by a test.
   - `TraceEvent` types are frozen and strict; mypy strict is clean.
2. **Per-intent scorer seam**
   - `Scorer` Protocol with a scripted fake; confidence is the minimum of kind, slots and scope.
   - Scorer timeout or error yields escalate, not an exception.
3. **Verified-caller gate**
   - Account-specific intents are blocked until 2 of 4 fields match by code.
   - A bypass-code claim does not unblock; 3 failures hand off.
   - Verification questions flow through the ask path; Jev is not involved.
4. **Per-intent decision with irreversible confirmation**
   - Irreversible intents never `Run` without an explicit, parameter-bound yes, even at 0.99.
   - Reversible writes and reads run at >= 0.85 without a yes.
   - A changed parameter makes the yes stale.
5. **Unclear-message pre-check and help menu**
   - Greeting, gibberish, empty and wrong-language inputs map to `UNCLEAR_OR_GREETING` without Jev.
   - `offer_help` returns a fixed menu; after 3 menus without a real intent, the next unclear turn escalates.
6. **Execution planner**
   - Independent reads share a wave; same-resource writes are serialized; dependencies are ordered.
   - Urgent intents come first; cycles are rejected.
7. **Executor with mock routes and failure policy**
   - Mock outcomes (success, slow, timeout, malformed, partial, raises) never leak an exception.
   - A failed intent skips its dependents; independent intents still run; nothing is undone.
   - Parallel waves are verified through the trace.
8. **Single customer reply**
   - One message holds results, one batched question set tied to intent ids, and errors with a human offer.
   - No question repeats a slot already answered.
9. **Multi-turn orchestrator**
   - Pending confirmations, the per-intent 3-try cap (escalates that intent only), change of mind, duplicate submit and cross-intent replies behave per the edge-case table.
   - `REQUEST_HUMAN` and injection fixtures behave per the table.
10. **Scenario suite with trace assertions**
    - Hand-written scenarios, each a message plus scripted scores plus mock outcomes, asserting the full trace.
    - A small curated set inspired by tau2 tasks (not all 97), every row of the edge-case table covered.
    - Runs inside `uv run pytest`, offline.
11. **Docs**
    - README section; `CLAUDE.md` current-work and layout updated; this PRD's Decisions finalized.

## Decisions
Confirmed:
- Thresholds 85 / 65 as fixed constants. Exactly 65 asks (the stated ranges overlapped at 65).
- Irreversible actions always need an explicit yes.
- Per-intent decisions; other intents still run when one escalates.
- Each intent escalates to a human after 3 ask attempts. This replaces the decomposer PRD's per-conversation round cap.
- Parallel only when safe.
- A failed intent is skipped, the rest run, the customer is told.
- Urgent priority only; no human alert.
- A verified caller for all account-specific intents.
- Direct means a fixed recipe; sub-agent means policy reasoning.
- Static help menu now; hand-written scenarios inspired by a curated subset of tau2.

My calls, open to change:
- A superseded intent restarts its try counter.
- Dependents of a failed intent are skipped; the failure message offers a human.
- One Jev call per message with per-intent questions.
- 3 failed verification attempts hand off.
- Idempotency key per (session, message id).
- Confirmation is bound to action and parameters and goes stale if either changes.
