# PRD: Intent decomposer and decision gate

Routing, execution, per-intent decisions, the verified-caller gate and the 0.85 / 0.65 bands are specified in `docs/prd-orchestrator-routing.md`. Where the two differ, that document wins.

## Why
A customer message often holds several requests: "My debit card was stolen so freeze it, dispute the $200 charge from yesterday, tell me my savings balance, and if the card's closed move what's left to checking." The orchestrator needs this as structured, validated intents before it can choose tools. A prompt that returns free text drops intents, merges or splits them differently on each run, loses dependencies ("if the card's closed..."), and can be steered by text inside the message. The orchestrator also needs a reliable answer to "do we know enough to act, do we need to ask, or is this a human's job?" before any banking tool runs. The tool wrapper already labels which tools are irreversible; this stage must respect that and never act on guesses.

## What
A pipeline in `backend/src/orchestrator/`:
1. **Decompose.** A LiteLLM call turns the message (plus conversation history) into a typed `Decomposition`.
2. **Validate.** Pure code checks the result.
3. **Decide.** A `Decider` returns `Proceed`, `AskCustomer` or `HandOff`. Implementations: a rule-based decider (default) and a Jev adapter.
4. **Ask.** If the decision is `AskCustomer`, an LLM phrases one batched message. The customer's reply goes back to step 1 and is merged into the existing intents. Each intent has its own try counter: after 3 ask attempts that intent hands off.

### Goals
- 3+ intents in one message are all captured, stable across runs, and ordered canonically.
- Dependencies, conflicts, missing details and out-of-scope requests are represented explicitly, not lost.
- Everything except the single LLM call is pure code, so it is testable offline with a fake LLM.
- Irreversible actions never proceed on unresolved details or low confidence.
- The decomposer and decider are swappable behind Protocols.

### Non-goals
- Choosing or executing banking tools (see `docs/prd-orchestrator-routing.md`).
- HTTP endpoints, UI, voice.
- Training a dedicated model (a comparison is in the evaluation issue).
- Changing tau2-bench or `banking_tools`.

### Definitions
- **Intent**: one customer goal, with a kind from a closed list, the details (slots) found for it, and which other intents it depends on.
- **Slot**: a detail an intent needs (which card, which account, an amount, a date).
- **Decision**: `Proceed` (enough is known), `AskCustomer` (questions tied to specific intents), `HandOff` (human needed).
- **Round**: one ask-and-reply cycle.

### Intent kinds (closed list; `CARD_CONTROLS`, `FEE_OR_INTEREST_ADJUSTMENT` and `UNCLEAR_OR_GREETING` added after mapping all 53 catalogued tools)
`FREEZE_CARD`, `UNFREEZE_CARD`, `ACTIVATE_CARD`, `ORDER_REPLACEMENT_CARD`, `CHANGE_OR_RESET_PIN`, `CLOSE_CARD`, `CARD_CONTROLS`, `DISPUTE_TRANSACTION`, `CREDIT_LIMIT_REQUEST`, `FEE_OR_INTEREST_ADJUSTMENT`, `TRANSFER_FUNDS`, `PAY_CREDIT_CARD`, `OPEN_ACCOUNT`, `CLOSE_ACCOUNT`, `ACCOUNT_BALANCE_OR_HISTORY`, `LOOKUP_INFO`, `UPDATE_PROFILE`, `REPORT_FRAUD_OR_LOSS`, `REQUEST_HUMAN`, `UNCLEAR_OR_GREETING`, `OTHER`. Anything outside the list is `OTHER`; the model may not invent kinds.

## How
Package `backend/src/orchestrator/`, tests in `backend/tests/orchestrator/`. Same style as `banking_tools`: frozen strict pydantic models, `StrEnum`, mypy strict, offline tests.

- `types.py`: `IntentKind`, `Intent` (`id`, `kind`, `summary` in the customer's words, `slots`, `missing_slots`, `depends_on`, `source_span`), `Decomposition` (`intents`, `conflicts`, `flags`, `schema_version`), `Decision` union (`Proceed`, `AskCustomer`, `HandOff`), `Decider` Protocol.
- `llm.py`: `LLMClient` Protocol, LiteLLM implementation, fake for tests.
- `decomposer.py`: `Decomposer` Protocol and `LiteLLMDecomposer`. Temperature 0, schema-constrained output, bounded retries, never raises on bad model output.
- `validate.py`: pure checks (unknown kinds, duplicate ids, dependency cycles, conflicts, required slots per kind, dropped-intent heuristic, injection flags).
- `sanitize.py`: redacts PIN, CVV and card numbers before logging and before any outbound call.
- `decider/rules.py` and `decider/jev.py`: the two deciders. Jev via `typesafe_sdk` `client.system_one(state=..., questions=...)`, pinned model, optional dependency, key from env var.
- `clarify.py`: LLM phrasing of batched questions.
- `loop.py`: state machine for rounds, cap and merging of answers.
- Reuses `banking_tools.get_registry()` for reversibility labels and the redaction approach in `banking_tools/session.py`.

### What "deterministic" means here
The LLM is not byte-identical run to run. Determinism comes from temperature 0, a strict schema, a closed taxonomy, canonical intent ordering, code validation after the model, and fixed fake-LLM tests for everything else.

### Constraints and risks
1. **Dropped or unstable intents.** The most likely failure with 3+ intents. Mitigation: dropped-intent heuristic, stability checks across N runs in the evaluation issue.
2. **Dependencies and conflicts.** Flat lists lose "if the card's closed...". Mitigation: `depends_on`, cycle checks, conflict records.
3. **Prompt injection in the message.** The message is data; output is schema-only; injection is flagged and never changes kinds.
4. **Jev limits** (from TypeSafe's own docs): weak at numbers and dates, distracted by large irrelevant input, sensitive to adversarial text, 64k context, text-only, early access, vendor-reported latency (70-500 ms) and price unverified. So deterministic checks run first, the model is pinned, thresholds are per consequence level, and it runs in shadow mode before enforcing. Confidence is never authorization.
5. **Jev cannot write questions.** A second LLM call phrases them.
6. **No Jev key yet.** Credentials are available from the TypeSafe console but not fetched; the rule-based decider ships first so nothing blocks.
7. **litellm** is only a transitive dependency via tau2 (`>=1.80.15,<1.82.7`); a direct pin must stay inside that range.
8. **Sensitive data** in messages must not reach logs or third parties.
9. **Dedicated multi-intent models.** Public work (MixATIS/MixSNIPS lineage) uses synthetic "and"-joined data, reports roughly 35-47% sentence accuracy there, and does not model dependencies or conflicts. No off-the-shelf splitter was found. Start with an LLM behind a Protocol and let the evaluation issue decide whether a fine-tuned small model, likely in a hybrid, is worth it.

## Issues and acceptance criteria
Issue text will be created in GitHub from this document; the criteria are repeated here.

1. **Add litellm as a direct dependency and an LLM client seam**
   - `litellm` is declared in `backend/pyproject.toml` within tau2's pin; `uv.lock` updated.
   - `LLMClient` Protocol with a LiteLLM implementation and a fake for tests.
   - `uv sync` and `uv run pytest` pass offline; mypy strict is clean.
2. **Intent types and closed taxonomy** (GitHub #18)
   - Frozen strict pydantic models for `Intent`, `Decomposition`, `Decision`.
   - Every non-excluded tool in the catalog maps to an `IntentKind` or to `OTHER`, checked by a test.
   - A JSON schema snapshot test exists.
   - A dependency on an unknown id, a duplicate id, or an unknown kind is rejected.
3. **Single-intent decomposer**
   - A message becomes one `Decomposition` through the fake LLM.
   - Tests assert temperature 0 and the schema are passed to LiteLLM.
   - Malformed model output is retried a bounded number of times, then returns a hand-off-safe result; no exception reaches the caller.
4. **Multi-intent decomposition, ordering and dependencies**
   - Fixtures with 3 and 4+ intents lose none, including one buried mid-sentence.
   - Intent order is canonical and stable.
   - `depends_on` is captured and cycles are rejected.
   - The stolen-card message from the Why section is a fixture.
5. **Deterministic validator**
   - Unit tests exist for conflicts, missing required slots, unknown kinds, the dropped-intent heuristic, and injection flags.
   - The validator is pure: no network, no LLM.
6. **Safety and sanitization**
   - PIN, CVV and card numbers are redacted before logging and before any outbound call.
   - Prompt-injection fixtures never change schema validity or kinds and are flagged.
   - Empty, gibberish, very long and non-English inputs produce a defined result.
7. **Decider protocol and rule-based decider** (superseded by routing PRD issue #10; not created)
   - `Decision` is a typed union; rules cover proceed, ask and hand-off.
   - An irreversible intent (per `banking_tools` labels) with unresolved slots never gets `Proceed`.
   - `REQUEST_HUMAN`, out-of-scope, and an intent reaching its 3-try cap produce `HandOff`.
8. **Clarification question generation** (superseded by routing PRD issue #14; not created)
   - One batched customer message per round, each question tied to intent ids.
   - No question for a slot already answered in history.
   - Tested with the fake LLM.
9. **Clarification loop** (superseded by routing PRD issue #15; not created)
   - Answers merge into existing intents without re-deriving the rest.
   - After 3 ask attempts for an intent, that intent is `HandOff`; other intents continue.
   - A customer changing their mind supersedes the earlier intent.
   - Fully deterministic with the fake LLM.
10. **Jev adapter** (the scorer seam is routing PRD issue #8; the adapter itself is still to be created later)
    - Implements `Decider`; model version pinned; key read from an env var.
    - `typesafe_sdk` is an optional dependency; importing the package without it works.
    - Timeout or error falls back to hand-off or the rule decider.
    - Implements the per-intent `Scorer` Protocol from the routing PRD; bands are fixed at 0.85 (proceed) and 0.65 (escalate below); high confidence never turns an irreversible action into `Proceed` with unresolved slots or without the customer's explicit yes.
    - Contract tests use a fake client; one live test is skipped without a key.
11. **Evaluation harness over tau2 tasks**
    - Runs the decomposer on user instructions from `tasks.json`; reports intent-count stability over N runs and precision/recall against expected tool actions.
    - Shadow-mode comparison of the rule decider and Jev.
    - Compares the LLM decomposer against a small supervised baseline to decide whether a dedicated model is worth adopting.
    - Opt-in; requires keys; not part of `uv run pytest`.
12. **Docs**
    - README section on the orchestrator and how to run it.
    - `CLAUDE.md` updated.
    - Decisions section of this PRD finalized.

## Decisions
Confirmed:
- Flow: LiteLLM decomposer, then Jev decides, then an LLM asks the customer, then back to the decomposer.
- Decision outcomes: proceed, ask customer, hand off to a human.
- Scope: decomposer and decision gate only; closed intent list from the banking domain.

My calls, open to change:
- The decomposer is a pure function that never talks to the customer; the orchestrator owns the loop.
- Try cap is 3 per intent, not per conversation (confirmed; see the routing PRD).
- Rule-based decider is the default until the Jev key is fetched; Jev is an optional dependency.
- Jev thresholds (0.85 / 0.65) stay uncalibrated placeholders until shadow-mode data exists.
- The intent list above is a draft derived from the 53 catalogued tools.
