# Banking customer service agent

A typed multi-agent customer service system for a bank, built on the `banking_knowledge` domain of tau2-bench. A customer writes in plain language; the system works out what they want, checks who they are, asks only what it needs, shows the exact changes it will make, and carries them out after one yes. The model never decides a rule, a fee or a step: it only reports what the customer said, and code does the rest.

## How it works

```
customer message
  -> decomposer (LLM, strict JSON)      splits it into typed requests ("intents"); sees the last 8 messages
  -> Jev scorer + code gates            how sure are we? who is calling (2 of 4 identity fields)? what is missing?
  -> decision per intent                run / ask one question / show a plan and ask for a yes / hand to a person
  -> plan (waves)                       urgent first, dependent requests wait, independent ones run together
  -> route
       direct recipes                   freeze, balance, referrals: no model
       specialists (6)                  debit, credit, accounts, fees, fraud composer, general answers
       hand-off                         a person, with a note written by code
  -> one reply, one question at a time
```

Each **specialist** is a configuration of one generic class. In plan mode a bounded model loop reads the customer's records (search tools for charges and cards, a records lookup, policy search) and ends by filling in a **workflow form** (what the customer said). Code then decides: validators check every tool call against the bank's written policy (the bank's own tools check very little), a dry run on a copy of the database shows the final state, and the customer sees one preview and gives one yes bound to those exact steps. Execution runs with no model at all and re-checks each step first. Nothing irreversible runs without that yes.

**Jev** (TypeSafe AI) is a hosted decision model that returns yes/no probabilities and no text. It scores how sure the system should be about each intent's kind, independent of the decomposer. A write also needs its kind score at 0.95 or more, or the customer is asked first. It can never authorize an action alone.

## Structure

| Path | What is there |
|---|---|
| `backend/src/banking_tools/` | typed wrapper over 53 tau2 bank tools: kinds, reversibility, audit log, per-session database copy |
| `backend/src/orchestrator/` | decomposer, Jev scorer, identity gate, decisions, plans, routes, replies, trace; evaluation harness in `evaluation/` |
| `backend/src/agent/` | specialists (`specialists/`), workflows with the bank's rules (`workflows/`), search tools (`credit_lookup.py`, `debit_lookup.py`, `overview.py`), the wiring (`assembly.py`), offline reports, the live harness (`live.py`), the free preflight (`preflight.py`), the simulated customer (`customer.py`) |
| `backend/src/main.py` | FastAPI: `POST /conversations/{id}/messages`, the test customers, and the demo page |
| `frontend/widget.html` | demo chat: pick one of 81 test customers on the left, talk to the agent on the right |
| `docs/` | PRDs, and one policy-decisions note per area with the document or data behind every rule |
| `tau2-bench/` | git submodule (the fork `asayenju/tau2-bench`); never edited |

## Run the demo

```bash
cd backend
PYTHONPATH=src:../tau2-bench/src uv run uvicorn main:app --port 8765
# open http://localhost:8765/demo   (needs OPENAI_KEY and JEV_KEY in the repo-root .env)
```

## Results so far

Offline replay of the benchmark's own reference trajectories through the real specialists, compared with tau2's database state (`uv run python -m agent.pilot`, `agent.credit_report`, `agent.preflight`):

- **60 of 62 gradeable tasks reproduce the reference state exactly**; the other 2 are documented departures where written policy and the reference disagree. 35 more of the 97 are not gradeable (customer-side only, human-transfer only, one broken reference).
- Debit 12/12, accounts 19/19, lost wallet 5/5, credit 17/18, fees 9/10. 0 writes outside an approved plan in every report.
- 2,500+ offline tests, `mypy --strict` on the typed packages.
- Decomposer pass@1 (older measurement, before the latest prompt changes): 0.92 tuning set, 0.97 held-out, 1.00 minimal pairs, 0.83 on the hardest set.
- Jev gate on 70 labelled intents (30 wrong ones made up by swapping the action): avoided 29 of 30 wrong, blocked 2 of 40 right (AUROC 0.97). A plain gpt-4o-mini yes/no on the same intents avoided 28 of 30 and blocked 9 of 40 (AUROC 0.95).

## Where it is failing so far

Be sceptical of the numbers above: replay shows the code is right when it is given the right facts, not that the model always supplies them.

- **Live pass rate is low.** The last full live run of the 12 debit tasks (gpt-4o, one-shot harness) passed 2. Several failures were the harness (the customer cannot answer a question), but the model also wrote wrong values (a dispute amount, a year, a design nobody asked for). Fixes since then are tested offline only; the live number is out of date.
- **Most paths have never run live.** Credit, fees, accounts, card activation and the PIN-lock score are covered by offline replay and unit tests only. The PIN-lock fraud score is built but not wired into a workflow. Nothing calls `transfer_to_human_agents` yet, so a hand-off ends the chat but transfers nobody.
- **Multi-turn conversation is patched, not solved.** The request reader sometimes misreads a short answer as a new request, a yes or a withdrawal; several cases are caught in code, but a question can still be asked twice when answers arrive out of order. Its run-to-run variation can turn a vague request into a hand-off.
- **Inferred rules.** Where a policy document and the reference answers disagree, the references win and the rule is marked inferred (the Entry-tier reorder wait, provisional credit for a cancelled subscription, the 12-month dispute caps). The debit limit cap is not enforced because the data has no ATM limit field.
- **The Jev result is thin.** The wrong intents are synthetic, there are only 70 intents, and its 0.95 threshold was set from about 10 wrong examples. It shows Jev blocks fewer right requests than a plain LLM, not that it beats one on accuracy.
- **Measurement gaps.** 12 live recordings are stale, 42 runnable tasks need three or more customer replies the one-shot harness cannot give (the simulated customer has only been tried on one task), and the decomposer scores predate its latest prompt changes.
- **Not production.** It runs on a copy of the benchmark data: no real accounts or login, no rate limits, monitoring or audit storage. Every live run costs real money and needs both keys.

---

## Setup

`tau2-bench` is a git submodule (the fork `asayenju/tau2-bench`), and the backend depends on it as an editable package.

```bash
git clone --recurse-submodules https://github.com/asayenju/banking_customer_service_agent.git
cd banking_customer_service_agent/backend
uv sync          # Python 3.12 (tau2 requires >=3.12,<3.14)
uv run pytest    # offline; no API keys needed
```

If `import tau2` fails with `No module named 'tau2'` on macOS, the editable-install `.pth` file in `backend/.venv` has been marked hidden (iCloud does this to `~/Documents`) and Python ignores it. `pytest` no longer depends on it, but scripts and the `tau2` command do. Fix: `chflags -R nohidden backend/.venv`.

## Banking tool wrapper (`backend/src/banking_tools/`)

One typed interface over the agent-side tools of tau2's `banking_knowledge` domain. Every tool has typed arguments, a kind, and, for writes, a reversibility.

```python
from banking_tools import BankingSession, get_registry

get_registry()["freeze_debit_card_3892"]
# ToolSpec(kind=write, reversibility=reversible, args_model=<pydantic model>, ...)

session = BankingSession()          # private copy of the banking database
result = session.call("freeze_debit_card_3892", {"card_id": "dbc_lj82d4f1a9_bluest"})
result.ok, result.text, result.error, result.audit_id
session.audit_log                   # one AuditEntry per executed write
```

### Labels

| Kind | Meaning |
|---|---|
| `read` | returns data, changes nothing |
| `write` | changes banking data; always `reversible` or `irreversible` |
| `handoff` | signals a transfer to a human, changes nothing |

**Reversible** means a tool in the set can restore the prior state, given what the caller read beforehand (for example `freeze_debit_card` and `unfreeze_debit_card`). Every other write is irreversible.

The full table of 53 catalogued tools, and the 5 excluded ones with reasons, is in [`docs/banking-tool-catalog.md`](docs/banking-tool-catalog.md). The labels themselves live in `backend/src/banking_tools/catalog.py`.

### How it differs from tau2

- **No unlock step.** In tau2, 44 tools must be unlocked and then called through `call_discoverable_agent_tool` with a JSON string. Here each is called directly with validated arguments.
- **No tau2 audit row.** tau2's dispatcher writes a row to `agent_discoverable_tools` for mutating tools. The wrapper does not; `session.audit_log` replaces it. The log redacts `pin`, `new_pin`, `current_pin` and `cvv`.
- **Numbers become floats**, as in tau2's dispatcher (`parse_int=float`).
- **Failures are values, not exceptions.** tau2 tools return `"Error: ..."` strings; the wrapper maps them to `ToolResult(ok=False, error=...)`. Invalid arguments and unknown tools fail the same way, and the tau2 tool is not called.
- **Isolation.** Each `BankingSession` deep-copies the database and serialises its own calls with a lock.
- **Out of scope:** customer-side (simulated user) tools and tau2's unlock, list and call dispatcher tools.

### Keeping it in sync with tau2

Tests compare the catalog and argument schemas with the installed tau2, so an upstream change fails the build:

- A tau2 tool was added, removed or relabelled: update `_LABELS` or `EXCLUDED` in `catalog.py`, then regenerate the table:
  `cd backend && PYTHONPATH=src uv run python -c "from banking_tools.catalog import render_markdown; open('../docs/banking-tool-catalog.md','w').write(render_markdown())"`
- A tool signature changed: review it, then `UPDATE_SNAPSHOTS=1 uv run pytest tests/test_args.py`.

### Checks

```bash
cd backend
uv run pytest
uv run mypy --strict src/banking_tools src/orchestrator
```

## Orchestrator (`backend/src/orchestrator/`)

The central agent's decision layer. It takes what a customer says, decides per intent whether to act, ask, or hand off to a human, runs the work, and answers once. Everything is plain code except two swappable parts: the LLM that splits a message into intents (the decomposer) and the Jev model that says how sure we are (the scorer). Both have scripted versions for tests and real ones (below); the test suite uses scripted tools and sub-agents, so it runs offline with no keys (the real specialists are in `backend/src/agent/`). Specs: [`docs/prd-orchestrator-routing.md`](docs/prd-orchestrator-routing.md) and [`docs/prd-decomposer.md`](docs/prd-decomposer.md).

```python
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route

orch = Orchestrator(
    decomposer=..., scorer=..., directory=...,        # Protocols; scripted fakes ship in the package
    runners={Route.DIRECT: ..., Route.SUB_AGENT: ...},  # mocks in orchestrator.mocks
)
turn = await orch.handle_turn("msg-1", "my card was stolen, freeze it and tell me my balance")
turn.reply.text      # one message for the whole turn
turn.decisions       # run / ask / escalate per intent, with the reason
turn.trace.events    # every choice, in order, with wave numbers
```

### What it decides

| Rule | Where |
|---|---|
| Confidence per intent is the weakest of kind, slots and scope. 0.85 or more proceeds, 0.65 to 0.85 asks, below 0.65 escalates. | `scoring.py`, `bands.py` |
| Greetings, gibberish and empty messages get a fixed menu without any model, then a human after 3 menus. | `unclear.py` |
| Account-specific intents need a verified caller: 2 of 4 identity fields, checked by code. Bypass codes and supervisor claims are ignored. 3 wrong answers lock the caller out. | `auth.py` |
| Irreversible actions never run without the customer's yes to the exact details, at any confidence. A changed detail makes the yes stale. | `decision.py` |
| Each intent escalates after 3 questions. Changing its details restarts the count; answering a different intent does not reset it. | `orchestrator.py` |
| Urgent intents (fraud or loss report, freeze) go first. Writes run together only when each names a different resource. Dependents of anything not running wait. A cycle hands off. | `plan.py` |
| A failed route is reported, its dependents are skipped, independent work carries on, nothing is undone. | `executor.py` |
| One reply per turn: results, problems with an offer of a human, escalations, and one question (the first missing detail; every yes to what is about to be done is still asked together). | `reply.py` |
| A repeated message id returns the stored result and runs nothing twice. | `orchestrator.py` |

Other modules: `types.py` (intents, decomposition, decisions), `taxonomy.py` (which tool serves which intent kind), `routes.py` (direct, sub-agent or hand-off per kind), `trace.py`, `decomposer.py` (the per-turn interface the LLM will fill), `mocks.py` (scripted routes: success, slow, timeout, malformed, partial, raises).

### Tests

```bash
cd backend
uv run pytest tests/orchestrator                  # the whole orchestrator, offline
uv run pytest tests/orchestrator/test_scenarios.py  # whole conversations, asserted on the trace
UPDATE_SNAPSHOTS=1 uv run pytest tests/orchestrator/test_intent_types.py  # after changing a type
```

`test_scenarios.py` has one scenario per row of the edge-case table in the PRD, plus seven inspired by tau2 banking tasks. A test reads that table, so adding a row without a scenario fails the build. New scenarios use the scripted parts in `tests/orchestrator/scenario_kit.py`.

### Real decomposer and Jev scorer

```python
from orchestrator.jev import make_jev_scorer          # needs: uv sync --extra jev
from orchestrator.llm import LiteLLMClient
from orchestrator.llm_decomposer import LLMDecomposer

orch = Orchestrator(
    decomposer=LLMDecomposer(LiteLLMClient()),        # OpenAI via LiteLLM, temperature 0, strict JSON schema
    scorer=make_jev_scorer(),                         # pinned jev-1.13.0
    directory=..., runners=...,
)
```

Keys are read from the repo-root `.env` (git-ignored; see `.env.example`): `OPENAI_KEY`, `JEV_KEY`, and optionally `DECOMPOSER_MODEL` (default `openai/gpt-4o-mini`). The decomposer redacts PINs, CVVs and card numbers before anything leaves the process, bounds retries, and on unusable output returns one `OTHER` intent so a person takes over; it never raises. `validate.py` adds missing required details, orders intents by position in the message, and flags dropped requests and injection attempts. Jev only sees the redacted message, kind and summary; a message flagged as injection is not sent and escalates. Live checks are opt-in: `RUN_LIVE=1 uv run pytest tests/orchestrator/test_jev.py`.

### Not built yet

LLM phrasing of the reply (a template today) and a real transfer to a human agent. The sub-agents and the evaluation harness are built; see "Where it is failing so far" above.
