# banking_customer_service_agent

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

The central agent's decision layer. It takes what a customer says, decides per intent whether to act, ask, or hand off to a human, runs the work, and answers once. Everything is plain code except two swappable parts: the LLM that splits a message into intents (the decomposer) and the Jev model that says how sure we are (the scorer). Both have scripted versions for tests and real ones (below); the tools and sub-agents are still mocks, so the test suite runs offline with no keys. Specs: [`docs/prd-orchestrator-routing.md`](docs/prd-orchestrator-routing.md) and [`docs/prd-decomposer.md`](docs/prd-decomposer.md).

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
| One reply per turn: results, problems with an offer of a human, escalations, and one batched set of questions. | `reply.py` |
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

Real sub-agents, LLM phrasing of the reply (a template today), and the opt-in evaluation harness (`docs/prd-decomposer.md` issue 11).
