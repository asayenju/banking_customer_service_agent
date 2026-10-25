# banking_customer_service_agent

## Setup

`tau2-bench` is a git submodule (the fork `asayenju/tau2-bench`), and the backend depends on it as an editable package.

```bash
git clone --recurse-submodules https://github.com/asayenju/banking_customer_service_agent.git
cd banking_customer_service_agent/backend
uv sync          # Python 3.12 (tau2 requires >=3.12,<3.14)
uv run pytest    # offline; no API keys needed
```

If `import tau2` fails with `No module named 'tau2'` on macOS, the editable-install `.pth` file in `backend/.venv` has been marked hidden and Python ignores it. Fix: `chflags -R nohidden backend/.venv`.

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
uv run mypy --strict src/banking_tools
```
