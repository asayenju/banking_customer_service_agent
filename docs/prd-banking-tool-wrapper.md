# PRD: Typed banking tool wrapper

## Why
The banking tools live in `tau2-bench/src/tau2/domains/banking_knowledge/tools.py` as plain methods that return strings. 44 of them are "discoverable": they can only be reached by unlocking them and calling them through a dispatcher that takes a JSON string. The backend has no typed way to list, validate or call them, and nothing says which tools only read, which change state, and which changes can be undone. A policy layer or LLM agent needs that to decide what to allow, confirm or block.

## What
One typed interface in `backend/src/banking_tools/` over the agent-side banking tools:
- Every tool has a typed argument model and a typed result.
- Every tool is labelled `read`, `write` or `handoff`.
- Every `write` is labelled `reversible` or `irreversible`.
- Discoverable tools are exposed directly, with no unlock step and no JSON-string arguments.

### Goals
- A registry that lists each tool with its kind, reversibility, description and argument schema.
- A session object that executes a tool call against its own copy of the banking database.
- Tests that fail when a tau2 tool is added, removed, relabelled or has its signature changed without the catalog being updated.

### Non-goals
- HTTP endpoints (can be mounted on FastAPI later).
- Customer-side (simulated user) tools and the unlock/call/list dispatcher tools.
- Running the tau2 benchmark or any LLM.
- Changing tau2-bench itself.

### Definitions
- **read**: returns data and changes no banking data.
- **write**: changes banking data.
- **handoff**: signals a transfer to a human; changes no banking data.
- **reversible**: a tool in the set can restore the prior state, given what the caller read beforehand. Everything else that writes is **irreversible**.

### Labels
- **Read:** get_current_time, get_user_information_by_id / by_name / by_email, get_referrals_by_user, get_credit_card_transactions_by_user, get_credit_card_accounts_by_user, get_debit_dispute_status_7483, get_atm_deposit_images_8473, get_user_dispute_history_7291, get_pending_replacement_orders_5765, get_closure_reason_history_8293, get_credit_limit_increase_history_4829, get_payment_history_6183, get_all_user_accounts_by_user_id_3847, get_bank_account_transactions_9173, get_debit_cards_by_account_id_7823.
- **Handoff:** transfer_to_human_agents, initial_transfer_to_human_agent_0218, initial_transfer_to_human_agent_1822, emergency_credit_bureau_incident_transfer_1114.
- **Write, reversible:** change_user_email, update_transaction_rewards_3847, set_debit_card_recurring_block_7382, freeze_debit_card_3892, unfreeze_debit_card_3893, change_debit_card_pin_6285.
- **Write, irreversible:** log_verification, file_credit_card_transaction_dispute_4829, file_debit_card_transaction_dispute_6281, order_replacement_credit_card_7291, log_credit_card_closure_reason_4521, apply_statement_credit_8472, apply_credit_card_account_flag_6147, close_credit_card_account_7834, pay_credit_card_from_checking_9182, submit_credit_limit_increase_request_7392, approve_credit_limit_increase_5847, deny_credit_limit_increase_5848, open_bank_account_4821, close_bank_account_7392, transfer_funds_between_bank_accounts_7291, apply_checking_account_credit_5829, apply_savings_account_credit_6831, submit_interest_discrepancy_report_7294, order_debit_card_5739, activate_debit_card_8291, activate_debit_card_8292, activate_debit_card_8293, close_debit_card_4721, clear_debit_card_fraud_alert_4892, reset_debit_card_pin_6284, request_temporary_debit_card_limit_increase_8374.
- **Excluded:** example_agent_tool_0000 (placeholder), unlock_/give_/call_/list_discoverable_* tools, all customer-side tools.
- **Judgement calls to confirm:** transfer_funds and pay_credit_card (a reverse transfer restores balances but not history), reset_debit_card_pin (old PIN is not stored), request_temporary_debit_card_limit_increase (expires on its own, no revert tool). All are irreversible under the strict definition.

## How
Package `backend/src/banking_tools/`:
- `types.py`: `ToolKind`, `Reversibility`, `ToolSpec`, `ToolResult`.
- `catalog.py`: the labelled table.
- `args.py`: pydantic argument models generated from the tau2 method signatures.
- `session.py`: `BankingSession` with its own deep-copied `TransactionalDB`, a lock, and `call(name, args) -> ToolResult`.
- `__init__.py`: `get_registry()` and `BankingSession`.

### Constraints and risks found during exploration
1. **Number handling.** tau2's dispatcher parses JSON with `parse_int=float` so database IDs and hashes do not depend on how a number was written. Direct calls must convert numeric arguments the same way.
2. **Errors are strings.** tau2 tools never raise; 203 return strings starting with `Error:`. The wrapper turns these into `ok=False`.
3. **Audit rows.** Bypassing the dispatcher means tau2's `agent_discoverable_tools` row is not written. The wrapper keeps its own audit log and documents the difference.
4. **Decorator labels are unreliable.** `example_agent_tool_0000` is marked WRITE but does nothing; unlock is marked mutating but only changes memory. The catalog is the source of truth, and tests cross-check the decorator with a listed set of known exceptions.
5. **Shared mutable database.** Each session gets its own copy, and calls on one session are serialised.
6. **Environment.** The backend needs Python 3.12 or 3.13 (tau2 requires `>=3.12,<3.14`), and `tau2-bench` is a gitlink with no `.gitmodules`, so fresh clones get an empty folder.

## Issues and acceptance criteria
Issue text is in the GitHub issues created from this document; the criteria are repeated here.

1. **Make tau2 a resolvable dependency**
   - Backend `requires-python` is `>=3.12,<3.14`, with tau2-bench as an editable path dependency and `uv.lock` updated.
   - `uv sync` succeeds from a clean clone.
   - `python -c "import tau2.domains.banking_knowledge"` succeeds in the backend environment without API keys.
   - `tau2-bench` is registered in `.gitmodules` (or pinned by git URL), and the choice is documented.
2. **Core types**
   - A `write` ToolSpec without a reversibility fails validation.
   - A `read` or `handoff` ToolSpec with a reversibility fails validation.
   - `ToolResult` carries `ok`, `text`, `error`, `tool` and `audit_id`.
   - All new code is fully type-annotated and passes the type checker.
3. **Labelled tool catalog**
   - Every tool in `KnowledgeTools.get_tools()` and `get_discoverable_tools()` is catalogued or in an explicit exclusion list.
   - A test fails if a tool is unlabelled, removed or newly added upstream.
   - Where the tau2 decorator disagrees with the label, the tool is in a documented exceptions list.
   - A reviewable table of all labels is in the repo.
4. **Typed argument models**
   - Every catalogued tool has a pydantic model built from its tau2 signature.
   - Missing, extra and wrongly typed arguments are rejected with a clear message.
   - Numeric arguments are converted to float where tau2's dispatcher would.
   - A snapshot test fails when an upstream signature changes.
5. **Session and executor**
   - Two sessions never share state.
   - Concurrent calls on one session do not corrupt it.
   - A tool result starting with `Error:` produces `ok=False`.
   - Every write call appends an audit entry; reads do not.
   - For a sample of write tools, the database hash after a wrapper call equals the hash after the same call through tau2's dispatcher.
6. **Tests and docs**
   - `uv run pytest` passes offline with no API key.
   - The README documents the registry, the labels, and the unlock/audit bypass.

## Decisions
- tau2-bench is registered in `.gitmodules` pointing at the fork `https://github.com/asayenju/tau2-bench.git`.
- The four judgement calls under Labels are accepted as irreversible.
