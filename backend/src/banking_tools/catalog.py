"""Hand-reviewed labels for every agent-side banking tool.

tau2's own decorator labels are not trusted as the source of truth (see
docs/prd-banking-tool-wrapper.md); the tables below are. tests/test_catalog.py
checks them against tau2 so an added, removed or relabelled upstream tool fails
the build.

"Reversible" means a tool in this set can restore the prior state, given what
the caller read beforehand. Every other write is irreversible.
"""

from typing import Literal, Optional

from tau2.domains.banking_knowledge.tools import KnowledgeTools

from banking_tools.args import build_args_model
from banking_tools.types import Reversibility, ToolKind, ToolSpec

Source = Literal["visible", "discoverable"]

R, W, H = ToolKind.READ, ToolKind.WRITE, ToolKind.HANDOFF
REV, IRREV = Reversibility.REVERSIBLE, Reversibility.IRREVERSIBLE

# name -> (source, kind, reversibility)
_LABELS: dict[str, tuple[Source, ToolKind, Optional[Reversibility]]] = {
    # --- visible: read ---
    "get_current_time": ("visible", R, None),
    "get_user_information_by_id": ("visible", R, None),
    "get_user_information_by_name": ("visible", R, None),
    "get_user_information_by_email": ("visible", R, None),
    "get_referrals_by_user": ("visible", R, None),
    "get_credit_card_transactions_by_user": ("visible", R, None),
    "get_credit_card_accounts_by_user": ("visible", R, None),
    # --- visible: write ---
    "change_user_email": ("visible", W, REV),
    "log_verification": ("visible", W, IRREV),
    # --- visible: handoff ---
    "transfer_to_human_agents": ("visible", H, None),
    # --- discoverable: read ---
    "get_debit_dispute_status_7483": ("discoverable", R, None),
    "get_atm_deposit_images_8473": ("discoverable", R, None),
    "get_user_dispute_history_7291": ("discoverable", R, None),
    "get_pending_replacement_orders_5765": ("discoverable", R, None),
    "get_closure_reason_history_8293": ("discoverable", R, None),
    "get_credit_limit_increase_history_4829": ("discoverable", R, None),
    "get_payment_history_6183": ("discoverable", R, None),
    "get_all_user_accounts_by_user_id_3847": ("discoverable", R, None),
    "get_bank_account_transactions_9173": ("discoverable", R, None),
    "get_debit_cards_by_account_id_7823": ("discoverable", R, None),
    # --- discoverable: handoff ---
    "initial_transfer_to_human_agent_0218": ("discoverable", H, None),
    "initial_transfer_to_human_agent_1822": ("discoverable", H, None),
    "emergency_credit_bureau_incident_transfer_1114": ("discoverable", H, None),
    # --- discoverable: write, reversible ---
    "update_transaction_rewards_3847": ("discoverable", W, REV),
    "set_debit_card_recurring_block_7382": ("discoverable", W, REV),
    "freeze_debit_card_3892": ("discoverable", W, REV),
    "unfreeze_debit_card_3893": ("discoverable", W, REV),
    "change_debit_card_pin_6285": ("discoverable", W, REV),
    # --- discoverable: write, irreversible ---
    "file_credit_card_transaction_dispute_4829": ("discoverable", W, IRREV),
    "file_debit_card_transaction_dispute_6281": ("discoverable", W, IRREV),
    "order_replacement_credit_card_7291": ("discoverable", W, IRREV),
    "log_credit_card_closure_reason_4521": ("discoverable", W, IRREV),
    "apply_statement_credit_8472": ("discoverable", W, IRREV),
    "apply_credit_card_account_flag_6147": ("discoverable", W, IRREV),
    "close_credit_card_account_7834": ("discoverable", W, IRREV),
    "pay_credit_card_from_checking_9182": ("discoverable", W, IRREV),
    "submit_credit_limit_increase_request_7392": ("discoverable", W, IRREV),
    "approve_credit_limit_increase_5847": ("discoverable", W, IRREV),
    "deny_credit_limit_increase_5848": ("discoverable", W, IRREV),
    "open_bank_account_4821": ("discoverable", W, IRREV),
    "close_bank_account_7392": ("discoverable", W, IRREV),
    "transfer_funds_between_bank_accounts_7291": ("discoverable", W, IRREV),
    "apply_checking_account_credit_5829": ("discoverable", W, IRREV),
    "apply_savings_account_credit_6831": ("discoverable", W, IRREV),
    "submit_interest_discrepancy_report_7294": ("discoverable", W, IRREV),
    "order_debit_card_5739": ("discoverable", W, IRREV),
    "activate_debit_card_8291": ("discoverable", W, IRREV),
    "activate_debit_card_8292": ("discoverable", W, IRREV),
    "activate_debit_card_8293": ("discoverable", W, IRREV),
    "close_debit_card_4721": ("discoverable", W, IRREV),
    "clear_debit_card_fraud_alert_4892": ("discoverable", W, IRREV),
    "reset_debit_card_pin_6284": ("discoverable", W, IRREV),
    "request_temporary_debit_card_limit_increase_8374": ("discoverable", W, IRREV),
}

# tau2 tools deliberately left out of the wrapper, with the reason.
EXCLUDED: dict[str, str] = {
    "example_agent_tool_0000": "placeholder that does nothing",
    "list_discoverable_agent_tools": "dispatcher: the wrapper lists tools itself",
    "unlock_discoverable_agent_tool": "dispatcher: the wrapper has no unlock step",
    "call_discoverable_agent_tool": "dispatcher: tools are called directly",
    "give_discoverable_user_tool": "hands a tool to the simulated customer",
}


def _description(name: str) -> str:
    doc = getattr(KnowledgeTools, name).__doc__ or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    return first or name


def build_catalog() -> dict[str, ToolSpec]:
    """One ToolSpec per labelled tool, keyed by tool name."""
    return {
        name: ToolSpec(
            name=name,
            kind=kind,
            reversibility=reversibility,
            description=_description(name),
            source=source,
            args_model=build_args_model(name),
        )
        for name, (source, kind, reversibility) in _LABELS.items()
    }


def render_markdown() -> str:
    """The reviewable label table, as checked in at docs/banking-tool-catalog.md."""
    lines = [
        "# Banking tool catalog",
        "",
        "Generated by `banking_tools.catalog.render_markdown()`; do not edit by hand.",
        "",
        "| Tool | Source | Kind | Reversibility |",
        "|---|---|---|---|",
    ]
    for spec in build_catalog().values():
        rev = spec.reversibility.value if spec.reversibility else "-"
        lines.append(f"| `{spec.name}` | {spec.source} | {spec.kind.value} | {rev} |")
    lines += ["", "## Excluded", "", "| Tool | Reason |", "|---|---|"]
    lines += [f"| `{name}` | {why} |" for name, why in EXCLUDED.items()]
    return "\n".join(lines) + "\n"
