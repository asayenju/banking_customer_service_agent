"""Which of the 97 tau2 banking tasks each specialist can be tested on, and which cannot.

The manifest is generated from ``tasks.json`` and committed as ``evals/manifest.json``;
a test regenerates it and fails on any drift, so the counts the PRD quotes cannot go
stale unnoticed.

A task's cluster set is the clusters of the agent-side write tools in its gold actions
(discoverable tools unwrapped, reads ignored). A task inside one cluster is "wholly"
that cluster's; one that spans two is graded on the DB-state gate too. Everything else
is excluded from that gate, each with a stated reason.

Regenerate with ``uv run python -m banking_tools.manifest``.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from banking_tools.catalog import get_registry
from banking_tools.types import ToolKind

ROOT = Path(__file__).resolve().parents[3]
TASKS_PATH = ROOT / "tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
MANIFEST_PATH = ROOT / "backend/evals/manifest.json"

_WRAPPER = "call_discoverable_agent_tool"
_HUMAN_TRANSFER = frozenset(
    {
        "transfer_to_human_agents",
        "initial_transfer_to_human_agent_0218",
        "initial_transfer_to_human_agent_1822",
        "emergency_credit_bureau_incident_transfer_1114",
    }
)

Cluster = Literal["debit", "credit", "accounts", "fees", "profile"]

CLUSTER_TOOLS: Mapping[Cluster, frozenset[str]] = {
    "debit": frozenset(
        {
            "freeze_debit_card_3892",
            "unfreeze_debit_card_3893",
            "close_debit_card_4721",
            "order_debit_card_5739",
            "activate_debit_card_8291",
            "activate_debit_card_8292",
            "activate_debit_card_8293",
            "change_debit_card_pin_6285",
            "reset_debit_card_pin_6284",
            "clear_debit_card_fraud_alert_4892",
            "set_debit_card_recurring_block_7382",
            "request_temporary_debit_card_limit_increase_8374",
            "file_debit_card_transaction_dispute_6281",
        }
    ),
    "credit": frozenset(
        {
            "file_credit_card_transaction_dispute_4829",
            "order_replacement_credit_card_7291",
            "log_credit_card_closure_reason_4521",
            "close_credit_card_account_7834",
            "apply_statement_credit_8472",
            "apply_credit_card_account_flag_6147",
            "submit_credit_limit_increase_request_7392",
            "approve_credit_limit_increase_5847",
            "deny_credit_limit_increase_5848",
        }
    ),
    "accounts": frozenset(
        {
            "open_bank_account_4821",
            "close_bank_account_7392",
            "transfer_funds_between_bank_accounts_7291",
            "pay_credit_card_from_checking_9182",
        }
    ),
    "fees": frozenset(
        {
            "apply_checking_account_credit_5829",
            "apply_savings_account_credit_6831",
            "submit_interest_discrepancy_report_7294",
            "update_transaction_rewards_3847",
        }
    ),
    "profile": frozenset({"change_user_email"}),
}
# Writes that belong to no specialist: identity logging is done by the orchestrator.
NON_CLUSTER_WRITES = frozenset({"log_verification"})

PILOT = "debit"
# Tasks whose correct outcome is a refusal or hand-off, or that try to talk the agent
# into one. The expected result is "no unsafe write".
TRAPS = frozenset(
    f"task_{n:03d}" for n in (8, 14, 16, 35, 49, 52, 99)
)

Category = Literal[
    "wholly_one_cluster",
    "spans_clusters",
    "human_transfer_only",
    "customer_side_or_reads_only",
]


class TaskEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    basis: Literal["DB", "ACTION"]
    category: Category
    clusters: tuple[Cluster, ...]
    has_overlay: bool
    human_transfer: bool
    trap: bool
    pilot: bool
    # Why the task is outside the DB-state gate; None when it is gradeable.
    excluded_reason: Optional[str]


def _cluster_of(tool: str) -> Optional[Cluster]:
    for cluster, tools in CLUSTER_TOOLS.items():
        if tool in tools:
            return cluster
    return None


def _agent_tools(task: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
        if action.get("requestor") != "assistant":
            continue
        name = action["name"]
        if name == _WRAPPER:
            name = (action.get("arguments") or {}).get("agent_tool_name", "")
        names.append(name)
    return names


def classify(task: Mapping[str, Any]) -> TaskEntry:
    task_id = str(task["id"])
    tools = _agent_tools(task)
    clusters = tuple(sorted({c for t in tools if (c := _cluster_of(t))}))
    basis: Literal["DB", "ACTION"] = (
        "ACTION" if (task.get("evaluation_criteria") or {}).get("reward_basis") == ["ACTION"] else "DB"
    )
    overlay = bool(((task.get("initial_state") or {}).get("initialization_data") or {}))
    human = any(t in _HUMAN_TRANSFER for t in tools)

    category: Category
    reason: Optional[str] = None
    if not clusters:
        category = "human_transfer_only" if human else "customer_side_or_reads_only"
        reason = (
            "ACTION basis: assert the hand-off, the transfer tool and no domain write"
            if human
            else "no agent-side write: customer-side tools or reads only (staged simulation)"
        )
    elif len(clusters) == 1:
        category = "wholly_one_cluster"
    else:
        category = "spans_clusters"
    if task_id == "task_005":
        reason = "anomalous gold: the supervisor bypass code cannot be reproduced honestly"
    elif task_id == "task_083":
        reason = (
            "ACTION basis, and the gold's dispute calls omit customer_max_liability_amount, "
            "which the bank's tool requires: a typed call cannot reproduce it"
        )
    elif basis == "ACTION" and reason is None:
        reason = "ACTION basis: compare tool calls, not the database"
    return TaskEntry(
        task_id=task_id,
        basis=basis,
        category=category,
        clusters=clusters,
        has_overlay=overlay,
        human_transfer=human,
        trap=task_id in TRAPS,
        pilot=clusters == (PILOT,),
        excluded_reason=reason,
    )


def build_manifest(tasks: Sequence[Mapping[str, Any]]) -> list[TaskEntry]:
    return [classify(t) for t in tasks]


def uncovered_write_tools() -> list[str]:
    """Catalog writes that are in no cluster: a new tau2 tool must be placed on purpose."""
    covered = set().union(*CLUSTER_TOOLS.values()) | NON_CLUSTER_WRITES
    return sorted(
        name for name, spec in get_registry().items() if spec.kind is ToolKind.WRITE and name not in covered
    )


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = json.loads(path.read_text())
    return tasks


def render(entries: Sequence[TaskEntry]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in entries], indent=1) + "\n"


def load_manifest(path: Path = MANIFEST_PATH) -> list[TaskEntry]:
    return [TaskEntry.model_validate(row) for row in json.loads(path.read_text())]


def main() -> None:
    MANIFEST_PATH.write_text(render(build_manifest(load_tasks())))
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
