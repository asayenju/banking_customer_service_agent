"""A task's starting database: the pristine DB with the task's overlay applied.

tau2 builds each task's world by merging ``initial_state.initialization_data``
into the base database (``agent_data`` first, then ``user_data``, both into the
one shared database) with ``update_pydantic_model_with_dict``. This module uses
that same function so a session started from a task has the hash tau2 gets.

Tasks may also carry ``initialization_actions``. None of the banking tasks do, so
rather than silently diverging from tau2 we refuse them.
"""

from collections.abc import Mapping
from typing import Any

from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.utils.pydantic_utils import update_pydantic_model_with_dict


def apply_initial_state(db: TransactionalDB, task: Mapping[str, Any]) -> TransactionalDB:
    """Return ``db`` with the task's overlay merged in. ``db`` itself is not changed."""
    state = task.get("initial_state") or {}
    if state.get("initialization_actions"):
        raise ValueError(
            f"{task.get('id')}: initialization_actions are not supported "
            "(tau2 would run them, and we do not)"
        )
    data = state.get("initialization_data") or {}
    for key in ("agent_data", "user_data"):
        overlay = data.get(key)
        if overlay:
            db = update_pydantic_model_with_dict(db, overlay)
    return db
