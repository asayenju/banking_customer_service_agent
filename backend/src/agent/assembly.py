"""The whole system, wired: one customer conversation with every specialist behind the orchestrator.

``build_conversation`` is the only place the parts meet. For one conversation it creates:

  - a ``BankingSession`` (its own copy of the database; sessions never share state);
  - one ``SecretVault``, shared by every specialist and by the orchestrator's secret capture, so a PIN typed
    to one is swapped for a reference before any model sees it and is resolved only when a plan runs;
  - the six specialists (debit, credit, accounts, fees, fraud, general), each a configuration of
    ``Specialist`` over the same database, behind the single ``SubAgentDispatcher``;
  - the direct recipes (freeze, unfreeze, profile, balances, lookups, greeting) for the DIRECT route;
  - containment, so the cards a lost-wallet report names are frozen before its plan;
  - a directory over the database's own customers for the 2-of-4 identity check.

The decomposer, the scorer and the tool-calling model are passed in, so tests and the live app use the same
wiring and differ only in those three. ``live_conversation`` supplies the real ones (keys from ``.env``).

The HAND_OFF route has no runner on purpose: a hand-off kind never runs, it escalates (``decision.decide``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent.containment import WalletContainment
from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.llm_tools import ToolCallingLLMClient
from agent.recipes.direct import DirectRunner
from agent.specialist import Specialist, SpecialistConfig
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.specialists.credit import CREDIT_SPECIALIST
from agent.specialists.debit import DEBIT_SPECIALIST
from agent.specialists.fees import FEES_SPECIALIST
from agent.specialists.fraud import FRAUD_SPECIALIST
from agent.specialists.general import GENERAL_ANSWERER
from agent.toolbox import SecretVault
from banking_tools.session import BankingSession
from knowledge.index import KnowledgeIndex
from orchestrator.auth import CustomerRecord, Directory, IdentityField, claim_matches
from orchestrator.context import scoped
from orchestrator.decision_log import DecisionLog, JsonlLog
from orchestrator.decomposer import Decomposer
from orchestrator.executor import ROUTE_TIMEOUTS
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.scoring import Scorer

# Every specialist the dispatcher can name, and the configuration that is it.
SPECIALISTS: Mapping[Name, SpecialistConfig] = {
    Name.DEBIT: DEBIT_SPECIALIST,
    Name.CREDIT: CREDIT_SPECIALIST,
    Name.ACCOUNTS: ACCOUNTS_SPECIALIST,
    Name.FEES: FEES_SPECIALIST,
    Name.FRAUD: FRAUD_SPECIALIST,
    Name.GENERAL: GENERAL_ANSWERER,
}


class DatabaseDirectory:
    """Finds the customer a caller claims to be among the database's own, by user id, email or phone.

    It only names a candidate; ``orchestrator.auth.verify`` still needs two matching fields, so a guessed
    email alone verifies nobody."""

    def __init__(self, database: Any) -> None:
        self._db = database

    def _record(self, row: Mapping[str, Any]) -> CustomerRecord:
        return CustomerRecord(
            user_id=str(row["user_id"]),
            date_of_birth=str(row.get("date_of_birth") or ""),
            email=str(row.get("email") or ""),
            phone_number=str(row.get("phone_number") or ""),
            address=str(row.get("address") or ""),
        )

    async def find(self, claims: Mapping[str, str]) -> Optional[CustomerRecord]:
        users = [u for u in self._db.users.data.values() if isinstance(u, Mapping) and u.get("user_id")]
        user_id = claims.get("user_id", "").strip().casefold()
        for row in users:
            record = self._record(row)
            if user_id and user_id == record.user_id.casefold():
                return record
            for field in (IdentityField.EMAIL, IdentityField.PHONE_NUMBER):
                claimed = claims.get(field.value, "").strip()
                if claimed and claim_matches(field, claimed, record):
                    return record
        return None


@dataclass(frozen=True)
class Conversation:
    """One customer's conversation: send each message to ``orchestrator.handle_turn``."""

    orchestrator: Orchestrator
    session: BankingSession
    vault: SecretVault
    dispatcher: SubAgentDispatcher
    specialists: Mapping[Name, Specialist]


# Passive record of every decision from real use of the demo (git-ignored); the calibration report reads it.
DECISION_LOG_PATH = Path(__file__).resolve().parents[2] / "evals" / "decisions.jsonl"


def build_conversation(
    *,
    decomposer: Decomposer,
    scorer: Scorer,
    llm: ToolCallingLLMClient,
    index: KnowledgeIndex,
    session: Optional[BankingSession] = None,
    directory: Optional[Directory] = None,
    route_timeouts: Optional[Mapping[Route, float]] = None,
    decision_log: Optional[DecisionLog] = None,
) -> Conversation:
    shared = session if session is not None else BankingSession()
    vault = SecretVault()
    specialists = {
        name: Specialist(config, llm=llm, index=index, database=shared.database, vault=vault)
        for name, config in SPECIALISTS.items()
    }
    dispatcher = SubAgentDispatcher(specialists, RecordKinds(shared.database))
    orchestrator = Orchestrator(
        decomposer=decomposer,
        scorer=scorer,
        directory=directory if directory is not None else DatabaseDirectory(shared.database),
        runners={Route.DIRECT: DirectRunner(shared.database), Route.SUB_AGENT: dispatcher},
        route_timeouts=route_timeouts or ROUTE_TIMEOUTS,
        session_for=scoped(shared),
        capture_secrets=vault.capture,
        containment=WalletContainment(shared.database),
        products_for=lambda user: product_summary(shared.database, user),
        decision_log=decision_log,
    )
    return Conversation(orchestrator, shared, vault, dispatcher, specialists)


def product_summary(db: Any, user_id: str) -> str:
    """The kinds of product a customer holds, in a line for Jev: types and levels only, never ids, balances or digits."""
    debit = sorted({str(a.get("level")) for a in db.accounts.data.values() if a.get("user_id") == user_id and a.get("class") == "checking"})
    savings = sorted({str(a.get("level")) for a in db.accounts.data.values() if a.get("user_id") == user_id and a.get("class") != "checking"})
    credit = sorted({str(a.get("card_type")) for a in db.credit_card_accounts.data.values() if a.get("user_id") == user_id})
    parts = [("checking accounts with debit cards", debit), ("savings accounts", savings), ("credit cards", credit)]
    return "; ".join(f"{label}: {', '.join(names)}" for label, names in parts if names)


def live_conversation(session: Optional[BankingSession] = None) -> Conversation:  # pragma: no cover - needs keys
    """The real decomposer, Jev scorer and tool-calling model (keys from the repo-root ``.env``)."""
    from agent.llm_tools import LiteLLMToolClient
    from knowledge.index import get_index
    from orchestrator.jev import make_jev_scorer
    from orchestrator.llm import LiteLLMClient
    from orchestrator.llm_decomposer import LLMDecomposer

    return build_conversation(
        decomposer=LLMDecomposer(LiteLLMClient()),
        scorer=make_jev_scorer(),
        llm=LiteLLMToolClient(),
        index=get_index(),
        session=session,
        decision_log=DecisionLog(JsonlLog(DECISION_LOG_PATH)),
    )
