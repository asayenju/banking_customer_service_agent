"""A five-question live check of the general answerer, on the same cost controls as the pilot.

Four ordinary product questions should each get a short answer that cites documents the
answerer really retrieved; one question needs the customer's own account, which the
answerer has no access to, so it should hand over to a person. Anything else is a failure
worth reading.

Same rules as ``agent.live``: nothing is sent without ``--yes`` after an estimate, a dollar
cap is enforced on real usage, and every conversation is recorded so a second run is free.

Run with ``uv run python -m agent.spot`` (prints the estimate), then add ``--yes``.
"""

import argparse
import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.live import (
    DEFAULT_CASSETTES,
    BudgetExceeded,
    CassetteClient,
    CassetteMismatch,
    LiveRunRefused,
    MeteredClient,
    cassette_path,
    estimate_task,
    price_for,
)
from agent.llm_tools import ToolCallingLLMClient
from agent.specialist import Specialist
from agent.specialists.general import GENERAL_ANSWERER
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from knowledge.index import KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind

DEFAULT_CAP = 0.30
_CITED = re.compile(r"\[(doc_[^\]]+)\]")


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    expect: str  # "answer" or "hand_off"


QUESTIONS: tuple[Question, ...] = (
    Question("blue-fee", "What is the monthly fee on the Blue checking account, and how can I avoid it?", "answer"),
    Question("foreign-fee", "Does the Gold Rewards credit card charge foreign transaction fees?", "answer"),
    Question("overdraft", "What is the overdraft fee on a Green checking account?", "answer"),
    Question("subscription", "What are the benefits of a Rho-Bank+ subscription?", "answer"),
    Question("my-balance", "What is my current checking balance?", "hand_off"),
)


@dataclass(frozen=True)
class SpotResult:
    key: str
    ok: bool
    outcome: str
    cited: tuple[str, ...]
    dollars: float
    note: str = ""


def estimate_dollars(price_in: float, price_out: float) -> float:
    calls, tokens_in, tokens_out = estimate_task(1, 2)
    return round(len(QUESTIONS) * (tokens_in * price_in + tokens_out * price_out) / 1_000_000, 4)


def run_spot(
    make_client: Callable[[str], ToolCallingLLMClient],
    index: KnowledgeIndex,
    *,
    cassettes: Path,
    model: str,
    price_in: float,
    price_out: float,
    cap: float,
    rerecord: bool = False,
) -> list[SpotResult]:
    results: list[SpotResult] = []
    spent = 0.0
    for number, question in enumerate(QUESTIONS):
        path = cassette_path(cassettes, "spot", number, model)
        fresh = rerecord or not path.exists()
        if spent >= cap:
            results.append(SpotResult(question.key, False, "skipped", (), 0.0, "cost cap reached"))
            continue
        meter = MeteredClient(make_client(model), price_in=price_in, price_out=price_out, cap=cap - spent) if fresh else None
        client = CassetteClient(meter, path, rerecord=rerecord)
        shared = BankingSession()
        specialist = Specialist(GENERAL_ANSWERER, llm=client, index=index, database=shared.database, vault=SecretVault())
        intent = Intent(id=f"spot-{number}", kind=IntentKind.GENERAL_QUESTION, summary=question.text)
        context = RouteContext(intent_id=intent.id, verified_user_id=None, session=scoped(shared)(SessionRequest(intent.id)), mode="plan")
        try:
            outcome = asyncio.run(specialist.run(intent, context))
            client.save()
        except (BudgetExceeded, CassetteMismatch) as exc:
            results.append(SpotResult(question.key, False, "stopped", (), meter.dollars if meter else 0.0, f"{type(exc).__name__}: {exc}"))
            spent += meter.dollars if meter else 0.0
            continue
        dollars = meter.dollars if meter else 0.0
        spent += dollars
        cited = tuple(_CITED.findall(outcome.text))
        if question.expect == "answer":
            ok = outcome.status is OutcomeStatus.DONE and bool(cited)
            note = "" if ok else f"expected a cited answer, got {outcome.status.value} {outcome.error or ''}".strip()
        else:
            ok = outcome.status is OutcomeStatus.FAILED and "human agent" in (outcome.error or "")
            note = "" if ok else f"expected a hand-off, got {outcome.status.value}"
        results.append(SpotResult(question.key, ok, outcome.status.value, cited, dollars, note))
    return results


def render(results: Sequence[SpotResult]) -> str:
    lines = [f"{sum(r.ok for r in results)}/{len(results)} as expected, ${sum(r.dollars for r in results):.4f} spent"]
    for r in results:
        lines.append(f"- {r.key}: {'ok' if r.ok else 'NOT AS EXPECTED'} ({r.outcome}; cites {len(r.cited)}){' ' + r.note if r.note else ''}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin wrapper over tested parts
    import sys

    from agent.llm_tools import LiteLLMToolClient
    from knowledge.index import get_index
    from orchestrator.config import agent_model

    parser = argparse.ArgumentParser(prog="agent.spot", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--price-in", type=float, default=None)
    parser.add_argument("--price-out", type=float, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_CAP)
    parser.add_argument("--cassettes", type=Path, default=DEFAULT_CASSETTES)
    parser.add_argument("--rerecord", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    model = args.model or agent_model()
    try:
        price_in, price_out = price_for(model, args.price_in, args.price_out)
        need = estimate_dollars(price_in, price_out)
        unrecorded = [n for n in range(len(QUESTIONS)) if args.rerecord or not cassette_path(args.cassettes, "spot", n, model).exists()]
        if unrecorded and need > args.max_cost_usd:
            raise LiveRunRefused(f"estimated ${need:.2f} is over the cap of ${args.max_cost_usd:.2f}: raise --max-cost-usd")
        if unrecorded and not args.yes:
            raise LiveRunRefused(f"this would spend about ${need:.2f} on {len(unrecorded)} question(s); re-run with --yes to go ahead")
    except LiveRunRefused as refusal:
        print(f"not started: {refusal}")
        return 2
    results = run_spot(lambda m: LiteLLMToolClient(m), get_index(), cassettes=args.cassettes, model=model, price_in=price_in, price_out=price_out, cap=args.max_cost_usd, rerecord=args.rerecord)
    print(render(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
