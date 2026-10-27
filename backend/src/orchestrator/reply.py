"""One message to the customer for the whole turn.

Whatever happened to however many intents, the customer gets a single reply:
what got done, what went wrong (with an offer of a human), what a human is
taking over, what is waiting on something else, and last, one batched set of
questions so the thing they have to answer comes at the end.

The wording here is a plain template so tests are exact. An LLM can phrase it
later behind the same structure. Internal reasons and error strings never reach
the customer: they stay in the trace.
"""

from collections.abc import Mapping, Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict

from orchestrator.auth import IDENTITY_SLOT
from orchestrator.decision import CLARIFY_SLOT, CONFIRM_SLOT, Action, IntentDecision
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.plan import Waiting
from orchestrator.types import Intent, Question

FALLBACK = "Is there anything I can help you with?"
IDENTITY_PROMPT = (
    "To verify your identity, please give me any two of: date of birth, email, "
    "phone number, or address on the account."
)
EARLIER_STEP = "the earlier step"


class Reply(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    text: str
    # Deduplicated, each tied to an intent and carrying its wording.
    questions: tuple[Question, ...]
    escalated_intent_ids: tuple[str, ...]
    failed_intent_ids: tuple[str, ...]


def _summary(intents: Mapping[str, Intent], intent_id: str) -> str:
    intent = intents.get(intent_id)
    return intent.summary if intent is not None else "that request"


def _already_answered(intents: Mapping[str, Intent], question: Question) -> bool:
    intent = intents.get(question.intent_id)
    return intent is not None and bool(intent.slots.get(question.slot, "").strip())


def _wording(intents: Mapping[str, Intent], question: Question) -> str:
    what = _summary(intents, question.intent_id)
    if question.slot == IDENTITY_SLOT:
        return IDENTITY_PROMPT
    if question.slot == CONFIRM_SLOT:
        return f"Please confirm: {what} (yes or no)?"
    if question.slot == CLARIFY_SLOT:
        return f"Can you tell me more about: {what}?"
    return f"What is the {question.slot.replace('_', ' ')} for: {what}?"


def _batch(
    intents: Mapping[str, Intent], decisions: Sequence[IntentDecision]
) -> tuple[Question, ...]:
    seen: set[tuple[Optional[str], str]] = set()
    batch: list[Question] = []
    for decision in decisions:
        if decision.action is not Action.ASK:
            continue
        for question in decision.questions:
            if _already_answered(intents, question):
                continue
            # One identity question serves every blocked intent.
            key = (None if question.slot == IDENTITY_SLOT else question.intent_id, question.slot)
            if key in seen:
                continue
            seen.add(key)
            batch.append(
                Question(
                    intent_id=question.intent_id,
                    slot=question.slot,
                    text=_wording(intents, question),
                )
            )
    return tuple(batch)


def _result_line(intents: Mapping[str, Intent], outcome: RouteOutcome) -> Optional[str]:
    what = _summary(intents, outcome.intent_id)
    if outcome.status is OutcomeStatus.DONE:
        return outcome.text or f"Done: {what}"
    if outcome.status is OutcomeStatus.PARTIAL:
        return f"I only finished part of this ({what}): {outcome.text}"
    return None


def compose_reply(
    intents: Mapping[str, Intent],
    *,
    outcomes: Sequence[RouteOutcome] = (),
    decisions: Sequence[IntentDecision] = (),
    waiting: Sequence[Waiting] = (),
) -> Reply:
    lines: list[str] = []
    for outcome in outcomes:
        line = _result_line(intents, outcome)
        if line:
            lines.append(line)

    failed_ids = tuple(o.intent_id for o in outcomes if o.status is not OutcomeStatus.DONE)
    if failed_ids:
        problems = "; ".join(_summary(intents, i) for i in failed_ids)
        lines.append(
            f"I ran into a problem with: {problems}. I can connect you with a human agent for that."
        )

    escalated_ids = tuple(d.intent_id for d in decisions if d.action is Action.ESCALATE)
    if escalated_ids:
        handed = "; ".join(_summary(intents, i) for i in escalated_ids)
        lines.append(f"I'm handing this to a human agent: {handed}.")

    for hold in waiting:
        blockers = " and ".join(
            _summary(intents, b) if b in intents else EARLIER_STEP for b in hold.blocked_by
        )
        lines.append(f"Still to do: {_summary(intents, hold.intent_id)} once {blockers} is done.")

    questions = _batch(intents, decisions)
    if questions:
        lines.append("To go on, I need a little more information:")
        lines.extend(f"- {q.text}" for q in questions)

    return Reply(
        text="\n".join(lines) if lines else FALLBACK,
        questions=questions,
        escalated_intent_ids=escalated_ids,
        failed_intent_ids=failed_ids,
    )
