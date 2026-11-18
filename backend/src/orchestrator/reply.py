"""One message to the customer for the whole turn.

Whatever happened to however many intents, the customer gets a single reply:
what got done, what went wrong (with an offer of a human), what a human is
taking over, what is waiting on something else, and last, one question (the
first of whatever is missing) so the thing they have to answer comes at the end.

The wording here is a plain template so tests are exact. An LLM can phrase it
later behind the same structure. Internal reasons and error strings never reach
the customer: they stay in the trace.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict

from orchestrator.auth import IDENTITY_SLOT
from orchestrator.conflicts import heads_up
from orchestrator.capabilities import LABEL
from orchestrator.decision import CLARIFY_SLOT, CONFIRM_SLOT, Action, IntentDecision
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.plain import plain_text
from orchestrator.plan import Waiting
from orchestrator.types import Choice, Intent, Question

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
    # Buttons the chat may show next to lines of the text (see ``types.Choice``).
    choices: tuple[Choice, ...] = ()


def _summary(intents: Mapping[str, Intent], intent_id: str) -> str:
    intent = intents.get(intent_id)
    return intent.summary if intent is not None else "that request"


def _already_answered(intents: Mapping[str, Intent], question: Question) -> bool:
    intent = intents.get(question.intent_id)
    return intent is not None and bool(intent.slots.get(question.slot, "").strip())


_NOTE_TO_SELF = re.compile(r"\b(?:the )?customer\b|\blikely\b|\bappears?\b|\bseems?\b|\breferenc|\bpresumably\b|\bpossibly\b", re.IGNORECASE)


def _speakable(summary: str) -> bool:
    """A summary that reads as the request itself ("sort out my fees"), short and in the customer's terms, and not as a
    model's note about the customer ("The customer is referencing a charge, likely in relation to...")."""
    return bool(summary) and len(summary) <= 100 and not _NOTE_TO_SELF.search(summary)


def wording(intents: Mapping[str, Intent], question: Question) -> str:
    what = _summary(intents, question.intent_id).strip().rstrip(".?! ")
    if question.slot == IDENTITY_SLOT:
        return IDENTITY_PROMPT
    if question.slot == CONFIRM_SLOT:
        return f"Please confirm: {what} (yes or no)?"
    if question.slot == CLARIFY_SLOT:
        if _speakable(what):
            return f"To be sure I have this right: {what}. Is that what you want? If not, tell me more."
        intent = intents.get(question.intent_id)
        if intent is not None and intent.kind in LABEL:
            # The model's note about the request is not for the customer's eyes ("The customer is referencing a charge,
            # likely in relation to..."): say it in words written here.
            return f"To be sure I have this right: you would like me to {LABEL[intent.kind]}. Is that what you want? If not, tell me more."
        return "I want to be sure I understand. What would you like me to help with?"
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
            # Text a route or the plan preview built is the exact wording to show. Everything else about the question
            # (its kind: "yes_no" tells the chat to offer Yes and No) is kept as it was.
            batch.append(question.model_copy(update={"text": question.text or wording(intents, question)}))
    return tuple(batch)


def _result_line(intents: Mapping[str, Intent], outcome: RouteOutcome) -> Optional[str]:
    what = _summary(intents, outcome.intent_id)
    if outcome.status is OutcomeStatus.DONE:
        return plain_text(outcome.text) if outcome.text else f"Done: {what}"
    if outcome.status is OutcomeStatus.PARTIAL:
        return f"I only finished part of this ({what}): {plain_text(outcome.text or '')}"
    return None


def compose_reply(
    intents: Mapping[str, Intent],
    *,
    outcomes: Sequence[RouteOutcome] = (),
    decisions: Sequence[IntentDecision] = (),
    waiting: Sequence[Waiting] = (),
    one_at_a_time: bool = True,
) -> Reply:
    lines: list[str] = []
    for outcome in outcomes:
        line = _result_line(intents, outcome)
        if line:
            lines.append(line)

    failed_ids = tuple(
        o.intent_id
        for o in outcomes
        if o.status
        not in (OutcomeStatus.DONE, OutcomeStatus.NEEDS_INFO, OutcomeStatus.NEEDS_CONFIRMATION)
    )
    if failed_ids:
        problems = "; ".join(_summary(intents, i) for i in failed_ids)
        lines.append(
            f"I ran into a problem with: {problems}. I can connect you with a human agent for that."
        )

    escalated_ids = tuple(d.intent_id for d in decisions if d.action is Action.ESCALATE)
    if escalated_ids:
        handed = "; ".join(_summary(intents, i) for i in escalated_ids)
        lines.append(f"I'm handing this to a human agent: {handed}.")

    deferred_ids = tuple(d.intent_id for d in decisions if d.action is Action.DEFER)
    if deferred_ids:
        lines.append(
            "I have noted that you want to speak to a person. Your cards come first, so I will finish protecting "
            "them, and then connect you."
        )

    for hold in waiting:
        blockers = " and ".join(
            _summary(intents, b) if b in intents else EARLIER_STEP for b in hold.blocked_by
        )
        lines.append(f"Still to do: {_summary(intents, hold.intent_id)} once {blockers} is done.")

    questions = _batch(intents, decisions)
    if not questions and not lines:
        # A route asked again for a detail the customer has already given words for. Saying nothing would leave them
        # stuck on "anything I can help with?", so the question is asked again as it was.
        again = next((q for d in decisions if d.action is Action.ASK for q in d.questions if q.slot not in (IDENTITY_SLOT,)), None)
        if again is not None:
            questions = (again.model_copy(update={"text": again.text or wording(intents, again)}),)
    if one_at_a_time:
        # Never a list of questions: the first missing detail now. Whatever else is missing is asked again, one at a
        # time, when the request runs again with this answer (a route recomputes what it still needs each time).
        # A yes to what is about to be done is not a question to spread out: every one of those stays.
        details = [q for q in questions if q.slot != CONFIRM_SLOT]
        confirms = [q for q in questions if q.slot == CONFIRM_SLOT]
        if confirms:
            # What is about to be done comes first, and alone: a plan and a question in one reply leave the customer (and the
            # model reading their answer) unsure which one a yes or a date belongs to. The question waits its turn.
            questions = tuple(confirms)
        else:
            # A concrete missing detail for a request that can move comes before a "what do you mean?" about one that cannot.
            details.sort(key=lambda q: q.slot == CLARIFY_SLOT)
            questions = tuple(details[:1])
    if questions:
        planned = {intents[q.intent_id].kind for q in questions if q.slot == CONFIRM_SLOT and q.intent_id in intents}
        waiting_kinds = {
            intents[d.intent_id].kind
            for d in decisions
            if d.action is Action.ASK and d.intent_id in intents and not any(q.slot == CONFIRM_SLOT for q in d.questions)
        }
        lines.extend(heads_up(planned, waiting_kinds))  # what the plan on screen will do to something else they asked for
        lines.append("To go on, I need a little more information:")
        lines.extend(f"- {q.text}" for q in questions)

    return Reply(
        text="\n".join(lines) if lines else FALLBACK,
        questions=questions,
        escalated_intent_ids=escalated_ids,
        failed_intent_ids=failed_ids,
        choices=tuple(c for outcome in outcomes for c in outcome.choices),
    )
