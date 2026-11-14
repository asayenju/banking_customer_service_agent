"""The same question is never shown twice the same way (the Carlos loop)."""

from scenario_kit import GOOD_CLAIMS, I, S, World, interp

from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.orchestrator import MAX_ASKS
from orchestrator.repair import numbered
from orchestrator.types import IntentKind as K
from orchestrator.types import Question

LIST = "Which card do you mean?\n- the card ending 3847 (active)\n- the card ending 8165 (active)"


class AsksWhichCard:
    async def run(self, intent, ctx):
        q = Question(intent_id=intent.id, slot="card_id", text=LIST)
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(q,))


def close(**slots):
    return I("a", K.GENERAL_QUESTION, slots)


def loop():
    answers = [interp(close(), claims=GOOD_CLAIMS)] + [interp(close(card_id=f"hmm {n}")) for n in range(MAX_ASKS)]
    w = World(answers, [[S("a", 0.99)]] * (MAX_ASKS + 1), sub=AsksWhichCard())
    return [w.turn("cancel some debit cards")] + [w.turn(f"hmm {n}") for n in range(MAX_ASKS)]


def test_no_question_text_is_shown_twice_and_the_steps_come_in_order():
    results = loop()
    shown = [r.reply.questions[0].text for r in results[:-1]]
    assert len(set(shown)) == len(shown) == MAX_ASKS
    assert shown[0] == LIST
    assert "reply with just a number" in shown[1] and "1. the card ending 3847" in shown[1] and "2. the card ending 8165" in shown[1]
    assert "person" in shown[2]


def test_a_person_only_comes_after_the_repairs_failed():
    results = loop()
    assert [r.decisions[0].action.value for r in results] == ["ask"] * MAX_ASKS + ["escalate"]


def test_numbering_a_question_without_a_list_rewords_it():
    assert numbered("What is the card id for: close it?").startswith("Sorry")
