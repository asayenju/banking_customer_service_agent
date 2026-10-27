import pytest
from pydantic import ValidationError

from banking_tools import ToolKind, get_registry
from orchestrator.plan import (
    READ_ONLY_KINDS,
    URGENT_KINDS,
    Plan,
    PlanRejected,
    Waiting,
    plan,
)
from orchestrator.taxonomy import TOOL_INTENT
from orchestrator.types import Conflict, Intent, IntentKind

K = IntentKind


def I(id, kind=K.LOOKUP_INFO, slots=None, depends_on=()):
    return Intent(id=id, kind=kind, summary="s", slots=slots or {}, depends_on=depends_on)


def card(id, kind=K.CLOSE_CARD, card_id="c1", depends_on=()):
    return I(id, kind, {"card_id": card_id}, depends_on)


def waves(result):
    assert isinstance(result, Plan), result
    return result.waves


def test_independent_reads_share_one_wave_in_input_order():
    assert waves(plan([I("a"), I("b"), I("c", K.ACCOUNT_BALANCE_OR_HISTORY)])) == (("a", "b", "c"),)


def test_a_dependency_puts_the_dependent_in_a_later_wave():
    assert waves(plan([I("a"), I("b", depends_on=("a",))])) == (("a",), ("b",))


def test_a_chain_runs_one_step_per_wave():
    result = plan([I("a"), I("b", depends_on=("a",)), I("c", depends_on=("b",))])
    assert waves(result) == (("a",), ("b",), ("c",))


def test_a_diamond_runs_the_middle_in_parallel():
    result = plan(
        [
            I("a"),
            I("b", depends_on=("a",)),
            I("c", depends_on=("a",)),
            I("d", depends_on=("b", "c")),
        ]
    )
    assert waves(result) == (("a",), ("b", "c"), ("d",))


def test_dependencies_are_honoured_even_when_listed_after_the_dependent():
    assert waves(plan([I("b", depends_on=("a",)), I("a")])) == (("a",), ("b",))


def test_two_writes_on_the_same_card_are_serialized_in_input_order():
    result = plan([card("a", K.FREEZE_CARD), card("b", K.CLOSE_CARD)])
    assert waves(result) == (("a",), ("b",))


def test_two_writes_on_different_known_cards_run_together():
    result = plan([card("a", K.CLOSE_CARD, "c1"), card("b", K.CLOSE_CARD, "c2")])
    assert waves(result) == (("a", "b"),)


def test_a_write_with_no_resource_id_is_serialized_against_every_other_write():
    result = plan([I("a", K.CLOSE_CARD), card("b", K.CLOSE_CARD, "c2")])
    assert waves(result) == (("a",), ("b",))


def test_two_writes_with_no_resource_ids_are_serialized():
    assert waves(plan([I("a", K.CLOSE_CARD), I("b", K.TRANSFER_FUNDS)])) == (("a",), ("b",))


def test_a_user_id_alone_does_not_identify_a_resource():
    a = I("a", K.CLOSE_CARD, {"user_id": "u1"})
    b = I("b", K.TRANSFER_FUNDS, {"user_id": "u1"})
    assert waves(plan([a, b])) == (("a",), ("b",))


def test_any_shared_id_is_enough_to_conflict():
    a = I("a", K.TRANSFER_FUNDS, {"source_account_id": "x1", "destination_account_id": "x2"})
    b = I("b", K.CLOSE_ACCOUNT, {"account_id": "x2"})
    assert waves(plan([a, b])) == (("a",), ("b",))


def test_ids_are_compared_ignoring_case_and_spaces():
    result = plan([card("a", card_id="C1"), card("b", card_id=" c1 ")])
    assert waves(result) == (("a",), ("b",))


def test_a_read_never_conflicts_with_a_write():
    assert waves(plan([card("a", K.CLOSE_CARD), I("b", K.ACCOUNT_BALANCE_OR_HISTORY)])) == (
        ("a", "b"),
    )


def test_urgent_intents_go_before_independent_work():
    result = plan([I("a", K.ACCOUNT_BALANCE_OR_HISTORY), card("b", K.FREEZE_CARD)])
    assert waves(result) == (("b",), ("a",))


def test_a_fraud_report_is_urgent_too():
    result = plan([I("a"), I("b", K.REPORT_FRAUD_OR_LOSS, {"card_id": "c1"})])
    assert waves(result) == (("b",), ("a",))


def test_urgent_intents_that_conflict_keep_input_order():
    first = I("a", K.REPORT_FRAUD_OR_LOSS, {"card_id": "c1"})
    second = card("b", K.FREEZE_CARD, "c1")
    assert waves(plan([first, second])) == (("a",), ("b",))
    assert waves(plan([second, first])) == (("b",), ("a",))


def test_urgent_does_not_jump_its_own_dependency():
    result = plan([I("r"), card("f", K.FREEZE_CARD, depends_on=("r",)), I("x")])
    assert waves(result) == (("r", "x"), ("f",))


def test_an_intent_whose_dependency_is_not_running_waits():
    result = plan([I("a"), I("b", depends_on=("gone",))])
    assert isinstance(result, Plan)
    assert result.waves == (("a",),)
    assert result.waiting == (Waiting(intent_id="b", blocked_by=("gone",)),)


def test_waiting_spreads_to_everything_downstream():
    result = plan([I("b", depends_on=("gone",)), I("c", depends_on=("b",)), I("a")])
    assert isinstance(result, Plan)
    assert result.waves == (("a",),)
    assert result.waiting == (
        Waiting(intent_id="b", blocked_by=("gone",)),
        Waiting(intent_id="c", blocked_by=("b",)),
    )


def test_a_dependency_finished_earlier_is_satisfied():
    result = plan([I("b", depends_on=("a",))], done=frozenset({"a"}))
    assert waves(result) == (("b",),)


@pytest.mark.parametrize(
    "intents",
    [
        [I("a", depends_on=("b",)), I("b", depends_on=("a",))],
        [I("a", depends_on=("a",))],
        [I("a"), I("b", depends_on=("c",)), I("c", depends_on=("d",)), I("d", depends_on=("b",))],
        [I("a", depends_on=("gone", "b")), I("b", depends_on=("a",))],
    ],
)
def test_a_dependency_cycle_rejects_the_whole_plan(intents):
    result = plan(intents)
    assert isinstance(result, PlanRejected)
    assert result.reason == "dependency cycle"


def test_a_rejection_names_the_intents_on_the_cycle():
    result = plan([I("ok"), I("a", depends_on=("b",)), I("b", depends_on=("a",))])
    assert isinstance(result, PlanRejected) and result.intent_ids == ("a", "b")


def test_an_intent_downstream_of_a_cycle_is_not_named_as_part_of_it():
    cycle = [I("a", depends_on=("b",)), I("b", depends_on=("a",))]
    result = plan([*cycle, I("after", depends_on=("a",))])
    assert isinstance(result, PlanRejected) and result.intent_ids == ("a", "b")


def test_a_self_dependency_is_a_cycle_of_one():
    result = plan([I("a", depends_on=("a",)), I("b")])
    assert isinstance(result, PlanRejected) and result.intent_ids == ("a",)


def test_a_conflict_flagged_by_the_decomposer_serializes_even_two_reads():
    flagged = [Conflict(intent_ids=("a", "b"), reason="same thing twice")]
    assert waves(plan([I("a"), I("b")], conflicts=flagged)) == (("a",), ("b",))


def test_nothing_to_run_is_an_empty_plan():
    assert plan([]) == Plan(waves=(), waiting=())


def test_every_intent_is_planned_or_waiting_exactly_once():
    intents = [
        I("a"),
        card("b", K.CLOSE_CARD),
        card("c", K.FREEZE_CARD, depends_on=("a",)),
        I("d", depends_on=("gone",)),
        I("e", K.TRANSFER_FUNDS, {"account_id": "x"}),
    ]
    result = plan(intents)
    assert isinstance(result, Plan)
    seen = [i for wave in result.waves for i in wave] + [w.intent_id for w in result.waiting]
    assert sorted(seen) == ["a", "b", "c", "d", "e"]


def test_the_same_input_gives_the_same_plan():
    intents = [I("a"), card("b"), card("c", K.FREEZE_CARD), I("d", depends_on=("a",))]
    assert plan(intents) == plan(intents)


def test_the_read_only_list_agrees_with_the_tool_catalog():
    writes = {
        TOOL_INTENT[name]
        for name, spec in get_registry().items()
        if spec.kind is ToolKind.WRITE
    }
    assert not (READ_ONLY_KINDS & writes), "a kind with a write tool is listed as read-only"


def test_urgent_kinds_are_the_protective_ones():
    assert URGENT_KINDS == {K.REPORT_FRAUD_OR_LOSS, K.FREEZE_CARD}


def test_plan_models_are_frozen_and_strict():
    with pytest.raises(ValidationError):
        Plan(waves=(), waiting=()).waves = (("a",),)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlanRejected(reason="r", intent_ids=(), surprise="x")
