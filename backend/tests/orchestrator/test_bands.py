import pytest

from orchestrator.bands import ASK_AT, PROCEED_AT, Band, band


def test_thresholds_are_the_agreed_numbers():
    assert (PROCEED_AT, ASK_AT) == (0.85, 0.65)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (1.0, Band.PROCEED),
        (0.99, Band.PROCEED),
        (0.85, Band.PROCEED),
        (0.8499, Band.ASK),
        (0.75, Band.ASK),
        (0.65, Band.ASK),
        (0.6499, Band.ESCALATE),
        (0.5, Band.ESCALATE),
        (0.0, Band.ESCALATE),
    ],
)
def test_band_boundaries(confidence, expected):
    assert band(confidence) is expected


@pytest.mark.parametrize("garbage", [-0.1, 1.0001, 85.0, float("nan"), float("inf")])
def test_a_confidence_outside_zero_to_one_never_proceeds(garbage):
    assert band(garbage) is Band.ESCALATE


def test_each_risk_class_has_its_own_exact_edges():
    from orchestrator.bands import BANDS, Band, Risk, band

    for risk, (proceed, ask) in BANDS.items():
        assert band(proceed, risk) is Band.PROCEED
        assert band(proceed - 0.0001, risk) is Band.ASK
        assert band(ask, risk) is Band.ASK
        assert band(ask - 0.0001, risk) is Band.ESCALATE
        assert band(float("nan"), risk) is Band.ESCALATE
        assert 0 < ask < proceed < 1
    assert BANDS[Risk.READ][0] < BANDS[Risk.REVERSIBLE][0] < BANDS[Risk.IRREVERSIBLE][0], "the cost of a mistake sets the bar"


def test_every_kind_has_a_risk_class_and_only_freezes_and_reads_skip_the_yes():
    from orchestrator.bands import Risk
    from orchestrator.decision import NO_YES_NEEDED, risk_of
    from orchestrator.types import IntentKind

    assert risk_of(IntentKind.UPDATE_PROFILE) is Risk.IRREVERSIBLE
    assert risk_of(IntentKind.FREEZE_CARD) is Risk.REVERSIBLE
    assert risk_of(IntentKind.REPORT_FRAUD_OR_LOSS) is Risk.URGENT
    assert IntentKind.UPDATE_PROFILE not in NO_YES_NEEDED
    assert all(risk_of(k) in (Risk.READ, Risk.REVERSIBLE) for k in NO_YES_NEEDED)
