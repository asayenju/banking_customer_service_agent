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
