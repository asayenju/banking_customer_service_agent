import pytest
from pydantic import ValidationError

from banking_tools import Reversibility, ToolKind, ToolResult, ToolSpec


def spec(kind, reversibility=None):
    return ToolSpec(
        name="t",
        kind=kind,
        reversibility=reversibility,
        description="d",
        source="visible",
    )


def test_write_requires_reversibility():
    with pytest.raises(ValidationError, match="must declare a reversibility"):
        spec(ToolKind.WRITE)


@pytest.mark.parametrize("kind", [ToolKind.READ, ToolKind.HANDOFF])
def test_non_write_rejects_reversibility(kind):
    with pytest.raises(ValidationError, match="must not declare a reversibility"):
        spec(kind, Reversibility.REVERSIBLE)


def test_valid_specs():
    assert spec(ToolKind.READ).reversibility is None
    assert spec(ToolKind.HANDOFF).reversibility is None
    assert spec(ToolKind.WRITE, Reversibility.IRREVERSIBLE).kind is ToolKind.WRITE


def test_spec_is_frozen():
    with pytest.raises(ValidationError):
        spec(ToolKind.READ).name = "x"


def test_result_fields_and_consistency():
    ok = ToolResult(tool="t", ok=True, text="fine", audit_id="a1")
    assert (ok.ok, ok.error, ok.audit_id) == (True, None, "a1")
    bad = ToolResult(tool="t", ok=False, text="Error: x", error="x")
    assert bad.error == "x"
    with pytest.raises(ValidationError):
        ToolResult(tool="t", ok=True, text="x", error="boom")
    with pytest.raises(ValidationError):
        ToolResult(tool="t", ok=False, text="x")
