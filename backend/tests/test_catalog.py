from pathlib import Path

import pytest
from tau2.domains.banking_knowledge.environment import get_db
from tau2.domains.banking_knowledge.tools import KnowledgeTools

from banking_tools.catalog import EXCLUDED, build_catalog, render_markdown
from banking_tools.types import ToolKind

DOC = Path(__file__).parents[2] / "docs" / "banking-tool-catalog.md"

# tau2 decorator exceptions: catalogued tools whose tau2 label disagrees with ours.
# Empty today; add a tool here, with a reason, instead of loosening the check.
DECORATOR_EXCEPTIONS: dict[str, str] = {}


@pytest.fixture(scope="module")
def toolkit() -> KnowledgeTools:
    return KnowledgeTools(get_db())


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def test_every_tau2_tool_is_catalogued_or_excluded(toolkit, catalog):
    upstream = set(toolkit.get_tools()) | set(toolkit.get_discoverable_tools())
    known = set(catalog) | set(EXCLUDED)
    assert upstream - known == set(), "new tau2 tools need a label or an exclusion"
    assert known - upstream == set(), "catalog names tools tau2 no longer has"


def test_catalog_and_exclusions_do_not_overlap(catalog):
    assert set(catalog) & set(EXCLUDED) == set()


def test_source_matches_tau2(toolkit, catalog):
    visible = set(toolkit.get_tools())
    for name, spec in catalog.items():
        assert (spec.source == "visible") == (name in visible), name


def test_labels_agree_with_tau2_decorator(toolkit, catalog):
    expected = {
        ToolKind.READ: ("read", False),
        ToolKind.WRITE: ("write", True),
        ToolKind.HANDOFF: ("generic", False),
    }
    for name, spec in catalog.items():
        if name in DECORATOR_EXCEPTIONS:
            continue
        tau2_type, mutates = expected[spec.kind]
        assert toolkit.tool_type(name).value == tau2_type, name
        assert toolkit.tool_mutates_state(name) == mutates, name


def test_decorator_exceptions_are_real(catalog):
    assert set(DECORATOR_EXCEPTIONS) <= set(catalog)


def test_every_description_is_non_empty(catalog):
    for name, spec in catalog.items():
        assert spec.description.strip(), name


def test_checked_in_table_is_current():
    assert DOC.read_text() == render_markdown(), (
        "docs/banking-tool-catalog.md is stale; regenerate it from render_markdown()"
    )
