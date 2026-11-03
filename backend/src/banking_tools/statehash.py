"""Compare two end states the way tau2's evaluator means to, and say what differs.

tau2 scores a run by hashing the whole database. That is exact, but a mismatch says
nothing about where or why, and cosmetic differences (``"$12.00"`` against
``"12.00"``, ``33`` against ``33.0``) count as failures. This module reports two
levels:

- strict: the full tau2 hash is equal;
- semantic: the changes each side made to the starting database are equal after
  normalisation, row by row and field by field.

A strict-only mismatch with a semantic match points at a fidelity detail, not at the
agent. The semantic comparison ignores row order (tables are keyed by primary key)
and, by default, the verification log, which is asserted separately.

Ids are compared literally. tau2's ids are SHA-256 of the arguments, so equal
arguments give equal ids; an id mismatch means the arguments differed.
"""

import re
from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict
from tau2.domains.banking_knowledge.data_model import TransactionalDB

VERIFICATION_TABLE = "verification_history"
TOOL_LOG_TABLE = "agent_discoverable_tools"

# Bare numeric strings are money only in these fields (elsewhere they can be ids).
BARE_MONEY_FIELDS = frozenset({("accounts", "current_holdings")})

_DOLLARS = re.compile(r"^(-?)\$([\d,]+(?:\.\d+)?)$")
_BARE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")
_POINTS = re.compile(r"^([\d,]+) points$")
_US_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def _cents(text: str) -> int:
    """Integer cents of a decimal string, without float rounding surprises."""
    negative = text.startswith("-")
    whole, _, frac = text.lstrip("-").replace(",", "").partition(".")
    cents = int(whole or "0") * 100 + int((frac + "00")[:2])
    return -cents if negative else cents


def normalise(value: Any, *, table: str = "", field: str = "") -> Any:
    """A value in a form where cosmetic differences disappear."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else round(value, 2)
    if isinstance(value, str):
        if (m := _DOLLARS.match(value)) is not None:
            return {"cents": _cents(m.group(1) + m.group(2))}
        if (table, field) in BARE_MONEY_FIELDS and _BARE.match(value):
            return {"cents": _cents(value)}
        if (m := _POINTS.match(value)) is not None:
            return {"points": int(m.group(1).replace(",", ""))}
        if _US_DATE.match(value):
            return datetime.strptime(value, "%m/%d/%Y").date().isoformat()
        return value
    if isinstance(value, Mapping):
        return {k: normalise(v, table=table, field=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v, table=table, field=field) for v in value]
    return value


class RowChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: str
    pk: str
    kind: Literal["added", "removed", "changed"]
    # field -> (before, after), normalised; for an added row before is None.
    fields: dict[str, tuple[Any, Any]]


class Mismatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: str
    pk: str
    problem: str


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    strict_equal: bool
    semantic_equal: bool
    mismatches: tuple[Mismatch, ...]

    @property
    def strict_only(self) -> bool:
        """The hashes differ but no domain change does: look for a fidelity detail."""
        return self.semantic_equal and not self.strict_equal


def _tables(db: TransactionalDB) -> dict[str, dict[str, dict[str, Any]]]:
    dumped = db.model_dump()
    return {
        name: table["data"]
        for name, table in dumped.items()
        if isinstance(table, dict) and "data" in table
    }


def domain_delta(
    before: TransactionalDB,
    after: TransactionalDB,
    *,
    ignore_verification: bool = True,
    read_log_names: Collection[str] = (),
) -> dict[tuple[str, str], RowChange]:
    """Every row that was added, removed or changed between two databases."""
    start, end = _tables(before), _tables(after)
    skipped = {VERIFICATION_TABLE} if ignore_verification else set()
    changes: dict[tuple[str, str], RowChange] = {}
    for table in sorted(set(start) | set(end)):
        if table in skipped:
            continue
        rows_before, rows_after = start.get(table, {}), end.get(table, {})
        for pk in sorted(set(rows_before) | set(rows_after)):
            old, new = rows_before.get(pk), rows_after.get(pk)
            if table == TOOL_LOG_TABLE and (old or new or {}).get("tool_name") in set(read_log_names):
                continue
            if old == new:
                continue
            if old is None:
                kind: Literal["added", "removed", "changed"] = "added"
                fields = {k: (None, normalise(v, table=table, field=k)) for k, v in (new or {}).items()}
            elif new is None:
                kind = "removed"
                fields = {k: (normalise(v, table=table, field=k), None) for k, v in old.items()}
            else:
                kind = "changed"
                fields = {}
                for key in sorted(set(old) | set(new)):
                    a = normalise(old.get(key), table=table, field=key)
                    b = normalise(new.get(key), table=table, field=key)
                    if a != b:
                        fields[key] = (a, b)
                if not fields:
                    continue  # only cosmetic differences between the two versions of this row
            changes[(table, pk)] = RowChange(table=table, pk=pk, kind=kind, fields=fields)
    return changes


def compare(
    start: TransactionalDB,
    expected: TransactionalDB,
    actual: TransactionalDB,
    *,
    ignore_verification: bool = True,
    read_log_names: Collection[str] = (),
) -> Comparison:
    """Strict (tau2 hash) and semantic (normalised delta from ``start``) equality."""
    want = domain_delta(start, expected, ignore_verification=ignore_verification, read_log_names=read_log_names)
    got = domain_delta(start, actual, ignore_verification=ignore_verification, read_log_names=read_log_names)
    mismatches: list[Mismatch] = []
    for key in sorted(set(want) | set(got)):
        table, pk = key
        a: Optional[RowChange] = want.get(key)
        b: Optional[RowChange] = got.get(key)
        if a is None:
            mismatches.append(Mismatch(table=table, pk=pk, problem="unexpected change"))
        elif b is None:
            mismatches.append(Mismatch(table=table, pk=pk, problem="expected change is missing"))
        elif a.kind != b.kind or a.fields != b.fields:
            names = sorted(set(a.fields) | set(b.fields))
            differing = [n for n in names if a.fields.get(n) != b.fields.get(n)]
            mismatches.append(Mismatch(table=table, pk=pk, problem=f"{a.kind} vs {b.kind}; fields differ: {', '.join(differing)}"))
    return Comparison(
        strict_equal=expected.get_hash() == actual.get_hash(),
        semantic_equal=not mismatches,
        mismatches=tuple(mismatches),
    )
