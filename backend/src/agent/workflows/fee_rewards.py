"""Correcting the rewards on card purchases after an approved cash back dispute.

The customer's own cash back dispute is a step they take themselves; once it has been reviewed and approved
(``cash_back_disputes``: status RESOLVED, resolution APPROVED) the purchase's rewards are recalculated here
from the card's earning rules (``agent/rewards_review.py``) and corrected if they differ. The dispute's
``expected_rewards`` and the recorded value are never trusted. A purchase whose dispute is still being
reviewed, or never made, is not touched, and the customer is told what to do.

The model reports which purchases the customer named (none named means every approved dispute). The plan has
one step per purchase that is wrong; a purchase already right is said to be right.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.rewards_audit import audit
from agent.rewards_review import points_text, review_transaction, stored_points
from agent.specialist import ValidationContext
from agent.specialists import fees_rules as rules
from agent.specialists.accounts_rules import dollars
from agent.specialists.credit_rules import day, money
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Reply, Step, WorkflowContext
from orchestrator.types import IntentKind


class RewardsFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    purchases: list[str] = Field(
        description=(
            "The purchases the customer says earned the wrong rewards, each as the request's details or the customer "
            "name it (a merchant, a date or an amount). Leave empty if the customer did not name any."
        )
    )


def _describe(row: Mapping[str, Any]) -> str:
    when = day(row.get("transaction_date"))
    amount = money(row.get("transaction_amount"))
    shown = f"${amount:,.2f}" if amount is not None else str(row.get("transaction_amount"))
    return f"your {shown} {row.get('merchant_name')} purchase" + (f" on {when:%m/%d/%Y}" if when else "")


def _matches(row: Mapping[str, Any], named: str) -> bool:
    text = named.strip().lower()
    if not text:
        return False
    when = day(row.get("transaction_date"))
    amount = money(row.get("transaction_amount"))
    return (
        text == str(row.get("transaction_id")).lower()
        or text in str(row.get("merchant_name") or "").lower()
        or (when is not None and text in f"{when:%m/%d/%Y}")
        or (amount is not None and text.replace("$", "").replace(",", "") in {f"{amount:.2f}", f"{amount:g}"})
    )


class RewardsCorrection:
    name = "rewards_correction"
    kinds = frozenset({IntentKind.FEE_OR_INTEREST_ADJUSTMENT})
    summary = "The customer says a card purchase earned the wrong rewards or cash back and wants it corrected."
    facts = RewardsFacts

    def _submit_disputes(self, ctx: WorkflowContext, facts: RewardsFacts, found: str) -> Decision:
        """No dispute exists yet: look at every purchase on the customer's cards (or the ones they named), and plan a cash back
        dispute for each one the card's earning rules say is wrong. The customer is never asked to find them or to submit
        anything: they see what is wrong and approve the submissions with one yes. A correction follows once approved."""
        db = ctx.database
        rows = [r for r in db.credit_card_transaction_history.data.values() if r.get("user_id") == ctx.user_id]
        named = [r for r in rows if any(_matches(r, n) for n in facts.purchases)] if facts.purchases else rows
        result = audit(db, ctx.user_id, named)
        if not result.wrong:
            text = f"I went through {len(named)} purchase{'s' if len(named) != 1 else ''} on your cards and recalculated the rewards from each card's earning rules. Every one I could recalculate already matches."
            if result.undecided:
                text += f" I could not recalculate {result.undecided} (refunds, or a card with no rewards rules on file)."
            return Reply(text)
        steps = []
        for row, review in result.wrong:
            was = review.stored if review.stored is not None else 0
            note = (
                f"Submit a cash back dispute for {_describe(row)} (on file {points_text(was)}, the card's rules give "
                f"{points_text(review.expected)})"
            )
            steps.append(Step(rules.DISPUTE_TOOL, {"user_id": ctx.user_id, "transaction_id": str(row["transaction_id"])}, note=note))
        return Plan(tuple(steps))

    @staticmethod
    def _findings(db: Any, user_id: str, named: Sequence[str]) -> str:
        """What the card's earning rules say about each purchase the customer picked by its id, as a read: nothing is
        changed here, and a correction still waits for an approved dispute. The same recalculation the correction uses."""
        lines: list[str] = []
        for text in named:
            row = db.credit_card_transaction_history.data.get(text.strip())
            if not isinstance(row, Mapping) or row.get("user_id") != user_id:
                continue
            review = review_transaction(db, row, user_id)
            if review.cannot_decide is not None:
                lines.append(f"{_describe(row)}: I cannot recalculate it ({review.cannot_decide}).")
            elif review.wrong:
                lines.append(f"{_describe(row)} shows {row.get('rewards_earned')}, but the card's earning rules give {points_text(review.expected)}.")
            else:
                lines.append(f"{_describe(row)} already shows the right rewards ({row.get('rewards_earned')}).")
        return "I checked it from the card's earning rules. " + " ".join(lines) + " " if lines else ""

    def decide(self, ctx: WorkflowContext, facts: RewardsFacts) -> Decision:
        db = ctx.database
        found = self._findings(db, ctx.user_id, facts.purchases)
        disputes = [d for d in db.cash_back_disputes.data.values() if isinstance(d, Mapping) and d.get("user_id") == ctx.user_id]
        if not disputes:
            return self._submit_disputes(ctx, facts, found)
        approved = [d for d in disputes if rules.approved_dispute(db, ctx.user_id, str(d.get("transaction_id")))]
        rows = [r for r in (db.credit_card_transaction_history.data.get(str(d.get("transaction_id"))) for d in approved) if isinstance(r, Mapping) and r.get("user_id") == ctx.user_id]
        if not rows:
            return Reply(found + "Your cash back dispute is still under review. I can correct the rewards once it has been approved.")
        chosen: list[Mapping[str, Any]] = []
        for named in facts.purchases:
            found = [r for r in rows if _matches(r, named)]
            if len(found) != 1:
                options = "\n".join(f"{n}. {_describe(r)}" for n, r in enumerate(rows, start=1))
                return NeedInfo((Ask("purchases", f"Which purchase do you mean?\n{options}"),))
            if found[0] not in chosen:
                chosen.append(found[0])
        chosen = chosen or rows

        steps: list[Step] = []
        right: list[str] = []
        left_out: list[str] = []
        decided = 0
        for row in chosen:
            review = review_transaction(db, row, ctx.user_id)
            if review.cannot_decide is not None:
                left_out.append(f"{_describe(row)}: {review.cannot_decide}")
                continue
            decided += 1
            if not review.wrong:
                right.append(_describe(row))
                continue
            args = {"transaction_id": str(row["transaction_id"]), "new_rewards_earned": points_text(review.expected)}
            if (why := rules.rewards_update(ValidationContext(db, ctx.user_id, ctx.today), args)) is not None:
                left_out.append(f"{_describe(row)}: {why}")
                continue
            was = row.get("rewards_earned")
            promo = " (the card's double-rewards promotion applied)" if review.multiplier == 2 else ""
            note = f"Change the rewards on {_describe(row)} from {was} to {points_text(review.expected)}{promo}"
            steps.append(Step(rules.REWARDS_TOOL, args, note=note))
        if not steps:
            if decided == 0 and left_out:
                return HandOff("; ".join(left_out))
            text = "I recalculated the rewards from the card's earning rules."
            if right:
                text += " Already correct: " + "; ".join(right) + "."
            if left_out:
                text += " Not changed: " + "; ".join(left_out) + "."
            return Reply(text)
        if right or left_out:
            first = steps[0]
            steps[0] = Step(first.tool, first.args, note=f"{first.note} (not changed: {'; '.join(right + left_out)})")
        return Plan(tuple(steps))
