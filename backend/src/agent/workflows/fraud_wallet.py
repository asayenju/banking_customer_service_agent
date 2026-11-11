"""The lost or stolen wallet protocol: one combined plan over debit and credit.

Sources: "Lost/Stolen Card - Cross-Product Security Protocol" (030), "Closing/Cancelling a Debit Card"
(025), "Freezing and Unfreezing" (026), "Debit Card Replacement Options by Account Tier" (029) and the
credit replacement and dispute documents. The decisions and their evidence are in
``docs/fraud-policy-decisions.md``.

The composer has no tools of its own. It builds each part with the builders the debit and credit
specialists already use (``DebitReplacement``, ``CreditReplacement``, ``CreditDispute``) and joins their
steps in one order: unfreeze and close, reorder, replace credit cards, then dispute. The customer
approves all of it with one yes. The model reports only what the customer said; code decides:

  - nothing is closed or replaced until the customer says the loss is definite (a freeze is the answer
    to a possible loss, and it is made before this runs);
  - a card with a pending purchase stays frozen; a card with a pending refund waits for the customer's
    accept that the refund goes to checking (025);
  - an Entry account whose card is closed in this plan is not reordered today (029, the 48 hour wait);
  - the credit cards are always asked about (030), and nothing is replaced that the customer did not name.
"""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agent.cards import describe, describe_credit, owned_credit_accounts
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from agent.workflows.credit_dispute import CreditDispute, CreditDisputeFacts, CreditDisputeRequest
from agent.workflows.credit_replacement import CreditReplacement, CreditReplacementFacts, NewCreditCard
from agent.workflows.debit_replacement import Closure, DebitReplacement, NewCard, ReplacementFacts
from agent.workflows.tiers import tier_of_level
from orchestrator.types import IntentKind

LossReason = Literal["lost", "stolen", "fraud_suspected"]
WAIT_HOURS = 48


class Replacement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    delivery_option: Literal["STANDARD", "EXPEDITED", "RUSH", "FASTEST_FREE"]
    card_design: Literal["CLASSIC", "PREMIUM", "CUSTOM", "BEST_FREE"]
    # The address the customer gave, or null for the one on file.
    shipping_address: Optional[str]
    # True only if the customer agreed to pay an excess replacement fee when one applies.
    accepts_excess_fee: bool


class WalletCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    # True only if the customer agreed that a pending refund is credited to the checking account.
    accepts_refund_to_checking: bool
    # What the customer wants as a new card for this one, or null if they want none now.
    replacement: Optional[Replacement]


class WalletCredit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_card_account_id: str
    expedited: bool
    # True only if the customer agreed to pay the expedited fee when one applies.
    accepts_expedite_fee: bool
    shipping_address: Optional[str]


class WalletFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: LossReason
    # True only if the customer said the cards are gone for sure, not "I think I lost them".
    loss_is_definite: bool
    # The debit cards that were in the wallet. A card the customer found is not listed.
    debit_cards: list[WalletCard]
    # The answer to "was a credit card in the wallet?"
    credit_answer: Literal["some_were_in_the_wallet", "none_were_in_the_wallet", "not_asked_yet"]
    credit_cards: list[WalletCredit]
    # Fraudulent credit card charges the customer names.
    credit_disputes: list[CreditDisputeFacts]


def _pending(db: Any, account_id: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """The account's pending purchases and its pending refunds."""
    rows = [
        t
        for t in db.bank_account_transaction_history.data.values()
        if t.get("account_id") == account_id and str(t.get("status")).lower() == "pending"
    ]
    refunds = [t for t in rows if t.get("type") == "refund"]
    return [t for t in rows if t.get("type") != "refund"], refunds


def _level(db: Any, account_id: str) -> str:
    return str(db.accounts.data.get(account_id, {}).get("level") or "account")


class WalletLoss:
    name = "wallet_loss"
    kinds = frozenset({IntentKind.REPORT_FRAUD_OR_LOSS})
    summary = (
        "The customer lost a wallet or had cards stolen, or reports their cards as lost, stolen or "
        "compromised. List the debit cards that were in the wallet (not any they found), say whether the "
        "loss is definite, and whether a credit card was also in it. Report what they want for each new "
        "card and which fraudulent credit card charges they name. The rules, fees and order are worked out."
    )
    facts = WalletFacts

    def decide(self, ctx: WorkflowContext, facts: WalletFacts) -> Decision:
        db = ctx.database
        credit_open = owned_credit_accounts(db, ctx.user_id)
        asks: list[Ask] = []
        if not facts.debit_cards and facts.credit_answer != "some_were_in_the_wallet":
            asks.append(Ask("cards_in_the_wallet", _which_cards(db, ctx.user_id)))
        if credit_open and facts.credit_answer == "not_asked_yet":
            listing = ", ".join(describe_credit(a) for a in credit_open)
            asks.append(
                Ask(
                    "credit_cards_in_the_wallet",
                    f"I see you also have {listing}. Was it in the wallet too? If it was, I can order a "
                    "replacement with a new number to protect you.",
                )
            )
        if facts.credit_answer == "none_were_in_the_wallet" and facts.credit_cards:
            return Refuse("the customer said no credit card was in the wallet, but credit cards are listed")
        if facts.credit_answer == "some_were_in_the_wallet" and not facts.credit_cards:
            asks.append(Ask("which_credit_cards", "Which credit cards were in the wallet?"))
        if not facts.loss_is_definite:
            if asks:
                return NeedInfo(tuple(asks))
            return Reply(
                "Nothing is closed or replaced yet. The cards you named are frozen, so nobody can use them, and "
                "you can unfreeze one if you find it. Tell me when you are sure they are gone and I will close "
                "them and order replacements."
            )
        steps: list[Step] = []
        notes: list[str] = []
        needs: list[Ask] = list(asks)

        debit = self._debit(ctx, facts, steps, notes, needs)
        if isinstance(debit, (Refuse, HandOff)):
            return debit
        credit = self._credit(ctx, facts, steps, notes, needs)
        if isinstance(credit, (Refuse, HandOff)):
            return credit
        disputes = self._disputes(ctx, facts, steps, notes, needs)
        if isinstance(disputes, (Refuse, HandOff)):
            return disputes
        if needs:
            return NeedInfo(tuple(needs))
        if not steps:
            return Reply(" ".join(notes) or "There is nothing to close or replace.")
        return Plan(_noted(steps, notes))

    # ------------------------------------------------------------------ the debit cards

    def _debit(
        self, ctx: WorkflowContext, facts: WalletFacts, steps: list[Step], notes: list[str], needs: list[Ask]
    ) -> Optional[Decision]:
        db = ctx.database
        seen: set[str] = set()
        for choice in facts.debit_cards:
            card = db.debit_cards.data.get(choice.card_id)
            account = db.accounts.data.get(str(card.get("account_id"))) if card else None
            if card is None or account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{choice.card_id} is not one of this customer's debit cards")
            if choice.card_id in seen:
                return Refuse(f"{choice.card_id} is listed twice")
            seen.add(choice.card_id)
            label = describe(db, card)
            if card.get("status") == "CLOSED":
                notes.append(f"{label[0].upper()}{label[1:]} is already closed.")
                continue
            purchases, refunds = _pending(db, str(card["account_id"]))
            if purchases:
                if card.get("status") == "ACTIVE":
                    steps.append(Step("freeze_debit_card_3892", {"card_id": choice.card_id}, f"Freeze {label}"))
                notes.append(
                    f"{label[0].upper()}{label[1:]} stays frozen, not closed: it has purchases still pending, "
                    "and a card with pending purchases cannot be closed until they settle."
                )
                continue
            if refunds and not choice.accepts_refund_to_checking:
                needs.append(
                    Ask(
                        "accepts_refund_to_checking",
                        f"{label[0].upper()}{label[1:]} has a refund still pending. It cannot be closed until the "
                        "refund processes, unless you agree that it is credited to your checking account instead. "
                        "Do you agree?",
                    )
                )
                continue
            replacement = choice.replacement
            tier = tier_of_level(_level(db, str(card["account_id"])))
            if replacement is not None and tier == "Entry":
                earliest = ctx.today + timedelta(hours=WAIT_HOURS)
                notes.append(
                    f"A new card for your {_level(db, str(card['account_id']))} cannot be ordered today: this tier "
                    f"waits {WAIT_HOURS} hours after a card is closed. It can be ordered from "
                    f"{earliest:%m/%d/%Y}."
                )
                replacement = None
            result = self._one_card(ctx, choice, replacement, facts.reason)
            if isinstance(result, Reply):
                notes.append(result.text)
                result = self._one_card(ctx, choice, None, facts.reason)
            if isinstance(result, NeedInfo):
                needs.extend(result.questions)
                continue
            if isinstance(result, (Refuse, HandOff)):
                return result
            assert isinstance(result, Plan)
            steps.extend(_describe_debit(db, s, label) for s in result.steps)
        return None

    def _one_card(
        self, ctx: WorkflowContext, choice: WalletCard, replacement: Optional[Replacement], reason: LossReason
    ) -> Decision:
        db = ctx.database
        account_id = str(db.debit_cards.data[choice.card_id]["account_id"])
        return DebitReplacement().decide(
            ctx,
            ReplacementFacts(
                closures=[Closure(card_id=choice.card_id, reason=reason)],
                new_cards=[]
                if replacement is None
                else [
                    NewCard(
                        account_id=account_id,
                        delivery_option=replacement.delivery_option,
                        card_design=replacement.card_design,
                        shipping_address=replacement.shipping_address,
                        accepts_excess_fee=replacement.accepts_excess_fee,
                    )
                ],
            ),
        )

    # ------------------------------------------------------------------ the credit cards

    def _credit(
        self, ctx: WorkflowContext, facts: WalletFacts, steps: list[Step], notes: list[str], needs: list[Ask]
    ) -> Optional[Decision]:
        db = ctx.database
        for item in facts.credit_cards:
            account = db.credit_card_accounts.data.get(item.credit_card_account_id)
            if account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{item.credit_card_account_id} is not one of this customer's credit card accounts")
            result = CreditReplacement().decide(
                ctx,
                CreditReplacementFacts(
                    replacements=[
                        NewCreditCard(
                            credit_card_account_id=item.credit_card_account_id,
                            reason=facts.reason,
                            expedited=item.expedited,
                            accepts_expedite_fee=item.accepts_expedite_fee,
                            shipping_address=item.shipping_address,
                        )
                    ]
                ),
            )
            if isinstance(result, Reply):
                notes.append(result.text)
            elif isinstance(result, NeedInfo):
                needs.extend(result.questions)
            elif isinstance(result, (Refuse, HandOff)):
                return result
            else:
                assert isinstance(result, Plan)
                label = describe_credit(account)
                steps.extend(
                    Step(s.tool, s.args, f"Order a replacement for {label} ({facts.reason}); the old card is cancelled at once")
                    for s in result.steps
                )
        return None

    # ------------------------------------------------------------------ disputes of credit charges

    def _disputes(
        self, ctx: WorkflowContext, facts: WalletFacts, steps: list[Step], notes: list[str], needs: list[Ask]
    ) -> Optional[Decision]:
        if not facts.credit_disputes:
            return None
        result = CreditDispute().decide(ctx, CreditDisputeRequest(disputes=facts.credit_disputes))
        if isinstance(result, Reply):
            notes.append(result.text)
        elif isinstance(result, NeedInfo):
            needs.extend(result.questions)
        elif isinstance(result, (Refuse, HandOff)):
            return result
        else:
            assert isinstance(result, Plan)
            for s in result.steps:
                txn = ctx.database.credit_card_transaction_history.data.get(str(s.args.get("transaction_id")), {})
                steps.append(Step(s.tool, s.args, f"Dispute the {txn.get('merchant_name', 'card')} charge ({str(s.args.get('dispute_reason')).replace('_', ' ')})"))
        return None

    # ------------------------------------------------------------------ after the plan ran

    def after(self, ctx: WorkflowContext, steps: Sequence[Step]) -> str:
        """What is still true for the customer once the plan has run."""
        db = ctx.database
        closed = {str(s.args["card_id"]) for s in steps if s.tool == "close_debit_card_4721"}
        ordered = {str(s.args["account_id"]) for s in steps if s.tool == "order_debit_card_5739"}
        lines: list[str] = []
        for card_id in sorted(closed):
            account_id = str(db.debit_cards.data[card_id]["account_id"])
            if account_id not in ordered and tier_of_level(_level(db, account_id)) == "Entry":
                lines.append(
                    f"Your {_level(db, account_id)} has no new card yet; it can be ordered from "
                    f"{ctx.today + timedelta(hours=WAIT_HOURS):%m/%d/%Y}."
                )
        frozen = [c for c in db.debit_cards.data.values() if c.get("user_id") == ctx.user_id and c.get("status") == "FROZEN"]
        if frozen:
            lines.append("Still frozen: " + ", ".join(describe(db, c) for c in frozen) + ".")
        if ordered:
            lines.append("When a new card arrives, tell me and I will activate it.")
        return " ".join(lines)


def _which_cards(db: Any, user_id: str) -> str:
    from agent.cards import owned_cards

    lines = [f"- {describe(db, c)}" for c in owned_cards(db, user_id)]
    lines += [f"- {describe_credit(a)}" for a in owned_credit_accounts(db, user_id)]
    return "Which cards were in the wallet?\n" + "\n".join(lines) if lines else "Which cards were in the wallet?"


def _describe_debit(db: Any, step: Step, label: str) -> Step:
    if step.tool == "unfreeze_debit_card_3893":
        return Step(step.tool, step.args, f"Unfreeze {label} so it can be closed")
    if step.tool == "close_debit_card_4721":
        return Step(step.tool, step.args, f"Close {label} ({str(step.args.get('reason')).replace('_', ' ')}); it cannot be reopened")
    if step.tool == "order_debit_card_5739":
        fees = float(step.args.get("delivery_fee") or 0) + float(step.args.get("design_fee") or 0) + float(step.args.get("excess_replacement_fee") or 0)
        return Step(
            step.tool,
            step.args,
            f"Order a new card for your {_level(db, str(step.args.get('account_id')))} "
            f"({str(step.args.get('delivery_option')).lower()} delivery, ${fees:.2f} charged to the account)",
        )
    return step


def _noted(steps: list[Step], notes: list[str]) -> tuple[Step, ...]:
    """Not-done notes ride on the first step's note, which is the text the customer approves."""
    if not notes:
        return tuple(steps)
    first = steps[0]
    extra = " Not included: " + " ".join(notes)
    return (Step(first.tool, first.args, (first.note + "." + extra) if first.note else extra.strip()), *steps[1:])
