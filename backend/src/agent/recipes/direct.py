"""The six requests the orchestrator sends down its DIRECT route, done in plain code.

Freeze or unfreeze a card, change the email on file, read balances and history, look
something up, answer a greeting. None of them needs policy reasoning, so none of them
calls a model: this module imports no LLM client, and a test keeps it that way.

Every recipe works for the verified customer only. Cards and accounts are chosen from that
customer's own records; another customer's id is treated as not found. Every call goes
through the code guard. Replies are written here from the database, in fixed wording, so
nothing a tool or a customer typed is repeated back except the customer's own new email.

Choosing a card follows the same rule as the specialists (``agent.cards``): one candidate
means no question, several means a list written by code, and the customer may answer with
the last four digits.
"""

import re
from collections.abc import Mapping
from typing import Any, Optional

from agent.cards import Card, describe, owned_cards, resolve, which_card
from agent.guard import DbOwnership, Guard, GuardedSession
from orchestrator.context import RouteContext
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind, Question
from orchestrator.unclear import offer_help

K = IntentKind
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
HISTORY_ROWS = 10


def _done(intent: Intent, text: str) -> RouteOutcome:
    return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text=text)


def _failed(intent: Intent, error: str) -> RouteOutcome:
    return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.FAILED, error=error)


def _ask(intent: Intent, slot: str, text: str) -> RouteOutcome:
    return RouteOutcome(
        intent_id=intent.id,
        status=OutcomeStatus.NEEDS_INFO,
        questions=(Question(intent_id=intent.id, slot=slot, text=text),),
    )


def _money(value: Any) -> str:
    try:
        amount = float(str(value).replace("$", "").replace(",", ""))
    except ValueError:
        return str(value)
    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


class DirectRunner:
    proposes_plans = False

    def __init__(self, database: Any) -> None:
        self._db = database

    async def run(self, intent: Intent, ctx: RouteContext) -> RouteOutcome:
        if intent.kind is K.UNCLEAR_OR_GREETING:
            return _done(intent, offer_help())
        user = ctx.verified_user_id
        if ctx.session is None or (user is None and intent.kind is not K.LOOKUP_INFO):
            return _failed(intent, "no verified customer or session for this request")
        guarded = GuardedSession(
            ctx.session, Guard(kind=intent.kind, user_id=user, ownership=DbOwnership(self._db))
        )
        if intent.kind is K.FREEZE_CARD:
            return self._set_card(intent, guarded, str(user), freeze=True)
        if intent.kind is K.UNFREEZE_CARD:
            return self._set_card(intent, guarded, str(user), freeze=False)
        if intent.kind is K.UPDATE_PROFILE:
            return self._email(intent, guarded, str(user))
        if intent.kind is K.ACCOUNT_BALANCE_OR_HISTORY:
            return self._accounts(intent, guarded, str(user))
        if intent.kind is K.LOOKUP_INFO:
            return self._lookup(intent, guarded, user)
        return _failed(intent, "this is not a direct request")

    # ------------------------------------------------------------------ freeze and unfreeze

    def _set_card(self, intent: Intent, guarded: GuardedSession, user: str, *, freeze: bool) -> RouteOutcome:
        verb, tool, wanted = (
            ("frozen", "freeze_debit_card_3892", "ACTIVE") if freeze else ("unfrozen", "unfreeze_debit_card_3893", "FROZEN")
        )
        everyone = owned_cards(self._db, user)
        candidates = [c for c in everyone if c.get("status") == wanted]
        named = (intent.slots.get("card_id") or "").strip()
        card: Optional[Card]
        if named:
            card = resolve(named, everyone)
            if card is None:
                return which_card(intent, self._db, candidates or everyone) if (candidates or everyone) else _done(intent, "You have no cards on file.")
        elif len(candidates) == 1:
            card = candidates[0]
        elif not candidates:
            return _done(intent, f"You have no card that can be {verb} right now.")
        else:
            return which_card(intent, self._db, candidates)
        label = describe(self._db, card)
        if card.get("status") == ("FROZEN" if freeze else "ACTIVE"):
            return _done(intent, f"{label[0].upper()}{label[1:]} is already {'frozen' if freeze else 'active'}.")
        if card.get("status") != wanted:
            return _done(intent, f"{label[0].upper()}{label[1:]} cannot be {verb}: it is {str(card.get('status')).lower()}.")
        result = guarded.call(tool, {"card_id": card["card_id"]})
        if not result.ok:
            return _failed(intent, f"the bank would not {'freeze' if freeze else 'unfreeze'} the card")
        return _done(intent, f"{label[0].upper()}{label[1:]} is now {verb}.")

    # ------------------------------------------------------------------ the email on file

    def _email(self, intent: Intent, guarded: GuardedSession, user: str) -> RouteOutcome:
        new = (intent.slots.get("new_email") or intent.slots.get("email") or "").strip()
        if not _EMAIL.match(new):
            return _ask(intent, "new_email", "What email address should I put on your account?")
        # The customer is always the verified one, whatever a slot says.
        result = guarded.call("change_user_email", {"user_id": user, "new_email": new})
        if not result.ok:
            return _failed(intent, "the bank would not change the email")
        return _done(intent, f"The email on your account is now {new}.")

    # ------------------------------------------------------------------ balances and history

    def _accounts(self, intent: Intent, guarded: GuardedSession, user: str) -> RouteOutcome:
        mine = {k: v for k, v in self._db.accounts.data.items() if v.get("user_id") == user}
        named = (intent.slots.get("account_id") or "").strip()
        if named:
            if named not in mine:
                names = ", ".join(a.get("level", "an account") for a in mine.values()) or "none"
                return _done(intent, f"I can't find that account among yours. Your accounts: {names}.")
            guarded.call("get_bank_account_transactions_9173", {"account_id": named})
            rows = sorted(
                (t for t in self._db.bank_account_transaction_history.data.values() if t.get("account_id") == named),
                key=lambda t: _sort_key(str(t.get("date"))),
                reverse=True,
            )[:HISTORY_ROWS]
            level = mine[named].get("level", "your account")
            if not rows:
                return _done(intent, f"There are no transactions on your {level}.")
            lines = [f"Recent activity on your {level}:"] + [
                f"- {str(t.get('date'))[:10]}  {_money(t.get('amount'))}  {t.get('description', '')}" for t in rows
            ]
            return _done(intent, "\n".join(lines))
        guarded.call("get_all_user_accounts_by_user_id_3847", {"user_id": user})
        cards = [c for c in self._db.credit_card_accounts.data.values() if c.get("user_id") == user]
        lines = ["Your accounts:"]
        lines += [f"- {a.get('level')} ({a.get('class')}): {_money(a.get('current_holdings'))}" for a in mine.values()]
        lines += [f"- {c.get('card_type')}: balance {_money(c.get('current_balance'))}" for c in cards]
        return _done(intent, "\n".join(lines) if len(lines) > 1 else "I don't see any accounts on your profile.")

    # ------------------------------------------------------------------ lookups

    def _lookup(self, intent: Intent, guarded: GuardedSession, user: Optional[str]) -> RouteOutcome:
        text = intent.summary.lower()
        if re.search(r"\b(time|date|today|day)\b", text):
            result = guarded.call("get_current_time", {})
            return _done(intent, str(result.text).replace("\n", " ").strip()[:200])
        if user is None:
            return _failed(intent, "no verified customer for this lookup")
        if "referral" in text:
            guarded.call("get_referrals_by_user", {"user_id": user})
            rows = [r for r in self._db.referrals.data.values() if r.get("referrer_id") == user]
            if not rows:
                return _done(intent, "You have no referrals on file.")
            return _done(
                intent,
                "Your referrals:\n"
                + "\n".join(f"- {r.get('referred_account_type')}: {str(r.get('referral_status')).lower().replace('_', ' ')}" for r in rows),
            )
        if "card" in text:
            cards = owned_cards(self._db, user, statuses=None)
            return _done(intent, "Your cards:\n" + "\n".join(f"- {describe(self._db, c)}" for c in cards) if cards else "You have no cards on file.")
        guarded.call("get_user_information_by_id", {"user_id": user})
        row: Mapping[str, Any] = self._db.users.data.get(user, {})
        return _done(
            intent,
            f"On your profile: {row.get('name', 'name not on file')}, {row.get('email', 'no email')}, {row.get('phone_number', 'no phone')}.",
        )


def _sort_key(date_text: str) -> tuple[int, int, int, str]:
    """MM/DD/YYYY [time] as something that sorts by date, newest last."""
    parts = date_text[:10].split("/")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[2]), int(parts[0]), int(parts[1]), date_text[10:]
    return 0, 0, 0, date_text
