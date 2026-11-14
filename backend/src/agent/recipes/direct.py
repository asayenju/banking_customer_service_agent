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

from agent.cards import Card, describe, named_cards, owned_cards, resolve, which_card
from agent.entities import resolve_cards
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
        said = " ".join(
            [intent.summary, *(str(v) for k, v in intent.slots.items() if k in ("card_id", "card_ids", "cards", "card_or_account_id", "account_id", "card"))]
        )
        named, every = named_cards(said, everyone, self._db)
        if not (named or every):
            # "the second one", "2", "those": counted along the list this customer was just shown.
            answer = " ".join(str(intent.slots.get(k) or "") for k in ("card_id", "card_ids", "cards", "card"))
            picked = resolve_cards(answer, candidates, self._db)
            named, every = list(picked.cards), picked.every
        if named or every:
            return self._set_each(intent, guarded, named, freeze=freeze, every=every)
        gave_something = any((intent.slots.get(k) or "").strip() for k in ("card_id", "card_ids", "cards", "card_or_account_id"))
        if not candidates:
            return _done(intent, f"You have no card that can be {verb} right now.")
        if len(candidates) == 1 and not gave_something:
            return self._set_each(intent, guarded, candidates, freeze=freeze, every=False)
        # They named something that is not one of their cards, or did not say: never pick one for them.
        return which_card(intent, self._db, candidates, allow_all=True)

    def _set_each(self, intent: Intent, guarded: GuardedSession, cards: list[Card], *, freeze: bool, every: bool) -> RouteOutcome:
        """Freeze or unfreeze each named card that can be, and say what happened to every one of them."""
        verb, tool, wanted = (
            ("frozen", "freeze_debit_card_3892", "ACTIVE") if freeze else ("unfrozen", "unfreeze_debit_card_3893", "FROZEN")
        )
        lines: list[str] = []
        for card in cards:
            label = describe(self._db, card)
            cap = f"{label[0].upper()}{label[1:]}"
            if card.get("status") == ("FROZEN" if freeze else "ACTIVE"):
                lines.append(f"{cap} is already {'frozen' if freeze else 'active'}.")
            elif card.get("status") != wanted:
                lines.append(f"{cap} cannot be {verb}: it is {str(card.get('status')).lower()}.")
            elif not guarded.call(tool, {"card_id": card["card_id"]}).ok:
                return _failed(intent, f"the bank would not {'freeze' if freeze else 'unfreeze'} {label}")
            else:
                lines.append(f"{cap} is now {verb}.")
        if not lines:
            return _done(intent, f"You have no card that can be {verb} right now.")
        return _done(intent, "\n".join(lines))

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

    def _which_card(self, intent: Intent, user: str) -> tuple[str, Any]:
        """The card or account a customer's words name: ("credit", card type), ("debit", account id), ("ask", labels)
        or ("none", None). "Silver card" is the Silver Rewards Card, not also the Business Silver one: of the cards whose
        distinctive words are all there, the one that matched the most words wins; a tie is a question."""
        said = " ".join([intent.summary, *(v for v in intent.slots.values() if isinstance(v, str))]).lower()
        words = set(re.findall(r"[a-z0-9]+", said))
        generic = {"card", "cards", "rewards", "reward", "account", "back", "cash"}
        want_credit, want_debit = "credit" in words, "debit" in words
        found: list[tuple[int, str, Any, str]] = []

        def consider(kind: str, key: Any, label: str) -> None:
            # "EcoCard" is "eco card" to a customer.
            parts = [w[:-4] if w.endswith("card") and len(w) > 4 else w for w in re.findall(r"[a-z0-9]+", label.lower())]
            distinct = set(parts) - generic
            if distinct and distinct <= words:
                found.append((len(distinct), kind, key, label))

        if not want_debit:
            for card_type in sorted({str(c.get("card_type")) for c in self._db.credit_card_accounts.data.values() if c.get("user_id") == user}):
                consider("credit", card_type, card_type)
        if not want_credit:
            for account_id, account in self._db.accounts.data.items():
                if account.get("user_id") == user and account.get("class") == "checking":
                    consider("debit", account_id, str(account.get("level")))
            for card in owned_cards(self._db, user, statuses=None):
                last_four = str(card.get("last_4_digits") or "")
                if last_four and last_four in words:
                    found.append((9, "debit", str(card.get("account_id")), last_four))
        if not found:
            return "none", None
        best = max(f[0] for f in found)
        top = [f for f in found if f[0] == best]
        if len({(f[1], f[2]) for f in top}) > 1:
            return "ask", [f[3] for f in top]
        return top[0][1], top[0][2]

    def _credit_history(self, intent: Intent, user: str, card_type: str) -> RouteOutcome:
        rows = sorted(
            (
                t
                for t in self._db.credit_card_transaction_history.data.values()
                if t.get("user_id") == user and t.get("credit_card_type") == card_type
            ),
            key=lambda t: _sort_key(str(t.get("transaction_date"))),
            reverse=True,
        )[:HISTORY_ROWS]
        if not rows:
            return _done(intent, f"There are no charges on your {card_type}.")
        lines = [f"Recent charges on your {card_type}:"] + [
            f"- {str(t.get('transaction_date'))[:10]}  {_money(t.get('transaction_amount'))}  {t.get('merchant_name', '')}"
            f" ({str(t.get('status', '')).lower()})"
            for t in rows
        ]
        return _done(intent, "\n".join(lines))

    def _accounts(self, intent: Intent, guarded: GuardedSession, user: str) -> RouteOutcome:
        mine = {k: v for k, v in self._db.accounts.data.items() if v.get("user_id") == user}
        named = (intent.slots.get("account_id") or "").strip()
        if named not in mine:
            kind, picked = self._which_card(intent, user)
            if kind == "credit":
                guarded.call("get_credit_card_transactions_by_user", {"user_id": user})
                return self._credit_history(intent, user, str(picked))
            if kind == "debit":
                named = str(picked)
            elif kind == "ask":
                listing = "\n".join(f"- {label}" for label in picked)
                return _ask(intent, "card_or_account_id", f"Which card do you mean?\n{listing}")
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
