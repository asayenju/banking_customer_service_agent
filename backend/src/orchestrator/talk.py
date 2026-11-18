"""What was said about each open request, kept apart from the request's details.

The details of a request (its slots) say what a customer's yes covers: they are fingerprinted, and a plan the customer
saw is bound to them. A transcript is not part of that. Keeping the exchange inside the slots meant a change in the
chat history could make a plan the customer had just seen look stale, and the same plan was asked about again.

So the exchanges live here. Each request keeps a sliding window: the most recent ``WINDOW`` exchanges, whole ones only
(never a string cut in the middle), each capped, and the window as a whole capped. Text is redacted when it is stored.
The orchestrator hands the window to a route as the ``conversation_so_far`` detail of the intent it runs; that key is
listed in ``decision.TRANSIENT_SLOTS``, so it never changes a fingerprint.
"""

from collections import defaultdict, deque

from orchestrator.sanitize import redact

KEY = "conversation_so_far"
WINDOW = 6  # exchanges kept per request
ASKED_CHARS = 300
SAID_CHARS = 500
TOTAL_CHARS = 1500


class Talk:
    def __init__(self) -> None:
        self._exchanges: defaultdict[str, deque[tuple[str, str]]] = defaultdict(lambda: deque(maxlen=WINDOW))

    def add(self, intent_id: str, asked: str, said: str) -> None:
        self._exchanges[intent_id].append((redact(" ".join(asked.split()))[:ASKED_CHARS], redact(said.strip())[:SAID_CHARS]))

    def render(self, intent_id: str) -> str:
        """The window as plain lines, oldest first; the oldest whole exchanges go first when it is too long."""
        entries = [f"Asked: {a}\nCustomer: {s}" for a, s in self._exchanges.get(intent_id, ())]
        while len(entries) > 1 and sum(len(e) + 1 for e in entries) > TOTAL_CHARS:
            entries.pop(0)
        return "\n".join(entries)

    def forget(self, intent_id: str) -> None:
        self._exchanges.pop(intent_id, None)
