"""The sliding window of what was said about a request."""

from orchestrator.talk import SAID_CHARS, TOTAL_CHARS, WINDOW, Talk


def test_it_keeps_the_most_recent_exchanges_whole_and_oldest_first():
    talk = Talk()
    for n in range(WINDOW + 3):
        talk.add("x", f"question {n}?", f"answer {n}")
    lines = talk.render("x").splitlines()
    assert len(lines) == WINDOW * 2
    assert lines[0] == f"Asked: question 3?" and lines[-1] == f"Customer: answer {WINDOW + 2}"


def test_a_long_window_drops_whole_exchanges_never_part_of_one():
    talk = Talk()
    for n in range(WINDOW):
        talk.add("x", f"q{n}", ("word " * 200)[:SAID_CHARS])
    text = talk.render("x")
    assert len(text) <= TOTAL_CHARS
    for block in text.split("\nAsked: "):
        assert block.strip().startswith(("Asked: ", "q")) and "Customer: " in block


def test_each_request_has_its_own_window_and_forgetting_clears_it():
    talk = Talk()
    talk.add("a", "q", "one")
    talk.add("b", "q", "two")
    assert "one" in talk.render("a") and "two" not in talk.render("a")
    talk.forget("a")
    assert talk.render("a") == "" and "two" in talk.render("b")


def test_secrets_are_redacted_when_stored():
    talk = Talk()
    talk.add("x", "what is your pin?", "my pin is 7351 and cvv 123")
    assert "7351" not in talk.render("x")
