"""The buttons the demo page shows for a question, checked on the real functions in widget.html (needs node; skipped without)."""

import json
import pathlib
import shutil
import subprocess

import pytest

WIDGET = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "widget.html"
CARDS = (
    "Which card do you mean?\n- the card ending 8765 (pending, on your Green Account)\n"
    "- your Business Silver Rewards Card (credit card)\n- your Platinum Rewards Card (credit card)"
)
SCRIPT = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const grab = n => { const a = html.indexOf('function ' + n); return html.slice(a, html.indexOf('\n}\n', a) + 3); };
const mk = (tag, cls, txt) => ({ tag, cls, txt, children: [], append(...c) { this.children.push(...c); }, querySelectorAll() { return this.children; }, set onclick(f) { this._c = f; } });
const { questionOptions, choiceBar } = new Function('el', 'send', ['questionOptions', 'choiceBar'].map(grab).join('\n') + ';return {questionOptions, choiceBar}')(mk, () => {});
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map(c => { const bar = choiceBar(questionOptions(c.question), c.asks); return bar ? bar.children.map(b => b.txt) : null; })));
"""


def buttons(*cases):
    run = subprocess.run(["node", "-e", SCRIPT, str(WIDGET), json.dumps(cases)], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_question_with_options_gets_one_button_per_option_and_nothing_else_gets_buttons():
    ask = [{"slot": "card_id", "kind": ""}]
    got = buttons(
        {"question": CARDS, "asks": ask},
        {"question": "Sorry, I didn't catch which you meant. Please reply with just a number:\n1. your A\n2. your B", "asks": ask},
        {"question": "How much would you like to increase your credit limit by?", "asks": ask},
        {"question": "Please confirm: close the card (yes or no)?", "asks": [{"slot": "confirmation", "kind": "yes_no"}]},
        {"question": "Recent charges on your card:", "asks": []},
    )
    assert got[0] == [
        "the card ending 8765 (pending, on your Green Account)",
        "your Business Silver Rewards Card (credit card)",
        "your Platinum Rewards Card (credit card)",
    ]
    assert got[1] == ["your A", "your B"]
    assert got[2] is None, "an amount is typed"
    assert got[3] == ["Yes", "No"]
    assert got[4] is None, "a result the agent reports is not a question"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_done_reply_that_also_asks_a_question_is_split_so_the_question_is_not_inside_the_done_card():
    script = r"""
    const fs = require('fs');
    const html = fs.readFileSync(process.argv[1], 'utf8');
    const grab = n => { const a = html.indexOf('function ' + n); return html.slice(a, html.indexOf('\n}\n', a) + 3); };
    const start = html.match(/const QUESTIONS_START = [^\n]+\n/)[0];
    const splitDone = new Function(start + grab('splitDone') + ';return splitDone')();
    const cases = JSON.parse(process.argv[2]);
    console.log(JSON.stringify(cases.map(splitDone)));
    """
    ask = "To go on, I need a little more information:\n- I'm still not sure I have this right. (I still need an answer to this.)"
    cases = ["Done: Order a replacement card. The old card will be cancelled.\n" + ask, "Done: Order a replacement card.", "Which card do you mean?", "Done: A\nDone: B.\n" + ask]
    run = subprocess.run(["node", "-e", script, str(WIDGET), json.dumps(cases)], capture_output=True, text=True, check=True)
    one, only_done, no_done, two = json.loads(run.stdout)
    assert one["done"] == "Done: Order a replacement card. The old card will be cancelled." and one["rest"] == ask
    assert only_done == {"done": "Done: Order a replacement card.", "rest": ""}
    assert no_done == {"done": "", "rest": "Which card do you mean?"}, "a reply that is not a Done is left alone"
    assert two["done"] == "Done: A\nDone: B." and two["rest"] == ask
