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
