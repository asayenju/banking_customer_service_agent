# Banking customer service agent

A typed multi-agent customer service system for a bank, built on the `banking_knowledge` domain of tau2-bench. A customer writes in plain language; the system works out what they want, checks who they are, asks only what it needs, shows the exact changes it will make, and carries them out after one yes. The model never decides a rule, a fee or a step: it only reports what the customer said, and code does the rest.

Two things this project is mostly about: **where Jev, a non-LLM decision model, sits in the pipeline and what it is trusted with**, and **context engineering**, which turned out to be the hardest part. Both have their own section below.

## How it works

```
customer message
  -> decomposer (LLM, strict JSON)      splits it into typed requests ("intents"); sees the last 8 messages
                                        and the question currently on screen
  -> Jev scorer + code gates            how sure are we? who is calling (2 of 4 identity fields)? what is missing?
  -> decision per intent                run / ask one question / show a plan and ask for a yes / hand to a person
  -> plan (waves)                       urgent first, dependent requests wait, independent ones run together
  -> route
       direct recipes                   freeze, balance, charge lists, referrals: no model
       specialists (6)                  debit, credit, accounts, fees, fraud composer, general answers
       hand-off                         a person, with a note written by code
  -> one reply, one question at a time
```

Each **specialist** is a configuration of one generic class. In plan mode a bounded model loop reads the customer's records (search tools for charges and cards, a records lookup, policy search) and ends by filling in a **workflow form** (what the customer said). Code then decides: validators check every tool call against the bank's written policy (the bank's own tools check very little), a dry run on a copy of the database shows the final state, and the customer sees one preview and gives one yes bound to those exact steps. Execution runs with no model at all and re-checks each step first. Nothing irreversible runs without that yes.

## Where Jev is included

Jev (TypeSafe AI) is a hosted decision model. It answers yes/no questions with a probability and writes no text. It is used in exactly one place: **the gate between "the decomposer found a request" and "the system acts on it"** (`orchestrator/jev.py`, `JevScorer`). Nothing else in the system calls it.

```
decomposer (LLM) finds "dispute this charge"
        |
        v
   Jev is asked, per request:   "is that kind right, given what the assistant can do for it?"   -> probability
                                "is this something a retail bank's customer service handles?"  -> probability
        |
   code adds a third score:     details complete?  (plain code: 1.0 when nothing is missing, else 0.0)
        |
        v
   confidence = the weakest of the three  ->  decision layer (proceed / ask / hand off, by risk class)
```

**What Jev is shown** (`_state` in `jev.py`): the redacted customer message, the decomposer's reading (kind, summary, the words it pointed at), what the assistant can really do for that kind, the *names* of the details found and missing, the last four messages, the kinds of product the customer holds, and the customer's other open requests with any order they asked for. **Never** a detail value, an id, an amount, a date or a digit. A message flagged as a possible injection is not sent at all, and its requests score zero.

**What Jev is trusted with, and what it is not:**

| Jev decides | Code decides |
|---|---|
| how sure we are that the request was read correctly | whether the details are complete (Jev is weak at numbers and dates) |
| a second reading ("X or Y?") when it is torn between two kinds | who the caller is, which tools may run, which records are theirs |
| (logged only) whether the customer asked for a person, answered the question on screen, or is frustrated | the plan, the fees, the rules, and the customer's yes |

- **It never authorizes anything alone.** An irreversible action still needs the customer's yes to the exact plan, at any confidence. Thresholds depend on what a wrong run costs (read, reversible, irreversible, urgent; `orchestrator/bands.py`, numbers are placeholders until `agent.calibration` has real data), and a low score *asks the customer what they mean* instead of handing over.
- **Why not just ask an LLM?** Jev returns numbers only, so a customer's words cannot steer it into saying or doing something, and its mistakes are independent of the decomposer's. It is pinned to `jev-1.13.0` because `jev-latest` moves, runs with a 4 second timeout, and a failure is recorded as the exception type only, never text.
- **Extras behind switches, off by default:** `JEV_CHOICE=1` asks one question over every request kind, to find the runner-up reading and offer "X or Y?"; `JEV_SIGNALS=1` asks for the person, answer and frustration signals, which are logged and compared with the code's own checks and decide nothing; `JEV_STATE=facts` adds application facts (verified, a plan waiting, how often this was asked) as words and yes/no only.
- **A design we tried and reverted.** We let a model decide what to do next with each request and called Jev only as an optional tool. It was reverted: Jev gates every request again.

Measured on 70 labelled intents (40 right, 30 wrong; the wrong ones are a plausible action swapped in, so this tests obvious mistakes):

| Gate | Wrong intents avoided | Right intents blocked | AUROC |
|---|---|---|---|
| Jev, kind score at 0.95 (the earlier write bar) | 29 of 30 (97%) | 2 of 40 (5%) | 0.97 |
| Plain gpt-4o-mini yes/no, P(yes) at 0.95 | 28 of 30 (93%) | 9 of 40 (22%) | 0.95 |

Both catch nearly all wrong actions. Jev's edge is blocking far fewer right requests, which means fewer needless clarifying questions. It is a small sample, so read it as direction, not a precise gap. The measuring scripts are the `jev` command of `orchestrator.evaluation`. Every decision is logged beside what the old single gate would have done (`evals/decisions.jsonl`), so the bands can be tuned from real use. See `docs/prd-decision-layer.md`.

## Context engineering: the hardest part

Most of the bugs found while using the demo were not wrong rules or wrong tools. They were the system and the customer disagreeing about *what the conversation was about right now*: which question was open, which request an answer belonged to, what had already been done. Getting a model to read the right thing at the right moment, and getting code to hold everything that must be exact, was harder than any workflow.

**The principle:** the conversation is state held in code, not a transcript every part reads. Each part is handed only the slice it needs, in the form it can use, and anything that must be exact (what a yes covers, which question is on screen, what is finished) lives in typed state, never in prose a model has to interpret.

### Problems before, and how it works now

| Problem we hit | What it looked like | How it works now |
|---|---|---|
| **An answer became a new request** | The customer answered "no" or "I noticed just now" and the decomposer made a fresh request from it, so two requests existed for one dispute | The decomposer is told the question on screen for each waiting request and says whether the message answers it. Code binds a short reply to the one open question, folds a copy of the same request into the open one, and treats "no" as an answer, not a withdrawal |
| **Questions came as a list** | Several questions in one reply, answers mixed up, a yes spread across them | One question at a time. The code holds only the question that was shown, so the next reply binds to it. A yes to what is about to be done is never spread out |
| **A yes meant different things** | A plain yes confirmed the wrong plan, or a question, or nothing | A yes is bound to a fingerprint of the exact steps. Plans are shown before questions, the decomposer is told which plans are waiting, and a changed detail makes the yes stale |
| **Finished work was re-made** | A request already done came back as new | The decomposer is told which requests are finished, and a list offered to pick from is not counted as finished |
| **Specialists had no memory** | Each run started cold, so earlier answers were asked again | Each specialist gets the request, the customer's answers as a short conversation (`conversation_so_far`), the last 8 messages, and a ledger of what has really been done and what a plan will undo |
| **Jev saw only the message** | It judged "dispute this" with no idea what the assistant could do or what was just said | Jev is told the decomposer's finding, the assistant's capabilities for that kind, the recent conversation, products held and the other open requests |
| **The model guessed what records hold** | It asked the customer for a transaction id, or made up a date | Look first, then ask: search tools over the customer's own records, no questions about an id the records hold, and a date the model writes is checked against the customer's own words |
| **The customer was asked to type what could be tapped** | "Which charge?" in words, then a typo | The server offers the buttons (a card, then that card's recent charges with a Dispute button each). A press sends only the offer's id, so no model reads it, and it still passes the identity check, the plan preview and the customer's yes |
| **Secrets in text** | A PIN typed in a message travelled to the model and the log | Swapped for a reference such as `{secret:pin}` before any model, trace or reply sees it; the real value goes back in only at the guarded tool call |

### What a conversation holds

One `Conversation` per id (`agent/assembly.py`):
- The caller's verification state (2 of 4 identity fields), set by code only.
- Every open request: kind, a short summary, and its details. Code adds `said` (the customer's own words when they reported a loss) and `conversation_so_far`.
- The score each request earned, how many times it has been asked about, and the one question currently on screen.
- The approved plan and its fingerprint, the steps already done, and the stored result of every message id (a repeated id runs nothing twice).
- The last 8 messages (each cut to 600 characters, secrets hidden), one `BankingSession` (a private copy of the database and an audit log) and one `SecretVault`.

### What each part is given

| Part | It sees |
|---|---|
| Decomposer | the message, the open requests, the last 8 messages, the question on screen for each waiting request, the plans waiting for a yes, and the requests already finished |
| Jev | the redacted message, the decomposer's reading, the assistant's capabilities for that kind, detail names (never values), the last 4 messages, products held, and the other open requests |
| Specialist | a fresh run each time: the request, the recent conversation, what has been done, policy search, and the customer's own records through search tools |
| Route | its request plus the recent conversation; it never sees other requests in the same turn |

Between specialists nothing is passed directly. They share only the database, so a card one specialist closes is already closed when the next reads it.

### How we found the problems

Every turn through the demo is recorded as one flat record (`orchestrator/observe.py`): decisions and why, events, outcomes, agent runs, latency and a `failure_class` worked out in code (`repeated_ask`, `workflow_refused`, `ask_limit_escalation`, `decomposer_failed`, ...). The records hold names, kinds, reasons and counts, never message text. That is safe, and it is also a limit: a repeated question shows *that* it repeated and which intent held it, not what the customer typed. The demo's Debug panel, `GET /debug/summary` and `uv run python -m orchestrator.observe replay <conversation>` show them.

### Still open

- **The decomposer still sometimes makes a new request out of an answer.** The fixes above catch most cases, but a customer's reply that is long, reads as an affirmation, or arrives while two questions are open can still slip through. The symptom is a stale "what do you mean?" that keeps coming back; a thank-you now closes it, but the cause is not closed.
- **Relative dates.** "I noticed just now" has to become a calendar date. The model sometimes guesses one; the workflow refuses a date the customer's words do not back, and the retry costs time and a repeated question. The date should be worked out in code.
- **Stale clarifications.** A "what do you mean?" is closed at a thank-you but not yet when its own request finishes.
- **Identity re-asks** are counted as `repeated_ask` in the observability records, which makes that class noisier than it should be.
- **Calibration.** The decision bands rest on little labelled data. `agent.calibration` reads the decision log; no band number should change before it has been run on real use.

## What we learned from using the demo

Offline replays passed long before the demo felt right. Each item below came from a real conversation with a test customer, and each was fixed in code, not in a prompt.

| What happened | What we changed |
|---|---|
| A debit dispute asked "which transaction?" in words | The agent asks **which card** first, then lists **that card's charges with a Dispute button each**, built in code with no model (`agent/debit_charges.py`, `specialists/debit.py`). A press sends only the offer's id. Credit works the same way |
| A customer said the cash back looked wrong and the agent asked them to pick the purchase | The agent now **finds the wrong rewards itself**: it reads the customer's cards and purchases and recalculates each one from the card's earning rules (`agent/rewards_audit.py`, `rewards_review.py`). For each wrong purchase it plans a `submit_cash_back_dispute_0589` step (the customer-side tool, added to the wrapper with source "user"), shows the plan, and submits after one yes |
| A fee, interest or rewards complaint went to the wrong place or was dropped | Saying an amount "looks wrong" is a request for a review (`FEE_OR_INTEREST_ADJUSTMENT`, `orchestrator/policy.py`), so it goes to the fees specialist. The credit-card lookup also lists credit cards now |
| A customer named a card they do not hold ("my Gold Rewards card") and the agent went looking | Code checks the customer's own cards first and says plainly that they do not have that card (`_card_not_held` in `recipes/direct.py`). A "false verification" button on the test customers lets a tester try exactly this |
| "Thank you, that's all" opened a new request or left an old question hanging | A thank-you closes the chat kindly and clears any stale clarification (`unclear.py`, `orchestrator.py`) |
| Test customers had no first line, or many were near-duplicates | Every customer card has an opening line taken from the script (or written by hand), and the debit and fees lists keep only distinct customers. Accounts and credit come first in the demo |
| Pressing a customer card redrew the whole list | Each card is built once, opens and closes with an animation, and its brief is cached |

Smaller lessons that kept repeating:
- **Look before asking.** If the records can answer, a question to the customer is a bug. Every "which one?" became either a lookup or a button.
- **Buttons beat typed answers**, and the server must be the one that offers them. A button that carries only an id cannot be spoofed into a different request.
- **Put the logic where it can be tested.** The rewards check, the card-not-held check and the charge list are plain functions with offline tests. The model is asked only to report what the customer said.
- **Offline first, live last.** The default test run blocks the model API on purpose, so a live call never happens by accident. 2,765 tests passed at last count, and `mypy --strict` is clean on the typed packages.
- **Restart after every change.** The server keeps the old code in memory, so a fix looks like it did nothing until the process restarts.
- **Observability that records no text is safe but blunt.** It tells you a question repeated, not what the customer wrote.

## Structure

| Path | What is there |
|---|---|
| `backend/src/banking_tools/` | typed wrapper over 53 tau2 bank tools: kinds, reversibility, audit log, per-session database copy |
| `backend/src/orchestrator/` | decomposer, Jev scorer, identity gate, decisions, plans, routes, replies, trace, observability; evaluation harness in `evaluation/` |
| `backend/src/agent/` | specialists (`specialists/`), workflows with the bank's rules (`workflows/`), search tools (`credit_lookup.py`, `debit_lookup.py`, `debit_charges.py`, `overview.py`), the wiring (`assembly.py`), offline reports, the live harness (`live.py`), the free preflight (`preflight.py`), the simulated customer (`customer.py`) |
| `backend/src/main.py` | FastAPI: `POST /conversations/{id}/messages`, the test customers, and the demo page |
| `frontend/widget.html` | demo chat: pick one of the test customers on the left, talk to the agent on the right; buttons appear only where the server offered them |
| `docs/` | PRDs, and one policy-decisions note per area with the document or data behind every rule |
| `tau2-bench/` | git submodule (the fork `asayenju/tau2-bench`); never edited |

## Run the demo

```bash
cd backend
PYTHONPATH=src:../tau2-bench/src uv run uvicorn main:app --port 8765
# open http://localhost:8765/demo   (needs OPENAI_KEY and JEV_KEY in the repo-root .env)
```

Restart the server after any code change; the running process keeps the old code.

## Deploying

The backend runs on **Railway** and the demo page can run on **Vercel**. The page works from either: served by the backend at `/demo` it uses relative URLs, and anywhere else it calls the Railway backend (the `API` constant at the top of the script in `frontend/widget.html`).

**Backend on Railway**
- The `Dockerfile` at the repo root builds it. `tau2-bench` is a submodule, so the build fetches the pinned commit from GitHub instead of uploading its 880 MB of data (update `TAU2_COMMIT` in the Dockerfile when the submodule moves).
- Set `OPENAI_KEY` and `JEV_KEY` as Railway variables. The app reads real environment variables first, and there is no `.env` in the image.
- Deploy with `railway up --service backend --ci` from the repo root. It is not connected to GitHub, so a push does not deploy.
- `.railwayignore` keeps the submodule, caches and local logs out of the upload.

**Page on Vercel**
- Import the repo, set the root directory to `frontend`, framework "Other", no build command. `frontend/vercel.json` maps `/` to `widget.html`.
- The backend allows browser calls only from the Vercel page's origin and a local server (`CORSMiddleware` in `main.py`). A new domain, or Vercel preview URLs, must be added to that list.

**Things to know**
- CORS only controls browsers. Anyone with the backend URL can still call the API directly and spend OpenAI credits, so set a spending limit on Railway (workspace Usage settings) and a monthly budget on OpenAI, and consider a password or rate limit.
- `evals/turns.jsonl` and `decisions.jsonl` are written to the container's disk, which is wiped on every deploy. Add a Railway volume if the history must survive. The Debug button still works live.
- Railway Hobby allows 50 projects and includes $5 of usage a month. This backend is one small container.

## Results so far

Offline replay of the benchmark's own reference trajectories through the real specialists, compared with tau2's database state (`uv run python -m agent.pilot`, `agent.credit_report`, `agent.preflight`):

- **60 of 62 gradeable tasks reproduce the reference state exactly**; the other 2 are documented departures where written policy and the reference disagree. 35 more of the 97 are not gradeable (customer-side only, human-transfer only, one broken reference).
- Debit 12/12, accounts 19/19, lost wallet 5/5, credit 17/18, fees 9/10. 0 writes outside an approved plan in every report.
- 2,700+ offline tests, `mypy --strict` on the typed packages.
- Decomposer pass@1 (older measurement, before the latest prompt changes): 0.92 tuning set, 0.97 held-out, 1.00 minimal pairs, 0.83 on the hardest set.

These are offline and scripted. The context problems above were found in live use of the demo, which the offline replays do not exercise.

## System design

- **The model reports, code decides.** A model never sets a fee, a limit, a date rule or a step. It fills in a typed workflow form with what the customer said, and the workflow works out the rest from the bank's written policy (`agent/workflows/`). Every write tool also has a validator that runs when the plan is made and again just before each step.
- **One yes to an exact plan.** The customer sees the steps in plain words, and their yes is bound to a fingerprint of those details. Reads and reversible actions such as a freeze need no yes; everything irreversible does.
- **Plan first, then act with no model.** A plan is dry-run on a deep copy of the database. Execution replays only the approved steps through a guard (the request's allow-list of tools and the customer's ownership of every record), then reads back the result.
- **One generic specialist.** Debit, credit, accounts, fees, fraud and the general answerer are configurations of one `Specialist` class. The lost-wallet composer joins the debit and credit workflows into one preview, and freezes the named cards first because a freeze is reversible.
- **Look first, then ask.** Specialists have search tools over the customer's own records. A dispute may not ask the customer anything until a search has been used.
- **Failures are values.** Tools return `ok=False` results, a failed route is reported while independent work carries on, and anything unclear or failing ends in a hand-off with a note written by code.
- **Evidence before spend.** Offline replay, a free preflight over all 97 tasks, and a capped live harness with recorded model responses mean live runs are the last step, not the first.

---

## Setup

`tau2-bench` is a git submodule (the fork `asayenju/tau2-bench`), and the backend depends on it as an editable package.

```bash
git clone --recurse-submodules https://github.com/asayenju/banking_customer_service_agent.git
cd banking_customer_service_agent/backend
uv sync          # Python 3.12 (tau2 requires >=3.12,<3.14)
uv run pytest    # offline; no API keys needed
```

If `import tau2` fails with `No module named 'tau2'` on macOS, the editable-install `.pth` file in `backend/.venv` has been marked hidden (iCloud does this to `~/Documents`) and Python ignores it. `pytest` no longer depends on it, but scripts and the `tau2` command do. Fix: `chflags -R nohidden backend/.venv`.

## Banking tool wrapper (`backend/src/banking_tools/`)

One typed interface over the agent-side tools of tau2's `banking_knowledge` domain. Every tool has typed arguments, a kind, and, for writes, a reversibility.

```python
from banking_tools import BankingSession, get_registry

get_registry()["freeze_debit_card_3892"]
# ToolSpec(kind=write, reversibility=reversible, args_model=<pydantic model>, ...)

session = BankingSession()          # private copy of the banking database
result = session.call("freeze_debit_card_3892", {"card_id": "dbc_lj82d4f1a9_bluest"})
result.ok, result.text, result.error, result.audit_id
session.audit_log                   # one AuditEntry per executed write
```

### Labels

| Kind | Meaning |
|---|---|
| `read` | returns data, changes nothing |
| `write` | changes banking data; always `reversible` or `irreversible` |
| `handoff` | signals a transfer to a human, changes nothing |

**Reversible** means a tool in the set can restore the prior state, given what the caller read beforehand (for example `freeze_debit_card` and `unfreeze_debit_card`). Every other write is irreversible.

The full table of 53 catalogued tools, and the 5 excluded ones with reasons, is in [`docs/banking-tool-catalog.md`](docs/banking-tool-catalog.md). The labels themselves live in `backend/src/banking_tools/catalog.py`.

### How it differs from tau2

- **No unlock step.** In tau2, 44 tools must be unlocked and then called through `call_discoverable_agent_tool` with a JSON string. Here each is called directly with validated arguments.
- **No tau2 audit row.** tau2's dispatcher writes a row to `agent_discoverable_tools` for mutating tools. The wrapper does not; `session.audit_log` replaces it. The log redacts `pin`, `new_pin`, `current_pin` and `cvv`.
- **Numbers become floats**, as in tau2's dispatcher (`parse_int=float`).
- **Failures are values, not exceptions.** tau2 tools return `"Error: ..."` strings; the wrapper maps them to `ToolResult(ok=False, error=...)`. Invalid arguments and unknown tools fail the same way, and the tau2 tool is not called.
- **Isolation.** Each `BankingSession` deep-copies the database and serialises its own calls with a lock.
- **Out of scope:** customer-side (simulated user) tools and tau2's unlock, list and call dispatcher tools.

### Keeping it in sync with tau2

Tests compare the catalog and argument schemas with the installed tau2, so an upstream change fails the build:

- A tau2 tool was added, removed or relabelled: update `_LABELS` or `EXCLUDED` in `catalog.py`, then regenerate the table:
  `cd backend && PYTHONPATH=src uv run python -c "from banking_tools.catalog import render_markdown; open('../docs/banking-tool-catalog.md','w').write(render_markdown())"`
- A tool signature changed: review it, then `UPDATE_SNAPSHOTS=1 uv run pytest tests/test_args.py`.

### Checks

```bash
cd backend
uv run pytest
uv run mypy --strict src/banking_tools src/orchestrator
```

## Orchestrator (`backend/src/orchestrator/`)

The central agent's decision layer. It takes what a customer says, decides per intent whether to act, ask, or hand off to a human, runs the work, and answers once. Everything is plain code except two swappable parts: the LLM that splits a message into intents (the decomposer) and the Jev model that says how sure we are (the scorer). Both have scripted versions for tests and real ones (below); the test suite uses scripted tools and sub-agents, so it runs offline with no keys (the real specialists are in `backend/src/agent/`). Specs: [`docs/prd-orchestrator-routing.md`](docs/prd-orchestrator-routing.md) and [`docs/prd-decomposer.md`](docs/prd-decomposer.md).

```python
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route

orch = Orchestrator(
    decomposer=..., scorer=..., directory=...,        # Protocols; scripted fakes ship in the package
    runners={Route.DIRECT: ..., Route.SUB_AGENT: ...},  # mocks in orchestrator.mocks
)
turn = await orch.handle_turn("msg-1", "my card was stolen, freeze it and tell me my balance")
turn.reply.text      # one message for the whole turn
turn.decisions       # run / ask / escalate per intent, with the reason
turn.trace.events    # every choice, in order, with wave numbers
```

### What it decides

| Rule | Where |
|---|---|
| Confidence per intent is the weakest of kind, slots and scope. 0.85 or more proceeds, 0.65 to 0.85 asks, below 0.65 escalates. | `scoring.py`, `bands.py` |
| Greetings, gibberish and empty messages get a fixed menu without any model, then a human after 3 menus. | `unclear.py` |
| Account-specific intents need a verified caller: 2 of 4 identity fields, checked by code. Bypass codes and supervisor claims are ignored. 3 wrong answers lock the caller out. | `auth.py` |
| Irreversible actions never run without the customer's yes to the exact details, at any confidence. A changed detail makes the yes stale. | `decision.py` |
| Each intent escalates after 3 questions. Changing its details restarts the count; answering a different intent does not reset it. | `orchestrator.py` |
| Urgent intents (fraud or loss report, freeze) go first. Writes run together only when each names a different resource. Dependents of anything not running wait. A cycle hands off. | `plan.py` |
| A failed route is reported, its dependents are skipped, independent work carries on, nothing is undone. | `executor.py` |
| One reply per turn: results, problems with an offer of a human, escalations, and one question (the first missing detail; every yes to what is about to be done is still asked together). | `reply.py` |
| A repeated message id returns the stored result and runs nothing twice. | `orchestrator.py` |

Other modules: `types.py` (intents, decomposition, decisions), `taxonomy.py` (which tool serves which intent kind), `routes.py` (direct, sub-agent or hand-off per kind), `trace.py`, `decomposer.py` (the per-turn interface the LLM will fill), `mocks.py` (scripted routes: success, slow, timeout, malformed, partial, raises).

### Tests

```bash
cd backend
uv run pytest tests/orchestrator                  # the whole orchestrator, offline
uv run pytest tests/orchestrator/test_scenarios.py  # whole conversations, asserted on the trace
UPDATE_SNAPSHOTS=1 uv run pytest tests/orchestrator/test_intent_types.py  # after changing a type
```

`test_scenarios.py` has one scenario per row of the edge-case table in the PRD, plus seven inspired by tau2 banking tasks. A test reads that table, so adding a row without a scenario fails the build. New scenarios use the scripted parts in `tests/orchestrator/scenario_kit.py`.

### Real decomposer and Jev scorer

```python
from orchestrator.jev import make_jev_scorer          # needs: uv sync --extra jev
from orchestrator.llm import LiteLLMClient
from orchestrator.llm_decomposer import LLMDecomposer

orch = Orchestrator(
    decomposer=LLMDecomposer(LiteLLMClient()),        # OpenAI via LiteLLM, temperature 0, strict JSON schema
    scorer=make_jev_scorer(),                         # pinned jev-1.13.0
    directory=..., runners=...,
)
```

Keys are read from the repo-root `.env` (git-ignored; see `.env.example`): `OPENAI_KEY`, `JEV_KEY`, and optionally `DECOMPOSER_MODEL` (default `openai/gpt-4o-mini`). The decomposer redacts PINs, CVVs and card numbers before anything leaves the process, bounds retries, and on unusable output returns one `OTHER` intent so a person takes over; it never raises. `validate.py` adds missing required details, orders intents by position in the message, and flags dropped requests and injection attempts. Jev only sees the redacted message, kind and summary; a message flagged as injection is not sent and escalates. Live checks are opt-in: `RUN_LIVE=1 uv run pytest tests/orchestrator/test_jev.py`.

### Not built yet

LLM phrasing of the reply (a template today) and a real transfer to a human agent. The sub-agents and the evaluation harness are built.
