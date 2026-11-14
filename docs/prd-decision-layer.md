# PRD: decision layer (dialogue repair, then Jev used as a decision layer)

## Context
Live use showed the system failing in conversations, not in rules:
- **Carlos, "cancel some debit cards"**: four identical "Which card do you mean?" replies, then a hand-off. "evergreen and green accounts", "3847 and 8165", "those" and a pasted list line all failed, because the close path (`debit.prepare`) accepts one exact card id or four digits.
- **Lost wallet**: a "yes" or "yes the silver card as well" was read as a new request, scored alone by Jev (0.75 to 0.81) and handed to a person.
- **Fatima, "I want to see my silver reward card"**: Jev kind 0.53 gave a hand-off for a harmless read.
- **Zoom dispute**: a reply to "tell me more" was scored alone and escalated.

Jev's intended pattern (the Stackademic article and TypeSafe's docs; vendor claims, not measurements): Jev judges typed questions (Choice, Score, Noul) against rich application state; thresholds live in code and differ by the cost of a mistake; simple cases are deterministic code; an LLM runs only on the chosen path; outcomes are logged to tune. We use one Noul per intent on a short summary, one 0.85/0.65 band for everything, and log nothing.

Decisions made with the owner:
1. Order: dialogue repair first, then Jev, then measurement.
2. Low confidence on a first look never goes to a person. Ask one question built from the top readings. A person only after two failed repairs, an explicit request for one, or fraud/security cases with no tool.
3. `NO_YES_NEEDED` shrinks to freeze and unfreeze (plus reads). Email change, recurring block and PIN change always show the exact change and need a yes.
4. No LLM tests. Everything is verified offline with scripted decomposer, Jev and model fakes. Calibration uses only decision logs that the owner's own use of the demo records passively.

## What (goals)
- A request that names cards, accounts or charges in ordinary language resolves correctly: sets, quantifiers, partial names, references to the list just shown.
- No question is repeated verbatim; the repair strategy changes each time.
- Jev gets richer typed questions and its answers feed risk-tiered thresholds in one table.
- Low confidence asks with options built by code.
- Every decision is logged with its Jev answers and outcome.

Non-goals: replacing Jev or the decomposer, an LLM choosing actions (tried and reverted), changing any bank rule or specialist workflow, any live LLM or Jev run.

## Challenged assumptions
1. Low score does not always mean an unclear request: often a vague summary, or a short reply scored alone. Asking is right; handing over is not.
2. More context for Jev is not proven to help (the Silver-card kind score went down, 0.81 to 0.75, with conversation and products added). Each enrichment is behind a flag, in shadow.
3. A Choice distribution over 22 kinds is not known to be calibrated; it ranks readings first, thresholds on it come after calibration.
4. A lower threshold is safe only where the customer says yes to an exact plan, or the action is undoable (freeze). An email change is neither.
5. Jev returns numbers, no reasons; any wording of a doubt comes from code templates.
6. Synthetic wrong intents measure obvious mistakes; calibration needs real conversations.
7. Escalating after N asks created loops; the ladder must change strategy at each step.

## How
**Phase 1 (offline, no Jev change).** `agent/entities.py`: `resolve_cards(text, shown, owned, db)` built on `agent/cards.named_cards`, wired into debit and credit `prepare` and the direct recipes. A repair ladder per intent in the orchestrator (ask with candidates, reword with numbered options, offer the likely reading or a person, hand off) with a guard against repeating a question. An interpretation echo before a multi-card or ambiguous action, and one freeze-or-close question for "cancel".

**Phase 2 (behind flags).** `orchestrator/jev_state.py` builds state from facts (types and booleans only). A Choice question over kinds; sibling kinds with the same route and risk class are one reading. A risk-tiered bands table in `bands.py` (READ 0.70/0.40, REVERSIBLE 0.80/0.55, IRREVERSIBLE 0.85/0.60, URGENT 0.70/0.40 as placeholders). Parallel signals in shadow: asks for a person, answers the question on screen, frustration 0 to 4.

**Phase 3.** One redacted JSONL decision log, a shadow gate beside the live one, and an offline calibration report that makes no network call.

## Constraints and risks
- Jev gets no detail values and no secrets; redaction unchanged.
- Every new Jev call has a timeout and a deterministic fallback (ask with options). Tests use scripted fakes.
- At most 4 Jev questions per intent, in parallel.
- Irreversible changes still need the customer's yes to the exact plan at any score; ownership, validators and dry run unchanged.
- Band numbers stay placeholders until the owner's own sessions are logged. A lower READ band can run a wrong read; a read changes nothing and the answer says what it showed.

## Edge cases (each is a test)
See the table in the plan; summary: "green" is not "Light Green"; "those/both/the second" resolve against the list on screen; a pasted list line; a card not theirs, closed or already frozen is reported and never swapped; "actually only X" replaces the earlier selection; "hmm" re-asks differently; Jev down means ask with options, never a hand-off; two kinds on one route are one reading; each intent in a multi-intent message is decided alone; an injection-flagged message is not sent to Jev; a request for a person during a protection plan is still deferred; a frustrated customer is offered a person after one failed repair; an email change needs a yes at any score.

## Issues and acceptance criteria (label `decision-layer`)
**S0 Conversation fixtures** (blocks the rest)
- [ ] Scripted scenarios for Carlos (cancel, two cards, "those"), the wallet yes loop, Fatima silver card, Zoom clarification, Marriott search; each states the expected final state and reply.
- [ ] Each fails on the current code for a documented reason (list below).
- [ ] Offline in `pytest`.

**S1 Shared entity resolver**
- [ ] `resolve_cards` passes every card row of the edge-case table.
- [ ] Never returns a card that is not the customer's; never picks for them.
- [ ] Credit variant resolves type names, account ids, last four.

**S2 Resolver wired in everywhere a card is named**
- [ ] Debit and credit `prepare`, direct recipes and history use one implementation.
- [ ] The Carlos scenario passes: Evergreen and Green closed together with one plan and one yes.
- [ ] A closed, frozen or unknown card named is reported, not substituted.

**S3 Repair ladder**
- [ ] No question string is shown twice for one intent (property test).
- [ ] Ladder steps asserted in order.
- [ ] A numbered answer resolves against the options shown.

**S4 Interpretation echo and "cancel"**
- [ ] Multi-card or ambiguous actions echo what was understood before the plan.
- [ ] "Cancel" on an ACTIVE card with no reason asks freeze versus close once; either answer proceeds without re-asking.
- [ ] Freeze stays no-yes; close shows the plan and needs a yes.

**S5 Jev state with facts**
- [ ] `jev_state.build` returns only types and booleans (a test scans values for digits, ids, amounts).
- [ ] Behind `JEV_STATE=facts`; default unchanged until S10.
- [ ] Golden snapshots of what Jev would be sent for each S0 fixture (no model call).

**S6 Choice question and route equivalence**
- [ ] Choice criteria come from `capabilities.py`; every kind once.
- [ ] Sibling kinds with the same route and risk class are one reading.
- [ ] Top two readings give a templated alternative question.

**S7 Risk-tiered bands, ask not escalate, email needs a yes**
- [ ] One table in `bands.py`; exact-edge and NaN tests per class.
- [ ] Below the ask band on a first look: ask with options.
- [ ] `UPDATE_PROFILE` needs a yes to the exact change; tests updated.

**S8 Parallel signals in shadow**
- [ ] Three signals requested in parallel with timeouts.
- [ ] Logged, never change a decision in this slice.
- [ ] A Jev failure leaves behavior unchanged.

**S9 Decision log and shadow gate**
- [ ] One redacted JSONL record per decision, outcome filled on the next turn.
- [ ] Shadow gate beside the live one; disagreements counted, never acted on.
- [ ] No PIN, CVV, card number or detail value in any record (test).

**S10 Calibration report and final thresholds**
- [ ] Offline report: per class AUROC, ask rate, wrong-run rate, sweep table.
- [ ] No network call (a test blocks them); states how many logged decisions it rests on.
- [ ] Final table and flag defaults recorded under Decisions.

**S11 Docs**
- [ ] This PRD gets "what changed while building"; README and CLAUDE.md updated.

Dependencies: S1 after S0; S2 after S1; S3 after S0; S4 after S2, S3; S5 after S0; S6 after S5; S7 after S3, S6; S8 after S5; S9 after S7; S10 after S8, S9; S11 after S10.

## Decisions
- Initial band numbers per risk class are placeholders; S10 sets them.
- Whether the frustration or "answers the question" signals replace heuristics is decided by S10 evidence.
- Margin for treating two readings as ambiguous starts at 0.15.

## What changed while building it
- Built offline only: S1 to S10 in code with scripted fakes; no model or Jev call was made. S0 is `tests/orchestrator/test_conversations.py` (Fatima, wallet, Zoom, scorer down, email) plus `tests/agent/test_debit_prepare.py` (Carlos).
- The ask line of every class now only separates "ask" from "log as low"; both ask. A person comes from the ask limit (3 questions) after the ladder reworded the question twice, from a request for a person, and for kinds that always hand off.
- IRREVERSIBLE keeps 0.85/0.65 (the plan said 0.60 for the ask line): since low scores ask instead of escalating, the ask line no longer changes behavior.
- `cancel`: asking freeze or close is in the debit specialist's `prepare` (`cancel_means` detail). Choosing freeze closes nothing and tells the customer how to freeze; it does not switch the request to a freeze, because the specialist's allowed writes follow the original kind.
- Echo (S4): a multi-card close is shown by the plan preview that already lists each step and needs a yes; a separate echo sentence was not added.
- Credit and direct freeze/unfreeze take "the second one", "2" and "those" (`ordinal_picks`, `resolve_cards`); the credit side does not use the full resolver because its cards have no last-four field on the card record.
- S8 signals are asked in the same Jev call as the kind question when `JEV_SIGNALS=1`, logged in `signals`, and decide nothing.
- Not done: GitHub issues for the slices; any threshold change from data (the decision log is empty until the demo is used); the decomposer still does not fill card details for the fraud containment.
