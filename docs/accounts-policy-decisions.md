# Accounts and money movement: policy decisions

Slice 0 (Issue #58) of `docs/prd-accounts-money-specialist.md`. Every decision cites its evidence: a document, the bank's tool code (`tau2-bench/src/tau2/domains/banking_knowledge/tools.py`), the data, or a task. Statements about what the tools do are tests in `backend/tests/agent/test_accounts_policy.py`, so this note cannot drift from them. Where the tool, the documents and the gold disagree, the documented policy wins (project owner's decision).

## 1. Rules by product

Documents: `doc_bank_accounts_bank_accounts_(general)_001` to `_008` (opening and closing), `_010` (transfers). Today in every task is 11/14/2025.

### Opening
| Product | Rules (all must hold) | Exact class string |
|---|---|---|
| Personal checking (001) | verified; age 18 or more; at most 4 personal checking accounts; no checking account closed for cause in the last 6 months | product page name + " Account" (`Blue Account`, `Purple Account`); the two Green pages are `Green Account` (checking) and `Green Account` (savings), told apart by type |
| Personal savings (002) | verified; an ACTIVE or OPEN checking account held 14 days or more; fewer than 5 personal savings accounts; no account in collections, no negative balance | product page name + " Account" (`Silver Plus Account`, `Gold Plus Account`) |
| Business checking (003) | verified; an OPEN personal checking account with a balance of at least $500; at most 6 business checking accounts; no account with status CLOSED | the product name **without** "Account" (`Navy Blue`, `Sky Blue`, `Cobalt Blue`); see section 8 |
| Business savings (004) | verified; an OPEN business checking account open 30 days or more with a balance of at least $2,500; fewer than 4 business savings accounts; no negative balance on any account | product page name ending "Account" (`Gold Saver Account`, `Silver Plus Saver Account`) |

After opening a personal or business savings account the agent asks whether to transfer the opening deposit now (from checking, or from the qualifying business checking). If the customer says no, they have 30 days to fund it or it is closed (002, 004).

### Closing (005 to 008)
Requirements for every product: the account is OPEN, there are no pending transactions, and the balance is $0, or exactly the early-closure fee if one applies (section 3). A business checking account cannot close while a linked business savings account is OPEN (007); a business savings account cannot close if it is the customer's last business checking link (008).

| Product | Tier: fee, window, notice period, approval |
|---|---|
| Personal checking | Entry (Light Blue, Light Green, Green Fee-Free): $15, 30 days, 0 days. Mid (Blue, Green): $25, 60 days, 3 days. Premium (Evergreen): $50, 90 days, 7 days. Elite (Bluest): $100, 180 days, 14 days |
| Personal savings | Entry (Bronze): $20, 60 days, 1 day. Mid (Silver, Silver Plus): $35, 90 days, 5 days. Premium (Gold, Gold Plus, Gold Years): $75, 180 days, 10 days. Elite (Platinum, Platinum Plus, Diamond Elite): $150, 270 days, 21 days, manager approval |
| Business checking | Entry (Navy Blue): $50, 60 days, 7 days. Mid (True Blue, Sky Blue): $100, 90 days, 14 days. Premium (Cobalt Blue): $200, 180 days, 21 days, supervisor review. Elite (credit lines over $100,000): $400, 270 days, 30 days, manager approval |
| Business savings | Entry (Bronze Saver): $75, 90 days, 10 days. Mid (Silver Saver): $125, 120 days, 14 days. Premium (Gold Saver): $250, 180 days, 21 days, supervisor review. Elite (Platinum Reserve): $500, 270 days, 30 days, manager approval |

**Decision.** An approval requirement (manager or supervisor) is a **hand-off**, not a refusal: the agent has no approval tool, so it writes a note for a person.

## 2. What the bank's tools enforce

| Rule in the documents | `open_bank_account_4821` | `close_bank_account_7392` | `transfer_funds_between_bank_accounts_7291` |
|---|---|---|---|
| Same owner for both accounts (010) | n/a | n/a | **not checked** |
| Source open and funds sufficient | n/a | n/a | checked |
| Amount above zero, accounts differ | n/a | n/a | checked |
| Savings: checking 14 days old | checked | n/a | n/a |
| Savings: cap of 5; no negative balance or collections | **not checked** | n/a | n/a |
| Checking cap of 4, business caps 6 and 4 | **not checked** | n/a | n/a |
| Age 18 | **not checked** | n/a | n/a |
| Business checking: no CLOSED account; OPEN personal checking | checked | n/a | n/a |
| Business checking: $500 in personal checking | **not checked** | n/a | n/a |
| Business savings: 30-day business checking | checked | n/a | n/a |
| Business savings: $2,500; no negative balance | negative balance checked, $2,500 **not checked** | n/a | n/a |
| Exact class string | **not checked** (any string opens) | n/a | n/a |
| Early-closure fee is deducted | n/a | **checked, never debited** | n/a |
| Savings early-closure fee | n/a | **never fires** (tests `class == "savings"`, the data says `"saving"`) | n/a |
| Business early-closure fee, notice period, pending transactions, approval tiers, linked business savings | n/a | **not checked** | n/a |
| Fee waiver needs authority | n/a | **not checked** (any caller can set `waive_early_closure_fee`) | n/a |
| A transaction row is written | n/a | n/a | **no row written** |
| Repeating the call is safe | **silent no-op** ("Failed to open account", reported as a success by the wrapper) | refuses a second close | **moves the money twice** |

Other quirks, all shown in the tests:
- A new account is stored with `account_type` and `account_class`; every other tool reads `class` and `level`. The new account is invisible to class-based tools.
- Its id is a hash of user, type and class, so the id is unknown until the step has run (later steps use `{{step_N.field}}`) and opening the same type and class twice is the silent no-op above.
- Balances are written as `"$x.xx"` by transfer and `"x.xx"` by others; both parse.

**Decision.** Every rule the documents state is a validator we write (Slice 2). A rule for which the data has no field cannot be enforced and is listed in section 8.

## 3. Early-closure fee

- **Documents (005 to 008):** if a fee applies the balance must be at least the fee, otherwise $0. "The fee is deducted directly from the account balance and there is no alternative payment method."
- **Tool:** checks that `balance - fee == 0` and then closes. It never debits the fee. The check is an equality, so a balance above the fee is refused with "Account balance must be $0.00".
- **Task 061** shows the intended flow: the Green account is 44 days old (fee $25, window 60) with $0, the customer deposits a $25 check into it, and the agent closes it. The $25 stays in the closed account. Moving the $25 out would make the close fail.

**Decisions.**
1. The fee-sized balance stays in the account. The specialist never moves it and never sets `waive_early_closure_fee`.
2. Inside the fee window with a balance above the fee, the specialist plans a transfer of `balance - fee` first, leaving exactly the fee.
3. Inside the window with a balance below the fee, the account cannot close. The reply tells the customer how much to add and that a check can be deposited (customer-side, section 5).
4. Outside the window the whole balance moves out first.
5. A request to waive the fee is a hand-off: no document gives the agent authority.

## 4. Transfers

- **Ownership.** The documents require both accounts to belong to the same customer (010); the tool does not check. `agent/guard.py` already refuses an account id (`source_account_id`, `destination_account_id`, `account_id`, `checking_account_id`) that does not belong to the verified customer, as an unknown record; the validator repeats the check and says so in words the customer can read. The data has no joint accounts and no co-owner field, so "owned" means `user_id` equals the verified customer.
- **Task 016 is not a transfer** (it is a credit referral; its gold submits a transaction on a friend's card). Third-party destinations are tested on synthetic fixtures.
- **Amount in the yes.** The orchestrator already binds the customer's yes to the intent's slots (amount included). The plan the specialist proposes is shown with the same amount and accounts.
- **No limits, cutoffs or daily caps exist** in the transfer document or the tool for internal transfers; none is invented. External transfers (2 to 3 business days) have no tool and are a hand-off (Slice 8).
- **Retry.** The tool moves money twice on a repeat. A step already completed for the intent is never run again (step ids and PARTIAL resume, `docs/prd-specialist-agents.md`).

## 5. Customer-side steps in the gold

`deposit_check_3847` is a customer-side tool: the agent hands it over (`give_discoverable_user_tool`) and the customer calls it (tasks 055, 057, 061).
- **Tier-1** already replays a gold action whose `requestor` is `user` directly through `KnowledgeUserTools` (`agent/tier1.py` lines 111 and 343) and ends the surrounding request there, so a close that follows a deposit (061) is its own segment. Nothing in the specialist changes for this.
- **Live and production:** the specialist never calls a customer-side tool. When a deposit would unblock a close, it tells the customer what to deposit and why; the close is planned on a later turn from the balance it then finds.
- The same applies to `apply_for_credit_card` (058, 059, 063 to 069) and `submit_referral`: customer-side, replayed directly.

## 6. Boundaries with other specialists

| Tasks | Owner |
|---|---|
| 055 to 071, 075, 076 (19) | this specialist |
| 043, 045, 048, 051 | credit (#41): they pay a card from checking inside a credit flow, with the credit specialist's own pay validator |
| 072 to 074, 093 to 097 | fees, interest, rewards (#43): checking and savings credits and the interest report |
| 098 to 102 | referral advice (customer-side `submit_referral`); no account write |
| 004, 008, 012, 014, 032 to 035 | transfer-only ACTION tasks: hand-off negatives (Slice 8) |
| 077, 080, 081, 083 and the debit tasks | debit and fraud specialists |

A standalone "pay my card" goes to this specialist (`dispatcher.FIXED` already maps PAY_CREDIT_CARD to ACCOUNTS); the credit workflows call the pay tool themselves and are not rerouted.

## 7. Where the gold departs from policy

I checked, by hand, the start state and the gold writes of the 19 tasks against the rules in section 1 (checking accounts' age, balances and statuses; business checking age and balance; caps; closes inside fee windows). No gold *open* breaks a document rule:
- 060, 061: the savings is opened while a checking account at least 14 days old still exists (Green, 120 and 44 days); closing first would leave only a 10-day-old account.
- 056: the Navy Blue account is 105 days old with $3,200; 062: Evergreen holds $3,500 and no account is CLOSED when the business checking is opened; 071: Cobalt Blue is 60 days old with $12,500.
- 055, 057, 070, 075, 076: no cap or age rule is reached.

One departure is recorded, as a reading rather than a measurement: **the notice periods** (section 1) are not enforced, because every gold close (060, 061, 062, 065 to 069) happens in the same conversation as the request, and the tool agrees. Treating the notice as a delay would make every one of them depart. The hand-off for approval tiers does not apply to any gold close (Green, Light Blue, Evergreen and Bronze have none).

Slice 2 turned the hand check into a test (`backend/tests/agent/test_accounts_rules.py`, `test_every_write_in_the_reference_answer_passes_the_validators_in_order`): every account write of the 19 tasks passes the validators, in order, on the state the earlier writes leave, and each read-back passes. So no gold write departs from the documented rules, and `evals/policy_departures.json` gets no account entry. If a later rule change makes a gold write fail, the test names the task and the rule, and the departure is recorded there with its reason, not fitted around.

## 8. Not derivable

- **Notice period.** Documents list it per tier but say nothing about what it means for a same-day request; section 7 explains why it is not enforced. A person decides if a customer asks for a future-dated close.
- **Collections and "closed for cause".** The data has no field for either, no task mentions them, and every account is OPEN. The rules cannot be checked and are not invented; a customer who says they are in collections is a hand-off.
- **Pending transactions.** Every bank transaction in the data is `posted`. The validator treats any status other than `posted` as pending (so it works if one appears).
- **The "Account" suffix.** Documents 001, 002 and 004 say the class must end with "Account", but business checking is written `Navy Blue`, `Sky Blue`, `Cobalt Blue` in document 003 and in every gold open. The rule used: the string is the one the product page names for that type, and business checking has no suffix.
- **Products with no closing rules.** Documents 005 to 008 list no tier for Purple, Dark Green and Gold Years checking, for Green savings, and for the business products Beige, Hunter Green, Lime Green, World Blue, Diamond Vault, Emerald Saver and Gold Plus Saver. Closing one is a hand-off ("no closing rules on file"), not a guess. (Gold Years has a checking product page but is listed under savings in document 006; the validator follows the page's type.)
- **Elite business checking** is defined by "credit lines over $100,000", which the data does not hold. It is treated as a hand-off if it ever appears.
- **A templating error on a Gold savings page** (`doc_savings_accounts_gold_account_005`: "Transfer fees are Yes for this account tier, reducing the cost of moving funds"). No validator uses transfer fees; the candidate builder (Slice 3) cites the page's other fields only where they read sensibly, and flags the rest.
- **Age limits per product** are on the product pages (Gold Years: at least 62; Dark Green: 17 through 26). The bank's tool checks none of them and the opening documents (001, 002) state only age 18. Slice 3 treats a product page's age limit as part of that product's eligibility; where pages for one product disagree, the product is not offered and the disagreement is listed for a person.
