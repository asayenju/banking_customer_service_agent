"""Which customer goal each catalogued banking tool serves.

tests/orchestrator/test_taxonomy.py checks this table against the tool catalog,
so adding, removing or renaming a tool there fails the build until this is
updated. A tool whose purpose is internal rather than a customer goal maps to
OTHER.
"""

from orchestrator.types import IntentKind

K = IntentKind

TOOL_INTENT: dict[str, IntentKind] = {
    # --- lookups ---
    "get_current_time": K.LOOKUP_INFO,
    "get_user_information_by_id": K.LOOKUP_INFO,
    "get_user_information_by_name": K.LOOKUP_INFO,
    "get_user_information_by_email": K.LOOKUP_INFO,
    "get_referrals_by_user": K.LOOKUP_INFO,
    "get_debit_cards_by_account_id_7823": K.LOOKUP_INFO,
    # --- balances and history ---
    "get_credit_card_transactions_by_user": K.ACCOUNT_BALANCE_OR_HISTORY,
    "get_credit_card_accounts_by_user": K.ACCOUNT_BALANCE_OR_HISTORY,
    "get_payment_history_6183": K.ACCOUNT_BALANCE_OR_HISTORY,
    "get_all_user_accounts_by_user_id_3847": K.ACCOUNT_BALANCE_OR_HISTORY,
    "get_bank_account_transactions_9173": K.ACCOUNT_BALANCE_OR_HISTORY,
    # --- profile ---
    "change_user_email": K.UPDATE_PROFILE,
    # --- internal, not a customer goal ---
    "log_verification": K.OTHER,
    # --- human hand-off ---
    "transfer_to_human_agents": K.REQUEST_HUMAN,
    "initial_transfer_to_human_agent_0218": K.REQUEST_HUMAN,
    "initial_transfer_to_human_agent_1822": K.REQUEST_HUMAN,
    "emergency_credit_bureau_incident_transfer_1114": K.REQUEST_HUMAN,
    # --- disputes ---
    "get_debit_dispute_status_7483": K.DISPUTE_TRANSACTION,
    "get_atm_deposit_images_8473": K.DISPUTE_TRANSACTION,
    "get_user_dispute_history_7291": K.DISPUTE_TRANSACTION,
    "file_credit_card_transaction_dispute_4829": K.DISPUTE_TRANSACTION,
    "file_debit_card_transaction_dispute_6281": K.DISPUTE_TRANSACTION,
    # --- card lifecycle ---
    "freeze_debit_card_3892": K.FREEZE_CARD,
    "unfreeze_debit_card_3893": K.UNFREEZE_CARD,
    "activate_debit_card_8291": K.ACTIVATE_CARD,
    "activate_debit_card_8292": K.ACTIVATE_CARD,
    "activate_debit_card_8293": K.ACTIVATE_CARD,
    "get_pending_replacement_orders_5765": K.ORDER_REPLACEMENT_CARD,
    "order_replacement_credit_card_7291": K.ORDER_REPLACEMENT_CARD,
    "order_debit_card_5739": K.ORDER_REPLACEMENT_CARD,
    "get_closure_reason_history_8293": K.CLOSE_CARD,
    "log_credit_card_closure_reason_4521": K.CLOSE_CARD,
    "close_credit_card_account_7834": K.CLOSE_CARD,
    "close_debit_card_4721": K.CLOSE_CARD,
    "change_debit_card_pin_6285": K.CHANGE_OR_RESET_PIN,
    "reset_debit_card_pin_6284": K.CHANGE_OR_RESET_PIN,
    # --- card controls ---
    "set_debit_card_recurring_block_7382": K.CARD_CONTROLS,
    "clear_debit_card_fraud_alert_4892": K.CARD_CONTROLS,
    # --- limits ---
    "get_credit_limit_increase_history_4829": K.CREDIT_LIMIT_REQUEST,
    "submit_credit_limit_increase_request_7392": K.CREDIT_LIMIT_REQUEST,
    "approve_credit_limit_increase_5847": K.CREDIT_LIMIT_REQUEST,
    "deny_credit_limit_increase_5848": K.CREDIT_LIMIT_REQUEST,
    "request_temporary_debit_card_limit_increase_8374": K.CREDIT_LIMIT_REQUEST,
    # --- fees, interest and credits ---
    "update_transaction_rewards_3847": K.FEE_OR_INTEREST_ADJUSTMENT,
    "apply_statement_credit_8472": K.FEE_OR_INTEREST_ADJUSTMENT,
    "apply_credit_card_account_flag_6147": K.FEE_OR_INTEREST_ADJUSTMENT,
    "apply_checking_account_credit_5829": K.FEE_OR_INTEREST_ADJUSTMENT,
    "apply_savings_account_credit_6831": K.FEE_OR_INTEREST_ADJUSTMENT,
    "submit_interest_discrepancy_report_7294": K.FEE_OR_INTEREST_ADJUSTMENT,
    # --- money movement and accounts ---
    "pay_credit_card_from_checking_9182": K.PAY_CREDIT_CARD,
    "transfer_funds_between_bank_accounts_7291": K.TRANSFER_FUNDS,
    "open_bank_account_4821": K.OPEN_ACCOUNT,
    "close_bank_account_7392": K.CLOSE_ACCOUNT,
}


def intent_for_tool(tool: str) -> IntentKind:
    return TOOL_INTENT.get(tool, K.OTHER)
