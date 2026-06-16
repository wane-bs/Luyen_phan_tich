import os
import sqlite3
import pandas as pd
import numpy as np

def run_feature_engineering():
    db_path = os.path.join("data", "raw", "local_replica.db")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    print("Connecting to local SQLite database...")
    conn = sqlite3.connect(db_path)

    try:
        # 1. Load users data
        print("Loading users table...")
        df_users = pd.read_sql_query(
            "SELECT id as client_id, current_age, yearly_income, total_debt, credit_score, gender FROM users", 
            conn
        )
        # Encode gender: Female=1, Male=0
        df_users['gender_encoded'] = df_users['gender'].apply(lambda x: 1 if str(x).lower() == 'female' else 0)
        # Calculate DTI
        df_users['dti'] = df_users['total_debt'] / df_users['yearly_income'].replace(0, np.nan)
        df_users['dti'] = df_users['dti'].fillna(0)

        # 2. Historical Time Anchor & Dormant Cards (Q9)
        print("Determining Historical Time Anchor and filtering dormant cards...")
        # Get max transaction date
        df_max_date = pd.read_sql_query("SELECT MAX(date) as max_date FROM transactions", conn)
        t_max = pd.to_datetime(df_max_date['max_date'].iloc[0])
        print(f"   - Historical Time Anchor (T_max): {t_max}")

        # Get cards and transactions to identify used cards
        df_all_cards = pd.read_sql_query("SELECT id as card_id, client_id, credit_limit, card_type, acct_open_date FROM cards", conn)
        df_all_cards['acct_open_date'] = pd.to_datetime(df_all_cards['acct_open_date'])

        df_tx_cards = pd.read_sql_query("SELECT DISTINCT card_id FROM transactions", conn)
        used_card_ids = set(df_tx_cards['card_id'])

        # Identify dormant cards: open > 2 years ago relative to T_max and never used
        two_years_before_t_max = t_max - pd.DateOffset(years=2)
        df_all_cards['is_dormant'] = df_all_cards.apply(
            lambda r: 1 if (r['card_id'] not in used_card_ids and r['acct_open_date'] < two_years_before_t_max) else 0,
            axis=1
        )

        # Calculate dormant card ratio per user for Credit Cards
        df_credit_cards = df_all_cards[df_all_cards['card_type'] == 'Credit'].copy()
        df_user_card_stats = df_credit_cards.groupby('client_id').agg(
            total_credit_cards=('card_id', 'count'),
            dormant_credit_cards=('is_dormant', 'sum')
        ).reset_index()
        df_user_card_stats['dormant_card_ratio'] = (df_user_card_stats['dormant_credit_cards'] / df_user_card_stats['total_credit_cards']).fillna(0)

        # Calculate total credit limit using only ACTIVE (non-dormant) Credit Cards
        df_active_credit_cards = df_credit_cards[df_credit_cards['is_dormant'] == 0]
        df_limits = df_active_credit_cards.groupby('client_id').agg(
            total_credit_limit=('credit_limit', 'sum')
        ).reset_index()

        print(f"   - Dormant cards identified: {df_all_cards['is_dormant'].sum()} cards")

        # 3. Process transactions
        print("Loading and processing transactions...")
        # Load transactions with their card types joined
        df_txs = pd.read_sql_query(
            """
            SELECT t.client_id, t.card_id, t.amount, t.use_chip, t.errors, t.mcc, t.date, t.merchant_state, c.card_type 
            FROM transactions t
            LEFT JOIN cards c ON t.client_id = c.client_id AND t.card_id = c.id
            """,
            conn
        )
        
        # Parse date and extract month
        df_txs['date'] = pd.to_datetime(df_txs['date'])
        df_txs['month'] = df_txs['date'].dt.to_period('M').astype(str)

        # Split spend vs refunds
        df_txs['spend_amount'] = df_txs['amount'].apply(lambda x: x if x > 0 else 0)
        df_txs['refund_amount'] = df_txs['amount'].apply(lambda x: abs(x) if x < 0 else 0)
        df_txs['is_refund'] = df_txs['amount'].apply(lambda x: 1 if x < 0 else 0)

        # Errors checks
        df_txs['is_insufficient_balance'] = df_txs['errors'].apply(
            lambda x: 1 if pd.notnull(x) and 'Insufficient Balance' in str(x) else 0
        )
        df_txs['is_security_error'] = df_txs['errors'].apply(
            lambda x: 1 if pd.notnull(x) and 'Bad' in str(x) else 0
        )

        # Transaction types
        df_txs['is_online'] = df_txs['use_chip'].apply(lambda x: 1 if str(x) == 'Online Transaction' else 0)

        # Essential MCC List
        essential_mccs = {1711, 4784, 4814, 4900, 5411, 5499, 5541, 5912} # Standard essential categories
        df_txs['is_essential'] = df_txs['mcc'].apply(lambda x: 1 if x in essential_mccs else 0)

        # 4. Spatiotemporal Fraud Signal (Q10)
        print("Analyzing spatiotemporal fraud signals (multiple states in 24h)...")
        df_temp = df_txs[['client_id', 'date', 'merchant_state']].dropna().sort_values(by=['client_id', 'date']).copy()
        
        fraud_clients = set()
        for client_id, group in df_temp.groupby('client_id'):
            times = group['date'].values
            states = group['merchant_state'].values
            n = len(group)
            if n < 5:
                continue
                
            left = 0
            state_window = {}
            for right in range(n):
                state_window[states[right]] = state_window.get(states[right], 0) + 1
                
                # Shrink window if delta time > 24 hours
                while (times[right] - times[left]) > np.timedelta64(24, 'h'):
                    st = states[left]
                    state_window[st] -= 1
                    if state_window[st] == 0:
                        del state_window[st]
                    left += 1
                    
                if len(state_window) >= 5:
                    fraud_clients.add(client_id)
                    break
        
        # Aggregate user-level transactional features
        print("Aggregating transaction features to user-level...")
        user_tx_agg = df_txs.groupby('client_id').agg(
            total_tx_count=('amount', 'count'),
            net_spend=('spend_amount', 'sum'),
            total_refund=('refund_amount', 'sum'),
            refund_tx_count=('is_refund', 'sum'),
            insufficient_balance_count=('is_insufficient_balance', 'sum'),
            security_error_count=('is_security_error', 'sum'),
            online_tx_count=('is_online', 'sum'),
            essential_spend=('spend_amount', lambda s: sum(s * df_txs.loc[s.index, 'is_essential']))
        ).reset_index()

        # Calculate rates
        user_tx_agg['refund_rate'] = user_tx_agg['refund_tx_count'] / user_tx_agg['total_tx_count']
        user_tx_agg['insufficient_balance_rate'] = user_tx_agg['insufficient_balance_count'] / user_tx_agg['total_tx_count']
        user_tx_agg['online_tx_rate'] = user_tx_agg['online_tx_count'] / user_tx_agg['total_tx_count']
        user_tx_agg['essential_spend_ratio'] = user_tx_agg['essential_spend'] / user_tx_agg['net_spend'].replace(0, np.nan)
        user_tx_agg['essential_spend_ratio'] = user_tx_agg['essential_spend_ratio'].fillna(0)
        
        # Add spatiotemporal fraud signal flag
        user_tx_agg['has_spatiotemporal_fraud_signal'] = user_tx_agg['client_id'].apply(lambda cid: 1 if cid in fraud_clients else 0)
        print(f"   - Clients triggered spatiotemporal fraud signal: {len(fraud_clients)}")

        # 5. Calculate Monthly CUR and Dynamic CUR 30 Days (Q8)
        print("Calculating credit utilization rate (CUR) metrics...")
        # Filter transactions on Credit cards
        df_credit_txs = df_txs[df_txs['card_type'] == 'Credit'].copy()
        
        # Monthly spending on Credit cards per user
        df_monthly_spend = df_credit_txs.groupby(['client_id', 'month'])['spend_amount'].sum().reset_index()
        
        # Merge with active total credit limits
        df_monthly_cur = pd.merge(df_monthly_spend, df_limits, on='client_id', how='inner')
        df_monthly_cur['cur'] = df_monthly_cur['spend_amount'] / df_monthly_cur['total_credit_limit'].replace(0, np.nan)
        df_monthly_cur['cur'] = df_monthly_cur['cur'].fillna(0)

        # User-level monthly CUR statistics
        user_cur_stats = df_monthly_cur.groupby('client_id')['cur'].agg(
            max_monthly_cur='max',
            avg_monthly_cur='mean'
        ).reset_index()

        # Calculate Dynamic CUR 30 Days (Q8) lùi về từ T_max (2024-10-31)
        t_30d_start = t_max - pd.DateOffset(days=30)
        df_credit_txs_30d = df_credit_txs[(df_credit_txs['date'] >= t_30d_start) & (df_credit_txs['date'] <= t_max)]
        df_spend_30d = df_credit_txs_30d.groupby('client_id')['spend_amount'].sum().reset_index().rename(columns={'spend_amount': 'spend_30d'})
        
        # Merge spend 30d with active total credit limit to compute cur_30d
        df_cur_30d = pd.merge(df_spend_30d, df_limits, on='client_id', how='right').fillna(0)
        df_cur_30d['cur_30d'] = df_cur_30d['spend_30d'] / df_cur_30d['total_credit_limit'].replace(0, np.nan)
        df_cur_30d['cur_30d'] = df_cur_30d['cur_30d'].fillna(0)
        df_cur_30d = df_cur_30d[['client_id', 'cur_30d']]

        # 6. Merge all features together
        print("Merging all feature sets together...")
        final_df = pd.merge(df_users, df_limits, on='client_id', how='left')
        final_df = pd.merge(final_df, user_tx_agg, on='client_id', how='left')
        final_df = pd.merge(final_df, user_cur_stats, on='client_id', how='left')
        final_df = pd.merge(final_df, df_cur_30d, on='client_id', how='left')
        final_df = pd.merge(final_df, df_user_card_stats[['client_id', 'dormant_card_ratio']], on='client_id', how='left')

        # Fill missing values for users with no transactions or cards
        final_df = final_df.fillna(0)

        # 7. Labeling
        print("Labeling target variables...")
        # Default label: DTI > 3.0 OR max_monthly_cur > 0.20
        final_df['default'] = final_df.apply(
            lambda row: 1 if (row['dti'] > 3.0 or row['max_monthly_cur'] > 0.20) else 0,
            axis=1
        )

        # --- FRAUD LABEL: Multi-Signal Weighted Scoring ---
        # Tín hiệu mạnh (w=2): Lỗi bảo mật lặp lại có chủ đích, Giao dịch đa bang trong 24h
        # Tín hiệu phụ (w=1): Tỷ lệ hoàn tiền bất thường, Tỷ lệ giao dịch online cao bất thường
        # Ngưỡng theta được xác thực bởi Qwen 2.5 Math Validator
        FRAUD_SECURITY_THRESHOLD = 3      # Số lỗi bảo mật tối thiểu để coi là chủ đích
        FRAUD_REFUND_RATE_THRESHOLD = 0.10 # Tỷ lệ hoàn tiền bất thường
        FRAUD_ONLINE_RATE_THRESHOLD = 0.70 # Tỷ lệ online cao đáng ngờ

        # Đọc ngưỡng theta từ model_config.json (được Qwen 2.5 Math xác nhận)
        try:
            import json as _json
            _cfg_path = os.path.join("data", "configs", "model_config.json")
            with open(_cfg_path, "r") as _f:
                _cfg = _json.load(_f)
            FRAUD_SCORE_THETA = int(_cfg.get("fraud_label_threshold", 2))
            print(f"   - Đọc fraud_label_threshold={FRAUD_SCORE_THETA} từ model_config.json")
        except Exception:
            FRAUD_SCORE_THETA = 2  # Fallback: theta validated by Qwen
            print(f"   - Dùng fraud_label_threshold fallback={FRAUD_SCORE_THETA}")

        # Tính các tín hiệu thành phần
        final_df['security_error_count_heavy'] = (
            final_df['security_error_count'] >= FRAUD_SECURITY_THRESHOLD
        ).astype(int)

        final_df['high_online_rate'] = (
            final_df['online_tx_rate'] > FRAUD_ONLINE_RATE_THRESHOLD
        ).astype(int)

        # Fraud signal score tổng hợp (weighted)
        # Trọng số: security_heavy=2, spatiotemporal=2, refund_anomaly=1, online_anomaly=1
        final_df['fraud_signal_score'] = (
            2 * final_df['security_error_count_heavy'] +
            2 * final_df['has_spatiotemporal_fraud_signal'] +
            1 * (final_df['refund_rate'] > FRAUD_REFUND_RATE_THRESHOLD).astype(int) +
            1 * final_df['high_online_rate']
        )

        # Fraud label: fraud_signal_score >= theta
        final_df['fraud'] = (final_df['fraud_signal_score'] >= FRAUD_SCORE_THETA).astype(int)
        print(f"   - Fraud Signal Score distribution:\n{final_df['fraud_signal_score'].value_counts().sort_index().to_dict()}")
        print(f"   - Theta={FRAUD_SCORE_THETA} | Fraud labels assigned: {final_df['fraud'].sum()} / {len(final_df)}")

        # 8. Save results
        print("Saving feature matrix to SQLite and CSV...")
        # Save to SQLite
        final_df.to_sql("user_features_matrix", conn, if_exists="replace", index=False)
        
        # Save to CSV
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        csv_path = os.path.join("data", "processed", "user_features_matrix.csv")
        final_df.to_csv(csv_path, index=False)

        print("\n" + "="*50)
        print(" FEATURE ENGINEERING COMPLETED ")
        print("="*50)
        print(f"Total users processed: {len(final_df)}")
        print(f"Default class distribution: {final_df['default'].value_counts().to_dict()}")
        print(f"Default rate: {final_df['default'].mean()*100:.2f}%")
        print(f"Fraud class distribution (Multi-Signal): {final_df['fraud'].value_counts().to_dict()}")
        print(f"Fraud rate (Multi-Signal): {final_df['fraud'].mean()*100:.2f}%")
        print(f"Avg fraud_signal_score: {final_df['fraud_signal_score'].mean():.4f}")
        print(f"Saved to local SQLite table 'user_features_matrix' and CSV: {csv_path}")

    finally:
        conn.close()

if __name__ == "__main__":
    run_feature_engineering()
