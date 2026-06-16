import os
import sqlite3
import pandas as pd

def check_consistency(conn):
    print("\n" + "="*50)
    print(" SECTION 1: DATABASE CONSISTENCY VERIFICATION ")
    print("="*50)
    
    # 1. Duplicates check in primary keys
    print("\n1. Primary Key Uniqueness:")
    
    # Check users.id duplicates
    dup_users = pd.read_sql_query("SELECT id, COUNT(*) as cnt FROM users GROUP BY id HAVING cnt > 1", conn)
    print(f"   - Duplicate user IDs: {len(dup_users)}")
    
    # Check cards.id duplicates (Note: cards might be uniquely identified by id, or by combination client_id and card_id)
    dup_cards = pd.read_sql_query("SELECT id, COUNT(*) as cnt FROM cards GROUP BY id HAVING cnt > 1", conn)
    print(f"   - Duplicate card IDs: {len(dup_cards)}")
    
    # Check transactions.id duplicates
    dup_txs = pd.read_sql_query("SELECT id, COUNT(*) as cnt FROM transactions GROUP BY id HAVING cnt > 1", conn)
    print(f"   - Duplicate transaction IDs: {len(dup_txs)}")

    # 2. Referential Integrity Check
    print("\n2. Referential Integrity Checks:")
    
    # Check if cards.client_id references valid users.id
    orphan_cards = pd.read_sql_query(
        "SELECT COUNT(*) as cnt FROM cards WHERE client_id NOT IN (SELECT id FROM users)", conn
    ).iloc[0]['cnt']
    print(f"   - Cards with invalid client_id: {orphan_cards}")
    
    # Check if transactions.client_id references valid users.id
    orphan_txs_user = pd.read_sql_query(
        "SELECT COUNT(*) as cnt FROM transactions WHERE client_id NOT IN (SELECT id FROM users)", conn
    ).iloc[0]['cnt']
    print(f"   - Transactions with invalid client_id: {orphan_txs_user}")
    
    # Check if transactions (client_id, card_id) references valid cards (client_id, id)
    # Note: in this dataset, a user (client_id) has multiple cards indexed from 0 to N.
    orphan_txs_card = pd.read_sql_query(
        """
        SELECT COUNT(*) as cnt FROM transactions t
        WHERE NOT EXISTS (
            SELECT 1 FROM cards c 
            WHERE c.client_id = t.client_id AND c.id = t.card_id
        )
        """, conn
    ).iloc[0]['cnt']
    print(f"   - Transactions with invalid card reference (client_id + card_id): {orphan_txs_card}")
    
    # Check if transactions.mcc references valid mcc_codes.mcc_id
    orphan_mcc = pd.read_sql_query(
        "SELECT COUNT(*) as cnt FROM transactions WHERE mcc NOT IN (SELECT mcc_id FROM mcc_codes)", conn
    ).iloc[0]['cnt']
    print(f"   - Transactions with invalid MCC code: {orphan_mcc}")

    # 3. Missing/Null values in critical fields
    print("\n3. Missing/Null Values in Critical Fields:")
    
    null_users = pd.read_sql_query(
        "SELECT SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) as id_nulls, SUM(CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END) as score_nulls FROM users", conn
    ).iloc[0]
    print(f"   - Users Table Nulls: id_nulls={null_users['id_nulls']}, credit_score_nulls={null_users['score_nulls']}")
    
    null_cards = pd.read_sql_query(
        "SELECT SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) as id_nulls, SUM(CASE WHEN credit_limit IS NULL THEN 1 ELSE 0 END) as limit_nulls FROM cards", conn
    ).iloc[0]
    print(f"   - Cards Table Nulls: id_nulls={null_cards['id_nulls']}, credit_limit_nulls={null_cards['limit_nulls']}")
    
    null_txs = pd.read_sql_query(
        "SELECT SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) as id_nulls, SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) as amount_nulls FROM transactions", conn
    ).iloc[0]
    print(f"   - Transactions Table Nulls: id_nulls={null_txs['id_nulls']}, amount_nulls={null_txs['amount_nulls']}")

def check_descriptive_statistics(conn):
    print("\n" + "="*50)
    print(" SECTION 2: DESCRIPTIVE STATISTICS ")
    print("="*50)
    
    # 1. Users Table Statistics
    print("\n1. Users Profile Statistics:")
    df_users = pd.read_sql_query("SELECT current_age, yearly_income, total_debt, credit_score, gender FROM users", conn)
    
    # Process currency fields (strip '$' if present, but since it's numeric in db let's check)
    print(df_users.describe().round(2))
    print("\n   Gender distribution:")
    print(df_users['gender'].value_counts(normalize=True).round(4) * 100)

    # 2. Cards Table Statistics
    print("\n2. Cards Profile Statistics:")
    df_cards = pd.read_sql_query("SELECT credit_limit, card_brand, card_type, has_chip FROM cards", conn)
    
    # If credit_limit is stored with '$', clean it
    if df_cards['credit_limit'].dtype == object:
        df_cards['credit_limit'] = df_cards['credit_limit'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        
    print(df_cards[['credit_limit']].describe().round(2))
    print("\n   Card Brand distribution:")
    print(df_cards['card_brand'].value_counts())
    print("\n   Card Type distribution:")
    print(df_cards['card_type'].value_counts())
    print("\n   Has Chip ratio:")
    print(df_cards['has_chip'].value_counts(normalize=True).round(4) * 100)

    # 3. Transactions Table Statistics
    print("\n3. Transactions Statistics:")
    df_txs = pd.read_sql_query("SELECT amount, use_chip, date FROM transactions", conn)
    
    # Clean amount if object
    if df_txs['amount'].dtype == object:
        df_txs['amount'] = df_txs['amount'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        
    print(df_txs[['amount']].describe().round(2))
    print(f"\n   Transaction Date Range: {df_txs['date'].min()} to {df_txs['date'].max()}")
    print("\n   Transaction Method (use_chip) distribution:")
    print(df_txs['use_chip'].value_counts(normalize=True).round(4) * 100)

def main():
    sqlite_db_path = os.path.join("data", "raw", "local_replica.db")
    if not os.path.exists(sqlite_db_path):
        print(f"Error: Local database '{sqlite_db_path}' not found. Please run src/data_pipeline/replicate.py first.")
        return
        
    conn = sqlite3.connect(sqlite_db_path)
    try:
        check_consistency(conn)
        check_descriptive_statistics(conn)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
