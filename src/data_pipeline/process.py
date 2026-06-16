import os
import sqlite3
import pandas as pd

def process_local_dataset():
    sqlite_db_path = os.path.join("data", "raw", "local_replica.db")

    if not os.path.exists(sqlite_db_path):
        print(f"Error: Local SQLite database not found at '{sqlite_db_path}'.")
        print("Please run 'python src/data_pipeline/replicate.py' first to fork the datasets from your database.")
        return

    print(f"Connecting to local SQLite database at: {sqlite_db_path}")
    conn = sqlite3.connect(sqlite_db_path)

    try:
        # 1. List all tables in the local database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("\n=== Available local tables ===")
        if not tables:
            print("No tables found in the database. Please check your replication script.")
            return
        for t in tables:
            print(f"- {t}")

        # 2. Example: Querying and analyzing local tables
        print("\n=== Local Query Execution ===")
        for table in tables:
            print(f"\nAnalyzing local table: '{table}'")
            # Load table into Pandas
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", conn)
            
            print(f"First 5 rows of '{table}':")
            print(df.head())
            print(f"\nTable Info:")
            print(f"Total rows fetched in sample: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            
            # --- Place your local data processing / analysis logic below ---
            # e.g., df_clean = df.dropna()
            # df_clean.to_sql(f"{table}_processed", conn, if_exists="replace", index=False)
            
        print("\n=== Processing Complete ===")
        print("You can now write custom queries or build models using the tables in this SQLite database.")

    except Exception as e:
        print(f"An error occurred during local processing: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    process_local_dataset()
