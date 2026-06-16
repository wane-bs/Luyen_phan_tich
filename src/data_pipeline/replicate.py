import os
import json
import sqlite3
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_sql_server_connection():
    server = os.getenv("DB_SERVER", "localhost")
    database = os.getenv("DB_DATABASE")
    use_windows_auth = os.getenv("DB_USE_WINDOWS_AUTH", "True").lower() == "true"
    username = os.getenv("DB_USERNAME", "")
    password = os.getenv("DB_PASSWORD", "")
    db_library = os.getenv("DB_LIBRARY", "pymssql").lower()
    odbc_driver = os.getenv("DB_ODBC_DRIVER", "{ODBC Driver 17 for SQL Server}")

    if not database:
        raise ValueError("Error: DB_DATABASE environment variable is not set in .env")

    print(f"Connecting to SQL Server using {db_library.upper()}...")
    print(f"Server: {server} | Database: {database} | Windows Auth: {use_windows_auth}")

    if db_library == "pyodbc":
        import pyodbc
        if use_windows_auth:
            conn_str = f"DRIVER={odbc_driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
        else:
            conn_str = f"DRIVER={odbc_driver};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        return pyodbc.connect(conn_str)
        
    elif db_library == "pymssql":
        import pymssql
        if use_windows_auth:
            # For pymssql Windows Auth, we connect without username and password
            # (which delegates to Trusted Connection if running on Windows with Domain/Local login)
            return pymssql.connect(server=server, database=database)
        else:
            return pymssql.connect(server=server, user=username, password=password, database=database)
    else:
        raise ValueError(f"Unsupported DB_LIBRARY: '{db_library}'. Must be 'pyodbc' or 'pymssql'.")

def replicate_datasets():
    # Ensure local data directory exists
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    sqlite_db_path = os.path.join("data", "raw", "local_replica.db")

    # Read config.json
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    datasets = config.get("datasets", [])
    if not datasets:
        print("No datasets defined in config.json.")
        return

    # Connect to databases
    mssql_conn = None
    sqlite_conn = None
    try:
        mssql_conn = get_sql_server_connection()
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        print("Successfully connected to both source SQL Server and destination SQLite database.\n")

        for idx, dataset in enumerate(datasets, start=1):
            table_name = dataset.get("table_name")
            query = dataset.get("source_query")

            if not table_name or not query:
                print(f"[{idx}] Skipping invalid dataset entry (missing table_name or source_query).")
                continue

            print(f"[{idx}] Replicating dataset to local table: '{table_name}'...")
            print(f"    Running query: {query}")
            
            # Fetch data from SQL Server into pandas DataFrame
            df = pd.read_sql(query, mssql_conn)
            print(f"    Fetched {len(df)} rows and {len(df.columns)} columns.")

            # Write to SQLite
            df.to_sql(table_name, sqlite_conn, if_exists="replace", index=False)
            print(f"    Successfully saved to local SQLite database '{sqlite_db_path}' table '{table_name}'.\n")

        print("Replication finished successfully!")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during replication:")
        print(e)
        print("\nPlease check:")
        print("1. Your SQL Server credentials and host configuration in `.env`")
        print("2. The database service is active and accessible.")
        print("3. For pyodbc: ensure the specified driver is installed on your OS.")
    finally:
        if mssql_conn:
            mssql_conn.close()
        if sqlite_conn:
            sqlite_conn.close()

if __name__ == "__main__":
    replicate_datasets()
