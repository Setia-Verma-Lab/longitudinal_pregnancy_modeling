"""
Database utilities for PostgreSQL connections.
"""

from typing import Optional
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
from pathlib import Path


def load_config(config_path: str = "db_config.yaml") -> dict:
    """Load database configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Database config file not found: {config_path}\n"
            f"Please create {config_path} with your PostgreSQL connection details."
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_connection(config_path: str = "db_config.yaml"):
    """
    Create a PostgreSQL database connection.
    
    Returns:
        psycopg2 connection object
    """
    config = load_config(config_path)
    db_config = config['database']
    
    # Build connection parameters, only including auth if provided
    conn_params = {
        'host': db_config['host'],
        'port': db_config['port'],
        'dbname': db_config['dbname'],
    }
    
    # Add user/password only if present in config
    if 'user' in db_config and db_config['user']:
        conn_params['user'] = db_config['user']
    
    if 'password' in db_config and db_config['password']:
        conn_params['password'] = db_config['password']
    
    conn = psycopg2.connect(**conn_params)
    
    return conn


def query_to_df(
    query: str,
    conn,
    params: Optional[tuple] = None,
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame.
    
    Parameters:
        query: SQL query string
        conn: psycopg2 connection object
        params: Optional tuple of query parameters
        chunksize: If specified, return iterator of DataFrames
        
    Returns:
        pandas DataFrame with query results
    """
    if chunksize:
        return pd.read_sql_query(query, conn, params=params, chunksize=chunksize)
    else:
        return pd.read_sql_query(query, conn, params=params)


def execute_query(query: str, conn, params: Optional[tuple] = None) -> None:
    """
    Execute a SQL query without returning results (e.g., CREATE, INSERT, UPDATE).
    
    Parameters:
        query: SQL query string
        conn: psycopg2 connection object
        params: Optional tuple of query parameters
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()


def get_table_count(table_name: str, conn, schema: str = "public") -> int:
    """
    Get the number of rows in a table.
    
    Parameters:
        table_name: Name of the table
        conn: psycopg2 connection object
        schema: Schema name (default: public)
        
    Returns:
        Number of rows in the table
    """
    query = f"SELECT COUNT(*) as count FROM {schema}.{table_name}"
    result = query_to_df(query, conn)
    return int(result['count'].iloc[0])


def table_exists(table_name: str, conn, schema: str = "public") -> bool:
    """
    Check if a table exists in the database.
    
    Parameters:
        table_name: Name of the table
        conn: psycopg2 connection object
        schema: Schema name (default: public)
        
    Returns:
        True if table exists, False otherwise
    """
    query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = %s
        )
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table_name))
        return cur.fetchone()[0]
