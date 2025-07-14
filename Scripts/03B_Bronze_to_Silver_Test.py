import duckdb
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 03_Bronze_to_Silver.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"03_Bronze_to_Silver_Test_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('BronzeToSilver')
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger

def create_silver_test_table(duckdb_name: str = "assignment.duckdb") -> None:
    """
    Create silver schema test table by joining bronze tables
    """
    logger = logging.getLogger('BronzeToSilver')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Database path: {db_path}")

    # Check if database file exists
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))
    logger.info(f"Connected to DuckDB database: {db_path}")

    try:
        # Create silver schema if it doesn't exist
        conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
        logger.info("Created silver schema")

        # Check if all required bronze tables exist
        required_tables = ['sales_revenues', 'inflow_outflow', 'products_actbalance', 'soc_dem']
        for table in required_tables:
            result = conn.execute(f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'bronze' AND table_name = '{table}'
            """).fetchone()

            if result[0] == 0:
                logger.error(f"Required bronze table not found: bronze.{table}")
                raise ValueError(f"Required bronze table not found: bronze.{table}")
            else:
                logger.info(f"Found bronze.{table}")

        # Get column information for each table to build the SELECT statement
        logger.info("Analyzing table schemas...")

        # Get columns from each table
        tables_info = {}
        for table in required_tables:
            columns_result = conn.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'bronze' AND table_name = '{table}'
                ORDER BY ordinal_position
            """).fetchall()
            tables_info[table] = columns_result
            logger.info(f"bronze.{table} has {len(columns_result)} columns")

        # Build the SQL query with LEFT JOINs
        logger.info("Building JOIN query...")

        # Start with sales_revenues as master source
        query = """
        CREATE OR REPLACE TABLE silver.test AS
        SELECT
            sd.*,
            pa.* EXCLUDE (Client),
            io.* EXCLUDE (Client)
        FROM bronze.soc_dem sd
        INNER JOIN bronze.products_actbalance pa ON sd.Client = pa.Client
        INNER JOIN bronze.inflow_outflow io ON sd.Client = io.Client
        LEFT JOIN bronze.sales_revenues sr ON sd.Client = sr.Client
        WHERE sr.Client IS NULL
        """

        # Drop the table if it exists and create new one
        conn.execute("DROP TABLE IF EXISTS silver.test")

        # Execute the JOIN query
        logger.info("Executing JOIN query to create silver.test table...")
        conn.execute(query)
        logger.info("Successfully created silver.test table with JOINs")

        # Now handle NULL values - fill with 0 for all numeric columns except 'sex'
        logger.info("Handling NULL values...")

        # Get all columns from the newly created table
        columns_result = conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'silver' AND table_name = 'test'
            ORDER BY ordinal_position
        """).fetchall()

        # Build UPDATE statements for numeric columns (excluding 'sex')
        # numeric_columns = []
        # for col_name, data_type in columns_result:
        #     # Skip 'sex' column and only process numeric/integer columns
        #     if (col_name.lower() != 'sex' and
        #         any(numeric_type in data_type.upper() for numeric_type in
        #             ['INTEGER', 'BIGINT', 'DOUBLE', 'REAL', 'NUMERIC', 'DECIMAL', 'FLOAT'])):
        #         numeric_columns.append(col_name)

        # logger.info(f"Found {len(numeric_columns)} numeric columns to process for NULL handling")

        # # Create a single UPDATE query to handle all NULL values
        # if numeric_columns:
        #     # Build SET clauses for all numeric columns
        #     set_clauses = [f'"{col}" = COALESCE("{col}", 0)' for col in numeric_columns]
        #     update_query = f"""
        #     UPDATE silver.test
        #     SET {', '.join(set_clauses)}
        #     """

        #     conn.execute(update_query)
        #     logger.info(f"Successfully filled NULL values with 0 for numeric columns - {'||'.join(numeric_columns)}")

        # Verify the data was created correctly
        result = conn.execute("SELECT COUNT(*) as row_count FROM silver.test").fetchone()
        logger.info(f"Verified: {result[0]} rows in silver.test table")

        # Show table schema
        logger.info("Final table schema:")
        schema_info = conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'silver' AND table_name = 'test'
            ORDER BY ordinal_position
        """).fetchdf()
        logger.info(f"\n{schema_info}")

        # Show first few rows
        logger.info("First 3 rows of silver.test:")
        preview = conn.execute("SELECT * FROM silver.test LIMIT 3").fetchdf()
        logger.info(f"\n{preview}")

        # Show summary statistics
        logger.info("Data summary:")
        total_columns = len(columns_result)
        null_counts = conn.execute("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN Client IS NULL THEN 1 END) as null_clients
            FROM silver.test
        """).fetchone()

        logger.info(f"Total columns: {total_columns}")
        logger.info(f"Total rows: {null_counts[0]}")
        logger.info(f"NULL clients: {null_counts[1]}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        # Close the connection
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting Bronze to Silver transformation process")

    try:
        # Run transformation directly with database name
        create_silver_test_table(duckdb_name="assignment.duckdb")
        logger.info("✅ Bronze to Silver transformation process completed successfully")
    except Exception as e:
        logger.error(f"❌ Bronze to Silver transformation process failed: {e}")
        raise