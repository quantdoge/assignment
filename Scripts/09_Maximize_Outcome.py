import duckdb
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 05_Create_Predictions.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"05_Create_Predictions_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('CreatePredictions')
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

def create_prediction_tables(duckdb_name: str = "assignment.duckdb") -> None:
    """
    Create prediction schema tables by combining test predictions and revenue data
    """
    logger = logging.getLogger('CreatePredictions')

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
        # Create prediction schema if it doesn't exist
        conn.execute("CREATE SCHEMA IF NOT EXISTS prediction")
        logger.info("Created prediction schema")

        # Check if all required tables exist
        required_tables = [
            'test_predictions_cc_sales',
            'test_revenue_cc_predictions',
            'test_predictions_cl_sales',
            'test_revenue_cl_predictions',
            'test_predictions_mf_sales',
            'test_revenue_mf_predictions'
        ]

        for table in required_tables:
            result = conn.execute(f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = '{table}'
            """).fetchone()

            if result[0] == 0:
                logger.error(f"Required table not found: {table}")
                raise ValueError(f"Required table not found: {table}")
            else:
                logger.info(f"Found {table}")

        # Create test_predictions_combined table
        logger.info("Creating predictions.test_predictions_combined table...")

        combined_query = """
        CREATE OR REPLACE TABLE predictions.test_predictions_combined AS (
            SELECT
                cc_sales.Client AS client,
                cc_sales.predicted_probability AS cc_predicted_probability,
                cc_revenue.Predicted_Revenue_CC AS cc_predicted_revenue,
                CASE WHEN cc_sales.predicted_probability < 0.71 THEN 0 ELSE
                    cc_sales.predicted_probability * cc_revenue.Predicted_Revenue_CC
                END AS cc_expected_revenue,
                cl_sales.predicted_probability AS cl_predicted_probability,
                cl_revenue.Predicted_Revenue_CL AS cl_predicted_revenue,
                CASE WHEN cl_sales.predicted_probability < 0.71 THEN 0 ELSE
                    cl_sales.predicted_probability * cl_revenue.Predicted_Revenue_CL
                END AS cl_expected_revenue,
                mf_sales.predicted_probability AS mf_predicted_probability,
                mf_revenue.Predicted_Revenue_MF AS mf_predicted_revenue,
                CASE WHEN mf_sales.predicted_probability < 0.71 THEN 0 ELSE
                    mf_sales.predicted_probability * mf_revenue.Predicted_Revenue_MF
                END AS mf_expected_revenue
            FROM
                predictions.test_predictions_cc_sales AS cc_sales
            INNER JOIN predictions.test_revenue_cc_predictions AS cc_revenue ON cc_sales.Client = cc_revenue.Client
            INNER JOIN predictions.test_predictions_cl_sales AS cl_sales ON cc_sales.Client = cl_sales.Client
            INNER JOIN predictions.test_revenue_cl_predictions AS cl_revenue ON cc_sales.Client = cl_revenue.Client
            INNER JOIN predictions.test_predictions_mf_sales AS mf_sales ON cc_sales.Client = mf_sales.Client
            INNER JOIN predictions.test_revenue_mf_predictions AS mf_revenue ON cc_sales.Client = mf_revenue.Client
        )
        """

        conn.execute(combined_query)
        logger.info("Successfully created predictions.test_predictions_combined table")

        # Verify the combined table was created correctly
        result = conn.execute("SELECT COUNT(*) as row_count FROM predictions.test_predictions_combined").fetchone()
        logger.info(f"Verified: {result[0]} rows in predictions.test_predictions_combined table")

        # Create optimized_outcome table
        logger.info("Creating predictions.optimized_outcome table...")

        optimized_query = """
        CREATE OR REPLACE TABLE predictions.optimized_outcome AS (
            SELECT client,
                GREATEST(cc_expected_revenue, cl_expected_revenue, mf_expected_revenue) AS max_expected_revenue,
                CASE
                    WHEN GREATEST(cc_expected_revenue, cl_expected_revenue, mf_expected_revenue) = cc_expected_revenue THEN 'CC'
                    WHEN GREATEST(cc_expected_revenue, cl_expected_revenue, mf_expected_revenue) = cl_expected_revenue THEN 'CL'
                    ELSE 'MF'
                END AS best_product_type
            FROM predictions.test_predictions_combined
        )
        """

        conn.execute(optimized_query)
        logger.info("Successfully created predictions.optimized_outcome table")

        # Verify the optimized outcome table was created correctly
        result = conn.execute("SELECT COUNT(*) as row_count FROM predictions.optimized_outcome").fetchone()
        logger.info(f"Verified: {result[0]} rows in predictions.optimized_outcome table")

        # Show table schemas
        logger.info("predictions.test_predictions_combined schema:")
        schema_info = conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'prediction' AND table_name = 'test_predictions_combined'
            ORDER BY ordinal_position
        """).fetchdf()
        logger.info(f"\n{schema_info}")

        logger.info("predictions.optimized_outcome schema:")
        schema_info = conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'prediction' AND table_name = 'optimized_outcome'
            ORDER BY ordinal_position
        """).fetchdf()
        logger.info(f"\n{schema_info}")

        # Show first few rows of each table
        logger.info("First 3 rows of predictions.test_predictions_combined:")
        preview = conn.execute("SELECT * FROM predictions.test_predictions_combined LIMIT 3").fetchdf()
        logger.info(f"\n{preview}")

        logger.info("First 3 rows of predictions.optimized_outcome:")
        preview = conn.execute("SELECT * FROM predictions.optimized_outcome LIMIT 3").fetchdf()
        logger.info(f"\n{preview}")

        # Show summary statistics for optimized outcome
        logger.info("Product type distribution in optimized_outcome:")
        distribution = conn.execute("""
            SELECT
                best_product_type,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM predictions.optimized_outcome
            GROUP BY best_product_type
            ORDER BY count DESC
        """).fetchdf()
        logger.info(f"\n{distribution}")

        # Show revenue statistics
        logger.info("Revenue statistics:")
        revenue_stats = conn.execute("""
            SELECT
                ROUND(AVG(max_expected_revenue), 2) as avg_expected_revenue,
                ROUND(MIN(max_expected_revenue), 2) as min_expected_revenue,
                ROUND(MAX(max_expected_revenue), 2) as max_expected_revenue,
                ROUND(SUM(max_expected_revenue), 2) as total_expected_revenue
            FROM predictions.optimized_outcome
        """).fetchone()

        logger.info(f"Average expected revenue: {revenue_stats[0]}")
        logger.info(f"Minimum expected revenue: {revenue_stats[1]}")
        logger.info(f"Maximum expected revenue: {revenue_stats[2]}")
        logger.info(f"Total expected revenue: {revenue_stats[3]}")

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
    logger.info("Starting Prediction table creation process")

    try:
        # Run transformation directly with database name
        create_prediction_tables(duckdb_name="assignment.duckdb")
        logger.info("✅ Prediction table creation process completed successfully")
    except Exception as e:
        logger.error(f"❌ Prediction table creation process failed: {e}")
        raise