import duckdb
import pandas as pd
import os
import yaml
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

def setup_logging() -> logging.Logger:

    """
    Set up logging configuration to write to 01_Raw_to_Bronze.log
    """

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark= datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"01_Raw_to_Bronze_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('RawToBronze')
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

def ingest_from_xlsx(duckdb_name: str= Field(..., description="DuckDB database name"),
                     duckdb_schema: str = Field(..., description="DuckDB schema name"),
                     duckdb_table: str = Field(..., description="DuckDB table name"),
                     excel_path: str = Field(..., description="Path to the Excel file"),
                     excel_sheetname: str= Field(...,description="Excel Sheet Name")) -> None:
    """
    Initialize DuckDB and read Excel data into bronze schema
    """
    logger = logging.getLogger('RawToBronze')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define paths
    db_path = project_root / duckdb_name #"assignment.db"
    excel_path = project_root / excel_path #"Data" / "Raw" / "DataScientist_CaseStudy_Dataset.xlsx"

    logger.info(f"Database path: {db_path}")
    logger.info(f"Excel file path: {excel_path}")

    # Check if Excel file exists
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Initialize DuckDB connection with persistent database
    conn = duckdb.connect(str(db_path))
    logger.info(f"Connected to DuckDB database: {db_path}")

    try:
        # Create bronze schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {duckdb_schema}")
        logger.info(f"Created {duckdb_schema} schema")

        # Read the sheet from Excel file
        logger.info(f"Reading {excel_sheetname} sheet from Excel file...")
        df = pd.read_excel(excel_path, sheet_name=excel_sheetname)
        logger.info(f"Loaded {len(df)} rows from {excel_sheetname} sheet")
        logger.info(f"Columns: {list(df.columns)}")

        # Drop the table if it exists and create new one
        conn.execute(f"DROP TABLE IF EXISTS {duckdb_schema}.{duckdb_table}")

        # Insert data into DuckDB table
        conn.execute(f"CREATE TABLE {duckdb_schema}.{duckdb_table} AS SELECT * FROM df")
        logger.info(f"Created table {duckdb_schema}.{duckdb_table}")

        # Verify the data was inserted
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {duckdb_schema}.{duckdb_table}").fetchone()
        logger.info(f"Verified: {result[0]} rows inserted into {duckdb_schema}.{duckdb_table}")

        # Show first few rows
        logger.info(f"First 5 rows of {duckdb_schema}.{duckdb_table}")
        preview = conn.execute(f"SELECT * FROM {duckdb_schema}.{duckdb_table} LIMIT 5").fetchdf()
        logger.info(f"\n{preview}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        # Close the connection
        conn.close()
        logger.info("Database connection closed")

def load_config(config_path: str= Field(...,description="Path to ELT config file")) -> Dict:
    """
    Load configuration from YAML file
    """
    logger = logging.getLogger('RawToBronze')

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_file_path = project_root / config_path

    if not config_file_path.exists():
        logger.error(f"Config file not found: {config_file_path}")
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    logger.info(f"Successfully loaded configuration from {config_file_path}")
    return config

def run_ingestion_jobs(config_path: str= 'config.yaml') -> None:
    """
    Run multiple ingestion jobs based on YAML configuration
    """
    logger = logging.getLogger('RawToBronze')

    # Load configuration
    config = load_config(config_path)

    # Get the list of ingestion jobs
    jobs = config.get('ingestion_jobs', [])

    if not jobs:
        logger.warning("No ingestion jobs found in configuration")
        return

    logger.info(f"Found {len(jobs)} ingestion jobs to process")

    # Loop through each job configuration
    for i, job in enumerate(jobs, 1):
        job_name = job.get('job_name', f'Job_{i}')
        logger.info("=" * 50)
        logger.info(f"Processing {job_name} ({i}/{len(jobs)})")
        logger.info("=" * 50)

        try:
            # Extract parameters for this job
            duckdb_name = job['duckdb_name']
            duckdb_schema = job['duckdb_schema']
            duckdb_table = job['duckdb_table']
            excel_path = job['excel_path']
            excel_sheetname = job['excel_sheetname']

            # Run the ingestion for this job
            ingest_from_xlsx(
                duckdb_name=duckdb_name,
                duckdb_schema=duckdb_schema,
                duckdb_table=duckdb_table,
                excel_path=excel_path,
                excel_sheetname=excel_sheetname
            )

            logger.info(f"✅ Successfully completed {job_name}")

        except Exception as e:
            logger.error(f"❌ Failed to process {job_name}: {e}")
            # Continue with next job instead of stopping
            continue

    logger.info("=" * 50)
    logger.info("All ingestion jobs completed!")
    logger.info("=" * 50)

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting Raw to Bronze ingestion process")

    try:
        # Run all jobs from config file
        run_ingestion_jobs()
        logger.info("Raw to Bronze ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Raw to Bronze ingestion process failed: {e}")
        raise

    # Or run a single job (original way)
    # ingest_from_xlsx(
    #     duckdb_name="assignment.duckdb",
    #     duckdb_schema="bronze",
    #     duckdb_table="description",
    #     excel_path="Data/Raw/DataScientist_CaseStudy_Dataset.xlsx",
    #     excel_sheetname="Description"
    # )