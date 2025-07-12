import duckdb
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum
import yaml
import logging
from datetime import datetime

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / "Logs"

    # Create Logs directory if it doesn't exist
    logs_dir.mkdir(exist_ok=True)

    dt_mark= datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"02_Data_Quality_Check_{dt_mark}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class CheckType(str, Enum):
    UNIQUE = "unique"
    ENUM = "enum"
    POSITIVE = "positive"
    NON_NULL = "non_null"
    TYPE = "type"
    SYNC_NULL = "sync_null"
    SYNC_ZERO = "sync_zero"

class DataQualityCheck(BaseModel):
    """
    Pydantic model for data quality check configuration
    """
    database_name: str = Field(..., description="DuckDB database name")
    schema_name: str = Field(..., description="DuckDB schema name")
    table_name: str = Field(..., description="DuckDB table name")
    data_field_name: str = Field(..., description="Data field name to check")
    check_name: str = Field(..., description="Name for this check")
    is_check_unique: bool = Field(default=False, description="Check if field contains only unique values")
    is_check_enum: bool = Field(default=False, description="Check if field contains only values in enum list")
    enum_values: Optional[List[Any]] = Field(default=None, description="List of allowed enum values")
    is_check_positive: bool = Field(default=False, description="Check if field contains only positive values")
    is_check_null: bool = Field(default=False, description="Check if field contains only non-null values")
    is_check_type: bool = Field(default=False, description="Check if field is of a specific type")
    df_type: Optional[str] = Field(default=None, description="Expected DataFrame type for the field ('numerical' or 'datetime')")
    is_sync_null: bool = Field(default=False, description="Check if this field has synchronized null state with sync_null_field")
    sync_null_field: Optional[str] = Field(default=None, description="Field name to synchronize null state with")
    is_sync_zero: bool = Field(default=False, description="Check if this field has synchronized zero state with sync_zero_field")
    sync_zero_field: Optional[str] = Field(default=None, description="Field name to synchronize zero state with")


    @field_validator('enum_values')
    @classmethod
    def validate_enum_values(cls, v, info):
        if info.data.get('is_check_enum') and not v:
            raise ValueError("enum_values must be provided when is_check_enum is True")
        return v

    @field_validator('df_type')
    @classmethod
    def validate_df_type(cls, v, info):
        if info.data.get('is_check_type') and not v:
            raise ValueError("df_type must be provided when is_check_type is True")
        if v and v not in ['numerical', 'datetime', 'string']:
            raise ValueError("df_type must be either 'numerical', 'datetime', or 'string'")
        return v

    @field_validator('sync_null_field')
    @classmethod
    def validate_sync_null_field(cls, v, info):
        if info.data.get('is_sync_null') and not v:
            raise ValueError("sync_null_field must be provided when is_sync_null is True")
        return v

    @field_validator('sync_zero_field')
    @classmethod
    def validate_sync_zero_field(cls, v, info):
        if info.data.get('is_sync_zero') and not v:
            raise ValueError("sync_zero_field must be provided when is_sync_zero is True")
        return v

class DataQualityResult(BaseModel):
    """
    Pydantic model for data quality check results
    """
    database_name: str
    schema_name: str
    table_name: str
    data_field_name: str
    total_records: int
    check_type: CheckType
    records_passed: int
    records_failed: int
    check_name: str

def run_data_quality_check(check_config: DataQualityCheck) -> List[DataQualityResult]:
    """
    Run data quality checks on a specific field in DuckDB
    """
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / check_config.database_name

    if not db_path.exists():
        error_msg = f"Database file not found: {db_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Connect to DuckDB
    conn = duckdb.connect(str(db_path))
    results = []

    try:
        # Get total number of records
        total_query = f"""
        SELECT COUNT(*) as total_count
        FROM {check_config.schema_name}.{check_config.table_name}
        """
        total_records = conn.execute(total_query).fetchone()[0]

        logger.info(f"Checking field '{check_config.data_field_name}' in {check_config.schema_name}.{check_config.table_name}")
        logger.info(f"Total records: {total_records}")

        # Check unique values
        if check_config.is_check_unique:
            unique_query = f"""
            SELECT COUNT(*) as unique_count
            FROM (
                SELECT {check_config.data_field_name}, COUNT(*) as cnt
                FROM {check_config.schema_name}.{check_config.table_name}
                GROUP BY {check_config.data_field_name}
                HAVING COUNT(*) = 1
            )
            """
            unique_count = conn.execute(unique_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.UNIQUE,
                records_passed=unique_count,
                records_failed=total_records - unique_count,
                check_name=f"{check_config.check_name}_unique"
            ))

        # Check enum values
        if check_config.is_check_enum:
            # Convert enum values to SQL-compatible format
            enum_values_str = ", ".join([f"'{val}'" if isinstance(val, str) else str(val) for val in check_config.enum_values])

            enum_query = f"""
            SELECT COUNT(*) as enum_passed_count
            FROM {check_config.schema_name}.{check_config.table_name}
            WHERE {check_config.data_field_name} IN ({enum_values_str})
            OR {check_config.data_field_name} IS NULL
            """
            enum_passed = conn.execute(enum_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.ENUM,
                records_passed=enum_passed,
                records_failed=total_records - enum_passed,
                check_name=f"{check_config.check_name}_enum"
            ))

        # Check positive values
        if check_config.is_check_positive:
            positive_query = f"""
            SELECT COUNT(*) as positive_count
            FROM {check_config.schema_name}.{check_config.table_name}
            WHERE {check_config.data_field_name} >= 0
            OR {check_config.data_field_name} IS NULL
            """
            positive_count = conn.execute(positive_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.POSITIVE,
                records_passed=positive_count,
                records_failed=total_records - positive_count,
                check_name=f"{check_config.check_name}_positive"
            ))

        # Check non-null values
        if check_config.is_check_null:
            non_null_query = f"""
            SELECT COUNT(*) as non_null_count
            FROM {check_config.schema_name}.{check_config.table_name}
            WHERE {check_config.data_field_name} IS NOT NULL
            """
            non_null_count = conn.execute(non_null_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.NON_NULL,
                records_passed=non_null_count,
                records_failed=total_records - non_null_count,
                check_name=f"{check_config.check_name}_non_null"
            ))

        # Check data type
        if check_config.is_check_type:
            if check_config.df_type == 'numerical':
                # Check if field is numerical (int, float, decimal, etc.)
                type_query = f"""
                SELECT COUNT(*) as type_passed_count
                FROM {check_config.schema_name}.{check_config.table_name}
                WHERE ({check_config.data_field_name} IS NOT NULL
                AND TRY_CAST({check_config.data_field_name} AS DOUBLE) IS NOT NULL)
                OR {check_config.data_field_name} IS NULL
                """
            elif check_config.df_type == 'datetime':
                # Check if field is date or datetime
                type_query = f"""
                SELECT COUNT(*) as type_passed_count
                FROM {check_config.schema_name}.{check_config.table_name}
                WHERE ({check_config.data_field_name} IS NOT NULL
                AND (TRY_CAST({check_config.data_field_name} AS DATE) IS NOT NULL
                     OR TRY_CAST({check_config.data_field_name} AS TIMESTAMP) IS NOT NULL))
				OR {check_config.data_field_name} IS NULL
                """
            elif check_config.df_type == 'string':
                # Check if field is string/varchar type
                type_query = f"""
                SELECT COUNT(*) as type_passed_count
                FROM {check_config.schema_name}.{check_config.table_name}
                WHERE ({check_config.data_field_name} IS NOT NULL
                AND TRY_CAST({check_config.data_field_name} AS VARCHAR) IS NOT NULL)
                OR {check_config.data_field_name} IS NULL
                """

            type_passed = conn.execute(type_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.TYPE,
                records_passed=type_passed,
                records_failed=total_records - type_passed,
                check_name=f"{check_config.check_name}_type_{check_config.df_type}"
            ))

        # Check synchronized null values
        if check_config.is_sync_null:
            sync_null_query = f"""
            SELECT COUNT(*) as sync_null_passed_count
            FROM {check_config.schema_name}.{check_config.table_name}
            WHERE ({check_config.data_field_name} IS NULL AND {check_config.sync_null_field} IS NULL)
            OR ({check_config.data_field_name} IS NOT NULL AND {check_config.sync_null_field} IS NOT NULL)
            """
            sync_null_passed = conn.execute(sync_null_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.SYNC_NULL,
                records_passed=sync_null_passed,
                records_failed=total_records - sync_null_passed,
                check_name=f"{check_config.check_name}_sync_null_{check_config.sync_null_field}"
            ))

        # Check synchronized zero values
        if check_config.is_sync_zero:
            sync_zero_query = f"""
            SELECT COUNT(*) as sync_zero_passed_count
            FROM {check_config.schema_name}.{check_config.table_name}
            WHERE ({check_config.data_field_name} = 0 AND {check_config.sync_zero_field} = 0)
            OR ({check_config.data_field_name} != 0 AND {check_config.sync_zero_field} != 0)
            OR ({check_config.data_field_name} IS NULL OR {check_config.sync_zero_field} IS NULL)
            """
            sync_zero_passed = conn.execute(sync_zero_query).fetchone()[0]

            results.append(DataQualityResult(
                database_name=check_config.database_name,
                schema_name=check_config.schema_name,
                table_name=check_config.table_name,
                data_field_name=check_config.data_field_name,
                total_records=total_records,
                check_type=CheckType.SYNC_ZERO,
                records_passed=sync_zero_passed,
                records_failed=total_records - sync_zero_passed,
                check_name=f"{check_config.check_name}_sync_zero_{check_config.sync_zero_field}"
            ))

    except Exception as e:
        error_msg = f"Error during data quality check: {e}"
        logger.error(error_msg)
        raise
    finally:
        conn.close()

    return results

def save_results_to_duckdb(results: List[DataQualityResult],
                          database_name: str = "assignment.duckdb",
                          schema_name: str = "dq",
                          table_name: str = "dq_results") -> None:
    """
    Save data quality results to a DuckDB table
    """
    if not results:
        logger.warning("No results to save")
        return

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    db_path = project_root / database_name

    # Convert results to DataFrame
    results_data = []
    for result in results:
        results_data.append({
            'database_name': result.database_name,
            'schema_name': result.schema_name,
            'table_name': result.table_name,
            'data_field_name': result.data_field_name,
            'total_records': result.total_records,
            'check_type': result.check_type.value,
            'records_passed': result.records_passed,
            'records_failed': result.records_failed,
            'check_name': result.check_name,
            'check_timestamp': pd.Timestamp.now()
        })

    df = pd.DataFrame(results_data)

    # Connect to DuckDB
    conn = duckdb.connect(str(db_path))

    try:
        # Create quality schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        logger.info(f"Created/verified {schema_name} schema")

        # Drop existing table and create new one
        conn.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
        conn.execute(f"CREATE TABLE {schema_name}.{table_name} AS SELECT * FROM df")

        logger.info(f"Saved {len(results)} quality check results to {schema_name}.{table_name}")

        # Show summary
        summary = conn.execute(f"""
        SELECT
            check_type,
            COUNT(*) as total_checks,
            SUM(records_failed) as total_failures,
            SUM(total_records) as total_records_checked
        FROM {schema_name}.{table_name}
        GROUP BY check_type
        ORDER BY check_type
        """).fetchdf()

        logger.info("Quality Check Summary:")
        for _, row in summary.iterrows():
            logger.info(f"  {row['check_type']}: {row['total_checks']} checks, {row['total_failures']} failures, {row['total_records_checked']} records checked")

    except Exception as e:
        error_msg = f"Error saving results: {e}"
        logger.error(error_msg)
        raise
    finally:
        conn.close()

def load_quality_config(config_path: str = 'quality_config.yaml') -> List[DataQualityCheck]:
    """
    Load data quality check configuration from YAML file
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_file_path = project_root / config_path

    if not config_file_path.exists():
        error_msg = f"Config file not found: {config_file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    quality_checks = []
    for check_config in config.get('quality_checks', []):
        quality_checks.append(DataQualityCheck(**check_config))

    return quality_checks

def run_all_quality_checks(config_path: str = 'quality_config.yaml') -> None:
    """
    Run all data quality checks from configuration file
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA QUALITY CHECKS")
    logger.info("=" * 60)

    # Load configuration
    quality_checks = load_quality_config(config_path)

    if not quality_checks:
        logger.warning("No quality checks found in configuration")
        return

    logger.info(f"Found {len(quality_checks)} quality checks to process")

    all_results = []

    # Run each quality check
    for i, check in enumerate(quality_checks, 1):
        logger.info("=" * 60)
        logger.info(f"Running Quality Check {i}/{len(quality_checks)}: {check.check_name}")
        logger.info("=" * 60)

        try:
            results = run_data_quality_check(check)
            all_results.extend(results)

            for result in results:
                status = "PASSED" if result.records_failed == 0 else "FAILED"
                logger.info(f"{status} {result.check_type.value}: {result.records_passed}/{result.total_records} records passed")

        except Exception as e:
            logger.error(f"Failed to run quality check {check.check_name}: {e}")
            continue

    # Save all results to DuckDB
    if all_results:
        logger.info("=" * 60)
        logger.info("Saving results to DuckDB...")
        logger.info("=" * 60)
        save_results_to_duckdb(all_results)

    logger.info("=" * 60)
    logger.info("All quality checks completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Example: Run all checks from config file
    run_all_quality_checks()

    # Example: Run a single check programmatically
    # single_check = DataQualityCheck(
    #     database_name="assignment.duckdb",
    #     schema_name="bronze",
    #     table_name="soc_dem",
    #     data_field_name="customer_id",
    #     check_name="customer_id_manual_check",
    #     is_check_unique=True,
    #     is_check_null=True
    # )
    # results = run_data_quality_check(single_check)
    # save_results_to_duckdb(results)