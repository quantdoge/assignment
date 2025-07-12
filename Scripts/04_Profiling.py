import duckdb
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 04_Profiling.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"04_Profiling_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('Profiling')
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

def profile_numerical_data(
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "silver",
    table_name: str = "train",
    profile_name: str = "silver_train_profile"
) -> List[Dict[str, Any]]:
    """
    Profile numerical data types and return statistical metrics

    Args:
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the table
        table_name: Table name to profile
        profile_name: Name identifier for this profiling session

    Returns:
        List of dictionaries containing profiling metrics for each numerical column
    """
    logger = logging.getLogger('Profiling')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Starting numerical data profiling for {schema_name}.{table_name}")
    logger.info(f"Database path: {db_path}")

    # Check if database file exists
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))
    logger.info(f"Connected to DuckDB database: {db_path}")

    try:
        # Check if table exists
        table_check = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
        """).fetchone()

        if table_check[0] == 0:
            logger.error(f"Table not found: {schema_name}.{table_name}")
            raise ValueError(f"Table not found: {schema_name}.{table_name}")

        logger.info(f"Found table: {schema_name}.{table_name}")

        # Get all columns and their data types
        columns_result = conn.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()

        # Filter for numerical columns
        numerical_columns = []
        for col_name, data_type in columns_result:
            if any(numeric_type in data_type.upper() for numeric_type in
                   ['INTEGER', 'BIGINT', 'DOUBLE', 'REAL', 'NUMERIC', 'DECIMAL', 'FLOAT']):
                numerical_columns.append(col_name)

        logger.info(f"Found {len(numerical_columns)} numerical columns: {numerical_columns}")

        if not numerical_columns:
            logger.warning("No numerical columns found for profiling")
            return []

        # Profile each numerical column
        profiling_results = []

        for column in numerical_columns:
            logger.info(f"Profiling column: {column}")

            try:
                # Calculate all statistical metrics in a single query
                stats_query = f"""
                WITH stats AS (
                    SELECT
                        MIN("{column}") as min_value,
                        MAX("{column}") as max_value,
                        MEDIAN("{column}") as median_value,
                        AVG("{column}") as mean_value,
                        MODE() WITHIN GROUP (ORDER BY "{column}") as mode_value,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{column}") as p25,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{column}") as p75,
                        SKEWNESS("{column}") as skewness_value,
                        KURTOSIS("{column}") as kurtosis_value
                    FROM {schema_name}.{table_name}
                    WHERE "{column}" IS NOT NULL
                )
                SELECT
                    min_value,
                    max_value,
                    (max_value - min_value) as min_max_range,
                    median_value,
                    mean_value,
                    mode_value,
                    p25 as percentile_25,
                    p75 as percentile_75,
                    (p75 - p25) as iqr,
                    skewness_value,
                    kurtosis_value
                FROM stats
                """

                result = conn.execute(stats_query).fetchone()

                if result:
                    # Create profile dictionary with exact sequence as requested
                    profile_dict = {
                        'name_of_profile': profile_name,
                        'schema_name': schema_name,
                        'table_field_name': table_name,
                        'data_field_name': column,
                        'min_value': result[0],
                        'max_value': result[1],
                        'min_max_range': result[2],
                        'median_value': result[3],
                        'mean_value': result[4],
                        'mode': result[5],
                        '25_percentile_value': result[6],
                        '75_percentile_value': result[7],
                        'iqr': result[8],
                        'skewness': result[9],
                        'kurtosis': result[10]
                    }

                    profiling_results.append(profile_dict)
                    logger.info(f"Successfully profiled column: {column}")

                    # Log some key metrics
                    logger.info(f"  Min: {result[0]}, Max: {result[1]}, Mean: {result[4]:.4f}, Median: {result[3]}")

            except Exception as e:
                logger.error(f"Error profiling column {column}: {e}")
                continue

        logger.info(f"Profiling completed. Total columns profiled: {len(profiling_results)}")

        # Log summary of profiling results
        if profiling_results:
            logger.info("Profiling Summary:")
            for profile in profiling_results:
                logger.info(f"  Column: {profile['data_field_name']}")
                logger.info(f"    Range: {profile['min_value']} to {profile['max_value']}")
                logger.info(f"    Mean: {profile['mean_value']:.4f}, Median: {profile['median_value']}")
                logger.info(f"    Skewness: {profile['skewness']:.4f}, Kurtosis: {profile['kurtosis']}")

        return profiling_results

    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        raise
    finally:
        # Close the connection
        conn.close()
        logger.info("Database connection closed")

def profile_categorical_data(
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "silver",
    table_name: str = "train",
    profile_name: str = "silver_train_categorical_profile",
    categorical_columns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Profile categorical data types and analyze class balances

    Args:
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the table
        table_name: Table name to profile
        profile_name: Name identifier for this profiling session
        categorical_columns: List of specific categorical columns to profile. If None, auto-detect.

    Returns:
        List of dictionaries containing profiling metrics for each categorical column
    """
    logger = logging.getLogger('Profiling')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Starting categorical data profiling for {schema_name}.{table_name}")
    logger.info(f"Database path: {db_path}")

    # Check if database file exists
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))
    logger.info(f"Connected to DuckDB database: {db_path}")

    try:
        # Check if table exists
        table_check = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
        """).fetchone()

        if table_check[0] == 0:
            logger.error(f"Table not found: {schema_name}.{table_name}")
            raise ValueError(f"Table not found: {schema_name}.{table_name}")

        logger.info(f"Found table: {schema_name}.{table_name}")

        # If no specific columns provided, auto-detect categorical columns
        if categorical_columns is None:
            columns_result = conn.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()

            # Filter for categorical columns (VARCHAR, TEXT, BOOLEAN, and small INTEGER domains)
            categorical_columns = []
            for col_name, data_type in columns_result:
                if any(cat_type in data_type.upper() for cat_type in
                       ['VARCHAR', 'TEXT', 'CHAR', 'BOOLEAN']):
                    categorical_columns.append(col_name)
                elif 'INTEGER' in data_type.upper():
                    # Check if integer column has limited unique values (likely categorical)
                    unique_count = conn.execute(f"""
                        SELECT COUNT(DISTINCT "{col_name}") as unique_count
                        FROM {schema_name}.{table_name}
                        WHERE "{col_name}" IS NOT NULL
                    """).fetchone()[0]

                    # Consider as categorical if less than 20 unique values
                    if unique_count <= 20:
                        categorical_columns.append(col_name)

        logger.info(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")

        if not categorical_columns:
            logger.warning("No categorical columns found for profiling")
            return []

        # Profile each categorical column
        profiling_results = []

        for column in categorical_columns:
            logger.info(f"Profiling categorical column: {column}")

            try:
                # Get total count for the column (excluding nulls)
                total_count_query = f"""
                    SELECT COUNT(*) as total_count
                    FROM {schema_name}.{table_name}
                    WHERE "{column}" IS NOT NULL
                """
                total_count = conn.execute(total_count_query).fetchone()[0]

                # Get null count
                null_count_query = f"""
                    SELECT COUNT(*) as null_count
                    FROM {schema_name}.{table_name}
                    WHERE "{column}" IS NULL
                """
                null_count = conn.execute(null_count_query).fetchone()[0]

                # Get unique values and their counts
                value_counts_query = f"""
                    SELECT
                        "{column}" as category_value,
                        COUNT(*) as count,
                        ROUND(COUNT(*) * 100.0 / {total_count}, 2) as percentage
                    FROM {schema_name}.{table_name}
                    WHERE "{column}" IS NOT NULL
                    GROUP BY "{column}"
                    ORDER BY COUNT(*) DESC
                """

                value_counts = conn.execute(value_counts_query).fetchall()
                unique_values_count = len(value_counts)

                # Calculate class balance metrics
                if unique_values_count > 0:
                    # Get the most and least frequent categories
                    most_frequent_category = value_counts[0][0]
                    most_frequent_count = value_counts[0][1]
                    most_frequent_percentage = value_counts[0][2]

                    least_frequent_category = value_counts[-1][0]
                    least_frequent_count = value_counts[-1][1]
                    least_frequent_percentage = value_counts[-1][2]

                    # Calculate balance ratio (most frequent / least frequent)
                    balance_ratio = most_frequent_count / least_frequent_count if least_frequent_count > 0 else float('inf')

                    # Determine if data is balanced (commonly considered balanced if ratio < 2)
                    is_balanced = balance_ratio < 2.0

                    # Create detailed category distribution
                    category_distribution = {}
                    for category, count, percentage in value_counts:
                        category_distribution[str(category)] = {
                            'count': count,
                            'percentage': percentage
                        }

                else:
                    most_frequent_category = None
                    most_frequent_count = 0
                    most_frequent_percentage = 0
                    least_frequent_category = None
                    least_frequent_count = 0
                    least_frequent_percentage = 0
                    balance_ratio = 0
                    is_balanced = True
                    category_distribution = {}

                # Create profile dictionary
                profile_dict = {
                    'name_of_profile': profile_name,
                    'schema_name': schema_name,
                    'table_field_name': table_name,
                    'data_field_name': column,
                    'total_records': total_count + null_count,
                    'non_null_records': total_count,
                    'null_records': null_count,
                    'null_percentage': round((null_count / (total_count + null_count) * 100), 2) if (total_count + null_count) > 0 else 0,
                    'unique_values_count': unique_values_count,
                    'most_frequent_category': most_frequent_category,
                    'most_frequent_count': most_frequent_count,
                    'most_frequent_percentage': most_frequent_percentage,
                    'least_frequent_category': least_frequent_category,
                    'least_frequent_count': least_frequent_count,
                    'least_frequent_percentage': least_frequent_percentage,
                    'balance_ratio': balance_ratio,
                    'is_balanced': is_balanced,
                    'category_distribution': str(category_distribution)  # Store as string for database compatibility
                }

                profiling_results.append(profile_dict)
                logger.info(f"Successfully profiled categorical column: {column}")

                # Log key metrics
                logger.info(f"  Unique values: {unique_values_count}")
                logger.info(f"  Most frequent: {most_frequent_category} ({most_frequent_percentage}%)")
                logger.info(f"  Balance ratio: {balance_ratio:.2f} ({'Balanced' if is_balanced else 'Imbalanced'})")

            except Exception as e:
                logger.error(f"Error profiling categorical column {column}: {e}")
                continue

        logger.info(f"Categorical profiling completed. Total columns profiled: {len(profiling_results)}")

        # Log summary of profiling results
        if profiling_results:
            logger.info("Categorical Profiling Summary:")
            for profile in profiling_results:
                logger.info(f"  Column: {profile['data_field_name']}")
                logger.info(f"    Unique values: {profile['unique_values_count']}")
                logger.info(f"    Balance ratio: {profile['balance_ratio']:.2f}")
                logger.info(f"    Status: {'Balanced' if profile['is_balanced'] else 'Imbalanced'}")

        return profiling_results

    except Exception as e:
        logger.error(f"Error during categorical profiling: {e}")
        raise
    finally:
        # Close the connection
        conn.close()
        logger.info("Database connection closed")

def save_profiling_results(
    profiling_results: List[Dict[str, Any]],
    target_schema_name: str,
    target_table_name: str,
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save profiling results to a specified schema and table in DuckDB

    Args:
        profiling_results: List of profiling dictionaries from profile_numerical_data
        target_schema_name: Schema name where the results table will be created
        target_table_name: Table name for storing the profiling results
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('Profiling')

    if not profiling_results:
        logger.warning("No profiling results to save")
        return

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving profiling results to {target_schema_name}.{target_table_name}")
    logger.info(f"Database path: {db_path}")

    # Check if database file exists
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))
    logger.info(f"Connected to DuckDB database: {db_path}")

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")
        logger.info(f"Created/verified schema: {target_schema_name}")

        # Convert profiling results to DataFrame
        df = pd.DataFrame(profiling_results)
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

        # Drop the table if it exists and create new one
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{target_table_name}")
        logger.info(f"Dropped existing table if present: {target_schema_name}.{target_table_name}")

        # Save DataFrame to DuckDB table
        conn.execute(f"CREATE TABLE {target_schema_name}.{target_table_name} AS SELECT * FROM df")
        logger.info(f"Successfully created table: {target_schema_name}.{target_table_name}")

        # Verify the data was saved correctly
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {target_schema_name}.{target_table_name}").fetchone()
        logger.info(f"Verified: {result[0]} rows saved to {target_schema_name}.{target_table_name}")

        # Show sample of saved data
        logger.info("Sample of saved profiling data:")
        sample_data = conn.execute(f"SELECT * FROM {target_schema_name}.{target_table_name} LIMIT 3").fetchdf()
        logger.info(f"\n{sample_data}")

    except Exception as e:
        logger.error(f"Error saving profiling results: {e}")
        raise
    finally:
        # Close the connection
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting data profiling process")

    try:
        # Run numerical profiling
        numerical_results = profile_numerical_data(
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="train",
            profile_name="silver_train_numerical_profile"
        )

        logger.info(f"✅ Numerical data profiling completed successfully. Profiled {len(numerical_results)} columns")

        # Run categorical profiling
        categorical_results = profile_categorical_data(
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="train",
            profile_name="silver_train_categorical_profile",
            categorical_columns=['Sale_MF','Sale_CC','Sale_CL','Sex']
        )

        logger.info(f"✅ Categorical data profiling completed successfully. Profiled {len(categorical_results)} columns")

        # Save numerical results to DuckDB table
        if numerical_results:
            save_profiling_results(
                profiling_results=numerical_results,
                target_schema_name="profile",
                target_table_name="numerical_profiling_results",
                duckdb_name="assignment.duckdb"
            )
            logger.info("✅ Numerical profiling results saved to database successfully")

        # Save categorical results to DuckDB table
        if categorical_results:
            save_profiling_results(
                profiling_results=categorical_results,
                target_schema_name="profile",
                target_table_name="categorical_profiling_results",
                duckdb_name="assignment.duckdb"
            )
            logger.info("✅ Categorical profiling results saved to database successfully")

        # Print results summary to console
        if numerical_results:
            print(f"\nNumerical Profiling completed! Found {len(numerical_results)} numerical columns:")
            for result in numerical_results:
                print(f"  - {result['data_field_name']}: Range [{result['min_value']:.2f}, {result['max_value']:.2f}]")

        if categorical_results:
            print(f"\nCategorical Profiling completed! Found {len(categorical_results)} categorical columns:")
            for result in categorical_results:
                balance_status = "Balanced" if result['is_balanced'] else "Imbalanced"
                print(f"  - {result['data_field_name']}: {result['unique_values_count']} unique values ({balance_status})")

        print(f"\nResults saved to:")
        if numerical_results:
            print(f"  - profile.numerical_profiling_results")
        if categorical_results:
            print(f"  - profile.categorical_profiling_results")

    except Exception as e:
        logger.error(f"❌ Data profiling failed: {e}")
        raise