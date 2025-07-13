import duckdb
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import joblib
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration for prediction process
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"08_Model_Predict_Test_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('ModelPrediction')
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

def get_model_predictors_from_db(
    model_table_name: str = "final_xgboost_cl_model",
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "models"
) -> List[str]:
    """
    Get the predictor variables used in the trained model from database

    Args:
        model_table_name: Name of the model results table
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the model results

    Returns:
        List of predictor variable names
    """
    logger = logging.getLogger('ModelPrediction')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Getting predictor variables from {schema_name}.{model_table_name}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Check if table exists
        table_check = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = '{schema_name}' AND table_name = '{model_table_name}'
        """).fetchone()

        if table_check[0] == 0:
            logger.error(f"Model table not found: {schema_name}.{model_table_name}")
            raise ValueError(f"Model table not found: {schema_name}.{model_table_name}")

        # Get the predictive variables from the most recent model
        query = f"""
            SELECT predictive_variables
            FROM {schema_name}.{model_table_name}
            ORDER BY model_timestamp DESC
            LIMIT 1
        """
        result = conn.execute(query).fetchone()

        if result is None:
            logger.error("No model results found in the table")
            raise ValueError("No model results found in the table")

        # Parse the comma-separated predictor variables
        predictors_str = result[0]
        predictors = [var.strip() for var in predictors_str.split(',')]

        logger.info(f"Found {len(predictors)} predictor variables: {predictors}")
        return predictors

    except Exception as e:
        logger.error(f"Error getting predictor variables: {e}")
        raise
    finally:
        conn.close()

def load_test_data_from_duckdb(
    predictors: List[str],
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "silver",
    table_name: str = "test"
) -> pd.DataFrame:
    """
    Load test data from DuckDB with only the required predictor columns

    Args:
        predictors: List of predictor variable names to select
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the test table
        table_name: Test table name

    Returns:
        pandas DataFrame with test data
    """
    logger = logging.getLogger('ModelPrediction')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Loading test data from {schema_name}.{table_name}")
    logger.info(f"Selecting predictors: {predictors}")

    # Check if database file exists
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

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

        # Check which predictors exist in the table
        available_columns = conn.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
        """).fetchall()

        available_columns = [col[0] for col in available_columns]
        missing_predictors = [pred for pred in predictors if pred not in available_columns]

        if missing_predictors:
            logger.warning(f"Missing predictors in test table: {missing_predictors}")
            # Use only available predictors
            predictors = [pred for pred in predictors if pred in available_columns]
            logger.info(f"Using available predictors: {predictors}")

        # Also include Client column for identification
        columns_to_select = ['Client'] + predictors
        columns_str = ', '.join([f'"{col}"' for col in columns_to_select])

        # Load data into pandas DataFrame
        query = f"SELECT {columns_str} FROM {schema_name}.{table_name}"
        df = conn.execute(query).fetchdf()

        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Test data shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise
    finally:
        conn.close()

def load_trained_model(model_filename: str = "final_xgboost_cl_model_Sale_CL_20250714_020030.joblib") -> Pipeline:
    """
    Load the trained model from the Models directory

    Args:
        model_filename: Name of the model file to load

    Returns:
        Loaded sklearn Pipeline
    """
    logger = logging.getLogger('ModelPrediction')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define model path
    models_dir = project_root / "Models"
    model_path = models_dir / model_filename

    logger.info(f"Loading trained model from: {model_path}")

    # Check if model file exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        # List available models
        if models_dir.exists():
            available_models = list(models_dir.glob("*.joblib"))
            logger.info(f"Available models: {[m.name for m in available_models]}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load the model
        pipeline = joblib.load(model_path)
        logger.info("Successfully loaded trained model")

        # Log model information
        if hasattr(pipeline, 'steps'):
            logger.info(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")

        return pipeline

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_test_data(
    test_data: pd.DataFrame,
    predictors: List[str],
    categorical_variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply the same preprocessing as used in training

    Args:
        test_data: Test data DataFrame
        predictors: List of predictor variables
        categorical_variables: List of categorical variables

    Returns:
        Preprocessed test data
    """
    logger = logging.getLogger('ModelPrediction')

    logger.info("Preprocessing test data...")
    logger.info(f"Input shape: {test_data.shape}")

    # Create a copy of the data with only predictors
    data_clean = test_data[['Client'] + predictors].copy()

    # Handle missing values (same logic as in training script)
    for col in predictors:
        if data_clean[col].dtype in ['object', 'category']:
            data_clean[col] = data_clean[col].fillna('Unknown')
        else:
            data_clean[col] = data_clean[col].fillna(0)

    logger.info(f"Data after cleaning: {data_clean.shape}")

    # Check for any remaining null values
    null_counts = data_clean[predictors].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Remaining null values: {null_counts[null_counts > 0].to_dict()}")

    return data_clean

def make_predictions(
    pipeline: Pipeline,
    test_data: pd.DataFrame,
    predictors: List[str]
) -> pd.DataFrame:
    """
    Make predictions using the trained model

    Args:
        pipeline: Trained sklearn pipeline
        test_data: Preprocessed test data
        predictors: List of predictor variables

    Returns:
        DataFrame with predictions
    """
    logger = logging.getLogger('ModelPrediction')

    logger.info("Making predictions...")

    # Prepare features for prediction
    X_test = test_data[predictors]

    logger.info(f"Prediction input shape: {X_test.shape}")

    try:
        # Make predictions
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
        y_pred = pipeline.predict(X_test)  # Binary predictions

        # Create results DataFrame
        results = test_data[['Client']].copy()
        results['predicted_probability'] = y_pred_proba
        results['predicted_class'] = y_pred

        logger.info(f"Predictions completed. Results shape: {results.shape}")
        logger.info(f"Prediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")
        logger.info(f"Probability statistics: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

        return results

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def save_predictions_to_db(
    predictions: pd.DataFrame,
    target_schema_name: str = "models",
    target_table_name: str = "test_predictions_cl",
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save predictions to DuckDB database

    Args:
        predictions: DataFrame with predictions
        target_schema_name: Schema name for saving predictions
        target_table_name: Table name for saving predictions
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('ModelPrediction')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving predictions to {target_schema_name}.{target_table_name}")

    # Add timestamp
    predictions_with_timestamp = predictions.copy()
    predictions_with_timestamp['prediction_timestamp'] = datetime.now().isoformat()

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")
        logger.info(f"Created/verified schema: {target_schema_name}")

        # Drop existing table and create new one
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{target_table_name}")

        # Create table from DataFrame
        conn.execute(f"""
            CREATE TABLE {target_schema_name}.{target_table_name} AS
            SELECT * FROM predictions_with_timestamp
        """)

        logger.info(f"Successfully saved {len(predictions)} predictions to {target_schema_name}.{target_table_name}")

        # Verify the data was saved
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {target_schema_name}.{target_table_name}").fetchone()
        logger.info(f"Verified: {result[0]} records in table")

    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise
    finally:
        conn.close()

def save_predictions_to_csv(
    predictions: pd.DataFrame,
    filename: str = "test_predictions_cl.csv"
) -> None:
    """
    Save predictions to CSV file in Outputs directory

    Args:
        predictions: DataFrame with predictions
        filename: Name of the output CSV file
    """
    logger = logging.getLogger('ModelPrediction')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Outputs directory if it doesn't exist
    outputs_dir = project_root / "Outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_filename = f"{filename.replace('.csv', '')}_{timestamp}.csv"
    output_path = outputs_dir / timestamped_filename

    try:
        # Add timestamp column
        predictions_with_timestamp = predictions.copy()
        predictions_with_timestamp['prediction_timestamp'] = datetime.now().isoformat()

        # Save to CSV
        predictions_with_timestamp.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to CSV: {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {e}")
        raise

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting model prediction process for test data")

    try:
        final_xgboost_cl_model
        final_xgboost_cl_model_Sale_CL_20250714_020030
        test_predictions_cl_sales
        test_predictions_cl_sales.csv

        # Configuration
        model_table_name = "final_xgboost_cl_model"
        model_filename = "final_xgboost_cl_model_Sale_CL_20250714_020030.joblib"
        categorical_variables = []  # Update if there were categorical variables in training

        logger.info(f"Using model table: {model_table_name}")
        logger.info(f"Using model file: {model_filename}")

        # Step 1: Get predictor variables from database
        predictors = get_model_predictors_from_db(
            model_table_name=model_table_name,
            duckdb_name="assignment.duckdb",
            schema_name="models"
        )

        # Step 2: Load test data with only required predictors
        test_data = load_test_data_from_duckdb(
            predictors=predictors,
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="test"
        )

        # Step 3: Load the trained model
        trained_pipeline = load_trained_model(model_filename=model_filename)

        # Step 4: Preprocess test data (same as training)
        preprocessed_test_data = preprocess_test_data(
            test_data=test_data,
            predictors=predictors,
            categorical_variables=categorical_variables
        )

        # Step 5: Make predictions
        predictions = make_predictions(
            pipeline=trained_pipeline,
            test_data=preprocessed_test_data,
            predictors=predictors
        )

        # Step 6: Save predictions to database
        save_predictions_to_db(
            predictions=predictions,
            target_schema_name="models",
            target_table_name="test_predictions_cl_sales",
            duckdb_name="assignment.duckdb"
        )

        # Step 7: Save predictions to CSV
        csv_path = save_predictions_to_csv(
            predictions=predictions,
            filename="test_predictions_cl_sales.csv"
        )

        logger.info("✅ Model prediction process completed successfully")

        # Print summary
        print(f"\n🎯 Prediction Summary:")
        print(f"Model Used: {model_filename}")
        print(f"Predictors: {predictors}")
        print(f"Test Records: {len(predictions)}")
        print(f"Predictions with Class 1: {predictions['predicted_class'].sum()}")
        print(f"Predictions with Class 0: {(predictions['predicted_class'] == 0).sum()}")
        print(f"Average Probability: {predictions['predicted_probability'].mean():.4f}")
        print(f"Results saved to: models.test_predictions_cl")
        print(f"CSV saved to: {csv_path}")

        # Show sample predictions
        print(f"\nSample Predictions:")
        print(predictions.head(10))

    except Exception as e:
        logger.error(f"❌ Model prediction process failed: {e}")
        print(f"❌ Error: {e}")
        raise