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
from sklearn.preprocessing import StandardScaler

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 08_Predict_Regressor.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"08_Predict_Regressor_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('PredictRegressor')
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

def get_predictor_variables_from_db(
    model_table_name: str = "final_xgboost_cl_revenue_model",
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "models"
) -> List[str]:
    """
    Get the predictor variables used in the final model from the database

    Args:
        model_table_name: Name of the model table in the database
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the model table

    Returns:
        List of predictor variable names
    """
    logger = logging.getLogger('PredictRegressor')

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

        # Get the predictive variables from the most recent model record
        query = f"""
            SELECT predictive_variables
            FROM {schema_name}.{model_table_name}
            ORDER BY model_timestamp DESC
            LIMIT 1
        """

        result = conn.execute(query).fetchone()

        if result is None or result[0] is None:
            logger.error("No predictive variables found in the model table")
            raise ValueError("No predictive variables found in the model table")

        # Parse the comma-separated string into a list
        predictor_variables = [var.strip() for var in result[0].split(',')]

        logger.info(f"Found {len(predictor_variables)} predictor variables: {predictor_variables}")

        return predictor_variables

    except Exception as e:
        logger.error(f"Error getting predictor variables: {e}")
        raise
    finally:
        conn.close()

def load_test_data_from_duckdb(
    predictor_variables: List[str],
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "silver",
    table_name: str = "test"
) -> pd.DataFrame:
    """
    Load test data from DuckDB table with only the required predictor variables

    Args:
        predictor_variables: List of predictor variable names to select
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the table
        table_name: Table name to load

    Returns:
        pandas DataFrame with the test data
    """
    logger = logging.getLogger('PredictRegressor')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Loading test data from {schema_name}.{table_name}")
    logger.info(f"Selecting predictor variables: {predictor_variables}")

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

        # Check which predictor variables exist in the table
        existing_columns = conn.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
        """).fetchall()

        existing_columns = [col[0] for col in existing_columns]
        missing_columns = [var for var in predictor_variables if var not in existing_columns]

        if missing_columns:
            logger.warning(f"Missing predictor variables in test data: {missing_columns}")

        # Select only available predictor variables plus Client ID for tracking
        available_predictors = [var for var in predictor_variables if var in existing_columns]

        if not available_predictors:
            logger.error("No predictor variables found in test data")
            raise ValueError("No predictor variables found in test data")

        # Include Client column if available
        columns_to_select = available_predictors.copy()
        if 'Client' in existing_columns:
            columns_to_select = ['Client'] + columns_to_select

        # Build query to select required columns
        columns_str = ', '.join([f'"{col}"' for col in columns_to_select])
        query = f"SELECT {columns_str} FROM {schema_name}.{table_name}"

        df = conn.execute(query).fetchdf()

        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Data shape: {df.shape}")

        # Log basic statistics
        logger.info("Test data summary:")
        logger.info(f"Available predictor variables: {available_predictors}")

        return df

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise
    finally:
        conn.close()

def preprocess_test_data(
    data: pd.DataFrame,
    predictor_variables: List[str],
    categorical_variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply the same preprocessing as used in the training (from 07_Model_Final_Regressor.py)

    Args:
        data: Input test DataFrame
        predictor_variables: List of predictor variable names
        categorical_variables: List of categorical variables

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger('PredictRegressor')

    logger.info("Applying preprocessing to test data...")
    logger.info(f"Original data shape: {data.shape}")

    # Create a copy to avoid modifying original data
    data_clean = data.copy()

    # Handle missing values - same as in 07_Model_Final_Regressor.py
    for col in predictor_variables:
        if col in data_clean.columns:
            if data_clean[col].dtype in ['object', 'category']:
                data_clean[col] = data_clean[col].fillna('Unknown')
                logger.info(f"Filled missing values in categorical column '{col}' with 'Unknown'")
            else:
                data_clean[col] = data_clean[col].fillna(0)
                logger.info(f"Filled missing values in numerical column '{col}' with 0")

    logger.info(f"Data shape after preprocessing: {data_clean.shape}")
    logger.info("Preprocessing completed")

    return data_clean

def load_trained_model(model_filename: str) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Load the trained model and response scaler

    Args:
        model_filename: Name of the model file (e.g., 'final_xgboost_cl_revenue_model_Revenue_CL_20250714_021157.joblib')

    Returns:
        Tuple of (loaded pipeline, response scaler if exists)
    """
    logger = logging.getLogger('PredictRegressor')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define models directory
    models_dir = project_root / "Models"
    model_path = models_dir / model_filename

    logger.info(f"Loading trained model from: {model_path}")

    # Check if model file exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load the pipeline
        pipeline = joblib.load(model_path)
        logger.info("Successfully loaded trained pipeline")

        # Try to load response scaler if it exists
        response_scaler = None
        base_name = model_filename.replace('.joblib', '')
        scaler_filename = f"{base_name}_scaler.joblib"
        scaler_path = models_dir / scaler_filename

        if scaler_path.exists():
            response_scaler = joblib.load(scaler_path)
            logger.info(f"Successfully loaded response scaler from: {scaler_path}")
        else:
            logger.info("No response scaler found - predictions will be on original scale")

        return pipeline, response_scaler

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def make_predictions(
    pipeline: Any,
    test_data: pd.DataFrame,
    predictor_variables: List[str],
    response_scaler: Optional[StandardScaler] = None,
    prediction_column_name: str = "Predicted_Revenue"
) -> pd.DataFrame:
    """
    Make predictions on test data using the loaded model

    Args:
        pipeline: Trained sklearn pipeline
        test_data: Preprocessed test data
        predictor_variables: List of predictor variable names
        response_scaler: Response scaler if used during training
        prediction_column_name: Name for the prediction column in output

    Returns:
        DataFrame with predictions
    """
    logger = logging.getLogger('PredictRegressor')

    logger.info("Making predictions on test data...")

    # Prepare features for prediction
    available_predictors = [var for var in predictor_variables if var in test_data.columns]
    X_test = test_data[available_predictors]

    logger.info(f"Using {len(available_predictors)} predictor variables for prediction")
    logger.info(f"Test data shape for prediction: {X_test.shape}")

    try:
        # Make predictions
        predictions_scaled = pipeline.predict(X_test)
        logger.info(f"Generated {len(predictions_scaled)} predictions")

        # Apply inverse scaling if response scaler was used
        if response_scaler is not None:
            predictions = response_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            logger.info("Applied inverse scaling to predictions")
        else:
            predictions = predictions_scaled
            logger.info("No scaling applied to predictions")

        # Create results DataFrame
        results_df = test_data.copy()
        results_df[prediction_column_name] = predictions

        # Log prediction statistics
        logger.info("Prediction statistics:")
        logger.info(f"  Mean prediction: {predictions.mean():.4f}")
        logger.info(f"  Std prediction: {predictions.std():.4f}")
        logger.info(f"  Min prediction: {predictions.min():.4f}")
        logger.info(f"  Max prediction: {predictions.max():.4f}")

        return results_df

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def save_predictions_to_db(
    predictions_df: pd.DataFrame,
    target_schema_name: str = "predictions",
    target_table_name: str = "test_revenue_cl_predictions",
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save predictions to DuckDB database

    Args:
        predictions_df: DataFrame with predictions
        target_schema_name: Schema name for saving predictions
        target_table_name: Table name for saving predictions
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('PredictRegressor')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving predictions to {target_schema_name}.{target_table_name}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")
        logger.info(f"Created/verified schema: {target_schema_name}")

        # Drop table if exists and create new one
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{target_table_name}")

        # Register the DataFrame and create table
        conn.register('predictions_df', predictions_df)
        conn.execute(f"""
            CREATE TABLE {target_schema_name}.{target_table_name} AS
            SELECT * FROM predictions_df
        """)

        logger.info(f"Successfully saved {len(predictions_df)} predictions to {target_schema_name}.{target_table_name}")

        # Verify the data was saved
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {target_schema_name}.{target_table_name}").fetchone()
        logger.info(f"Verified: {result[0]} rows in {target_schema_name}.{target_table_name}")

    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise
    finally:
        conn.close()

def save_predictions_to_csv(
    predictions_df: pd.DataFrame,
    filename: str = "test_revenue_cl_predictions.csv"
) -> str:
    """
    Save predictions to CSV file

    Args:
        predictions_df: DataFrame with predictions
        filename: Name of the CSV file

    Returns:
        Path to saved CSV file
    """
    logger = logging.getLogger('PredictRegressor')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Results directory if it doesn't exist
    results_dir = project_root / "Results"
    results_dir.mkdir(exist_ok=True)

    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = filename.replace('.csv', '')
    csv_filename = f"{base_name}_{timestamp}.csv"
    csv_path = results_dir / csv_filename

    try:
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to CSV: {csv_path}")
        return str(csv_path)

    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {e}")
        raise

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting prediction process for test data")

    try:

        # model_filename:
        #     - final_xgboost_cl_revenue_model_Revenue_CL_20250714_021157.joblib
        #	  - final_xgboost_cc_revenue_model_Revenue_CC_20250714_021336.joblib

		# model_table_name:
        #     - final_xgboost_cl_revenue_model
        #	  - final_xgboost_cc_revenue_model

		# target_table_name:
      	# 	- test_revenue_cl_predictions
        #	 - test_revenue_cc_predictions

		# prediction_column_name:
		#     - Predicted_Revenue_CL
		#     - Predicted_Revenue_CC

        # filename:
        #     - test_revenue_cl_predictions.csv
        #     - test_revenue_cc_predictions.csv

        # Configuration
        model_filename = "final_xgboost_mf_revenue_model_Revenue_MF_20250714_021433.joblib"
        model_table_name = "final_xgboost_mf_revenue_model"
        categorical_variables = []  # Add if any categorical variables were used
        prediction_column_name = "Predicted_Revenue_MF"  # Configurable prediction column name

        logger.info(f"Using model: {model_filename}")
        logger.info(f"Prediction column name: {prediction_column_name}")

        # Step 1: Get predictor variables from database
        predictor_variables = get_predictor_variables_from_db(
            model_table_name=model_table_name,
            duckdb_name="assignment.duckdb",
            schema_name="models"
        )

        # Step 2: Load test data with only required predictors
        test_data = load_test_data_from_duckdb(
            predictor_variables=predictor_variables,
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="test"
        )

        # Step 3: Preprocess test data
        test_data_processed = preprocess_test_data(
            data=test_data,
            predictor_variables=predictor_variables,
            categorical_variables=categorical_variables
        )

        # Step 4: Load trained model
        pipeline, response_scaler = load_trained_model(model_filename)

        # Step 5: Make predictions
        predictions_df = make_predictions(
            pipeline=pipeline,
            test_data=test_data_processed,
            predictor_variables=predictor_variables,
            response_scaler=response_scaler,
            prediction_column_name=prediction_column_name
        )

        # Step 6: Save predictions to database
        save_predictions_to_db(
            predictions_df=predictions_df,
            target_schema_name="predictions",
            target_table_name="test_revenue_mf_predictions",
            duckdb_name="assignment.duckdb"
        )

        # Step 7: Save predictions to CSV
        csv_path = save_predictions_to_csv(
            predictions_df=predictions_df,
            filename="test_revenue_mf_predictions.csv"
        )

        logger.info("✅ Prediction process completed successfully")

        # Print summary
        print(f"\n🎯 Prediction Summary:")
        print(f"Model Used: {model_filename}")
        print(f"Predictor Variables: {predictor_variables}")
        print(f"Test Data Records: {len(predictions_df)}")
        print(f"Predictions Generated: {len(predictions_df[prediction_column_name])}")
        print(f"Mean Prediction: {predictions_df[prediction_column_name].mean():.4f}")
        print(f"Std Prediction: {predictions_df[prediction_column_name].std():.4f}")
        print(f"Predictions saved to database: predictions.test_revenue_cl_predictions")
        print(f"Predictions saved to CSV: {csv_path}")

    except Exception as e:
        logger.error(f"❌ Prediction process failed: {e}")
        print(f"❌ Error: {e}")
        raise