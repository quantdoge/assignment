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

import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import joblib

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 07_Model_Final_Regressor.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"07_Model_Final_Regressor_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('FinalRegressorTraining')
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

def load_data_from_duckdb(
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "silver",
    table_name: str = "train",
    col_non_zero: str = None
) -> pd.DataFrame:
    """
    Load data from DuckDB table into pandas DataFrame

    Args:
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the table
        table_name: Table name to load
        col_non_zero: Column name to filter for non-zero values

    Returns:
        pandas DataFrame with the loaded data
    """
    logger = logging.getLogger('FinalRegressorTraining')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Loading data from {schema_name}.{table_name}")
    logger.info(f"Database path: {db_path}")

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

        # Load data into pandas DataFrame
        if col_non_zero:
            query = f"SELECT * FROM {schema_name}.{table_name} WHERE {col_non_zero} != 0 AND {col_non_zero} IS NOT NULL"
            logger.info(f"Filtering data where {col_non_zero} is not zero/null")
        else:
            query = f"SELECT * FROM {schema_name}.{table_name}"

        df = conn.execute(query).fetchdf()

        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Data shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    finally:
        conn.close()

def get_best_parameters(
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "models",
    table_name: str = "xgboost_sales_cl_revenue_top"
) -> Dict[str, Any]:
    """
    Retrieve the best parameters from the training results table

    Args:
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the results table
        table_name: Table name with training results

    Returns:
        Dictionary with the best parameters
    """
    logger = logging.getLogger('FinalRegressorTraining')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Loading best parameters from {schema_name}.{table_name}")

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

        # Get the best result (lowest MAE score)
        query = f"""
            SELECT *
            FROM {schema_name}.{table_name}
            ORDER BY mae_score ASC
            LIMIT 1
        """
        best_result = conn.execute(query).fetchdf()

        if best_result.empty:
            logger.error("No results found in the training results table")
            raise ValueError("No results found in the training results table")

        # Extract parameters
        best_params = {}
        param_columns = ['max_depth', 'min_child_weight', 'gamma', 'reg_lambda',
                        'colsample_bytree', 'reg_alpha', 'learning_rate', 'n_estimators']

        for col in param_columns:
            if col in best_result.columns:
                best_params[col] = best_result[col].iloc[0]

        logger.info(f"Best MAE score: {best_result['mae_score'].iloc[0]:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params

    except Exception as e:
        logger.error(f"Error loading best parameters: {e}")
        raise
    finally:
        conn.close()

def train_final_xgboost_regressor(
    data: pd.DataFrame,
    response_variable: str,
    predictive_variables: List[str],
    categorical_variables: Optional[List[str]] = None,
    best_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    model_name: str = "final_xgboost_regressor",
    scale_response: bool = True
) -> Tuple[Pipeline, Dict[str, Any], StandardScaler]:
    """
    Train final XGBoost regressor using best parameters

    Args:
        data: Input DataFrame
        response_variable: Name of the target variable column
        predictive_variables: List of predictor variable names
        categorical_variables: List of categorical variables for one-hot encoding
        best_params: Dictionary with best parameters from hyperparameter tuning
        random_state: Random seed for reproducibility
        model_name: Name for the final model
        scale_response: Whether to scale the response variable

    Returns:
        Tuple of (trained pipeline, evaluation metrics, response scaler)
    """
    logger = logging.getLogger('FinalRegressorTraining')

    logger.info("Starting final XGBoost regressor training")
    logger.info(f"Response variable: {response_variable}")
    logger.info(f"Predictive variables: {predictive_variables}")
    logger.info(f"Categorical variables: {categorical_variables}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Scale response: {scale_response}")

    # Validate inputs
    if response_variable not in data.columns:
        raise ValueError(f"Response variable '{response_variable}' not found in data")

    missing_predictors = [var for var in predictive_variables if var not in data.columns]
    if missing_predictors:
        raise ValueError(f"Predictive variables not found in data: {missing_predictors}")

    # Prepare data
    logger.info("Preparing data...")

    # Handle missing values
    data_clean = data[predictive_variables + [response_variable]].copy()

    # Fill missing values
    for col in predictive_variables:
        if data_clean[col].dtype in ['object', 'category']:
            data_clean[col] = data_clean[col].fillna('Unknown')
        else:
            data_clean[col] = data_clean[col].fillna(0)

    # Remove rows with missing response variable
    data_clean = data_clean.dropna(subset=[response_variable])

    logger.info(f"Data after cleaning: {data_clean.shape}")

    # Prepare features and target
    X = data_clean[predictive_variables]
    y = data_clean[response_variable]

    # Ensure numeric target for regression
    if y.dtype == 'object':
        logger.warning("Target variable is object type, attempting to convert to numeric")
        y = pd.to_numeric(y, errors='coerce')
        y = y.dropna()
        X = X[y.index]

    logger.info(f"Target variable statistics BEFORE scaling: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

    # Scale response variable if requested
    response_scaler = None
    if scale_response:
        logger.info("Applying StandardScaler to response variable...")
        response_scaler = StandardScaler()
        y_scaled = pd.Series(
            response_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(),
            index=y.index,
            name=response_variable
        )
        y_training = y_scaled
        logger.info(f"Target variable statistics AFTER scaling: mean={y_training.mean():.4f}, std={y_training.std():.4f}, min={y_training.min():.4f}, max={y_training.max():.4f}")
    else:
        y_training = y
        logger.info("Response variable not scaled")

    # Set up preprocessing pipeline
    categorical_vars = categorical_variables if categorical_variables else []
    numerical_vars = [var for var in predictive_variables if var not in categorical_vars]

    logger.info(f"Numerical variables: {numerical_vars}")
    logger.info(f"Categorical variables: {categorical_vars}")

    # Create preprocessing pipeline
    preprocessors = []

    if numerical_vars:
        preprocessors.append(('num', StandardScaler(), numerical_vars))

    if categorical_vars:
        preprocessors.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_vars))

    if preprocessors:
        preprocessor = ColumnTransformer(preprocessors, remainder='passthrough')
    else:
        preprocessor = StandardScaler()

    # Set up XGBoost parameters
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'exact',
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 1
    }

    # Add best parameters if provided
    if best_params:
        for key, value in best_params.items():
            if key in ['max_depth', 'min_child_weight', 'gamma', 'reg_lambda',
                      'colsample_bytree', 'reg_alpha', 'learning_rate', 'n_estimators'] and value is not None:
                xgb_params[key] = value

    logger.info(f"XGBoost parameters: {xgb_params}")

    # Create XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(**xgb_params)

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb_regressor)
    ])

    # Train the final model
    logger.info("Training final model...")
    start_time = datetime.now()

    pipeline.fit(X, y_training)

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    logger.info(f"Model training completed in {training_time:.2f} seconds")

    # Make predictions on training data for evaluation
    y_pred_scaled = pipeline.predict(X)

    # Convert predictions back to original scale if response was scaled
    if scale_response and response_scaler is not None:
        y_pred = response_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = y  # Original unscaled values
    else:
        y_pred = y_pred_scaled
        y_actual = y

    # Calculate metrics on original scale
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    # Calculate additional metrics
    mean_actual = y_actual.mean()
    std_actual = y_actual.std()
    mae_percentage = (mae / mean_actual) * 100 if mean_actual != 0 else np.inf

    # Store evaluation metrics
    metrics = {
        'model_name': model_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'mae_percentage': mae_percentage,
        'mean_actual': mean_actual,
        'std_actual': std_actual,
        'training_time_seconds': training_time,
        'training_samples': len(X),
        'response_scaled': scale_response
    }

    logger.info("Final model evaluation metrics (on original scale):")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  MAE Percentage: {mae_percentage:.2f}%")
    logger.info(f"  Mean Actual: {mean_actual:.4f}")
    logger.info(f"  Std Actual: {std_actual:.4f}")

    return pipeline, metrics, response_scaler

def save_final_model(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    response_scaler: Optional[StandardScaler],
    model_name: str = "final_xgboost_regressor",
    response_variable: str = "target"
) -> str:
    """
    Save the final trained model and its metrics

    Args:
        pipeline: Trained sklearn pipeline
        metrics: Dictionary with evaluation metrics
        response_scaler: StandardScaler for response variable (if used)
        model_name: Name for the saved model
        response_variable: Name of the response variable for file naming

    Returns:
        Path to saved model file
    """
    logger = logging.getLogger('FinalRegressorTraining')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Models directory if it doesn't exist
    models_dir = project_root / "Models"
    models_dir.mkdir(exist_ok=True)

    # Create timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        # Save model using joblib (recommended for sklearn models)
        model_filename = f"{model_name}_{response_variable}_{timestamp}.joblib"
        model_path = models_dir / model_filename
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save response scaler if used
        if response_scaler is not None:
            scaler_filename = f"{model_name}_{response_variable}_{timestamp}_scaler.joblib"
            scaler_path = models_dir / scaler_filename
            joblib.dump(response_scaler, scaler_path)
            logger.info(f"Response scaler saved to: {scaler_path}")

        # Save model metrics
        metrics_filename = f"{model_name}_{response_variable}_{timestamp}_metrics.json"
        metrics_path = models_dir / metrics_filename

        import json
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_json[key] = value.item()
                else:
                    metrics_json[key] = value

            json.dump(metrics_json, f, indent=2)

        logger.info(f"Model metrics saved to: {metrics_path}")

        # Also save as pickle for backup
        pickle_filename = f"{model_name}_{response_variable}_{timestamp}.pkl"
        pickle_path = models_dir / pickle_filename

        model_data = {
            'pipeline': pipeline,
            'response_scaler': response_scaler,
            'metrics': metrics
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model pickle backup saved to: {pickle_path}")

        return str(model_path)

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def save_model_metrics_to_db(
    metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    response_variable: str,
    predictive_variables: List[str],
    target_schema_name: str = "models",
    target_table_name: str = "final_regressor_results",
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save final model metrics to DuckDB database

    Args:
        metrics: Dictionary with evaluation metrics
        best_params: Best parameters used for the model
        response_variable: Name of the response variable
        predictive_variables: List of predictive variables used
        target_schema_name: Schema name for the results table
        target_table_name: Table name for storing the results
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('FinalRegressorTraining')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving final model metrics to {target_schema_name}.{target_table_name}")

    # Prepare data for saving
    save_data = {
        'model_timestamp': datetime.now().isoformat(),
        'response_variable': response_variable,
        'predictive_variables': ','.join(predictive_variables),
        **metrics,
        **{f'param_{k}': v for k, v in best_params.items()}
    }

    # Convert to DataFrame
    results_df = pd.DataFrame([save_data])

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")
        logger.info(f"Created/verified schema: {target_schema_name}")

        # Create or append to table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_schema_name}.{target_table_name} AS
            SELECT * FROM results_df WHERE 1=0
        """)

        # Insert new data
        conn.execute(f"INSERT INTO {target_schema_name}.{target_table_name} SELECT * FROM results_df")

        logger.info(f"Successfully saved final model metrics to {target_schema_name}.{target_table_name}")

        # Verify the data was saved
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {target_schema_name}.{target_table_name}").fetchone()
        logger.info(f"Total records in table: {result[0]}")

    except Exception as e:
        logger.error(f"Error saving final model metrics: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting final XGBoost regressor training process")

    try:
        # Configuration - Update these variables as needed
        response_variable = "Revenue_MF"  # Revenue_MF, Revenue_CC, Revenue_CL
        predictive_variables_all = ["VolumeCred","VolumeCred_CA","TransactionsCred","TransactionsCred_CA","VolumeDeb","VolumeDeb_CA",
            "VolumeDebCash_Card","VolumeDebCashless_Card","VolumeDeb_PaymentOrder","TransactionsDeb","TransactionsDeb_CA",
            "TransactionsDebCash_Card","TransactionsDebCashless_Card","TransactionsDeb_PaymentOrder","Count_CA","Count_SA",
            "Count_MF","Count_OVD","Count_CC","Count_CL","ActBal_SA","ActBal_MF","ActBal_OVD","ActBal_CC",
            "ActBal_CL","Age","Tenure","Sex"
        ]
        predictive_variables = ["VolumeDebCash_Card","VolumeDeb","VolumeDebCashless_Card","VolumeDeb_PaymentOrder"]
        predictive_variables_cc = ["VolumeDebCashless_Card","Tenure","Age"]
        predictive_variables_cl = ['VolumeCred_CA','TransactionsDeb','Tenure','VolumeDeb_PaymentOrder','VolumeCred','VolumeDebCash_Card']

        categorical_variables = []  # Add categorical variables if any, e.g., ["Sex"]

        # Model configuration
        model_name = "final_xgboost_mf_revenue_model"  # final_xgboost_cl_revenue_model, final_xgboost_cc_revenue_model
        training_results_table = "xgboost_sales_mf_revenue_top"  # xgboost_sales_mf_revenue_top, xgboost_sales_cc_revenue_top, xgboost_sales_cl_revenue_top
        scale_response = False  # Whether to scale the response variable

        logger.info(f"Training final regressor for: {response_variable}")
        logger.info(f"Using predictive variables: {predictive_variables}")
        logger.info(f"Using categorical variables: {categorical_variables}")

        # Load training data
        data = load_data_from_duckdb(
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="train",
            col_non_zero=response_variable  # Filter for non-zero response values
        )

        # Get best parameters from previous training
        best_params = get_best_parameters(
            duckdb_name="assignment.duckdb",
            schema_name="models",
            table_name=training_results_table
        )

        # Train final model
        final_pipeline, metrics, response_scaler = train_final_xgboost_regressor(
            data=data,
            response_variable=response_variable,
            predictive_variables=predictive_variables,
            categorical_variables=categorical_variables,
            best_params=best_params,
            random_state=42,
            model_name=model_name,
            scale_response=scale_response
        )

        # Save the final model
        model_path = save_final_model(
            pipeline=final_pipeline,
            metrics=metrics,
            response_scaler=response_scaler,
            model_name=model_name,
            response_variable=response_variable
        )

        # Save metrics to database
        save_model_metrics_to_db(
            metrics=metrics,
            best_params=best_params,
            response_variable=response_variable,
            predictive_variables=predictive_variables,
            target_schema_name="models",
            target_table_name=model_name,
            duckdb_name="assignment.duckdb"
        )

        logger.info("✅ Final regressor training completed successfully")

        # Print summary
        print(f"\n🎯 Final Regressor Training Summary:")
        print(f"Response Variable: {response_variable}")
        print(f"Model Name: {model_name}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"MAE Percentage: {metrics['mae_percentage']:.2f}%")
        print(f"Training Time: {metrics['training_time_seconds']:.2f} seconds")
        print(f"Response Scaled: {metrics['response_scaled']}")
        print(f"Model saved to: Models/")
        print(f"Metrics saved to database: models.{model_name}")

    except Exception as e:
        logger.error(f"❌ Final regressor training failed: {e}")
        print(f"❌ Error: {e}")
        raise