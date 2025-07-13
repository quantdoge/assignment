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
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import joblib

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 06_Model_Final.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"06_Model_Final_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('FinalModelTraining')
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
    table_name: str = "train"
) -> pd.DataFrame:
    """
    Load data from DuckDB table into pandas DataFrame

    Args:
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the table
        table_name: Table name to load

    Returns:
        pandas DataFrame with the loaded data
    """
    logger = logging.getLogger('FinalModelTraining')

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
    table_name: str = "xgboost_cl_sales_top"
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
    logger = logging.getLogger('FinalModelTraining')

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

        # Get the best result (highest AUC score)
        query = f"""
            SELECT *
            FROM {schema_name}.{table_name}
            ORDER BY auc_score DESC
            LIMIT 1
        """
        best_result = conn.execute(query).fetchdf()

        if best_result.empty:
            logger.error("No results found in the training results table")
            raise ValueError("No results found in the training results table")

        # Extract parameters
        best_params = {}
        param_columns = ['max_depth', 'min_child_weight', 'gamma', 'reg_lambda',
                        'colsample_bytree', 'reg_alpha', 'learning_rate', 'n_estimators',
                        'scale_pos_weight']

        for col in param_columns:
            if col in best_result.columns:
                best_params[col] = best_result[col].iloc[0]

        logger.info(f"Best AUC score: {best_result['auc_score'].iloc[0]:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params

    except Exception as e:
        logger.error(f"Error loading best parameters: {e}")
        raise
    finally:
        conn.close()

def train_final_xgboost_classifier(
    data: pd.DataFrame,
    response_variable: str,
    predictive_variables: List[str],
    categorical_variables: Optional[List[str]] = None,
    best_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    model_name: str = "final_xgboost_model"
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train final XGBoost binary classifier using best parameters

    Args:
        data: Input DataFrame
        response_variable: Name of the target variable column
        predictive_variables: List of predictor variable names
        categorical_variables: List of categorical variables for one-hot encoding
        best_params: Dictionary with best parameters from hyperparameter tuning
        random_state: Random seed for reproducibility
        model_name: Name for the final model

    Returns:
        Tuple of (trained pipeline, evaluation metrics)
    """
    logger = logging.getLogger('FinalModelTraining')

    logger.info("Starting final XGBoost binary classifier training")
    logger.info(f"Response variable: {response_variable}")
    logger.info(f"Predictive variables: {predictive_variables}")
    logger.info(f"Categorical variables: {categorical_variables}")
    logger.info(f"Model name: {model_name}")

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

    # Ensure binary target
    unique_values = y.unique()
    if len(unique_values) != 2:
        logger.warning(f"Target variable has {len(unique_values)} unique values: {unique_values}")
        logger.info("Converting to binary by taking top 2 most frequent values")
        top_2_values = y.value_counts().head(2).index.tolist()
        y = y[y.isin(top_2_values)]
        X = X[y.index]

    # Convert to binary 0/1 if needed
    y_binary = pd.get_dummies(y, drop_first=True).iloc[:, 0] if y.dtype == 'object' else y

    logger.info(f"Target distribution: {y_binary.value_counts().to_dict()}")

    # Calculate scale_pos_weight for class imbalance if not provided
    if best_params is None or 'scale_pos_weight' not in best_params:
        pos_count = y_binary.sum()
        neg_count = len(y_binary) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    else:
        scale_pos_weight = best_params['scale_pos_weight']

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
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 1
    }

    # Add best parameters if provided
    if best_params:
        for key, value in best_params.items():
            if key in ['max_depth', 'min_child_weight', 'gamma', 'reg_lambda',
                      'colsample_bytree', 'reg_alpha', 'learning_rate', 'n_estimators',
                      'scale_pos_weight'] and value is not None:
                xgb_params[key] = value

    logger.info(f"XGBoost parameters: {xgb_params}")

    # Create XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(**xgb_params)

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb_classifier)
    ])

    # Train the final model
    logger.info("Training final model...")
    start_time = datetime.now()

    pipeline.fit(X, y_binary)

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    logger.info(f"Model training completed in {training_time:.2f} seconds")

    # Make predictions on training data for evaluation
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)

    # Calculate metrics
    auc_score = roc_auc_score(y_binary, y_pred_proba)

    # Get classification report
    class_report = classification_report(y_binary, y_pred, output_dict=True)

    # Get confusion matrix
    conf_matrix = confusion_matrix(y_binary, y_pred)

    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Store evaluation metrics
    metrics = {
        'model_name': model_name,
        'auc_score': auc_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'training_time_seconds': training_time,
        'training_samples': len(X),
        'positive_class_count': int(y_binary.sum()),
        'negative_class_count': int(len(y_binary) - y_binary.sum())
    }

    logger.info("Final model evaluation metrics:")
    logger.info(f"  AUC Score: {auc_score:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1_score:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return pipeline, metrics

def save_final_model(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    model_name: str = "final_xgboost_model",
    response_variable: str = "target"
) -> None:
    """
    Save the final trained model and its metrics

    Args:
        pipeline: Trained sklearn pipeline
        metrics: Dictionary with evaluation metrics
        model_name: Name for the saved model
        response_variable: Name of the response variable for file naming
    """
    logger = logging.getLogger('FinalModelTraining')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Models directory if it doesn't exist
    models_dir = project_root / "Models"
    models_dir.mkdir(exist_ok=True)

    # Create timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model using joblib (recommended for sklearn models)
    model_filename = f"{model_name}_{response_variable}_{timestamp}.joblib"
    model_path = models_dir / model_filename

    try:
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to: {model_path}")

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

        with open(pickle_path, 'wb') as f:
            pickle.dump(pipeline, f)

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
    target_table_name: str = "final_model_results",
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
    logger = logging.getLogger('FinalModelTraining')

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
    logger.info("Starting final XGBoost model training process")

    try:
        # Configuration - Update these variables as needed
        response_variable = "Sale_CL" #Sale_MF, Sale_CC
        predictive_variables_mf = ["TransactionsDebCashless_Card","Count_MF", "Age", "VolumeCred_CA"]
        predictive_variables_cc = ["VolumeDeb_PaymentOrder","ActBal_CA","TransactionsDebCashless_Card",
                                   "VolumeCred_CA","VolumeDebCash_Card","Tenure","ActBal_SA",
                                   "Age","VolumeCred","TransactionsDeb"]
        predictive_variables= ["Tenure","Age","VolumeCred_CA","VolumeDeb_CA","ActBal_CA"]

        categorical_variables = []  # Add categorical variables if any, e.g., ["Sex"]

        # Model configuration
        model_name = "final_xgboost_cl_model"  # final_xgboost_mf_model, final_xgboost_cc_model
        training_results_table = "xgboost_cl_sales_top"  # xgboost_mf_sales_top, xgboost_cc_sales_top

        logger.info(f"Training final model for: {response_variable}")
        logger.info(f"Using predictive variables: {predictive_variables}")
        logger.info(f"Using categorical variables: {categorical_variables}")

        # Load training data
        data = load_data_from_duckdb(
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="train"
        )

        # Get best parameters from previous training
        best_params = get_best_parameters(
            duckdb_name="assignment.duckdb",
            schema_name="models",
            table_name=training_results_table
        )

        # Train final model
        final_pipeline, metrics = train_final_xgboost_classifier(
            data=data,
            response_variable=response_variable,
            predictive_variables=predictive_variables,
            categorical_variables=categorical_variables,
            best_params=best_params,
            random_state=42,
            model_name=model_name
        )

        # Save the final model
        model_path = save_final_model(
            pipeline=final_pipeline,
            metrics=metrics,
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

        logger.info("✅ Final model training completed successfully")

        # Print summary
        print(f"\n🎯 Final Model Training Summary:")
        print(f"Response Variable: {response_variable}")
        print(f"Model Name: {model_name}")
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Training Time: {metrics['training_time_seconds']:.2f} seconds")
        print(f"Model saved to: Models/")
        print(f"Metrics saved to database: models.{model_name}")

    except Exception as e:
        logger.error(f"❌ Final model training failed: {e}")
        print(f"❌ Error: {e}")
        raise