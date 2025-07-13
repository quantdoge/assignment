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

# Dask and XGBoost imports
import dask
from dask.distributed import Client, as_completed, progress
from dask import delayed
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
from concurrent.futures import as_completed as futures_completed

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 05_Model_Training_Regressor.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"05_Model_Training_Regressor_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('ModelTrainingRegressor')
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

    Returns:
        pandas DataFrame with the loaded data
    """
    logger = logging.getLogger('ModelTrainingRegressor')

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
            df = conn.execute(query).fetchdf()
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

def setup_dask_client(n_workers: int = None, threads_per_worker: int = 2, memory_limit: str = '2GB') -> Client:
    """
    Set up Dask client for parallel processing

    Args:
        n_workers: Number of worker processes (default: number of CPU cores)
        threads_per_worker: Number of threads per worker
        memory_limit: Memory limit per worker

    Returns:
        Dask client instance
    """
    logger = logging.getLogger('ModelTrainingRegressor')

    try:
        # Create local cluster
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            silence_logs=False
        )

        logger.info(f"Dask client created successfully")
        logger.info(f"Dashboard link: {client.dashboard_link}")
        logger.info(f"Workers: {len(client.scheduler_info()['workers'])}")

        return client

    except Exception as e:
        logger.error(f"Failed to create Dask client: {e}")
        raise

@delayed
def train_single_kfold_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    k_folds: int,
    param_distributions: dict,
    n_iter: int,
    random_state: int,
    preprocessor,
    iteration_id: int
) -> dict:
    """
    Train XGBoost regressor with specific K-fold configuration using Dask delayed
    """
    logger = logging.getLogger('ModelTrainingRegressor')

    print(f"🔄 Starting iteration {iteration_id + 1} with {k_folds}-fold CV...")

    # Create XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        eval_metric='mae',
        tree_method='exact',  # Use 'exact' for better accuracy
        random_state=random_state,
        n_jobs=-1,  # Use 1 job per worker to avoid oversubscription
        verbosity=0
    )

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb_regressor)
    ])

    # Set up cross-validation
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=1,  # Use 1 job per worker
        random_state=random_state + iteration_id,  # Different seed for each iteration
        verbose=0
    )

    # Train model and measure time
    start_time = time.time()

    try:
        random_search.fit(X, y)
        end_time = time.time()
        runtime = end_time - start_time

        # Get best results
        best_score = random_search.best_score_  # This will be negative MAE
        best_params = random_search.best_params_

        # Extract XGBoost parameters from best_params
        xgb_params = {key.replace('xgb__', ''): value for key, value in best_params.items() if key.startswith('xgb__')}

        # Get feature names and importance from the best model
        best_model = random_search.best_estimator_

        # Transform features to get feature names after preprocessing
        X_transformed = best_model.named_steps['preprocessor'].fit_transform(X)

        # Get feature names after preprocessing
        feature_names = []
        if hasattr(best_model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Get feature importance from the trained XGBoost model
        xgb_model = best_model.named_steps['xgb']
        feature_importance = xgb_model.feature_importances_

        # Calculate explained variance (approximate using feature importance as proxy)
        # Note: XGBoost doesn't directly provide explained variance per feature
        # We'll use normalized feature importance as an approximation
        total_importance = np.sum(feature_importance)
        print('DEBUG >>>')
        print(f"Feature importance: {feature_importance}")
        print('<<<')

        explained_variance = feature_importance / total_importance if total_importance > 0 else feature_importance

        # Create feature importance ranking
        importance_ranks = np.argsort(-feature_importance) + 1  # Rank 1 = highest importance

        # Create feature importance data
        feature_data = []
        for i, feature_name in enumerate(feature_names):
            feature_data.append({
                'iteration_id': iteration_id,
                'feature_name': str(feature_name),
                'feature_importance': float(feature_importance[i]),
                'explained_variance': float(explained_variance[i]),
                'feature_importance_rank': int(importance_ranks[i])
            })

        # Store result
        result = {
            'iteration_id': iteration_id,
            'k_folds': k_folds,
            'max_depth': xgb_params.get('max_depth', None),
            'min_child_weight': xgb_params.get('min_child_weight', None),
            'gamma': xgb_params.get('gamma', None),
            'reg_lambda': xgb_params.get('reg_lambda', None),
            'colsample_bytree': xgb_params.get('colsample_bytree', None),
            'reg_alpha': xgb_params.get('reg_alpha', None),
            'learning_rate': xgb_params.get('learning_rate', None),
            'n_estimators': xgb_params.get('n_estimators', None),
            'runtime_seconds': runtime,
            'neg_mae_score': best_score,  # Negative MAE score
            'mae_score': -best_score,     # Positive MAE score
            'cv_std': random_search.cv_results_['std_test_score'][random_search.best_index_],
            'feature_data': feature_data  # Add feature importance data
        }

        print(f"✅ Completed iteration {iteration_id + 1}: MAE={-best_score:.4f}, Runtime={runtime:.1f}s")

        return result

    except Exception as e:
        print(f"❌ Failed iteration {iteration_id + 1}: {e}")
        return None

def train_xgboost_regressor_dask(
    data: pd.DataFrame,
    response_variable: str,
    predictive_variables: List[str],
    categorical_variables: Optional[List[str]] = None,
    n_iter: int = 50,
    random_state: int = 42,
    n_top_results: int = 10,
    dask_client: Client = None,
    iterations_per_kfold: int = 5  # Number of iterations per k-fold value
) -> pd.DataFrame:
    """
    Train XGBoost regressor with Dask parallel processing
    """
    logger = logging.getLogger('ModelTrainingRegressor')

    logger.info("Starting XGBoost regressor training with Dask")
    logger.info(f"Response variable: {response_variable}")
    logger.info(f"Predictive variables: {predictive_variables}")
    logger.info(f"Categorical variables: {categorical_variables}")

    # Validate inputs
    if response_variable not in data.columns:
        raise ValueError(f"Response variable '{response_variable}' not found in data")

    missing_predictors = [var for var in predictive_variables if var not in data.columns]
    if missing_predictors:
        raise ValueError(f"Predictive variables not found in data: {missing_predictors}")

    # Prepare data
    logger.info("Preparing data...")
    data_clean = data[predictive_variables + [response_variable]].copy()

    # Fill missing values
    for col in predictive_variables:
        if data_clean[col].dtype in ['object', 'category']:
            data_clean[col] = data_clean[col].fillna('Unknown')
        else:
            data_clean[col] = data_clean[col].fillna(0)

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

    # Apply StandardScaler to the response variable
    logger.info("Applying StandardScaler to response variable...")
    response_scaler = StandardScaler()
    y_scaled = pd.Series(
        response_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(),
        index=y.index,
        name=response_variable
    )

    logger.info(f"Target variable statistics AFTER scaling: mean={y_scaled.mean():.4f}, std={y_scaled.std():.4f}, min={y_scaled.min():.4f}, max={y_scaled.max():.4f}")
    logger.info(f"Response variable scaling completed successfully")

    # Set up preprocessing pipeline for features
    categorical_vars = categorical_variables if categorical_variables else []
    numerical_vars = [var for var in predictive_variables if var not in categorical_vars]

    preprocessors = []
    if numerical_vars:
        preprocessors.append(('num', StandardScaler(), numerical_vars))
    if categorical_vars:
        preprocessors.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_vars))

    preprocessor = ColumnTransformer(preprocessors, remainder='passthrough') if preprocessors else StandardScaler()

    # Define parameter distributions
    param_distributions = {
        'xgb__max_depth': list(range(3, 15)),  # Increased from 11
        'xgb__min_child_weight': list(range(1, 11)),  # Reduced from 26
        'xgb__gamma': np.linspace(0, 2, 21),  # Reduced from 10
        'xgb__reg_lambda': np.linspace(0.1, 2, 20),  # Changed to float range
        'xgb__colsample_bytree': np.linspace(0.6, 1.0, 5),  # Increased range
        'xgb__reg_alpha': np.linspace(0, 1, 11),  # Changed to float range
        'xgb__learning_rate': np.arange(0.01, 0.3, 0.02),  # Lowered minimum
        'xgb__n_estimators': [50, 100, 200, 300, 500, 1000]  # Added smaller values
    }

    k_fold_options = list(range(3, 6))  # 3 to 5

    logger.info("Setting up parallel training with Dask...")

    # Create delayed tasks
    delayed_tasks = []
    iteration_id = 0

    for k_folds in k_fold_options:
        for i in range(iterations_per_kfold):
            task = train_single_kfold_xgboost(
                X=X,
                y=y,  # Use scaled response variable
                k_folds=k_folds,
                param_distributions=param_distributions,
                n_iter=n_iter,
                random_state=random_state,
                preprocessor=preprocessor,
                iteration_id=iteration_id
            )
            delayed_tasks.append(task)
            iteration_id += 1

    total_iterations = len(delayed_tasks)
    logger.info(f"Created {total_iterations} parallel training tasks")
    print(f"🚀 Starting {total_iterations} parallel training iterations with scaled response variable...")

    # Execute tasks in parallel with progress tracking
    start_time = time.time()

    if dask_client:
        # Submit tasks to Dask cluster
        futures = dask_client.compute(delayed_tasks)

        # Track progress
        completed = 0
        results = []

        print(f"📊 Progress: 0/{total_iterations} iterations completed")

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                completed += 1

                # Update progress
                progress_pct = (completed / total_iterations) * 100
                print(f"📊 Progress: {completed}/{total_iterations} iterations completed ({progress_pct:.1f}%)")

            except Exception as e:
                logger.error(f"Task failed: {e}")
                completed += 1
    else:
        # Fallback to sequential execution
        results = []
        for i, task in enumerate(delayed_tasks):
            result = task.compute()
            if result is not None:
                results.append(result)
            progress_pct = ((i + 1) / total_iterations) * 100
            print(f"📊 Progress: {i + 1}/{total_iterations} iterations completed ({progress_pct:.1f}%)")

    end_time = time.time()
    total_runtime = end_time - start_time

    print(f"🎉 All iterations completed! Total runtime: {total_runtime:.1f} seconds")
    logger.info(f"Parallel training completed in {total_runtime:.2f} seconds")

    # Convert results to DataFrame
    results_df = pd.DataFrame([r for r in results if r is not None])

    if not results_df.empty:
        results_df = results_df.sort_values('mae_score', ascending=True).reset_index(drop=True)  # Sort by MAE ascending (lower is better)
        logger.info(f"Training completed. Generated {len(results_df)} successful results")
        logger.info(f"Best MAE score (on scaled target): {results_df.iloc[0]['mae_score']:.4f}")

        # Log top 5 results
        logger.info(f"Top 5 results (MAE on scaled target):")
        for idx, row in results_df.head().iterrows():
            logger.info(f"  Rank {idx+1}: MAE={row['mae_score']:.4f}, K-fold={row['k_folds']}, "
                       f"max_depth={row['max_depth']}, learning_rate={row['learning_rate']:.3f}")
    else:
        logger.warning("No successful training results generated")

    return results_df

def train_xgboost_regressor(
    data: pd.DataFrame,
    response_variable: str,
    predictive_variables: List[str],
    categorical_variables: Optional[List[str]] = None,
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1,
    n_top_results: int = 10,
) -> pd.DataFrame:
    """
    Train XGBoost regressor with hyperparameter optimization using RandomizedSearchCV

    Args:
        data: Input DataFrame
        response_variable: Name of the target variable column
        predictive_variables: List of predictor variable names
        categorical_variables: List of categorical variables for one-hot encoding
        n_iter: Number of parameter combinations to try in RandomizedSearchCV
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        n_top_results: Number of top results to store per k-fold

    Returns:
        DataFrame with different parameter combinations, runtime, and MAE scores
    """
    logger = logging.getLogger('ModelTrainingRegressor')

    logger.info("Starting XGBoost regressor training")
    logger.info(f"Response variable: {response_variable}")
    logger.info(f"Predictive variables: {predictive_variables}")
    logger.info(f"Categorical variables: {categorical_variables}")

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

    logger.info(f"Target variable statistics: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

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
        # One-hot encode categorical variables
        preprocessors.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_vars))

    if preprocessors:
        preprocessor = ColumnTransformer(preprocessors, remainder='passthrough')
    else:
        preprocessor = StandardScaler()

    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        'xgb__max_depth': list(range(3, 15)),  # Increased from 11
        'xgb__min_child_weight': list(range(1, 6)),  # Reduced from 26
        'xgb__gamma': np.linspace(0, 2, 21),  # Reduced from 10
        'xgb__reg_lambda': np.linspace(0.1, 2, 20),  # Changed to float range
        'xgb__colsample_bytree': np.linspace(0.7, 1.0, 6),  # Increased range
        'xgb__reg_alpha': np.linspace(0, 1, 11),  # Changed to float range
        'xgb__learning_rate': np.arange(0.01, 0.3, 0.02),  # Lowered minimum
        'xgb__n_estimators': [100, 200, 300, 500, 1000, 2000]  # Added smaller values
    }

    # K-Fold cross-validation options
    k_fold_options = list(range(3, 6))  # 3 to 5

    logger.info("Setting up hyperparameter optimization...")
    logger.info(f"Parameter search space size: {len(param_distributions)}")
    logger.info(f"Number of iterations: {n_iter}")

    # Store results
    results = []

    # Try different K-fold values
    for k_folds in k_fold_options:
        logger.info(f"Training with {k_folds}-fold cross-validation")

        # Create XGBoost regressor
        xgb_regressor = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            eval_metric='mae',
            tree_method='exact',  # Use 'exact' for better accuracy
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0
        )

        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('xgb', xgb_regressor)
        ])

        # Set up cross-validation
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0
        )

        # Train model and measure time
        start_time = time.time()

        try:
            logger.info(f"Starting RandomizedSearchCV with {k_folds}-fold CV...")
            random_search.fit(X, y)

            end_time = time.time()
            runtime = end_time - start_time

            # Get best results
            best_score = random_search.best_score_  # This will be negative MAE
            best_params = random_search.best_params_

            logger.info(f"Best MAE for {k_folds}-fold: {-best_score:.4f}")
            logger.info(f"Runtime: {runtime:.2f} seconds")
            logger.info(f"Best parameters: {best_params}")

            # Extract XGBoost parameters from best_params
            xgb_params = {key.replace('xgb__', ''): value for key, value in best_params.items() if key.startswith('xgb__')}

            # Store result
            result = {
                'k_folds': k_folds,
                'max_depth': xgb_params.get('max_depth', None),
                'min_child_weight': xgb_params.get('min_child_weight', None),
                'gamma': xgb_params.get('gamma', None),
                'reg_lambda': xgb_params.get('reg_lambda', None),
                'colsample_bytree': xgb_params.get('colsample_bytree', None),
                'reg_alpha': xgb_params.get('reg_alpha', None),
                'learning_rate': xgb_params.get('learning_rate', None),
                'n_estimators': xgb_params.get('n_estimators', None),
                'runtime_seconds': runtime,
                'neg_mae_score': best_score,
                'mae_score': -best_score,  # Convert to positive MAE
                'cv_std': random_search.cv_results_['std_test_score'][random_search.best_index_]
            }

            results.append(result)

            # Also store top N results for this K-fold
            cv_results = pd.DataFrame(random_search.cv_results_)
            top_results = cv_results.nlargest(n_top_results, 'mean_test_score')  # Top N results (least negative = best MAE)

            for idx, row in top_results.iterrows():
                params = row['params']
                xgb_params_detail = {key.replace('xgb__', ''): value for key, value in params.items() if key.startswith('xgb__')}

                detailed_result = {
                    'k_folds': k_folds,
                    'max_depth': xgb_params_detail.get('max_depth', None),
                    'min_child_weight': xgb_params_detail.get('min_child_weight', None),
                    'gamma': xgb_params_detail.get('gamma', None),
                    'reg_lambda': xgb_params_detail.get('reg_lambda', None),
                    'colsample_bytree': xgb_params_detail.get('colsample_bytree', None),
                    'reg_alpha': xgb_params_detail.get('reg_alpha', None),
                    'learning_rate': xgb_params_detail.get('learning_rate', None),
                    'n_estimators': xgb_params_detail.get('n_estimators', None),
                    'runtime_seconds': runtime,  # Same runtime for all combinations in this K-fold
                    'neg_mae_score': row['mean_test_score'],
                    'mae_score': -row['mean_test_score'],  # Convert to positive MAE
                    'cv_std': row['std_test_score'],
                    'rank': row['rank_test_score']
                }

                if detailed_result not in results:  # Avoid duplicates
                    results.append(detailed_result)

        except Exception as e:
            logger.error(f"Error training with {k_folds}-fold CV: {e}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Sort by MAE score ascending (lower is better)
        results_df = results_df.sort_values('mae_score', ascending=True).reset_index(drop=True)

        logger.info(f"Training completed. Generated {len(results_df)} result combinations")
        logger.info(f"Best MAE score: {results_df.iloc[0]['mae_score']:.4f}")

        # Log top 5 results
        logger.info(f"Top {n_top_results} results:")
        for idx, row in results_df.head().iterrows():
            logger.info(f"  Rank {idx+1}: MAE={row['mae_score']:.4f}, K-fold={row['k_folds']}, "
                       f"max_depth={row['max_depth']}, learning_rate={row['learning_rate']:.3f}")

    else:
        logger.warning("No successful training results generated")

    return results_df

def save_model_results(
    results_df: pd.DataFrame,
    target_schema_name: str = "models",
    target_table_name: str = "xgboost_regression_results",
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save model training results to a specified schema and table in DuckDB

    Args:
        results_df: DataFrame with model training results
        target_schema_name: Schema name where the results table will be created
        target_table_name: Table name for storing the results
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('ModelTrainingRegressor')

    if results_df.empty:
        logger.warning("No results to save")
        return

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving model results to {target_schema_name}.{target_table_name}")
    logger.info(f"Database path: {db_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")
        logger.info(f"Created/verified schema: {target_schema_name}")

        # Drop the table if it exists and create new one
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{target_table_name}")
        logger.info(f"Dropped existing table if present: {target_schema_name}.{target_table_name}")

        # Save DataFrame to DuckDB table
        conn.execute(f"CREATE TABLE {target_schema_name}.{target_table_name} AS SELECT * FROM results_df")
        logger.info(f"Successfully created table: {target_schema_name}.{target_table_name}")

        # Verify the data was saved correctly
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {target_schema_name}.{target_table_name}").fetchone()
        logger.info(f"Verified: {result[0]} rows saved to {target_schema_name}.{target_table_name}")

        # Show sample of saved data
        logger.info("Sample of saved model results:")
        sample_data = conn.execute(f"SELECT * FROM {target_schema_name}.{target_table_name} LIMIT 3").fetchdf()
        logger.info(f"\n{sample_data}")

    except Exception as e:
        logger.error(f"Error saving model results: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

def save_feature_importance_results(
    feature_data_list: List[Dict],
    target_schema_name: str = "models",
    target_table_name: str = "xgbreg_features",
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save feature importance results to a specified schema and table in DuckDB

    Args:
        feature_data_list: List of dictionaries containing feature importance data
        target_schema_name: Schema name where the results table will be created
        target_table_name: Table name for storing the results
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('ModelTrainingRegressor')

    if not feature_data_list:
        logger.warning("No feature importance data to save")
        return

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving feature importance results to {target_schema_name}.{target_table_name}")
    logger.info(f"Database path: {db_path}")

    # Convert to DataFrame
    features_df = pd.DataFrame(feature_data_list)

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")
        logger.info(f"Created/verified schema: {target_schema_name}")

        # Drop the table if it exists and create new one
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{target_table_name}")
        logger.info(f"Dropped existing table if present: {target_schema_name}.{target_table_name}")

        # Save DataFrame to DuckDB table
        conn.execute(f"CREATE TABLE {target_schema_name}.{target_table_name} AS SELECT * FROM features_df")
        logger.info(f"Successfully created table: {target_schema_name}.{target_table_name}")

        # Verify the data was saved correctly
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {target_schema_name}.{target_table_name}").fetchone()
        logger.info(f"Verified: {result[0]} rows saved to {target_schema_name}.{target_table_name}")

        # Show sample of saved data
        logger.info("Sample of saved feature importance results:")
        sample_data = conn.execute(f"SELECT * FROM {target_schema_name}.{target_table_name} LIMIT 5").fetchdf()
        logger.info(f"\n{sample_data}")

    except Exception as e:
        logger.error(f"Error saving feature importance results: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting XGBoost regressor training process with Dask")

    # Set up Dask client
    dask_client = None

    try:
        dask_client = setup_dask_client(
            n_workers=8,  # Adjust based on your CPU cores
            threads_per_worker=2,
            memory_limit='2GB'
        )

        print(f"Dask dashboard available at: {dask_client.dashboard_link}")

        # Load data from DuckDB
        data = load_data_from_duckdb(
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="train",
            col_non_zero="Revenue_CL" #Revenue_MF, Revenue_CC
        )
        print('DEBUG >>>')
        print(data.head(3))
        print('<<<')

        # Configuration for regression task
        response_variable = "Revenue_CL"  # Example continuous variable for regression
        predictive_variables_all = ["VolumeCred","VolumeCred_CA","TransactionsCred","TransactionsCred_CA","VolumeDeb","VolumeDeb_CA",
            "VolumeDebCash_Card","VolumeDebCashless_Card","VolumeDeb_PaymentOrder","TransactionsDeb","TransactionsDeb_CA",
            "TransactionsDebCash_Card","TransactionsDebCashless_Card","TransactionsDeb_PaymentOrder","Count_CA","Count_SA",
            "Count_MF","Count_OVD","Count_CC","Count_CL","ActBal_SA","ActBal_MF","ActBal_OVD","ActBal_CC",
            "ActBal_CL","Age","Tenure","Sex"
        ]
        predictive_variables_mf = ["VolumeDebCash_Card","VolumeDeb","VolumeDebCashless_Card","VolumeDeb_PaymentOrder"]
        predictive_variables_cc = ["VolumeDebCashless_Card","Tenure","Age"]
        predictive_variables= ['VolumeCred_CA','TransactionsDeb','Tenure','VolumeDeb_PaymentOrder','VolumeCred','VolumeDebCash_Card']

        categorical_variables_all = ["Sex"]
        categorical_variables = []

        logger.info("Starting regressor training with Dask...")

        # Train models with Dask
        results_df = train_xgboost_regressor_dask(
            data=data,
            response_variable=response_variable,
            predictive_variables=predictive_variables,
            categorical_variables=categorical_variables,
            n_iter=200,  # Reduced per iteration, but more parallel iterations
            random_state=42,
            dask_client=dask_client,
            iterations_per_kfold=10  # More iterations per k-fold for better exploration
        )

        logger.info(f"✅ Regressor training completed successfully. Generated {len(results_df)} result combinations")

        # Save results to database
        if not results_df.empty:
            save_model_results(
                results_df=results_df,
                target_schema_name="models",
                target_table_name="xgboost_sales_cl_revenue_top", # xgboost_sales_mf_revenue, xgboost_sales_mf_revenue_top, xgboost_sales_cc_revenue, xgboost_sales_cc_revenue_top, xgboost_sales_cl_revenue
                duckdb_name="assignment.duckdb"
            )
            logger.info("✅ Model results saved to database successfully")

            # Extract and save feature importance data
            all_feature_data = []
            for _, row in results_df.iterrows():
                if 'feature_data' in row and row['feature_data']:
                    all_feature_data.extend(row['feature_data'])

            if all_feature_data:
                save_feature_importance_results(
                    feature_data_list=all_feature_data,
                    target_schema_name="models",
                    target_table_name="xgbreg_sales_cl_revenue_top", #xgbreg_sales_mf_revenue, xgbreg_sales_mf_revenue_top, xgbreg_sales_cc_revenue, xgbreg_sales_cc_revenue_top, xgbreg_sales_cl_revenue
                    duckdb_name="assignment.duckdb"
                )
                logger.info("✅ Feature importance results saved to database successfully")
                print(f"💾 Saved {len(all_feature_data)} feature importance records to models.xgbreg_features")

            # Print summary
            print(f"\n🎯 Regressor Training Summary:")
            print(f"Total parameter combinations tested: {len(results_df)}")
            print(f"Best MAE score: {results_df.iloc[0]['mae_score']:.4f}")
            print(f"Best parameters:")
            best_row = results_df.iloc[0]
            for col in ['k_folds', 'max_depth', 'min_child_weight', 'gamma', 'reg_lambda',
                       'colsample_bytree', 'reg_alpha', 'learning_rate', 'n_estimators']:
                print(f"  {col}: {best_row[col]}")

            print(f"\nResults saved to:")
            print(f"  - Model results")
            print(f"  - Feature importance: models.xgbreg_features")

    except Exception as e:
        logger.error(f"❌ Regressor training failed: {e}")

    finally:
        if dask_client:
            dask_client.close()
            logger.info("Dask client closed")