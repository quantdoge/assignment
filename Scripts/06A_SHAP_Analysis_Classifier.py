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

# XGBoost and SHAP imports
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration to write to 06_SHAP_Analysis.log
    """
    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create Logs directory if it doesn't exist
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up logging configuration
    dt_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"06_SHAP_Analysis_{dt_mark}.log"

    # Create logger
    logger = logging.getLogger('SHAPAnalysis')
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
    logger = logging.getLogger('SHAPAnalysis')

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

def load_best_model_parameters(
    duckdb_name: str = "assignment.duckdb",
    schema_name: str = "models",
    table_name: str = "xgboost_mf_sales"
) -> Dict[str, Any]:
    """
    Load the best model parameters from the training results table

    Args:
        duckdb_name: Name of the DuckDB database file
        schema_name: Schema name containing the results table
        table_name: Table name with training results

    Returns:
        Dictionary with the best model parameters
    """
    logger = logging.getLogger('SHAPAnalysis')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Loading best model parameters from {schema_name}.{table_name}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Get the best model parameters (highest AUC score)
        query = f"""
        SELECT * FROM {schema_name}.{table_name}
        ORDER BY auc_score DESC
        LIMIT 1
        """

        best_params_df = conn.execute(query).fetchdf()

        if best_params_df.empty:
            logger.error(f"No model parameters found in {schema_name}.{table_name}")
            raise ValueError(f"No model parameters found in {schema_name}.{table_name}")

        best_params = best_params_df.iloc[0].to_dict()
        logger.info(f"Best model AUC score: {best_params['auc_score']:.4f}")
        logger.info(f"Best model parameters loaded successfully")

        return best_params

    except Exception as e:
        logger.error(f"Error loading best model parameters: {e}")
        raise
    finally:
        conn.close()

def build_xgboost_model_with_parameters(
    data: pd.DataFrame,
    response_variable: str,
    predictive_variables: List[str],
    categorical_variables: Optional[List[str]] = None,
    max_depth: int = 6,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    scale_pos_weight: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Build and train XGBoost model with specified parameters

    Args:
        data: Input DataFrame
        response_variable: Name of the target variable column
        predictive_variables: List of predictor variable names
        categorical_variables: List of categorical variables for one-hot encoding
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of instance weight needed in a child
        gamma: Minimum loss reduction required to make a split
        reg_lambda: L2 regularization term on weights
        colsample_bytree: Subsample ratio of columns when constructing each tree
        reg_alpha: L1 regularization term on weights
        learning_rate: Boosting learning rate
        n_estimators: Number of boosting rounds
        scale_pos_weight: Balancing of positive and negative weights
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (trained_model, X_train, X_test, y_train, y_test, preprocessor)
    """
    logger = logging.getLogger('SHAPAnalysis')

    logger.info("Building XGBoost model with specified parameters")
    logger.info(f"Parameters: max_depth={max_depth}, min_child_weight={min_child_weight}, "
                f"gamma={gamma}, reg_lambda={reg_lambda}, colsample_bytree={colsample_bytree}, "
                f"reg_alpha={reg_alpha}, learning_rate={learning_rate}, n_estimators={n_estimators}")

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
        top_2_values = y.value_counts().head(2).index.tolist()
        y = y[y.isin(top_2_values)]
        X = X[y.index]

    # Convert to binary 0/1 if needed
    y_binary = pd.get_dummies(y, drop_first=True).iloc[:, 0] if y.dtype == 'object' else y
    logger.info(f"Target distribution: {y_binary.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")

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

    # Create XGBoost classifier with specified parameters
    xgb_classifier = xgb.XGBClassifier(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_lambda=reg_lambda,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )

    # Create and train pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb_classifier)
    ])

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate model
    from sklearn.metrics import roc_auc_score, classification_report
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Model training completed")
    logger.info(f"Test AUC score: {auc_score:.4f}")

    return pipeline.named_steps['xgb'], X_train, X_test, y_train, y_test, preprocessor

def compute_shap_values(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor: ColumnTransformer,
    predictive_variables: List[str],
    sample_size: int = 1000,
    background_size: int = 100
) -> Tuple[shap.Explainer, np.ndarray, np.ndarray, List[str]]:
    """
    Compute SHAP values for the XGBoost model

    Args:
        model: Trained XGBoost classifier
        X_train: Training features
        X_test: Test features
        preprocessor: Fitted preprocessor
        predictive_variables: List of original feature names
        sample_size: Number of test samples to compute SHAP values for
        background_size: Number of background samples for SHAP explainer

    Returns:
        Tuple of (explainer, shap_values, transformed_data, feature_names)
    """
    logger = logging.getLogger('SHAPAnalysis')

    logger.info("Computing SHAP values...")
    logger.info(f"Sample size for SHAP: {sample_size}")
    logger.info(f"Background size for SHAP: {background_size}")

    # Transform the data using the fitted preprocessor
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    feature_names = []

    # Get feature names from the preprocessor
    if hasattr(preprocessor, 'transformers_'):
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_features = transformer.get_feature_names_out(features)
                    feature_names.extend(cat_features)
                else:
                    # Fallback for older sklearn versions
                    feature_names.extend([f"{feat}_{i}" for feat in features for i in range(len(transformer.categories_[0]))])
    else:
        feature_names = predictive_variables

    logger.info(f"Feature names after preprocessing: {len(feature_names)} features")

    # Sample data for SHAP computation
    if len(X_test_transformed) > sample_size:
        sample_indices = np.random.choice(len(X_test_transformed), sample_size, replace=False)
        X_sample = X_test_transformed[sample_indices]
    else:
        X_sample = X_test_transformed

    if len(X_train_transformed) > background_size:
        background_indices = np.random.choice(len(X_train_transformed), background_size, replace=False)
        X_background = X_train_transformed[background_indices]
    else:
        X_background = X_train_transformed

    logger.info(f"Using {len(X_sample)} samples for SHAP computation")
    logger.info(f"Using {len(X_background)} background samples")

    # Create SHAP explainer
    logger.info("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, X_background)

    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)

    logger.info(f"SHAP values computed successfully")
    logger.info(f"SHAP values shape: {shap_values.shape}")

    return explainer, shap_values, X_sample, feature_names

def create_shap_visualizations(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[str] = None,
    file_prefix: str = "shap"
) -> None:
    """
    Create and save SHAP visualizations

    Args:
        shap_values: Computed SHAP values
        X_sample: Sample data used for SHAP computation
        feature_names: List of feature names
        output_dir: Directory to save plots (if None, uses default)
        file_prefix: Prefix for output file names (default: "shap")
    """
    logger = logging.getLogger('SHAPAnalysis')

    # Set up output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "Outputs" / "SHAP_Analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving SHAP visualizations to: {output_dir}")
    logger.info(f"Using file prefix: {file_prefix}")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Summary plot (bar)
    logger.info("Creating SHAP summary plot (bar)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP value|)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{file_prefix}_summary_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Summary plot (beeswarm)
    logger.info("Creating SHAP summary plot (beeswarm)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (Feature Impact)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{file_prefix}_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Waterfall plot for first instance
    logger.info("Creating SHAP waterfall plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0],
                        base_values=shap_values.mean(axis=0).mean(),
                        data=X_sample[0],
                        feature_names=feature_names),
        show=False
    )
    plt.title("SHAP Waterfall Plot (First Instance)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{file_prefix}_waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Force plot for first few instances (save as HTML)
    logger.info("Creating SHAP force plots...")
    try:
        force_plot = shap.force_plot(
            base_value=shap_values.mean(axis=0).mean(),
            shap_values=shap_values[:min(5, len(shap_values))],
            features=X_sample[:min(5, len(X_sample))],
            feature_names=feature_names,
            show=False
        )
        shap.save_html(str(output_dir / f"{file_prefix}_force_plot.html"), force_plot)
    except Exception as e:
        logger.warning(f"Could not create force plot: {e}")

    logger.info("SHAP visualizations created successfully")

def save_shap_results(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    target_schema_name: str = "models",
    target_table_name: str = "shap_values",
    duckdb_name: str = "assignment.duckdb"
) -> None:
    """
    Save SHAP values and feature importance to DuckDB

    Args:
        shap_values: Computed SHAP values
        X_sample: Sample data used for SHAP computation
        feature_names: List of feature names
        target_schema_name: Schema name where the results table will be created
        target_table_name: Table name for storing the SHAP results
        duckdb_name: Name of the DuckDB database file
    """
    logger = logging.getLogger('SHAPAnalysis')

    # Get the project root directory (parent of Scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define database path
    db_path = project_root / duckdb_name

    logger.info(f"Saving SHAP results to {target_schema_name}.{target_table_name}")

    # Initialize DuckDB connection
    conn = duckdb.connect(str(db_path))

    try:
        # Create target schema if it doesn't exist
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema_name}")

        # Prepare SHAP summary data
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary_df = pd.DataFrame({
            'feature_name': feature_names,
            'mean_abs_shap_value': mean_abs_shap,
            'feature_rank': range(1, len(feature_names) + 1)
        })
        shap_summary_df = shap_summary_df.sort_values('mean_abs_shap_value', ascending=False).reset_index(drop=True)
        shap_summary_df['feature_rank'] = range(1, len(shap_summary_df) + 1)

        # Save summary table
        summary_table_name = f"{target_table_name}_summary"
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{summary_table_name}")
        conn.execute(f"CREATE TABLE {target_schema_name}.{summary_table_name} AS SELECT * FROM shap_summary_df")

        logger.info(f"Saved SHAP summary to {target_schema_name}.{summary_table_name}")

        # Prepare detailed SHAP values data
        shap_detailed_data = []
        for i in range(len(shap_values)):
            for j, feature_name in enumerate(feature_names):
                shap_detailed_data.append({
                    'instance_id': i,
                    'feature_name': feature_name,
                    'feature_value': X_sample[i, j],
                    'shap_value': shap_values[i, j]
                })

        shap_detailed_df = pd.DataFrame(shap_detailed_data)

        # Save detailed table
        detailed_table_name = f"{target_table_name}_detailed"
        conn.execute(f"DROP TABLE IF EXISTS {target_schema_name}.{detailed_table_name}")
        conn.execute(f"CREATE TABLE {target_schema_name}.{detailed_table_name} AS SELECT * FROM shap_detailed_df")

        logger.info(f"Saved detailed SHAP values to {target_schema_name}.{detailed_table_name}")

        # Verify the data was saved correctly
        summary_count = conn.execute(f"SELECT COUNT(*) FROM {target_schema_name}.{summary_table_name}").fetchone()[0]
        detailed_count = conn.execute(f"SELECT COUNT(*) FROM {target_schema_name}.{detailed_table_name}").fetchone()[0]

        logger.info(f"Verified: {summary_count} features in summary table")
        logger.info(f"Verified: {detailed_count} SHAP value records in detailed table")

    except Exception as e:
        logger.error(f"Error saving SHAP results: {e}")
        raise
    finally:
        conn.close()

def run_shap_analysis_with_parameters(
    response_variable: str,
    model_table_name: str ,
    generated_img_prefix: str,
    saved_to_table_name: str,
    predictive_variables: Optional[List[str]] = None,
    categorical_variables: Optional[List[str]] = None,
    max_depth: int = 6,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    use_best_params: bool = True,
    model_duckdb_name: str = "assignment.duckdb",
    model_schema_name: str = "models"
) -> None:
    """
    Main function to run SHAP analysis with specified or best parameters

    Args:
        response_variable: Name of the target variable column
        predictive_variables: List of predictor variable names (if None, uses default)
        categorical_variables: List of categorical variables for one-hot encoding (if None, uses default)
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of instance weight needed in a child
        gamma: Minimum loss reduction required to make a split
        reg_lambda: L2 regularization term on weights
        colsample_bytree: Subsample ratio of columns when constructing each tree
        reg_alpha: L1 regularization term on weights
        learning_rate: Boosting learning rate
        n_estimators: Number of boosting rounds
        use_best_params: Whether to use best parameters from training results
        model_duckdb_name: Name of the DuckDB database file for model parameters
        model_schema_name: Schema name containing the model parameters table
        model_table_name: Table name with model parameters
    """
    logger = logging.getLogger('SHAPAnalysis')

    try:
        # Load data
        logger.info("Loading training data...")
        data = load_data_from_duckdb(
            duckdb_name="assignment.duckdb",
            schema_name="silver",
            table_name="train"
        )

        # Set default variables if not provided
        if predictive_variables is None:
            predictive_variables = ["VolumeCred","VolumeCred_CA","TransactionsCred","TransactionsCred_CA","VolumeDeb","VolumeDeb_CA",
                "VolumeDebCash_Card","VolumeDebCashless_Card","VolumeDeb_PaymentOrder","TransactionsDeb","TransactionsDeb_CA",
                "TransactionsDebCash_Card","TransactionsDebCashless_Card","TransactionsDeb_PaymentOrder","Count_CA","Count_SA",
                "Count_MF","Count_OVD","Count_CC","Count_CL","ActBal_CA","ActBal_SA","ActBal_MF","ActBal_OVD","ActBal_CC",
                "ActBal_CL","Age","Tenure","Sex"
            ]

        if categorical_variables is None:
            categorical_variables = ["Sex"]

        logger.info(f"Using response variable: {response_variable}")
        logger.info(f"Using {len(predictive_variables)} predictive variables")
        logger.info(f"Using {len(categorical_variables)} categorical variables: {categorical_variables}")

        # Use best parameters if requested
        if use_best_params:
            logger.info("Loading best model parameters from training results...")

            best_params = load_best_model_parameters(
                duckdb_name=model_duckdb_name,
                schema_name=model_schema_name,
                table_name=model_table_name
            )

            max_depth = int(best_params['max_depth'])
            min_child_weight = int(best_params['min_child_weight'])
            gamma = float(best_params['gamma'])
            reg_lambda = float(best_params['reg_lambda'])
            colsample_bytree = float(best_params['colsample_bytree'])
            reg_alpha = float(best_params['reg_alpha'])
            learning_rate = float(best_params['learning_rate'])
            n_estimators = int(best_params['n_estimators'])
            scale_pos_weight = float(best_params['scale_pos_weight'])
        else:
            # Calculate scale_pos_weight for given parameters
            y = data[response_variable].dropna()
            pos_count = y.sum()
            neg_count = len(y) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        logger.info(f"Using parameters: max_depth={max_depth}, min_child_weight={min_child_weight}, "
                   f"gamma={gamma}, reg_lambda={reg_lambda}, colsample_bytree={colsample_bytree}, "
                   f"reg_alpha={reg_alpha}, learning_rate={learning_rate}, n_estimators={n_estimators}")

        # Build and train model
        logger.info("Building and training XGBoost model...")
        model, X_train, X_test, y_train, y_test, preprocessor = build_xgboost_model_with_parameters(
            data=data,
            response_variable=response_variable,
            predictive_variables=predictive_variables,
            categorical_variables=categorical_variables,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight
        )

        # Compute SHAP values
        logger.info("Computing SHAP values...")
        explainer, shap_values, X_sample, feature_names = compute_shap_values(
            model=model,
            X_train=X_train,
            X_test=X_test,
            preprocessor=preprocessor,
            predictive_variables=predictive_variables,
            sample_size=1000,
            background_size=100
        )

        # Create visualizations
        logger.info("Creating SHAP visualizations...")
        create_shap_visualizations(
            shap_values=shap_values,
            X_sample=X_sample,
            feature_names=feature_names,
            file_prefix=generated_img_prefix
        )

        # Save results to database
        logger.info("Saving SHAP results to database...")
        save_shap_results(
            shap_values=shap_values,
            X_sample=X_sample,
            feature_names=feature_names,
            target_schema_name="models",
            target_table_name=saved_to_table_name,
        )

        # Print summary
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[::-1][:10]

        print(f"\n🎯 SHAP Analysis Summary:")
        print(f"Response variable: {response_variable}")
        print(f"Model parameters used:")
        print(f"  max_depth: {max_depth}")
        print(f"  min_child_weight: {min_child_weight}")
        print(f"  gamma: {gamma}")
        print(f"  reg_lambda: {reg_lambda}")
        print(f"  colsample_bytree: {colsample_bytree}")
        print(f"  reg_alpha: {reg_alpha}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  n_estimators: {n_estimators}")
        print(f"\nTop 10 Most Important Features (by mean |SHAP value|):")
        for i, idx in enumerate(top_features_idx):
            print(f"  {i+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

        logger.info("✅ SHAP analysis completed successfully")

    except Exception as e:
        logger.error(f"❌ SHAP analysis failed: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    logger.info("Starting SHAP analysis")

    # Run SHAP analysis with best parameters from training (using default variables)
    run_shap_analysis_with_parameters(
		response_variable= "Sale_CL",
		model_table_name= "xgboost_cl_sales" ,
		use_best_params=True,
		model_duckdb_name = "assignment.duckdb",
		model_schema_name = "models",
		saved_to_table_name= "shap_values_cl",
		generated_img_prefix= "cl_shap"
  	)

    # Example of using custom variables:
    # custom_predictive_vars = ["Age", "Tenure", "Sex", "ActBal_CA", "ActBal_SA"]
    # custom_categorical_vars = ["Sex"]
    #
    # run_shap_analysis_with_parameters(
    #     response_variable="Sale_MF",
    #     predictive_variables=custom_predictive_vars,
    #     categorical_variables=custom_categorical_vars,
    #     use_best_params=False,
    #     max_depth=6,
    #     learning_rate=0.1
    # )