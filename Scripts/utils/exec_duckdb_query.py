import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import logging
from contextlib import contextmanager

def get_project_root() -> Path:
    """
    Get the project root directory (parent of Scripts folder)
    """
    script_dir = Path(__file__).parent.parent  # Go up from utils to Scripts
    return script_dir.parent  # Go up from Scripts to project root

def get_database_path(database_name: str = "assignment.duckdb") -> Path:
    """
    Get the full path to the database file
    """
    project_root = get_project_root()
    return project_root / database_name

def get_queried_output_path(folder_name: str= "Queried") -> Path:
    """
    Get the full path to the database file
    """
    project_root = get_project_root()
    return project_root / folder_name

@contextmanager
def get_duckdb_connection(database_name: str = "assignment.duckdb"):
    """
    Context manager for DuckDB connections
    """
    db_path = get_database_path(database_name)

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()

def execute_query(query: str,
                 database_name: str = "assignment.duckdb") -> Dict[str, Any]:

    try:
        with get_duckdb_connection(database_name) as conn:

                result = conn.execute(query).fetchdf()

                return {'status':1,
                        'query':result}

    except Exception as e:

        return {'status':0,
        		'query':str(e)}

def save_queried(query_result: Dict[str, Any],
				 folder_name: str = "Queried",
				 file_name: Optional[str] = None) -> Path:
	"""
	Save the queried result to a CSV file
	"""
	queried_path = get_queried_output_path(folder_name)
	queried_path.mkdir(parents=True, exist_ok=True)

	if file_name is None:
		file_name = f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

	file_path = queried_path / file_name

	if query_result['status'] == 1:
		query_result['query'].to_csv(file_path, index=False)
		return file_path
	else:
		raise ValueError(f"Query execution failed: {query_result['query']}")

# test_01_query= 'SELECT * FROM bronze.sales_revenues LIMIT 100'
# test_01= execute_query(query=test_01_query)
# save_queried(test_01,file_name='test.csv')

test_dq_query= 'SELECT * FROM dq.dq_results'
test_dq= execute_query(query=test_dq_query)
save_queried(test_dq,file_name='test_dq.csv')

test_silver_train_query= 'SELECT * FROM silver.train'
test_silver_train= execute_query(query=test_silver_train_query)
save_queried(test_silver_train,file_name='test_silver_train.csv')

test_silver_profiling_query= 'SELECT * FROM profile.numerical_profiling_results'
test_silver_profiling= execute_query(query=test_silver_profiling_query)
save_queried(test_silver_profiling,file_name='test_silver_profiling.csv')

test_silver_cat_profiling_query= 'SELECT * FROM profile.categorical_profiling_results'
test_silver_cat_profiling= execute_query(query=test_silver_cat_profiling_query)
save_queried(test_silver_cat_profiling,file_name='test_silver_cat_profiling.csv')
