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
save_queried(test_silver_profiling,file_name='test_silver_num_profiling.csv')

test_silver_cat_profiling_query= 'SELECT * FROM profile.categorical_profiling_results'
test_silver_cat_profiling= execute_query(query=test_silver_cat_profiling_query)
save_queried(test_silver_cat_profiling,file_name='test_silver_cat_profiling.csv')

test_mf_model_query= 'SELECT * FROM models.xgboost_mf_sales'
test_mf_model_profiling= execute_query(query=test_mf_model_query)
save_queried(test_mf_model_profiling,file_name='test_mf_model_profiling.csv')

# shap_mf_query= 'SELECT * FROM models.shap_analysis_mf_sales'
# shap_mf_result= execute_query(query=shap_mf_query)
# save_queried(shap_mf_result,file_name='shap_mf_result.csv')

test_mf_model_query= 'SELECT * FROM models.xgboost_mf_sales_top'
test_mf_model_profiling= execute_query(query=test_mf_model_query)
save_queried(test_mf_model_profiling,file_name='test_mf_model_top_profiling.csv')

mf_query= 'SELECT Sale_MF, TransactionsDebCashless_Card, Count_MF, * FROM silver.train WHERE Sale_MF=1'
mf_result= execute_query(query=mf_query)
save_queried(mf_result,file_name='mf_result.csv')

test_cl_model_query= 'SELECT * FROM models.xgboost_cl_sales'
test_cl_model_profiling= execute_query(query=test_cl_model_query)
save_queried(test_cl_model_profiling,file_name='test_cl_model_profiling.csv')

test_cl_model_query= 'SELECT * FROM models.xgboost_cl_sales_top'
test_cl_model_profiling= execute_query(query=test_cl_model_query)
save_queried(test_cl_model_profiling,file_name='test_cl_model_top_profiling.csv')

xgbreg_sales_mf_query= 'SELECT * FROM models.xgbreg_sales_mf_revenue'
xgbreg_sales_mf_result= execute_query(query=xgbreg_sales_mf_query)
save_queried(xgbreg_sales_mf_result,file_name='xgbreg_sales_mf_revenue.csv')

query='''
WITH AVG_CMF AS (
	SELECT 1 AS dummy, AVG(Revenue_MF) AS avg_revenue_mf
	FROM silver.train
    WHERE Revenue_MF!=0 AND Revenue_MF IS NOT NULL
), INT AS (
	SELECT a.*,
		   ABS(a.Revenue_MF - b.avg_revenue_mf) as abs_error
    FROM (SELECT 1 AS dummy, * FROM silver.train) a
    LEFT JOIN AVG_CMF b ON a.dummy=b.dummy
    WHERE a.Revenue_MF!=0 AND a.Revenue_MF IS NOT NULL
)
SELECT AVG(ABS_ERROR) AS mae_mf FROM INT
'''
query_result= execute_query(query=query)
print(query_result)

query='''
WITH AVG_CMF AS (
	SELECT 1 AS dummy, AVG(Revenue_CC) AS avg_revenue_cc
	FROM silver.train
    WHERE Revenue_CC!=0 AND Revenue_CC IS NOT NULL
), INT AS (
	SELECT a.*,
		   ABS(a.Revenue_CC - b.avg_revenue_cc) as abs_error
    FROM (SELECT 1 AS dummy, * FROM silver.train) a
    LEFT JOIN AVG_CMF b ON a.dummy=b.dummy
    WHERE a.Revenue_CC!=0 AND a.Revenue_CC IS NOT NULL
)
SELECT AVG(ABS_ERROR) AS mae_cc FROM INT
'''
query_result= execute_query(query=query)
print(query_result)

query='''
WITH AVG_CMF AS (
	SELECT 1 AS dummy, AVG(Revenue_CL) AS avg_revenue_cl
	FROM silver.train
    WHERE Revenue_CL!=0 AND Revenue_CL IS NOT NULL
), INT AS (
	SELECT a.*,
		   ABS(a.Revenue_CL - b.avg_revenue_cl) as abs_error
    FROM (SELECT 1 AS dummy, * FROM silver.train) a
    LEFT JOIN AVG_CMF b ON a.dummy=b.dummy
    WHERE a.Revenue_CL!=0 AND a.Revenue_CL IS NOT NULL
)
SELECT AVG(ABS_ERROR) AS mae_cl FROM INT
'''
query_result= execute_query(query=query)
print(query_result)

final_mf_sales_query= 'SELECT * FROM models.final_xgboost_mf_model'
final_mf_sales_result= execute_query(query=final_mf_sales_query)
save_queried(final_mf_sales_result,file_name='final_mf_sales_result.csv')

cl_sales_predictions_query= 'SELECT * FROM models.test_predictions_cl_sales'
cl_sales_predictions= execute_query(query=cl_sales_predictions_query)
save_queried(cl_sales_predictions,file_name='cl_sales_predictions_query.csv')

predict_combined_query = 'SELECT * FROM predictions.test_predictions_combined'
predict_combined_result = execute_query(query=predict_combined_query)
save_queried(predict_combined_result, file_name='predict_combined_result.csv')

predict_rank_query = 'SELECT * FROM predictions.optimized_outcome'
predict_rank_result = execute_query(query=predict_rank_query)
save_queried(predict_rank_result, file_name='predict_rank_result.csv')

predict_rank_query_02 = 'SELECT * FROM predictions.optimized_outcome WHERE max_expected_revenue > 0 ORDER BY max_expected_revenue DESC'
predict_rank_result_02 = execute_query(query=predict_rank_query_02)
save_queried(predict_rank_result_02, file_name='predict_rank_result_02.csv')

predict_rank_query_best = 'SELECT * FROM predictions.optimized_outcome WHERE max_expected_revenue > 0 ORDER BY max_expected_revenue DESC LIMIT 100'
predict_rank_result_best = execute_query(query=predict_rank_query_best)
save_queried(predict_rank_result_best, file_name='predict_rank_result_best.csv')

predict_rank_query_best_details= '''
WITH filtered AS (
	SELECT Client
	FROM predictions.optimized_outcome
	WHERE max_expected_revenue > 0
	ORDER BY max_expected_revenue DESC
	LIMIT 100
 )
 SELECT *
 FROM predictions.test_predictions_combined
 WHERE Client IN (SELECT Client FROM filtered)
 '''
predict_rank_result_best_details = execute_query(query=predict_rank_query_best_details)
save_queried(predict_rank_result_best_details, file_name='predict_rank_result_best_details.csv')