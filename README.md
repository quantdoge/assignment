./config.yaml:
Configuration file inputs to perform Raw to Bronze data ingestion at [./Scripts/01_Raw_to_Bronze.py]

./quality_config.yaml:
Configuration file inputs to perform data quality checks on the ingested raw data at [./Scripts/02_Data_Quality_Check.py]

./Logs:
Log folder storing all the logs of job runs for traceability and debugging purposes

./Data:
Data folder saving all the raw data (supposedly all raw data would be saved in .Data/Raw as landing zone)

./Models:
Model folder saving all the trained product propensity models and product revenue forecasting models

./Outputs:
Saving all the predicted probability of buying from trained propensity models and SHAP analysis outputs (summary bar, summary beeswarm chart)

./Results:
Saving all the predicted revenues from trained revenue forecasting models

./Queried:
Ad-hoc queries which produce csv files from DuckDB database at [./Scripts/utils/exec_duckdb_query.py]

./Scripts:
Stores all the scripts used for data quality checks, EDA, data pre-processing, model training, model evaluation and prediction works (running in sequences of prefix numbers)

++ 01_Raw_to_Bronze.py:
Ingest different raw Excel sheets into different bronze schema

++ 02_Data_Quality_Check.py:
Performing data quality checks on these ingested raw data

++ 03A_Bronze_to_Silver_Train.py:
Producing training datasets in the silver schema

++ 03B_Bronze_to_Silver_Test.py:
Producing test datasets in the silver schema

++ 04_Profiling.py:
Exploratory Data Analysis (EDA) on the training datasets

++ 05_Model_Training_Classifier.py:
Hyperparameter Tuning of Propensity Models (Probability of Buying) with randomized search, 1 model for 1 product

++ 05A_Model_Training_Classifier.py:
Hyperparameter Tuning of Propensity Models (Probability of Buying) with randomized search, 1 model for 1 product

++ 05B_Model_Training_Regressor.py:
Hyperparameter Tuning of Revenue Forecasting Models with randomized search, 1 model for 1 product, with the conditional probability that the customer is buying this product

++ 06A_SHAP_Analysis_Classifier.py:
SHAP analysis on the trained propensity models, getting the feature importance and effects of each feature trained

++ 06B_SHAP_Analysis_Regressor.py:
SHAP analysis on the trained revenue forecasting models, getting the feature importance and effects of each feature trained

++ 07A_Model_Final_Classifier.py:
Training propensity models (1 for each product) based on the hyperparameters that produce the best AUC-ROC score using features that are of importance based on SHAP Analysis

++ 07B_Model_Final_Regressor.py
Training revenue forecasting models (1 for each product where sales happened and revenue are non-zero) based on the hyperparameters that produce the best MAE score using features that are of importance based on SHAP Analysis

++ 08A_Predict_Classifier.py:
Using the trained propensity models to predict the probability of buying each product across all customers in the test set

++ 08B_Predict_Regressor.py
Using the trained revenue forecasting models to predict the predicted revenue for each product across all customers in the test set

++ 09_Maximize_Outcome.py

Across each of the 3 products (CC,CL,MF),
Stage 1: Assign a weight of 1 to the predicted probability of buying when this predicted probability is > p, else 0 => stage_1_weight (either 1 or 0)
Stage 2: Get the expected revenue for each product as stage_1_weight * predicted_probability * predicted_revenue => expected_revenue

For each customer, get the greatest expected_revenue across these 3 products, select top 100 customers with
max. expected_revenue


