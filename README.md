
# Mushroom Yield Project-day 1





## Environment Setup



1. Create virtual environment
  python -m venv venv



2. Activate virtual environment

  venv\\Scripts\\activate



3. Install dependencies
  pip install pandas numpy matplotlib scikit-learn jupyter



4. Run smoke test
  python src\\smoke\_test.py



## Project Overview-day 2

This project predicts mushroom yield using Python and machine learning.

## Folder Structure

- data/raw
- models
- outputs
- src



## Project overview ## -day3

loaded ingest.py file into src

 The objective of ingest.py in a data project is usually to load raw data and prepare it for further processing.
 
 ingest.py load the raw polyhouse sensor CSV, perform initial validation/conversion, and save it in a structured format (Parquet) for  downstream data processing and machine learning tasks.

rest of the workflow follows these steps
Raw Sensor Data
       ↓
ingest.py
       ↓
01_loaded.parquet
       ↓
clean.py / preprocess.py
       ↓
feature engineering
       ↓
train.py
       ↓
yield prediction model



## cleaning log ## -day4

loaded clean.py file into src

This script performs data cleaning and preprocessing using several common techniques:
1 Missing value analysis
2 Range-based filtering (data validation)
3 Null target removal
4 Forward-fill imputation
5 Deletion of missing targets
6 Deduplication

Overall Cleaning Strategy

This is a combination of:

1 Data Validation Cleaning
  Filters out out-of-range sensor readings.
2 Missing Value Treatment
  Forward-fill imputation for sensor data.
  Row deletion for missing target values.
3 Data Deduplication
  Removes duplicate timestamps.
4 Quality-Based Filtering
  Retains only records that satisfy predefined oyster polyhouse environmental conditions.

Duplicate records were identified using the timestamp column and removed while retaining the latest occurrence. A total of 0 duplicate records were removed, resulting in a final cleaned dataset containing 365 rows.

02_cleaned.parquet was successfully loaded and validated. The target column (yield_kg) contains 0 missing values, confirming that all records are suitable for downstream analysis and model training.