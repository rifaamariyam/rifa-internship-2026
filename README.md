
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
