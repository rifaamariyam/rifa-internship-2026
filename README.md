
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



## Data Quality Report Generation ## -day5

### Objective

The objective of this script is to perform an exploratory assessment of the cleaned polyhouse sensor dataset and automatically generate a data quality report.

### Tasks Performed

1. Loads the cleaned dataset (`02_cleaned.parquet`).
2. Calculates summary statistics for:

   * Temperature
   * Humidity
   * CO₂ concentration
   * Mushroom yield
3. Computes the coefficient of variation (CV) to measure relative variability.
4. Compares mean and median values to identify potential data skewness.
5. Generates human-readable insights describing the distribution of each feature.
6. Creates a Markdown report containing:

   * Dataset size
   * Date range
   * Summary statistics table
   * Distribution insights
7. Saves the report as:


reports/data_quality.md


### Output

The generated report provides a concise overview of data quality and feature distributions, helping validate the dataset before feature engineering, visualization, and machine learning model development.
## Data Quality Report Generation ## -day5

### Objective

The objective of this script is to perform an exploratory assessment of the cleaned polyhouse sensor dataset and automatically generate a data quality report.

### Tasks Performed

1. Loads the cleaned dataset (`02_cleaned.parquet`).
2. Calculates summary statistics for:

   * Temperature
   * Humidity
   * CO₂ concentration
   * Mushroom yield
3. Computes the coefficient of variation (CV) to measure relative variability.
4. Compares mean and median values to identify potential data skewness.
5. Generates human-readable insights describing the distribution of each feature.
6. Creates a Markdown report containing:

   * Dataset size
   * Date range
   * Summary statistics table
   * Distribution insights
7. Saves the report as:

reports/data_quality.md

### Output

The generated report provides a concise overview of data quality and feature distributions, helping validate the dataset before feature engineering, visualization, and machine learning model development.


## Exploratory Data Analysis (EDA) ## -day6

### Objective

Analyze the cleaned polyhouse sensor dataset to understand data distributions, variability, and relationships between environmental conditions and mushroom yield.

### Steps Performed

#### 1. Generated Summary Statistics

Computed descriptive statistics for:

* temperature_c
* humidity_pct
* co2_ppm
* yield_kg

Metrics generated:

* Count
* Mean
* Standard Deviation
* Minimum
* 25th Percentile
* Median
* 75th Percentile
* Maximum

#### 2. Calculated Coefficient of Variation (CV)

Calculated:


CV = Standard Deviation / Mean


Purpose:

* Compare variability across different sensor measurements.
* Identify features with higher relative dispersion.

#### 3. Analyzed Feature Distributions

Compared mean and median values for each feature to determine distribution shape:

* Mean > Median → Right-skewed
* Mean < Median → Left-skewed
* Mean ≈ Median → Approximately symmetric

#### 4. Generated Correlation Matrix

Computed the Pearson correlation matrix for:

* temperature_c
* humidity_pct
* co2_ppm
* yield_kg

Outputs:

reports/correlation_matrix.csv
reports/correlation_matrix.md

Purpose:

* Measure linear relationships between environmental variables and mushroom yield.

#### 5. Created Correlation Heatmap

Generated a heatmap visualization from the correlation matrix.

Output:

reports/figures/corr_heatmap.png

Purpose:

* Visualize positive and negative correlations among features.

#### 6. Created Scatter Plots

Generated scatter plots to examine relationships between yield and environmental conditions:

* Humidity (%) vs Yield (kg)
* Temperature (°C) vs Yield (kg)
* CO₂ (ppm) vs Yield (kg)

Output:

reports/figures/scatter_yield.png

Purpose:

* Identify trends, patterns, and potential outliers.

# # Feature Engineering & Scaling## -day7

## Objective

Prepare machine learning features from the cleaned polyhouse dataset and scale them to a common range using Min-Max Scaling. The resulting feature set will be used for model training in later tasks.

## Input Dataset

Source:

data/interim/02_cleaned.parquet


Target Variable:

yield_kg

## Feature Definitions

### 1. Temperature

Column:

temperature_c

Description:

Average temperature inside the polyhouse in degrees Celsius.

Biological Importance:

Temperature directly affects mushroom growth rate, metabolism, and fruiting body development.

### 2. Humidity

Column:

humidity_pct

Description:

Relative humidity percentage inside the polyhouse.

Biological Importance:

Oyster mushrooms require high humidity for healthy growth and yield production. Low humidity can reduce productivity and affect mushroom quality.

### 3. Carbon Dioxide

Column:

co2_ppm

Description:

Carbon dioxide concentration measured in parts per million (ppm).

Biological Importance:

CO₂ levels influence mushroom respiration and growth conditions. Extremely high or low concentrations may affect yield.

### 4. Temperature–Humidity Interaction Feature

Column:

temp_humid_interaction

Formula:

temp_humid_interaction =
(temperature_c × humidity_pct) / 100

Example:

temperature_c = 25
humidity_pct = 80

temp_humid_interaction =
(25 × 80) / 100
= 20

Biological Importance:

Mushroom growth depends on the combined effect of temperature and humidity rather than either variable independently. This engineered feature helps the model capture interactions between these environmental factors.


## Feature Matrix and Target

Feature Matrix (X):

[
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "temp_humid_interaction"
]


Target Variable (y):

yield_kg

## Scaling Method

Scaler Used:

MinMaxScaler()

Scaling Formula:

x_scaled =
(x - x_min) / (x_max - x_min)

Output Range:

[0, 1]

Purpose:

* Prevents large-scale variables from dominating smaller-scale variables.
* Improves compatibility with many machine learning algorithms.
* Produces comparable feature ranges.

## Saved Outputs

Processed Features:

data/processed/features.parquet


Saved Scaler:

models/minmax_scaler.joblib

## Validation Checks

The following checks are performed after feature engineering:

* Feature and target row counts match.
* No missing values remain after processing.
* All scaled features lie within the range [0, 1].
* Scaler object is successfully saved for future inference.

## Future Improvement

For learning purposes, the scaler is currently fitted on the full cleaned dataset.