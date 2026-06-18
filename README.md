
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

## Chronological Train/Test Split

### Objective

To prepare the mushroom yield dataset for machine learning by creating a chronological train/test split while preventing data leakage.

### Methodology

1. Loaded the cleaned dataset from:

   `data/interim/02_cleaned.parquet`

2. Sorted records by timestamp.

3. Applied an 80/20 chronological split:

   * First 80% of records → Training set
   * Last 20% of records → Test set

4. Verified that no test record occurred before the training cutoff date.

5. Applied MinMaxScaler:

   * Fitted only on training data
   * Applied to both training and test data

### Features Used

* temperature_c
* humidity_pct
* co2_ppm

### Target Variable

* yield_kg

### Leakage Prevention

The following assertion verifies that all test observations occur after the training period:

`assert test_start_date > train_end_date`

### Saved Artifacts

#### Model Assets

* models/minmax_scaler_train.joblib

#### Processed Data

* data/processed/train.csv
* data/processed/test.csv

#### NumPy Arrays

* data/processed/X_train.npy
* data/processed/X_test.npy
* data/processed/y_train.npy
* data/processed/y_test.npy

### Output Information Logged

The script logs:

* Train and test row counts
* Train period dates
* Test period dates
* Split cutoff date
* Leakage validation status
* X and y array shapes

### Execution

Run the script using:

`python src/train_test_split.py`

### Timeline Diagram

The chronological split can be visualized as:

|------------------- Training Set (80%) -------------------|------ Test Set (20%) ------|

2024-01-01                                            2024-10-18              2024-10-19                    2024-12-30
                                                        ↑
                                                   Split Cutoff

This timeline illustrates the separation between training and test windows and confirms that future observations are not used during model training.

### Seasonality Consideration

Because the dataset is split chronologically, the test period represents future observations that the model has not seen during training. If the average value of `yield_kg` in the test period differs significantly from the training period, evaluation metrics may decrease. Such differences can occur due to seasonality, environmental changes, or shifts in growing conditions over time.

This behavior is expected in real-world forecasting scenarios and does not indicate data leakage. Instead, it reflects the model's ability to generalize to future data under changing conditions.


## Baseline Linear Regression### -day 9

### Objective

Train a baseline Linear Regression model to predict mushroom yield using environmental sensor measurements and evaluate its performance on unseen test data.

### Features

| Feature | Description |
|----------|-------------|
| temperature_c | Temperature inside the polyhouse (°C) |
| humidity_pct | Relative humidity (%) |
| co2_ppm | Carbon dioxide concentration (ppm) |

### Target

| Target | Description |
|----------|-------------|
| yield_kg | Mushroom yield (kg) |

### Methodology

1. Loaded preprocessed train and test datasets.
2. Trained a Linear Regression model using `X_train` and `y_train`.
3. Generated predictions on the test set.
4. Computed evaluation metrics:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score
5. Inspected model coefficients to understand feature influence.
6. Saved the trained model and evaluation reports.

### Coefficient Interpretation

Since all features were scaled using MinMaxScaler, coefficient magnitudes can be compared directly.

- Positive coefficient → Higher feature value tends to increase yield.
- Negative coefficient → Higher feature value tends to decrease yield.
- Larger absolute coefficient → Greater influence on model predictions.

### Evaluation Metrics

- **MAE** measures average prediction error in kilograms.
- **RMSE** penalizes larger prediction errors.
- **R²** measures how much variation in yield is explained by the model.

### Saved Artifacts

#### Model

- `models/linear_regression.joblib`

#### Reports

- `reports/metrics_linear.json`
- `reports/metrics_linear.md`

### Execution

```bash
python src/train_linear_model.py
```

### Baseline Assessment

R² interpretation:

| R² Score | Assessment |
|-----------|------------|
| > 0.70 | Strong baseline |
| 0.50 – 0.70 | Reasonable baseline |
| < 0.50 | Additional feature engineering or advanced models recommended |

### Output

The script prints:

- MAE
- RMSE
- R² Score
- Feature coefficients
- Saved artifact locations

The resulting model serves as a baseline benchmark for future machine learning experiments on mushroom yield prediction.


# Linear Regression Diagnostics--day10

## Residual Definition

Residual = Actual Yield - Predicted Yield

A positive residual indicates the model underpredicted yield.

A negative residual indicates the model overpredicted yield.

---

## Diagnostic Findings

### Residuals vs Predicted Yield

The residuals are centered around zero, indicating that the model does not exhibit severe systematic bias.

Any visible funnel shape would indicate heteroscedasticity.

### Residuals vs Humidity

Residuals are randomly distributed across humidity values with no strong pattern.

A visible curve would indicate a nonlinear relationship not captured by Linear Regression.

---

## Recommendation

The baseline Linear Regression model provides a useful starting point.

If residual plots show curvature or increasing variance:

- Add engineered features.
- Try polynomial features.
- Evaluate nonlinear models such as Random Forest Regression.

Otherwise, continue with the linear baseline as a benchmark.



# Day 11 – Random Forest Regression Model

## Objective

To train and evaluate a Random Forest Regression model for mushroom yield prediction and compare its performance against the Linear Regression baseline.

---

## Dataset

### Training Data

* Samples: **292**

### Test Data

* Samples: **73**

Input files used:

* `data/processed/X_train.npy`
* `data/processed/y_train.npy`
* `data/processed/X_test.npy`
* `data/processed/y_test.npy`

The Random Forest model was trained exclusively on the training dataset and evaluated on the held-out test dataset.

---

## Methodology

### Baseline Model

A Linear Regression model was trained using the training data and evaluated on the test set.

Metrics computed:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

### Random Forest Model

A Random Forest Regressor was trained with the following configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Parameters:

| Parameter    | Value |
| ------------ | ----- |
| n_estimators | 100   |
| random_state | 42    |
| n_jobs       | -1    |

The model was fitted using:

```python
rf.fit(X_train, y_train)
```

Predictions were generated on the held-out test set and evaluated using the same metrics as the baseline model.

---

## Model Comparison

Performance of the Random Forest model was compared directly against the Linear Regression baseline.

Metrics included:

* MAE
* RMSE
* R²

Comparison results were exported for future reference.

Saved file:

```text
reports/model_comparison.csv
```

---

## Feature Importance Analysis

Random Forest feature importances were extracted using:

```python
rf.feature_importances_
```

The following environmental variables were evaluated:

* Temperature (`temperature_c`)
* Humidity (`humidity_pct`)
* CO₂ (`co2_ppm`)

Feature importance values indicate the relative contribution of each predictor to the model's predictions.

---

## Visualization

A horizontal bar chart was created to visualize feature importance rankings.

Saved figure:

```text
reports/figures/rf_feature_importance.png
```

The chart allows quick identification of the most influential environmental factor affecting mushroom yield predictions.

---

## Model Artifact

The trained Random Forest model was serialized using Joblib.

Saved model:

```text
models/random_forest.joblib
```

This artifact can be loaded later for inference, validation, or deployment.

---

## Interpretation

The feature with the highest importance score was identified as the strongest contributor to yield prediction.

Feature importance analysis provides insight into which environmental conditions have the greatest influence on the model's decisions.

---

## Complexity Assessment

Random Forest performance was compared against Linear Regression to determine whether the additional model complexity was justified.

Decision rule:

* Higher R² and lower prediction error → Random Forest likely justified.
* Similar performance → Linear Regression may remain preferable due to simplicity and interpretability.

This comparison helps balance predictive performance against model complexity.

---

## Saved Artifacts

### Model

```text
models/random_forest.joblib
```

### Comparison Table

```text
reports/model_comparison.csv
```

### Feature Importance Plot

```text
reports/figures/rf_feature_importance.png
```

---

## Deliverables Completed

* Random Forest trained using training data only.
* Test set evaluation completed.
* Performance compared against Linear Regression baseline.
* Feature importance values computed.
* Feature importance visualization generated.
* Trained model saved for reuse.
* Model comparison table exported.
* Complexity justification documented.

---

## Conclusion

A Random Forest Regression model was successfully trained and evaluated on the mushroom yield dataset. The model's predictive performance was compared with a Linear Regression baseline using MAE, RMSE, and R² metrics. Feature importance analysis provided insight into the influence of temperature, humidity, and CO₂ on yield prediction. The trained model, evaluation outputs, and visualization artifacts were saved to ensure reproducibility and future model analysis.


# Time Series Cross-Validation-day12

## Objective

To evaluate model stability and generalization performance using time-aware cross-validation while preventing data leakage from future observations.

## Methodology

1. Loaded the preprocessed training and test datasets:

   * `data/processed/X_train.npy`
   * `data/processed/X_test.npy`
   * `data/processed/y_train.npy`
   * `data/processed/y_test.npy`

2. Configured `TimeSeriesSplit` with 5 folds:

   * Training data was split chronologically.
   * Earlier observations were used to predict later observations.
   * No test data was used during cross-validation.

3. Evaluated two models:

   * Linear Regression
   * Random Forest Regressor (`n_estimators=100`)

4. Computed cross-validated Mean Absolute Error (MAE) for each fold.

5. Calculated:

   * Mean CV MAE
   * Standard deviation of CV MAE
   * Training MAE
   * Hold-out Test MAE

6. Compared cross-validation performance with final test-set performance.

7. Assessed overfitting by comparing training MAE and test MAE.

## Results

Cross-validation results were summarized for both models using:

* Mean CV MAE
* CV MAE Standard Deviation
* Training MAE
* Test MAE

Lower MAE values indicate better predictive performance.

The standard deviation across folds was used to assess model stability. Higher variance suggests model performance changes significantly across different time periods.

## Overfitting Analysis

Overfitting was evaluated by comparing training and test errors.

Guideline:

* Train MAE much lower than Test MAE → Potential overfitting
* Similar Train and Test MAE → Better generalization

Random Forest and Linear Regression were both examined using this criterion.

## Deliverables

### Saved Report

Cross-validation summary:

`reports/cv_results.md`

### Saved Visualization

Cross-validation MAE plot:

`reports/figures/cv_mae_scores.png`

### Input Files

Training and test datasets:

* `data/processed/X_train.npy`
* `data/processed/X_test.npy`
* `data/processed/y_train.npy`
* `data/processed/y_test.npy`

## Key Findings

* TimeSeriesSplit was used instead of random K-Fold to preserve chronological order.
* Cross-validation was performed exclusively on training data.
* Model performance was evaluated across multiple folds to estimate robustness.
* Variability across folds was analyzed using MAE standard deviation.
* Hold-out test performance was compared against cross-validation results.
* Overfitting risk was assessed by comparing training and test errors.
* Results provide a more reliable estimate of future model performance than a single train/test split.


Day 13 – Hyperparameter Tuning with GridSearchCV and TimeSeriesSplit
Objective
To optimize the Random Forest Regression model by tuning key hyperparameters using time-aware cross-validation while preventing data leakage from future observations.
---
Dataset
Training Data
Samples: 292
Test Data
Samples: 73
Input files used:
`data/processed/X_train.npy`
`data/processed/y_train.npy`
`data/processed/X_test.npy`
`data/processed/y_test.npy`
Only the training dataset was used during hyperparameter tuning.
---
Cross-Validation Strategy
A chronological cross-validation strategy was implemented using:
```python
TimeSeriesSplit(n_splits=3)
```
This ensures:
Temporal ordering is preserved.
Future observations are never used to predict past observations.
Data leakage is avoided.
The held-out test dataset was completely excluded from the tuning process.
---
Hyperparameter Grid
The following parameter combinations were evaluated:
Parameter	Values
n_estimators	50, 100, 200
max_depth	None, 8, 16
min_samples_leaf	1, 3, 5
Total combinations tested:
```text
27
```
Cross-validation folds:
```text
3
```
Total model fits:
```text
81
```
---
Grid Search Configuration
Grid Search was performed using:
RandomForestRegressor
GridSearchCV
TimeSeriesSplit
MAE scoring (`neg_mean_absolute_error`)
Parallel execution (`n_jobs=-1`)
Automatic refitting of the best model
---
Best Hyperparameters
The optimal parameter combination identified by Grid Search was:
```json
{
  "max_depth": 8,
  "min_samples_leaf": 5,
  "n_estimators": 100
}
```
Best Cross-Validation Performance
Metric	Value
CV MAE	0.466
---
Runtime
Grid Search execution time:
```text
5.22 seconds
```
The runtime remained well within the expected range for an internship-scale machine learning project.
---
Grid Boundary Analysis
The search reported:
```text
min_samples_leaf: best value is at upper edge (5)
```
This suggests that larger values of `min_samples_leaf` may potentially improve performance and could be explored in future tuning iterations.
---
Final Test Set Evaluation
After selecting the best hyperparameters, the model was automatically refit on the full training dataset and evaluated once on the held-out test set.
Test Metrics
Metric	Value
MAE	0.445
RMSE	0.562
R²	0.369
---
Saved Artifacts
Tuned Model
Best-performing Random Forest model:
```text
models/random_forest_tuned.joblib
```
Best Parameters
Selected hyperparameters:
```text
models/rf_best_params.json
```
Grid Search Transparency Log
First rows of GridSearchCV results:
```text
reports/gridsearch_cv_results_head.csv
```
Summary Metrics
Grid search and test evaluation summary:
```text
reports/gridsearch_summary.csv
```
---
Results Interpretation
The tuned Random Forest achieved a cross-validation MAE of 0.466.
Final test MAE improved to 0.445, indicating that the selected hyperparameters generalized well to unseen data.
A moderate R² score of 0.369 suggests the model captures part of the yield variability while leaving room for improvement.
The best value of `min_samples_leaf` occurred at the edge of the search space, indicating that additional tuning beyond the current grid may be beneficial.
The tuning process was completed in approximately 5 seconds, making the approach computationally efficient.
---
Conclusion
A time-aware Grid Search using TimeSeriesSplit successfully identified an improved Random Forest configuration:
`n_estimators = 100`
`max_depth = 8`
`min_samples_leaf = 5`
The tuned model achieved:
CV MAE = 0.466
Test MAE = 0.445
Test RMSE = 0.562
Test R² = 0.369
All tuning was performed exclusively on training data, and the test set was evaluated only once after model selection. The resulting model, parameter configuration, and search logs were saved for reproducibility and mentor review.
##Model Comparison and Champion Selection### --day14
Objective
Evaluate and compare the performance of three machine learning models for mushroom yield prediction:
Linear Regression
Random Forest (Default)
Random Forest (Tuned)
The comparison uses the same untouched chronological test set to ensure a fair evaluation and prevent data leakage.
---
Evaluation Methodology
The following metrics were used to assess model performance:
Cross-Validation MAE (CV MAE)
Test MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score
Training Time
Model Interpretability
All models were evaluated on the same test dataset generated during the chronological train/test split.
---
Model Comparison Table
Model	CV MAE	Test MAE	RMSE	R²	Training Time (s)	Interpretability
Linear Regression	Replace	Replace	Replace	Replace	Replace	High
Random Forest Default	Replace	Replace	Replace	Replace	Replace	Medium
Random Forest Tuned	Replace	Replace	Replace	Replace	Replace	Medium-Low


Champion Model## --day14
Selected Model: Replace with actual champion model
Selection Rationale
The champion model was selected based primarily on the lowest Test MAE while also considering RMSE, R² score, model complexity, and interpretability.
If multiple models achieved nearly identical MAE values, the simpler Linear Regression model would be preferred due to:
Greater transparency
Easier stakeholder communication
Simpler maintenance
Reduced deployment complexity
In this project, the selected champion model demonstrated the best balance between predictive performance and practical deployment considerations.
---
Predicted vs Actual Yield
A scatter plot comparing actual yield values against model predictions was generated for the champion model.
Saved Figure:
`reports/figures/pred\_vs\_actual.png`
Interpretation
Points close to the diagonal line indicate accurate predictions.
Larger deviations from the diagonal represent prediction errors.
A strong clustering around the diagonal suggests good model performance on unseen data.
---
Deployment Recommendation
The selected champion model is recommended for deployment as a decision-support tool for mushroom yield forecasting.
The model can assist growers by providing estimated yield predictions based on environmental sensor inputs.
---
Known Limitations and Edge Cases
Sensor Range Limitations
The model was trained on a limited range of environmental conditions. Predictions for temperature, humidity, or CO₂ values outside the observed training range may be unreliable.
Seasonality
The dataset covers a limited time period and may not fully capture long-term seasonal effects or environmental variations.
Unseen Conditions
Extreme growing conditions not represented in the training data may reduce prediction accuracy.
Synthetic Dataset Constraints
The project uses generated data for development purposes. Real-world mushroom farms may exhibit additional variability not captured by the synthetic dataset.
Operational Use
Model predictions should be considered advisory only.
The model is intended to support decision-making and should not replace grower expertise, operational experience, or field observations.
---
Deliverables
Generated artifacts:
`reports/model\_comparison.csv`
`reports/model\_comparison.md`
`reports/figures/pred\_vs\_actual.png`
These files provide a complete summary of model evaluation, champion selection, and deployment readiness assessment.

