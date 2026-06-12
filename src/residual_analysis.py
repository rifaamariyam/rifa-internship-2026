from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# Paths
# ----------------------------------

PROCESSED = Path("data/processed")
MODELS = Path("models")
FIGURES = Path("reports/figures")

FIGURES.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# Load data
# ----------------------------------

X_test = np.load(PROCESSED / "X_test.npy")
y_test = np.load(PROCESSED / "y_test.npy")

# ----------------------------------
# Load trained model
# ----------------------------------

model = joblib.load(
    MODELS / "linear_regression.joblib"
)

# ----------------------------------
# Predictions
# ----------------------------------

pred_test = model.predict(X_test)

# ----------------------------------
# Residuals
# residual = actual - predicted
# ----------------------------------

residuals = y_test - pred_test

# ----------------------------------
# Save residuals
# ----------------------------------

residual_df = pd.DataFrame({
    "actual_yield": y_test,
    "predicted_yield": pred_test,
    "residual": residuals
})

residual_df.to_csv(
    PROCESSED / "residuals_linear.csv",
    index=False
)

# ----------------------------------
# Print residuals
# ----------------------------------

print("\n===== RESIDUALS (FIRST 10 ROWS) =====")

print(
    residual_df.head(10).to_string(index=False)
)

print(
    f"\nResiduals saved to: "
    f"{PROCESSED / 'residuals_linear.csv'}"
)

# ----------------------------------
# Plot 1
# Residuals vs Predicted Yield
# ----------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(
    pred_test,
    residuals,
    alpha=0.5
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Yield (kg)")
plt.ylabel("Residual (kg)")
plt.title("Residuals vs Predicted Yield")

plt.tight_layout()

plt.savefig(
    FIGURES / "residuals_vs_predicted.png",
    dpi=150
)

plt.close()

# ----------------------------------
# Plot 2
# Residuals vs Humidity
# X_test[:,1]
# ----------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(
    X_test[:, 1],
    residuals,
    alpha=0.5
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Scaled Humidity")
plt.ylabel("Residual (kg)")
plt.title("Residuals vs Humidity")

plt.tight_layout()

plt.savefig(
    FIGURES / "residuals_vs_humidity.png",
    dpi=150
)

plt.close()

# ----------------------------------
# Plot 3
# Residuals vs Temperature
# X_test[:,0]
# ----------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(
    X_test[:, 0],
    residuals,
    alpha=0.5
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Scaled Temperature")
plt.ylabel("Residual (kg)")
plt.title("Residuals vs Temperature")

plt.tight_layout()

plt.savefig(
    FIGURES / "residuals_vs_temperature.png",
    dpi=150
)

plt.close()

# ----------------------------------
# Summary
# ----------------------------------

print("\n===== DIAGNOSTIC FILES =====")

print(
    f"Residual CSV : "
    f"{PROCESSED / 'residuals_linear.csv'}"
)

print(
    f"Plot 1       : "
    f"{FIGURES / 'residuals_vs_predicted.png'}"
)

print(
    f"Plot 2       : "
    f"{FIGURES / 'residuals_vs_humidity.png'}"
)

print(
    f"Plot 3       : "
    f"{FIGURES / 'residuals_vs_temperature.png'}"
)

print("\nDiagnostics completed successfully.")