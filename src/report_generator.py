import pandas as pd
from pathlib import Path

df = pd.read_parquet("data/interim/02_cleaned.parquet")

# Summary statistics
summary = df[
    ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]
].describe().T

summary["cv"] = summary["std"] / summary["mean"]

# Mean vs median analysis
insights = []

for col in ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]:
    mean_val = df[col].mean()
    median_val = df[col].median()

    if mean_val > median_val:
        insights.append(
            f"- {col}: mean ({mean_val:.2f}) > median ({median_val:.2f}) → slight right skew."
        )
    elif mean_val < median_val:
        insights.append(
            f"- {col}: mean ({mean_val:.2f}) < median ({median_val:.2f}) → slight left skew."
        )
    else:
        insights.append(
            f"- {col}: mean ≈ median → symmetric distribution."
        )

report = []

report.append("# Polyhouse Data Quality Report\n")
report.append(f"**Rows:** {len(df)}")
report.append(
    f"**Date Range:** {df['timestamp'].min()} → {df['timestamp'].max()}\n"
)

report.append("## Summary Statistics\n")
report.append(summary.to_markdown())

report.append("\n## Key Insights\n")
report.extend(insights)

Path("reports").mkdir(exist_ok=True)

Path("reports/data_quality.md").write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("Report generated: reports/data_quality.md")