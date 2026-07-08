# Import Libraries
import pandas as pd
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load Dataset
df = pd.read_csv("data/cleaned_behavioral_health.csv")

print("Dataset Loaded Successfully")

# -------------------------------
# Shapiro-Wilk Test
# -------------------------------
print("\nShapiro-Wilk Test")

for column in df.select_dtypes(include=['int64','float64']).columns:
    stat, p = shapiro(df[column])
    print(f"{column}")
    print("Statistic =", round(stat,4))
    print("P-value =", round(p,4))

# -------------------------------
# Variance Inflation Factor (VIF)
# -------------------------------

numeric_df = df.select_dtypes(include=['int64','float64'])

print("\nVariance Inflation Factor")

vif = pd.DataFrame()
vif["Feature"] = numeric_df.columns
vif["VIF"] = [variance_inflation_factor(numeric_df.values, i)
              for i in range(numeric_df.shape[1])]

print(vif)

print("\nAssumption Testing Completed")
