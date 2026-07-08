# Import Libraries
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data/cleaned_behavioral_health.csv")

# Features
X = df.drop("Mental_Health", axis=1)

# Load Trained Model
model = joblib.load("models/xgboost_model.pkl")

# Create SHAP Explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP Values
shap_values = explainer.shap_values(X)

# Summary Plot
shap.summary_plot(shap_values, X)

# Bar Plot
shap.summary_plot(shap_values, X, plot_type="bar")

print("SHAP Explainability Completed Successfully")
