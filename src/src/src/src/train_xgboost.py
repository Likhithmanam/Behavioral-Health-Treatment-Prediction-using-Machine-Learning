# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# Load Dataset
df = pd.read_csv("data/cleaned_behavioral_health.csv")

# Features and Target
X = df.drop("Mental_Health", axis=1)
y = df["Mental_Health"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# XGBoost Model
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# Hyperparameter Tuning
parameters = {
    "n_estimators": [100, 200],
    "max_depth": [5, 7],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=parameters,
    cv=3,
    scoring="accuracy"
)

# Train Model
grid.fit(X_train, y_train)

# Best Model
model = grid.best_estimator_

print("Best Parameters")
print(grid.best_params_)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "models/xgboost_model.pkl")

print("\nXGBoost Model Saved Successfully")
