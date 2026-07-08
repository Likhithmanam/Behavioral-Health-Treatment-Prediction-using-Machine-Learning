# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("data/cleaned_behavioral_health.csv")

# Features
X = df.drop(["Mood_Swings"], axis=1)

# Target
y = df[["Mood_Swings"]]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Multi Output Random Forest
model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Mood Prediction Accuracy")
print(accuracy_score(y_test, y_pred))

# Save Model
joblib.dump(model, "models/mood_prediction_model.pkl")

print("Mood Prediction Model Saved Successfully")
