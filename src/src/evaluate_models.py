# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import joblib

# Load Dataset
df = pd.read_csv("data/cleaned_behavioral_health.csv")

# Features
X = df.drop("Mental_Health", axis=1)

# Target
y = df["Mental_Health"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Load Models
logistic = joblib.load("models/logistic_model.pkl")
rf = joblib.load("models/random_forest_model.pkl")
xgb = joblib.load("models/xgboost_model.pkl")

models = {
    "Logistic Regression": logistic,
    "Random Forest": rf,
    "XGBoost": xgb
}

print("\nMODEL COMPARISON\n")

for name, model in models.items():

    y_pred = model.predict(X_test)

    print("--------------------------------")
    print(name)
    print("--------------------------------")

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(name + " Confusion Matrix")
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(X_test)

        fpr, tpr, threshold = roc_curve(
            y_test,
            probability[:,1]
        )

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,5))

        plt.plot(
            fpr,
            tpr,
            label=name + " AUC = %.3f" % roc_auc
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(name + " ROC Curve")
        plt.legend()
        plt.show()
