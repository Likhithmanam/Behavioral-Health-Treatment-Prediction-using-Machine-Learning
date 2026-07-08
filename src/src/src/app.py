from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load Trained Model
model = joblib.load("models/xgboost_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Behavioral Health Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return jsonify({
        "Prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
