import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Behavioral Health Prediction")

age = st.number_input("Age", 18, 80)

sleep = st.slider("Sleep Hours", 0, 12)

stress = st.slider("Stress Level", 1, 10)

work = st.slider("Work Hours", 0, 16)

activity = st.slider("Physical Activity", 0, 10)

social = st.slider("Social Interaction", 0, 10)

gender = st.selectbox("Gender", ["Male", "Female"])

df = pd.DataFrame({
    "Age":[age],
    "Sleep Hours":[sleep],
    "Stress Level":[stress],
    "Work Hours":[work],
    "Physical Activity":[activity],
    "Social Interaction":[social],
    "Gender":[gender]
})

df = pd.get_dummies(df)

if st.button("Predict"):

    X = scaler.transform(df)

    prediction = model.predict(X)

    if prediction[0] == 1:
        st.error("⚠ High Risk")
    else:
        st.success("✅ Healthy")
