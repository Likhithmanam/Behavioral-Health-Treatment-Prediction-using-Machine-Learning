# 🧠 Behavioral Health Prediction Using Machine Learning

> An end-to-end Machine Learning project designed to predict behavioral and mental health conditions using supervised learning algorithms. The project includes comprehensive data preprocessing, exploratory data analysis, model development, performance evaluation, explainable AI techniques, and deployment through a Flask API.

---

# 📌 Table of Contents

- Project Overview
- Problem Statement
- Objectives
- Dataset
- Project Workflow
- Technologies Used
- Machine Learning Models
- Exploratory Data Analysis
- Feature Engineering
- Model Evaluation
- Explainable AI (SHAP)
- Project Structure
- Installation
- Usage
- Results
- Future Enhancements
- Author

---

# 📖 Project Overview

Behavioral health disorders are becoming increasingly common, making early identification essential for timely intervention. This project leverages Machine Learning algorithms to analyze behavioral and demographic data and predict potential mental health conditions.

The project demonstrates the complete Machine Learning lifecycle, from raw data preprocessing to model deployment, while emphasizing model interpretability through SHAP (SHapley Additive Explanations).

---

# 🎯 Problem Statement

Mental health issues often remain undiagnosed due to limited awareness and delayed assessment. The objective of this project is to build an intelligent prediction system capable of identifying individuals at risk based on behavioral, demographic, and lifestyle-related features.

---

# 🎯 Objectives

- Clean and preprocess behavioral health data
- Perform Exploratory Data Analysis (EDA)
- Engineer meaningful features
- Train multiple Machine Learning models
- Compare model performances
- Select the best-performing model
- Explain predictions using SHAP
- Deploy the final model using Flask

---

# 📂 Dataset

The dataset consists of behavioral and demographic information collected from individuals.

### Features include:

- Age
- Gender
- Occupation
- Family History
- Mood Swings
- Sleep Pattern
- Work Stress
- Lifestyle Factors
- Treatment History
- Other Behavioral Indicators

---

# ⚙️ Project Workflow

```
Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
SHAP Explainability
     │
     ▼
Flask Deployment
```

---

# 🛠 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- SHAP
- Joblib
- Flask

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

# 🤖 Machine Learning Models

The following supervised learning algorithms were implemented and compared:

### Logistic Regression

Used as the baseline classification model.

### Random Forest Classifier

Applied for ensemble-based classification and improved prediction accuracy.

### XGBoost Classifier

Implemented as the final optimized model due to its superior predictive performance.

---

# 📊 Exploratory Data Analysis

The project includes several visualization techniques such as:

- Distribution Plots
- Count Plots
- Box Plots
- Correlation Heatmap
- Missing Value Analysis
- Feature Correlation Analysis

---

# ⚡ Feature Engineering

The preprocessing pipeline includes:

- Missing Value Handling
- Duplicate Removal
- Label Encoding
- Data Transformation
- Train-Test Split
- Feature Selection

---

# 📈 Model Evaluation

The trained models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

# 🔍 Explainable AI (SHAP)

To improve model transparency and interpretability, SHAP (SHapley Additive Explanations) was used.

The SHAP visualizations include:

- Feature Importance
- Summary Plot
- SHAP Value Distribution

These explanations provide insights into how each feature contributes to the prediction.

---

# 📁 Project Structure

```
Behavioral-Health-Prediction/

│
├── data/
│ ├── behavioral_health.csv
│ └── cleaned_behavioral_health.csv
│
├── models/
│ ├── logistic_model.pkl
│ ├── random_forest_model.pkl
│ ├── xgboost_model.pkl
│ └── mood_prediction_model.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── eda.py
│ ├── assumption_testing.py
│ ├── train_logistic.py
│ ├── train_randomforest.py
│ ├── train_xgboost.py
│ ├── mood_prediction.py
│ ├── evaluate_models.py
│ └── shap_explain.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Behavioral-Health-Prediction.git
```

Navigate to the project directory

```bash
cd Behavioral-Health-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

### Step 1

Preprocess the dataset

```bash
python src/data_preprocessing.py
```

### Step 2

Train the Machine Learning models

```bash
python src/train_logistic.py

python src/train_randomforest.py

python src/train_xgboost.py
```

### Step 3

Evaluate the models

```bash
python src/evaluate_models.py
```

### Step 4

Generate SHAP explanations

```bash
python src/shap_explain.py
```

### Step 5

Run the Flask API

```bash
python app.py
```

---

# 📌 Results

Among all implemented models, **XGBoost** achieved the best predictive performance and was selected as the final model for deployment.

The project demonstrates how advanced Machine Learning techniques can support early behavioral health assessment while maintaining model transparency through Explainable AI.

---

# 🔮 Future Enhancements

- Deep Learning-based prediction
- Streamlit Web Dashboard
- Real-time Prediction API
- Hyperparameter Optimization
- Cloud Deployment
- Automated Model Retraining
- Docker Containerization
