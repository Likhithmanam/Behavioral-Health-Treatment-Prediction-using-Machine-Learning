import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/cleaned_behavioral_health.csv")

print(df.head())

print(df.describe())

print(df.info())

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()

# Treatment Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Treatment", data=df)
plt.title("Treatment Distribution")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot")
plt.show()

print("EDA Completed")
