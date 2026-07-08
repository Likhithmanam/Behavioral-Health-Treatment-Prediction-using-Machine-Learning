import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv("data/behavioral_health.csv")

# Display Dataset Information
print(df.head())
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Fill Missing Values
imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Remove Duplicate Records
df.drop_duplicates(inplace=True)

# Encode Categorical Columns
encoder = LabelEncoder()

for column in df.select_dtypes(include="object"):
    df[column] = encoder.fit_transform(df[column])

# Save Clean Dataset
df.to_csv("data/cleaned_behavioral_health.csv", index=False)

print("Data Preprocessing Completed Successfully")
