import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(file))


csv_path = os.path.join(BASE_DIR, "insurance.csv")

print("Looking for CSV file at:")
print(csv_path)

if not os.path.exists(csv_path):
print("\nERROR: insurance.csv file not found!")
print("Please put insurance.csv in the same folder as main.py")
exit()

# Load Data

df = pd.read_csv(csv_path)

print("\nOriginal Data Loaded:", df.shape)

# Data Cleaning

df = df.drop_duplicates()

df = df.fillna(
df.median(numeric_only=True)
)

df = df.fillna(
df.mode().iloc[0]
)

print("After Cleaning:", df.shape)

# Pre-processing

label_encoders = {}

for c in df.select_dtypes(include="object").columns:
le = LabelEncoder()
df[c] = le.fit_transform(df[c])
label_encoders[c] = le

print("\nAfter Encoding:")
print(df.head())

# Feature Selection

corr = df.corr()["claim"].abs()

features = corr[corr > 0.1].index.drop("claim")

print("\nCorrelations:")
print(corr)

print("\nSelected Features:")
print(list(features))

# Input and Target

X = df[features]
y = df["claim"]

# Split Data

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Decision Tree

model = DecisionTreeClassifier(
criterion="gini",
max_depth=5,
random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Prediction

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print(
"\nInsurance Claim Prediction Accuracy: {:.2f}%".format(
accuracy * 100
)
)

# Save Model

model_path = os.path.join(
BASE_DIR,
"insurance_claim_model.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(model_path)
