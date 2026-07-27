# ===========================================
# HR Analytics - Logistic Regression
# Employee Attrition Prediction
# ===========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===========================================
# Load Dataset
# ===========================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# ===========================================
# Remove Unnecessary Columns
# ===========================================

drop_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(columns=drop_columns, inplace=True)

# ===========================================
# Encode All Categorical Columns
# ===========================================

# -------------------------------
# Convert Categorical Columns
# -------------------------------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Find all categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns

print("Categorical Columns:", cat_cols)

# Encode each categorical column
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nRemaining Object Columns:")
print(df.select_dtypes(include=["object"]).columns.tolist())

# -------------------------------
# Features and Target
# -------------------------------

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

print("\nData Types of X:")
print(X.dtypes)
# ===========================================
# Train-Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===========================================
# Logistic Regression Model
# ===========================================

model = LogisticRegression(
    max_iter=2000,
    solver="liblinear"
)

# ===========================================
# Train Model
# ===========================================

print("\nColumns still having object datatype:")
print(X_train.select_dtypes(include=["object"]).columns)

print("\nObject Columns Data:")
print(X_train.select_dtypes(include=["object"]).head())
print(X_train.dtypes)
model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ===========================================
# Prediction
# ===========================================

y_pred = model.predict(X_test)

# ===========================================
# Accuracy
# ===========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy : {:.2f}%".format(accuracy * 100))

# ===========================================
# Confusion Matrix
# ===========================================

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ===========================================
# Classification Report
# ===========================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ===========================================
# Sample Prediction
# ===========================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nSample Prediction")

if prediction[0] == 1:
    print("Employee is likely to Leave")
else:
    print("Employee is likely to Stay")