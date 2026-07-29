import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# ===================================
# Load Dataset
# ===================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ===================================
# Drop Unnecessary Columns
# ===================================

drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(columns=drop_cols, inplace=True)

# ===================================
# Encode Target Variable
# ===================================

df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# ===================================
# Encode Categorical Features
# ===================================

encoder = LabelEncoder()

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = encoder.fit_transform(df[col])

# ===================================
# Features and Target
# ===================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

# ===================================
# Train Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===================================
# XGBoost Model
# ===================================

model = XGBClassifier(

    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ===================================
# Train
# ===================================

model.fit(X_train, y_train)

# ===================================
# Prediction
# ===================================

y_pred = model.predict(X_test)

# ===================================
# Results
# ===================================

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print("\n========== XGBoost ==========")
print(f"Accuracy : {accuracy*100:.2f}%")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ===================================
# Sample Prediction
# ===================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction : Employee is likely to Leave")
else:
    print("\nPrediction : Employee is likely to Stay")
