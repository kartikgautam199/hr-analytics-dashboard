import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier

DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition-encoded.xls"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the HR dataset.

    NOTE: despite the .xls extension, this file's contents are plain CSV
    text, not a real Excel binary/OOXML file. pd.read_excel() will fail
    (or need an engine that doesn't apply here) — pd.read_csv() is correct.
    """
    return pd.read_csv(path)


def train_model(path: str = DATA_PATH) -> dict:
    """Train the XGBoost attrition classifier and return model + metrics.

    Wrapped in a function (rather than running at import time) so that
    app.py can cache the trained model with st.cache_resource instead of
    retraining on every Streamlit rerun.
    """
    df = load_data(path)

    # Drop columns that carry no predictive signal
    df = df.drop(
        columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
        errors="ignore",
    )

    # Encode categorical (string) columns. Attrition is already 0/1 in this
    # dataset, so it's left untouched here and just cast to int below.
    label_encoders = {}
    for col in df.columns:
        if col == "Attrition":
            continue
        if df[col].dtype.kind == "O":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": model,
        "accuracy": accuracy,
        "cm": cm,
        "report": report,
        "label_encoders": label_encoders,
        "feature_names": X.columns.tolist(),
    }


if __name__ == "__main__":
    # Standalone run: `python classification_model.py` — trains the model
    # once and prints results to the terminal.
    results = train_model()

    print("\n========== XGBOOST MODEL ==========")
    print(f"Accuracy : {results['accuracy'] * 100:.2f}%")

    print("\nConfusion Matrix")
    print(results["cm"])

    print("\nClassification Report")
    print(pd.DataFrame(results["report"]).transpose())

    sample = results["feature_names"]
    df = load_data()
    print("\nSample prediction on first row:")
    X_sample = df.drop(
        columns=["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
        errors="ignore",
    ).iloc[[0]].copy()
    for col, le in results["label_encoders"].items():
        X_sample[col] = le.transform(X_sample[col])
    prediction = results["model"].predict(X_sample)
    print("Employee is likely to Leave" if prediction[0] == 1 else "Employee is likely to Stay")
