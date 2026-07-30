import streamlit as st
import pandas as pd
import plotly.express as px

from classification_model import train_model

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

st.title("📊 HR Analytics Dashboard")
st.write("Employee Attrition Analysis")

DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition-encoded.xls"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # Despite the .xls extension, the file contents are plain CSV.
    df = pd.read_csv(path)
    # Attrition is already encoded as 0/1; keep a readable label alongside
    # it for charts and filters that read better as text.
    df["AttritionLabel"] = df["Attrition"].map({0: "No", 1: "Yes"})
    return df


@st.cache_resource
def get_model_results():
    # Cached so the model trains once per app session instead of on
    # every filter click / rerun.
    return train_model(DATA_PATH)


df = load_data(DATA_PATH)

st.dataframe(df, use_container_width=True)

st.sidebar.header("KARTIK GAUTAM")
st.sidebar.header("Filters")

department = st.sidebar.selectbox(
    "Select Department", ["All"] + sorted(df["Department"].unique().tolist())
)
gender = st.sidebar.selectbox(
    "Select Gender", ["All"] + sorted(df["Gender"].unique().tolist())
)
job_role = st.sidebar.selectbox(
    "Select Job Role", ["All"] + sorted(df["JobRole"].unique().tolist())
)

filtered_df = df.copy()

if department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == department]

if gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender]

if job_role != "All":
    filtered_df = filtered_df[filtered_df["JobRole"] == job_role]

total = len(filtered_df)
attrition = int((filtered_df["Attrition"] == 1).sum())
rate = (attrition / total * 100) if total else 0.0
avg_income = filtered_df["MonthlyIncome"].mean() if total else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", total)
col2.metric("Employees Left", attrition)
col3.metric("Attrition Rate", f"{rate:.2f}%")
col4.metric("Avg Monthly Income", f"{avg_income:.0f}" if total else "N/A")

if total == 0:
    st.warning("No employees match the selected filters.")
else:
    fig_pie = px.pie(
        filtered_df,
        names="AttritionLabel",
        title="Employee Attrition",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    dept = filtered_df.groupby("Department").size().reset_index(name="Employees")
    fig_dept = px.bar(
        dept,
        x="Department",
        y="Employees",
        color="Department",
        title="Employees by Department",
    )

    line_data = (
        filtered_df.groupby("YearsAtCompany")["MonthlyIncome"]
        .mean()
        .reset_index()
    )
    fig_line = px.line(
        line_data,
        x="YearsAtCompany",
        y="MonthlyIncome",
        title="Average Monthly Income by Years at Company",
        markers=True,
    )
    fig_line.update_layout(
        xaxis_title="Years at Company",
        yaxis_title="Average Monthly Income",
        template="plotly_white",
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.plotly_chart(fig_dept, use_container_width=True)

    job = filtered_df.groupby("JobRole").size().reset_index(name="Count")
    fig_job = px.bar(
        job,
        x="JobRole",
        y="Count",
        color="Count",
        title="Employees by Job Role",
    )
    st.plotly_chart(fig_job, use_container_width=True)

    fig_ot = px.histogram(
        filtered_df,
        x="OverTime",
        color="AttritionLabel",
        barmode="group",
        title="Attrition by OverTime",
    )
    st.plotly_chart(fig_ot, use_container_width=True)

st.subheader("Machine Learning Model")
st.write("**Model:** XGBoost")

with st.spinner("Training model..."):
    results = get_model_results()

st.success(f"Model Accuracy: {results['accuracy'] * 100:.2f}%")

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(
    results["cm"],
    index=["Actual Stay", "Actual Leave"],
    columns=["Predicted Stay", "Predicted Leave"],
)
st.dataframe(cm_df)

st.subheader("Classification Report")
report_df = pd.DataFrame(results["report"]).transpose()
st.dataframe(report_df)

st.title("🔮 Employee Attrition Prediction")

DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition-encoded.xls"

results = train_model(DATA_PATH)

model = results["model"]
label_encoders = results["label_encoders"]
feature_names = results["feature_names"]

df = pd.read_csv(DATA_PATH)

feature_source_df = df[feature_names]

VISIBLE_FEATURES = [
    "Age",
    "Department",
    "JobRole",
    "BusinessTravel",
    "MaritalStatus",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "YearsAtCompany",
    "NumCompaniesWorked",
    "JobSatisfaction",
    "WorkLifeBalance",
]

st.subheader("Enter Employee Details")

input_values = {}

for col in VISIBLE_FEATURES:

    if col in label_encoders:

        input_values[col] = st.selectbox(
            col,
            label_encoders[col].classes_
        )

    else:

        input_values[col] = st.slider(
            col,
            int(feature_source_df[col].min()),
            int(feature_source_df[col].max()),
            int(feature_source_df[col].median())
        )

if st.button("Predict"):

    default_row = {}

    for col in feature_names:

        if col in label_encoders:

            default_row[col] = feature_source_df[col].mode()[0]

        else:

            default_row[col] = feature_source_df[col].median()

    default_row.update(input_values)

    input_df = pd.DataFrame([default_row])[feature_names]

    for col, le in label_encoders.items():

        input_df[col] = le.transform(input_df[col])

    input_df = input_df.astype(float)

    probability = model.predict_proba(input_df)[0][1]

    # Use the threshold tuned in train_model() (via F1 on the test set)
    # instead of the default 0.5 cutoff, since the dataset is imbalanced
    # and 0.5 was almost never crossed for the "Leave" class.
    prediction = int(probability >= results["threshold"])

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Employee is likely to LEAVE")

    else:

        st.success("✅ Employee is likely to STAY")

    st.metric(
        "Probability of Leaving",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))
