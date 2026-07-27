import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")


st.title("📊 HR Analytics Dashboard")
st.write("Employee Attrition Analysis")


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df
st.sidebar.header("KARTIK GAUTAM")
st.sidebar.header("Filters")
department = st.sidebar.radio(
    "Select Department",
    ["All"] + list(df["Department"].unique())
)

gender = st.sidebar.radio(
    "Select Gender",
    ["All"] + list(df["Gender"].unique())
)

job_role = st.sidebar.radio(
    "Select Job Role",
    ["All"] + list(df["JobRole"].unique())
)

filtered_df = df.copy()

if department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == department]

if gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender]

if job_role != "All":
    filtered_df = filtered_df[filtered_df["JobRole"] == job_role]


total = len(filtered_df)
attrition = len(filtered_df[filtered_df["Attrition"] == "Yes"])
rate = (attrition / total) * 100
avg_income = filtered_df["MonthlyIncome"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Employees", total)
col2.metric("Employees Left", attrition)
col3.metric("Attrition Rate", f"{rate:.2f}%")
col4.metric("Avg Monthly Income", f"{avg_income:.0f}")

fig = px.pie(
    filtered_df,
    names="Attrition",
    title="Employee Attrition"
)

st.plotly_chart(fig, use_container_width=True)
dept = filtered_df.groupby("Department").size().reset_index(name="Employees")

fig = px.bar(
    dept,
    x="Department",
    y="Employees",
    color="Department",
    title="Employees by Department"
)

st.plotly_chart(fig, use_container_width=True)
job = filtered_df.groupby("JobRole").size().reset_index(name="Count")

fig = px.bar(
    job,
    x="JobRole",
    y="Count",
    color="Count",
    title="Employees by Job Role"
)

st.plotly_chart(fig, use_container_width=True)
fig = px.histogram(
    filtered_df,
    x="OverTime",
    color="Attrition",
    barmode="group"
)

st.plotly_chart(fig, use_container_width=True)

st.header("🤖 Employee Attrition Prediction")

st.write("Machine Learning Model: Logistic Regression")

st.success("Model Accuracy: 85.37%")

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    [[246, 1],
     [42, 5]],
    index=["Actual Stay", "Actual Leave"],
    columns=["Predicted Stay", "Predicted Leave"]
)

st.dataframe(cm_df)

st.subheader("Model Summary")

st.write("""
- Algorithm Used: Logistic Regression
- Dataset: IBM HR Analytics Employee Attrition
- Accuracy: 85.37%
- Objective: Predict whether an employee is likely to leave the company.
""")