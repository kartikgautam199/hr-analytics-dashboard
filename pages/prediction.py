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

