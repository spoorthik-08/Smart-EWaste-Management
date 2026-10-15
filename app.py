
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="Smart E-Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart E-Waste Management Using Data Science")
st.write(
    "Analyze electronic waste, predict recycling actions, "
    "and estimate future collection quantities."
)

DATA_FILE = "ewaste_dataset.csv"
RANDOM_STATE = 42

# --------------------------------------------------
# 2. CREATE SAMPLE DATASET
# --------------------------------------------------
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(RANDOM_STATE)

    categories = [
        "Mobile Phone", "Laptop", "Battery",
        "Television", "Charger", "Monitor", "Tablet"
    ]
    locations = [
        "Area A", "Area B", "Area C", "Area D", "Area E"
    ]
    conditions = ["Working", "Repairable", "Damaged"]

    # Sample records covering 24 months
    n = 800
    dates = pd.to_datetime(
        rng.choice(
            pd.date_range("2024-01-01", "2025-12-31", freq="D"),
            size=n
        )
    )

    category = rng.choice(categories, n)
    condition = rng.choice(
        conditions, n, p=[0.35, 0.35, 0.30]
    )
    age = rng.integers(1, 13, n)

    weight_map = {
        "Mobile Phone": 0.2,
        "Laptop": 2.0,
        "Battery": 0.8,
        "Television": 8.0,
        "Charger": 0.15,
        "Monitor": 4.0,
        "Tablet": 0.5
    }

    base_weight = np.array([weight_map[x] for x in category])
    quantity = rng.integers(1, 9, n)
    weight_kg = np.round(
        base_weight * quantity * rng.uniform(0.8, 1.2, n), 2
    )

    # Illustrative rule-generated labels for model training
    action = []
    for c, cond, a in zip(category, condition, age):
        if cond == "Working" and a <= 5:
            action.append("Reuse")
        elif cond == "Repairable" and a <= 8:
            action.append("Repair")
        else:
            action.append("Recycle")

    df = pd.DataFrame({
        "Date": dates,
        "Location": rng.choice(locations, n),
        "Category": category,
        "Condition": condition,
        "Age_Years": age,
        "Quantity": quantity,
        "Weight_kg": weight_kg,
        "Recommended_Action": action
    })

    df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    return df


if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
else:
    df = create_dataset()

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------
df = df.drop_duplicates().dropna().copy()
df["Month"] = df["Date"].dt.to_period("M").astype(str)

st.subheader("1. Dataset Overview")
st.write("Rows and columns:", df.shape)
st.dataframe(df.head(10), use_container_width=True)

with st.expander("View dataset statistics"):
    st.dataframe(df.describe(include="all").T,
                 use_container_width=True)

# --------------------------------------------------
# 4. SUMMARY METRICS
# --------------------------------------------------
total_items = int(df["Quantity"].sum())
total_weight = float(df["Weight_kg"].sum())
reuse_pct = 100 * (
    df["Recommended_Action"].eq("Reuse").mean()
)
recycle_pct = 100 * (
    df["Recommended_Action"].eq("Recycle").mean()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Items collected", f"{total_items:,}")
c3.metric("Total weight", f"{total_weight:,.1f} kg")
c4.metric("Reuse recommendation", f"{reuse_pct:.1f}%")

# --------------------------------------------------
# 5. DATA ANALYSIS AND GRAPHS
# --------------------------------------------------
st.subheader("2. Exploratory Data Analysis")

left, right = st.columns(2)

with left:
    category_data = (
        df.groupby("Category")["Weight_kg"]
        .sum().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    category_data.plot(kind="bar", ax=ax)
    ax.set_title("Total E-Waste Weight by Category")
    ax.set_xlabel("E-Waste Category")
    ax.set_ylabel("Weight (kg)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    action_counts = df["Recommended_Action"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    action_counts.plot(
        kind="pie", autopct="%1.1f%%", ax=ax
    )
    ax.set_title("Recommended Action Distribution")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

monthly = (
    df.groupby("Month")["Weight_kg"]
    .sum().reset_index()
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly["Month"], monthly["Weight_kg"], marker="o")
ax.set_title("Monthly E-Waste Collection")
ax.set_xlabel("Month")
ax.set_ylabel("Weight (kg)")
ax.tick_params(axis="x", rotation=70)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("3. Category Summary Table")
summary = (
    df.groupby("Category")
    .agg(
        Records=("Category", "size"),
        Items=("Quantity", "sum"),
        Total_Weight_kg=("Weight_kg", "sum"),
        Average_Weight_kg=("Weight_kg", "mean")
    )
    .round(2)
    .sort_values("Total_Weight_kg", ascending=False)
)
st.dataframe(summary, use_container_width=True)

# --------------------------------------------------
# 6. CLASSIFICATION MODEL
# Predict a suggested action
# --------------------------------------------------
st.subheader("4. Machine Learning: Action Classification")

features = ["Category", "Condition", "Age_Years"]
target = "Recommended_Action"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
    stratify=y
)

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     ["Category", "Condition"]),
    ("numeric", "passthrough", ["Age_Years"])
])

classifier = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=150, random_state=RANDOM_STATE,
        class_weight="balanced"
    ))
])

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.metric("Classification test accuracy", f"{accuracy:.1%}")
st.caption(
    "Important: labels are generated using sample rules. "
    "This accuracy measures how well the model learns those "
    "rules, not real-world recycling performance."
)

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T.round(3),
             use_container_width=True)

st.markdown("### Try a prediction")

with st.form("prediction_form"):
    chosen_category = st.selectbox(
        "Electronic item", sorted(df["Category"].unique())
    )
    chosen_condition = st.selectbox(
        "Item condition", ["Working", "Repairable", "Damaged"]
    )
    chosen_age = st.slider("Age of item (years)", 1, 12, 3)
    submit = st.form_submit_button("Predict action")

if submit:
    new_item = pd.DataFrame([{
        "Category": chosen_category,
        "Condition": chosen_condition,
        "Age_Years": chosen_age
    }])
    result = classifier.predict(new_item)[0]
    st.success(f"Model recommendation: {result}")
    st.write(
        "This is an educational prediction. Inspect the item "
        "and follow authorized recycling guidance where required."
    )

# --------------------------------------------------
# 7. FUTURE COLLECTION FORECAST
# --------------------------------------------------
st.subheader("5. Monthly E-Waste Quantity Forecast")

monthly["Month_Date"] = pd.to_datetime(monthly["Month"])
monthly["Month_Number"] = (
    monthly["Month_Date"].dt.month
)
monthly["Year"] = monthly["Month_Date"].dt.year
monthly["Time_Index"] = np.arange(len(monthly))

forecast_features = ["Time_Index", "Month_Number", "Year"]
Xf = monthly[forecast_features]
yf = monthly["Weight_kg"]

if len(monthly) >= 6:
    # Use chronological holdout to assess forecast error
    split_at = max(1, int(len(monthly) * 0.8))
    train_x = Xf.iloc[:split_at]
    test_x = Xf.iloc[split_at:]
    train_y = yf.iloc[:split_at]
    test_y = yf.iloc[split_at:]

    forecast_model = RandomForestRegressor(
        n_estimators=150, random_state=RANDOM_STATE
    )
    forecast_model.fit(train_x, train_y)

    if len(test_x) > 0:
        test_prediction = forecast_model.predict(test_x)
        mae = mean_absolute_error(test_y, test_prediction)
        st.metric("Chronological test MAE", f"{mae:.2f} kg")
        st.caption(
            "MAE is the average absolute forecast error on "
            "held-out historical months."
        )

    # Refit using all available months
    forecast_model.fit(Xf, yf)

    future_months = 3
    last_date = monthly["Month_Date"].max()
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=future_months, freq="MS"
    )

    future = pd.DataFrame({
        "Time_Index": np.arange(
            len(monthly), len(monthly) + future_months
        ),
        "Month_Number": future_dates.month,
        "Year": future_dates.year
    })

    future["Predicted_Weight_kg"] = np.maximum(
        forecast_model.predict(future[forecast_features]), 0
    )
    future["Month"] = future_dates.strftime("%Y-%m")

    st.dataframe(
        future[["Month", "Predicted_Weight_kg"]].round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly["Month"], monthly["Weight_kg"],
            marker="o", label="Historical sample")
    ax.plot(future["Month"], future["Predicted_Weight_kg"],
            marker="o", linestyle="--", label="Forecast")
    ax.set_title("Next 3 Months: Illustrative Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Weight (kg)")
    ax.tick_params(axis="x", rotation=65)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("At least 6 months of records are needed to forecast.")

# --------------------------------------------------
# 8. DOWNLOAD DATA AND REPORT
# --------------------------------------------------
st.subheader("6. Export Results")

st.download_button(
    "Download cleaned dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="ewaste_dataset.csv",
    mime="text/csv"
)

st.download_button(
    "Download category summary (CSV)",
    data=summary.to_csv().encode("utf-8"),
    file_name="ewaste_category_summary.csv",
    mime="text/csv"
)

st.info(
    "This project uses synthetic sample records. For real-world "
    "decisions, replace them with verified collection data and "
    "validate model recommendations with qualified recycling guidance."
)
