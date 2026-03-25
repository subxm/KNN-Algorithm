from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="KNN Tekworks Lab", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #18253d 0%, #101726 42%, #090e18 100%);
        color: #e8edf7;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #e8edf7;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #f4f7ff;
    }
    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        border: 1px solid rgba(196, 214, 255, 0.14);
        border-radius: 12px;
        background: rgba(12, 20, 34, 0.72);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0b1020 100%);
        border-right: 1px solid rgba(196, 214, 255, 0.12);
    }
    .hero {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(20, 30, 52, 0.98) 0%, rgba(50, 70, 105, 0.92) 50%, rgba(22, 31, 50, 0.98) 100%);
        color: #f7f9ff;
        border: 1px solid rgba(196, 214, 255, 0.18);
        box-shadow: 0 20px 45px rgba(6, 10, 20, 0.35);
    }
    .hero h1 {
        margin-bottom: 0.2rem;
        letter-spacing: 0.5px;
        color: #f9fbff;
    }
    .hero p {
        margin-top: 0.15rem;
        color: #d9e2f5;
        opacity: 0.95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="darkgrid")
plt.style.use("dark_background")

st.markdown(
    """
    <div class="hero">
        <h1>KNN Tekworks Regression Studio</h1>
        <p>Train, compare, and inspect KNN, Linear Regression, and Decision Tree models on your loan dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).with_name("task1_dataset.csv")
TARGET_COL = "target"


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def apply_iqr_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = df.copy()

    for col in ["income", "loan_amount", "credit_score", "annual_spend"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in ["income", "loan_amount", "credit_score", "num_transactions", "annual_spend"]:
        cleaned = apply_iqr_filter(cleaned, col)

    cleaned = cleaned.dropna()

    if "date" in cleaned.columns:
        parsed_date = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned["month"] = parsed_date.dt.month
        cleaned["day"] = parsed_date.dt.day
        cleaned["year"] = parsed_date.dt.year
        cleaned = cleaned.drop(columns=["date"])

    if TARGET_COL not in cleaned.columns:
        raise ValueError("Missing target column. Expected a column named 'target'.")

    y = cleaned[TARGET_COL]
    x = cleaned.drop(columns=[TARGET_COL])

    categorical_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=False)

    minmax_cols = [c for c in ["income", "loan_amount"] if c in x.columns]
    if minmax_cols:
        minmax_scaler = MinMaxScaler()
        x[minmax_cols] = minmax_scaler.fit_transform(x[minmax_cols])

    if "credit_score" in x.columns:
        standard_scaler = StandardScaler()
        x[["credit_score"]] = standard_scaler.fit_transform(x[["credit_score"]])

    return x, y


def evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    k: int,
) -> pd.DataFrame:
    models = {
        "KNN Regressor": KNeighborsRegressor(n_neighbors=k),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rows.append({"Model": model_name, "MSE": mse, "R2": r2})

    metrics = pd.DataFrame(rows).sort_values(by="MSE", ascending=True).reset_index(drop=True)
    return metrics


def knn_elbow_curve(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    k_min: int,
    k_max: int,
) -> pd.DataFrame:
    records = []
    for current_k in range(k_min, k_max + 1):
        model = KNeighborsRegressor(n_neighbors=current_k)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        records.append({"k": current_k, "MSE": mse})
    return pd.DataFrame(records)


with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    k_value = st.slider("K for KNN", min_value=1, max_value=20, value=5, step=1)
    test_size = st.slider("Test split ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    seed = st.number_input("Random state", min_value=0, max_value=1000, value=42, step=1)

raw_df = load_data(uploaded_file)

if raw_df.empty:
    st.error(
        "Dataset not found. Keep task1_dataset.csv in the same folder as streamlit.py, or upload a CSV from the sidebar."
    )
    st.stop()

if TARGET_COL not in raw_df.columns:
    st.error("Your CSV must include a 'target' column for regression.")
    st.stop()

st.subheader("Dataset Snapshot")
left, right = st.columns([2, 1])
with left:
    st.dataframe(raw_df.head(10), use_container_width=True)
with right:
    st.metric("Rows", f"{raw_df.shape[0]:,}")
    st.metric("Columns", f"{raw_df.shape[1]:,}")
    st.metric("Missing Values", f"{int(raw_df.isna().sum().sum()):,}")

with st.expander("Column Types"):
    dtypes = pd.DataFrame({"column": raw_df.columns, "dtype": raw_df.dtypes.astype(str)})
    st.dataframe(dtypes, use_container_width=True)

try:
    x, y = preprocess_dataframe(raw_df)
except ValueError as err:
    st.error(str(err))
    st.stop()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=int(seed)
)

metrics_df = evaluate_models(x_train, x_test, y_train, y_test, int(k_value))

st.subheader("Model Performance")
st.dataframe(
    metrics_df.style.format({"MSE": "{:.2f}", "R2": "{:.4f}"}),
    use_container_width=True,
)

best_model = metrics_df.loc[0, "Model"]
st.success(f"Best model by MSE: {best_model}")

curve_df = knn_elbow_curve(x_train, x_test, y_train, y_test, 1, 20)

plot_col, text_col = st.columns([3, 2])
with plot_col:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=curve_df, x="k", y="MSE", marker="o", ax=ax)
    ax.set_title("KNN Elbow Curve")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Squared Error")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

with text_col:
    optimal_k = int(curve_df.sort_values("MSE").iloc[0]["k"])
    st.info(f"Best k in tested range: {optimal_k}")
    st.write("Use this chart to choose a stable k with low validation error.")

st.subheader("Prediction Preview")
preview_count = st.slider("Rows to preview from test split", min_value=5, max_value=50, value=10, step=5)

knn_preview_model = KNeighborsRegressor(n_neighbors=int(k_value))
knn_preview_model.fit(x_train, y_train)
preview_pred = knn_preview_model.predict(x_test)

preview_df = pd.DataFrame(
    {
        "Actual": y_test.values[:preview_count],
        "Predicted (KNN)": preview_pred[:preview_count],
        "Absolute Error": np.abs(y_test.values[:preview_count] - preview_pred[:preview_count]),
    }
)

st.dataframe(preview_df.style.format("{:.2f}"), use_container_width=True)

st.caption(
    "Pipeline replicated from the notebook: missing-value imputation, IQR outlier filtering, encoding, scaling, train/test split, and model comparison."
)
