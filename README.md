# KNN Algorithm Lab

A hands-on machine learning mini-project that predicts `target` values from customer and loan-related features.

This repo now includes:

- A notebook-driven workflow (`Task-1.ipynb`) for analysis and experimentation.
- An interactive Streamlit app (`streamlit.py`) for quick training, comparison, and visualization.

---

## Why This Project

This project demonstrates a full regression pipeline, not just model fitting:

- Missing value treatment
- Outlier filtering with IQR
- Categorical encoding
- Feature scaling
- Train/test evaluation
- Model benchmarking (KNN vs Linear Regression vs Decision Tree)
- K tuning with an elbow curve

---

## Project Structure

```text
KNN Tekworks/
|- Task-1.ipynb
|- task1_dataset.csv
|- streamlit.py
|- README.md
```

---

## Dataset Overview

The dataset includes mixed feature types:

- Numerical: `age`, `income`, `loan_amount`, `credit_score`, `num_transactions`, `annual_spend`
- Categorical: `city`, `employment_type`, `loan_type`
- Date: `date`
- Target: `target`

Example use case: estimate a continuous loan-related output from demographic, financial, and behavioral features.

---

## Pipeline Implemented

Both notebook and app follow the same core logic:

1. Load CSV data.
2. Fill missing values (median) for:
   - `income`
   - `loan_amount`
   - `credit_score`
   - `annual_spend`
3. Remove outliers using IQR for:
   - `income`
   - `loan_amount`
   - `credit_score`
   - `num_transactions`
   - `annual_spend`
4. Convert `date` into `month`, `day`, `year` and drop `date`.
5. One-hot encode categorical columns.
6. Scale selected numeric columns:
   - MinMaxScaler: `income`, `loan_amount`
   - StandardScaler: `credit_score`
7. Split data into train/test sets.
8. Train and evaluate:
   - KNN Regressor
   - Linear Regression
   - Decision Tree Regressor
9. Compare models using:
   - Mean Squared Error (MSE)
   - R2 score
10. Tune KNN (`k = 1...20`) with elbow plot.

---

## Streamlit App Features

The app in `streamlit.py` includes:

- Upload your own CSV or use `task1_dataset.csv` by default.
- Sidebar controls for K value, test size, and random state.
- Dataset snapshot with quick health metrics.
- Model comparison table (sorted by MSE).
- Best model highlight.
- KNN elbow curve visualization.
- Prediction preview table (actual vs predicted + absolute error).

---

## Quick Start

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 3. Run the Streamlit app

```powershell
streamlit run streamlit.py
```

Open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## Notebook Workflow

To continue with notebook-based experimentation:

1. Open `Task-1.ipynb` in VS Code or Jupyter.
2. Ensure required packages are installed.
3. Run cells in order from top to bottom.
4. Compare outputs and plots with Streamlit results.

---

## Recommended Improvements

If you want to push this further, these are strong next steps:

- Add cross-validation for more robust model selection.
- Use `Pipeline` and `ColumnTransformer` to avoid leakage and simplify preprocessing.
- Add hyperparameter search (`GridSearchCV`) for KNN and Decision Tree.
- Save trained model artifacts with `joblib`.
- Add SHAP/permutation importance for explainability.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- Streamlit

---

## License

Use freely for learning and experimentation.
