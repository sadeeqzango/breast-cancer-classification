# Breast Cancer Classification (Benign vs Malignant)

A machine learning project that classifies breast tumours as **benign** or **malignant** from measurements of cell nuclei, comparing three models and saving the best one for reuse.

> ⚠️ **Disclaimer:** This is a learning/portfolio project. It is not a medical device and must not be used for real diagnosis.

## Results

The final model is a **Random Forest**, which scored highest on the held-out test set (114 samples).

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 96.49%   | 97.50%    | 92.86% | 95.12%   |
| Decision Tree       | 92.98%   | 90.48%    | 90.48% | 90.48%   |
| **Random Forest**   | **97.37%** | **100%** | 92.86% | **96.30%** |

**Reading the results:** The Random Forest had no false positives (precision 100%), but its recall of 92.86% means it missed 3 of the 42 malignant cases in the test set. In a medical setting, missed malignant cases are the costlier error, so improving recall would be the first priority for future work.

**Top predictive features (Random Forest):** `area_worst`, `concave_points_worst`, `radius_worst`, `perimeter_worst`, `concave_points_mean`.

## Dataset

- **File:** `data/breast_cancer.csv`
- **Size:** 569 samples, 30 numeric features, 1 target (`diagnosis`)
- **Target:** `0` = benign (357 samples), `1` = malignant (212 samples)
- **Features:** For each cell nucleus, 10 measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as the **mean**, **standard error (se)** and **worst** value.
- No missing values and no duplicate rows.

This matches the Breast Cancer Wisconsin (Diagnostic) dataset available from the UCI Machine Learning Repository.

## Approach

1. **Data exploration:** shape, data types, missing values, duplicates, class balance, boxplots, correlation heatmap
2. **Train/test split:** 80/20, stratified, `random_state=42`
3. **Preprocessing:** `StandardScaler` (fit on training data only) for Logistic Regression
4. **Modelling:** Logistic Regression, Decision Tree, Random Forest (100 trees)
5. **Evaluation:** accuracy, precision, recall, F1, classification reports, confusion matrices
6. **Interpretation:** Random Forest feature importances
7. **Deployment step:** best model saved with `joblib` and reloaded to predict on a new patient record

## Project Structure

```
.
├── data/
│   └── breast_cancer.csv
├── models/
│   └── breast_cancer_random_forest.pkl
├── Breast_Cancer_Classification.ipynb
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook Breast_Cancer_Classification.ipynb
```

> The notebook reads `breast_cancer.csv` from its own folder. If you keep the data in `data/`, change the path in the first code cell to `data/breast_cancer.csv`.

## Using the Saved Model

The Random Forest model expects the 30 features in the same column order as the dataset (everything except `diagnosis`).

```python
import joblib
import pandas as pd

model = joblib.load("models/breast_cancer_random_forest.pkl")

df = pd.read_csv("data/breast_cancer.csv")
sample = df.drop("diagnosis", axis=1).iloc[[0]]

print(model.predict(sample))          # 0 = benign, 1 = malignant
print(model.predict_proba(sample))    # [[P(benign), P(malignant)]]
```

Only load `.pkl` files from sources you trust, since pickle files can execute code when loaded.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter · joblib

## Limitations and Next Steps

- Small dataset (569 rows) evaluated on a single train/test split, so results can vary
- Add cross-validation for more reliable estimates
- Tune hyperparameters (e.g. `GridSearchCV`) and try other models (SVM, Gradient Boosting)
- Optimise for recall, for example by adjusting the decision threshold
- Wrap the model in a small web app (Streamlit or Flask)

## Author

**Abubakar Hamisu Zango**
Kano, Nigeria · [LinkedIn](https://linkedin.com/in/<your-handle>) · [GitHub](https://github.com/<your-username>)
