# ML Research — California Housing Prices

> Student research project (NIRS), Machine Learning Technologies course, BMSTU · Group IU5-64B

A comparative study of machine learning models on the California Housing dataset (1990 census). The notebook covers both a binary classification task (expensive vs. affordable housing) and a regression task (median house value prediction), with full EDA, feature engineering, hyperparameter tuning, and model comparison.

## Tasks

| Task | Target | Description |
|---|---|---|
| Classification | `price_class` | Above-median price → 1, otherwise → 0 |
| Regression | `MedHouseVal` | Median house value in $100k units |

## Notebook Contents

1. Dataset description and problem formulation
2. Data loading and preprocessing
3. Exploratory data analysis — pair plots, violin charts, geographic price map
4. Feature scaling (MinMaxScaler)
5. Correlation analysis
6. Baseline models (no tuning): LogR, KNN, SVC, DecisionTree, RandomForest, GradientBoosting
7. Hyperparameter search (GridSearchCV, 5-fold cross-validation)
8. Comparison of baseline vs. optimized models

## Results

**Best regression models (R²)**

| Model | MAE | MSE | R² |
|---|---|---|---|
| RandomForest | 0.325 | 0.243 | **0.817** |
| GradientBoosting | 0.362 | 0.265 | 0.800 |
| LinearRegression | 0.534 | 0.512 | 0.614 |

**Best classification models:** GradientBoosting and RandomForest achieved the highest ROC AUC after tuning.

## Running

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook nirs_california_housing.ipynb
```

Dataset source: `sklearn.datasets.fetch_california_housing`
