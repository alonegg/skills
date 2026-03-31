# Structured / Tabular Data Analysis Reference

Practical reference for end-to-end ML on structured data using scikit-learn, pandas, and numpy.

---

## 1. Data Profiling Checklist

Run these checks before any modeling work.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

# Shape and memory
print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Column types
print(df.dtypes.value_counts())

# Missing values — count and percentage
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
pd.DataFrame({"count": missing, "pct": missing_pct}).query("count > 0").sort_values("pct", ascending=False)

# Cardinality of categorical columns
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Target distribution (classification)
print(df["target"].value_counts(normalize=True))

# Target distribution (regression)
print(df["target"].describe())
```

**Checklist summary:**

| Check | Why it matters |
|---|---|
| Shape | Determines whether you need sampling or can fit in memory |
| Dtypes | Catches columns stored as wrong type (numeric as string) |
| Missing pattern | Drives imputation strategy |
| Cardinality | High-cardinality categoricals need special encoding |
| Target distribution | Imbalanced targets need resampling or weighted loss |
| Duplicates | Can inflate model performance if split across train/test |

---

## 2. EDA Patterns

### Univariate

```python
import matplotlib.pyplot as plt

# Numeric columns — histograms + box plots
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    df[col].hist(bins=50, ax=axes[0])
    axes[0].set_title(f"{col} — histogram")
    df[[col]].boxplot(ax=axes[1])
    axes[1].set_title(f"{col} — box plot")
    plt.tight_layout()
    plt.show()

# Categorical columns — value counts
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col].value_counts().head(20).plot.barh(title=f"{col} — top 20 values")
    plt.show()
```

### Bivariate

```python
# Correlation matrix (numeric only)
corr = df[numeric_cols].corr()
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# Scatter plots against target (regression)
for col in numeric_cols:
    if col != "target":
        plt.scatter(df[col], df["target"], alpha=0.3, s=5)
        plt.xlabel(col)
        plt.ylabel("target")
        plt.title(f"{col} vs target")
        plt.show()

# Grouped statistics (classification)
for col in numeric_cols:
    print(df.groupby("target")[col].agg(["mean", "median", "std"]))
```

### Target Analysis

```python
# Classification — class balance
counts = df["target"].value_counts()
ratio = counts.min() / counts.max()
print(f"Minority/majority ratio: {ratio:.3f}")
if ratio < 0.2:
    print("WARNING: severe class imbalance detected")

# Regression — distribution shape
from scipy import stats
skew = df["target"].skew()
print(f"Target skewness: {skew:.2f}")
if abs(skew) > 1:
    print("Consider log-transforming the target")
```

### Automated EDA

```python
# ydata-profiling (formerly pandas-profiling)
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
```

---

## 3. Missing Data Handling

### Detection Heuristics

| Mechanism | Definition | Heuristic test |
|---|---|---|
| MCAR | Missingness is random | Little's test; compare means of observed vs missing groups — no significant difference |
| MAR | Missingness depends on other observed columns | Logistic regression predicting missingness from other features — significant predictors exist |
| MNAR | Missingness depends on the missing value itself | Domain knowledge required; often seen in income, health data |

```python
# Quick MAR check: does missingness in col_A correlate with col_B values?
df["col_A_missing"] = df["col_A"].isnull().astype(int)
print(df.groupby("col_A_missing")["col_B"].mean())
# Large difference suggests MAR
```

### Imputation Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Mean / median (numeric, MCAR)
imp_median = SimpleImputer(strategy="median")

# Mode (categorical)
imp_mode = SimpleImputer(strategy="most_frequent")

# KNN imputation (MAR, moderate dataset size)
imp_knn = KNNImputer(n_neighbors=5, weights="distance")

# Iterative (MICE-style, best for MAR with complex patterns)
imp_iter = IterativeImputer(max_iter=10, random_state=42)

# Indicator column — preserves the information that a value was missing
from sklearn.impute import MissingIndicator
indicator = MissingIndicator(features="missing-only")
```

**When to use each:**

| Strategy | Best for | Watch out for |
|---|---|---|
| Drop rows | < 5% missing, MCAR | Loses data; biases sample if not MCAR |
| Drop columns | > 60% missing, low importance | Loses signal if column matters |
| Mean/median | MCAR, numeric, quick baseline | Distorts variance and correlations |
| Mode | Categorical with dominant category | Amplifies majority class |
| KNN | MAR, < 50k rows, numeric | Slow on large data; scale features first |
| Iterative | MAR, complex feature interactions | Slow; can overfit on small data |
| Indicator column | MNAR suspected | Use alongside another imputation method |

---

## 4. Feature Engineering Recipes

### Numeric Features

```python
# Log transform — reduces right skew
df["price_log"] = np.log1p(df["price"])  # log1p handles zeros

# Binning — discretize continuous variable
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 55, 100],
                       labels=["child", "young", "middle", "senior"])

# Polynomial and interaction terms
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(df[["feat_a", "feat_b"]])

# Manual interaction
df["area"] = df["length"] * df["width"]
```

### Categorical Features

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

# Label encoding — ordinal categoricals only (e.g., low/medium/high)
le = LabelEncoder()
df["size_enc"] = le.fit_transform(df["size"])

# One-hot — low cardinality (< 15 categories)
df_ohe = pd.get_dummies(df, columns=["color"], drop_first=True)

# Target encoding — high cardinality, supervised
te = TargetEncoder(cols=["zip_code"], smoothing=10)
df["zip_te"] = te.fit_transform(df["zip_code"], df["target"])

# Frequency encoding — unsupervised alternative to target encoding
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)
```

### Datetime Features

```python
df["dt"] = pd.to_datetime(df["timestamp"])

# Component extraction
df["hour"] = df["dt"].dt.hour
df["day_of_week"] = df["dt"].dt.dayofweek
df["month"] = df["dt"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Cyclical encoding — preserves periodicity (hour 23 is close to hour 0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lag features (time series in tabular form)
df = df.sort_values("dt")
df["sales_lag_1"] = df["sales"].shift(1)
df["sales_lag_7"] = df["sales"].shift(7)
df["sales_rolling_7"] = df["sales"].rolling(7).mean()
```

### Text Columns in Tabular Data

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF — sparse matrix, append to feature set
tfidf = TfidfVectorizer(max_features=200, stop_words="english")
text_features = tfidf.fit_transform(df["description"])

# Sentence embeddings (dense, higher quality)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["description"].tolist())
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
```

### Feature Selection

```python
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression, Lasso

# Mutual information (works for any relationship, not just linear)
mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_series.head(20))

# Recursive Feature Elimination with cross-validation
estimator = LogisticRegression(max_iter=1000, penalty="l2")
selector = RFECV(estimator, step=1, cv=5, scoring="roc_auc", n_jobs=-1)
selector.fit(X, y)
selected = X.columns[selector.support_].tolist()
print(f"Selected {len(selected)} features: {selected}")

# L1 regularization — embedded feature selection
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
importance = pd.Series(np.abs(lasso.coef_), index=X.columns)
selected_l1 = importance[importance > 0].index.tolist()
```

---

## 5. Task-Specific Pipelines

### Binary Classification (with class imbalance handling)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

numeric_features = ["age", "income"]
categorical_features = ["city", "gender"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# Option A: SMOTE oversampling (use imblearn Pipeline)
pipe_smote = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", GradientBoostingClassifier(random_state=42)),
])

# Option B: class weights (no resampling needed)
pipe_weighted = Pipeline([
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=42)),
])
# For models that support it, use sample_weight or class_weight
# XGBoost: scale_pos_weight = count_negative / count_positive

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_smote, X, y, cv=cv, scoring="roc_auc")
print(f"AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Multi-Class Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=42)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
# Key metrics: macro-averaged F1 (treats all classes equally),
# weighted F1 (accounts for class size)
```

### Regression

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pipe = Pipeline([
    ("prep", preprocessor),
    ("reg", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                       max_depth=5, random_state=42)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R2:   {r2_score(y_test, y_pred):.4f}")

# If target is skewed, train on log(target) and exponentiate predictions
y_train_log = np.log1p(y_train)
pipe.fit(X_train, y_train_log)
y_pred_log = pipe.predict(X_test)
y_pred_actual = np.expm1(y_pred_log)
```

### Multi-Label Classification

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# y_train shape: (n_samples, n_labels), each column is binary
multi_clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, random_state=42),
    n_jobs=-1,
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", multi_clf),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Metrics: use per-label and aggregated scores
from sklearn.metrics import f1_score
print(f"Micro F1: {f1_score(y_test, y_pred, average='micro'):.3f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
```

### Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method + silhouette analysis
inertias, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_, sample_size=5000))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, "o-")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(K_range, sil_scores, "o-")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Score")
plt.tight_layout()
plt.show()

best_k = K_range[np.argmax(sil_scores)]
print(f"Best k by silhouette: {best_k}")
```

### Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest — good default for tabular data
iso = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_iso"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal

# Local Outlier Factor — density-based
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df["anomaly_lof"] = lof.fit_predict(X_scaled)

# Inspect anomalies
anomalies = df[df["anomaly_iso"] == -1]
print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)")
print(anomalies.describe())
```

### Time Series with Tabular Features

```python
# Create lag and rolling features, then use standard tabular models
def create_ts_features(df, target_col, lags, windows):
    """Add lag and rolling statistics to a time-sorted dataframe."""
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    for win in windows:
        df[f"{target_col}_roll_mean_{win}"] = df[target_col].rolling(win).mean()
        df[f"{target_col}_roll_std_{win}"] = df[target_col].rolling(win).std()
    return df

df = df.sort_values("date")
df = create_ts_features(df, "sales", lags=[1, 7, 14, 28], windows=[7, 14, 30])
df = df.dropna()  # rows without enough history

# Time-based split — never shuffle time series
split_date = "2025-01-01"
train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

X_train = train.drop(columns=["sales", "date"])
y_train = train["sales"]
X_test = test.drop(columns=["sales", "date"])
y_test = test["sales"]

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
```

---

## 6. Common Pitfalls

### Data Leakage Patterns

**Preprocessing before splitting.** Fitting scalers, imputers, or encoders on the full dataset (including test data) leaks information from test into train.

```python
# WRONG — leaks test statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # fit on ALL data
X_train, X_test = X_scaled[:800], X_scaled[800:]

# CORRECT — fit only on train
X_train_raw, X_test_raw = X[:800], X[800:]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)  # fit on train only
X_test = scaler.transform(X_test_raw)        # transform test

# BEST — use a Pipeline so this is automatic
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier()),
])
# cross_val_score with this pipeline handles the split correctly
```

**Target encoding leakage.** Computing target encoding on train+test, or without internal cross-validation, leaks the target.

```python
# Use category_encoders with internal CV
from category_encoders import TargetEncoder
te = TargetEncoder(cols=["city"], smoothing=10)
# Always fit on train only
te.fit(X_train["city"], y_train)
X_train["city_enc"] = te.transform(X_train["city"])
X_test["city_enc"] = te.transform(X_test["city"])
```

### Feature-Target Leakage

Features that are created from or after the target event:

- A "days_since_cancellation" feature when predicting cancellation
- An "account_status" column that already encodes the outcome
- Aggregated features computed over a window that includes the prediction date

**How to detect:** Check if any single feature gives suspiciously high AUC (> 0.95). Inspect its definition and data generation process.

### Simpson's Paradox

A trend that appears in aggregated data reverses when the data is split by a confounding group.

```python
# Example: treatment appears harmful overall but helpful within each group
print(df.groupby("treatment")["outcome"].mean())          # misleading
print(df.groupby(["treatment", "severity"])["outcome"].mean())  # true picture
```

**Mitigation:** Always check results stratified by key subgroups before drawing conclusions.

### Temporal Leakage in Time-Dependent Data

Using future information to predict the past:

- Random train/test split on time series data (use time-based split instead)
- Lag features computed on the unsorted dataframe
- Rolling statistics that include the current row

```python
# WRONG — rolling mean includes current value
df["roll_mean"] = df["sales"].rolling(7).mean()

# CORRECT — shift to exclude current value
df["roll_mean"] = df["sales"].shift(1).rolling(7).mean()

# WRONG — random split on time data
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2)  # shuffles time

# CORRECT — time-based split
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

---

## Quick Reference: Choosing a Baseline Model

| Task | Fast baseline | Stronger default |
|---|---|---|
| Binary classification | LogisticRegression | GradientBoostingClassifier / XGBClassifier |
| Multi-class | LogisticRegression(multi_class="multinomial") | RandomForestClassifier |
| Regression | Ridge | GradientBoostingRegressor / XGBRegressor |
| Clustering | KMeans | HDBSCAN |
| Anomaly detection | IsolationForest | LocalOutlierFactor |
| Time series (tabular) | LinearRegression + lag features | LightGBM + lag/rolling features |
