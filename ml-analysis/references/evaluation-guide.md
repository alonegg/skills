# Model Evaluation Guide

Comprehensive reference for rigorous model assessment, explainability, and fairness auditing.

## Table of Contents
1. [Metrics by Task Type](#metrics-by-task-type)
2. [Validation Strategies](#validation-strategies)
3. [Diagnostic Visualizations](#diagnostic-visualizations)
4. [Explainability (XAI)](#explainability-xai)
5. [Fairness Audit](#fairness-audit)
6. [Error Analysis](#error-analysis)
7. [Overfitting/Underfitting Diagnosis](#overfittingunderfitting-diagnosis)

---

## Metrics by Task Type

### Binary Classification

| Metric | When to Use | Formula |
|--------|-------------|---------|
| **Accuracy** | Only if classes are balanced | (TP+TN)/(TP+TN+FP+FN) |
| **Precision** | When false positives are costly (spam filter) | TP/(TP+FP) |
| **Recall** | When false negatives are costly (disease detection) | TP/(TP+FN) |
| **F1** | Balance precision and recall | 2·P·R/(P+R) |
| **AUC-ROC** | Overall ranking ability, threshold-independent | Area under ROC curve |
| **AUC-PR** | Imbalanced datasets (prefer over ROC) | Area under PR curve |
| **Log Loss** | When probability calibration matters | -mean(y·log(p)+(1-y)·log(1-p)) |
| **MCC** | Imbalanced datasets, single summary metric | Matthews correlation coefficient |

```python
from sklearn.metrics import (classification_report, roc_auc_score,
    average_precision_score, log_loss, matthews_corrcoef, confusion_matrix)

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_prob):.4f}")
print(f"MCC:     {matthews_corrcoef(y_test, y_pred):.4f}")
```

### Multi-Class Classification

- **Macro avg**: treats all classes equally (use when all classes matter)
- **Micro avg**: weighted by support (same as accuracy for single-label)
- **Weighted avg**: weighted by class frequency (use when class importance ∝ frequency)
- **Top-K accuracy**: for many classes, check if correct label is in top K predictions

### Regression

| Metric | When to Use | Sensitive To |
|--------|-------------|-------------|
| **RMSE** | General purpose, penalizes large errors | Outliers |
| **MAE** | Robust to outliers | Less sensitive |
| **MAPE** | Percentage errors, interpretable | Near-zero values (divide by zero) |
| **R²** | Proportion of variance explained | Scale-independent |
| **Adjusted R²** | Multiple features (penalizes complexity) | |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
```

### Clustering

| Metric | Needs Labels? | Use For |
|--------|--------------|---------|
| **Silhouette** | No | Cluster separation quality (-1 to 1) |
| **Calinski-Harabasz** | No | Higher is better, ratio of between/within variance |
| **Davies-Bouldin** | No | Lower is better, cluster similarity measure |
| **Adjusted Rand Index** | Yes | Agreement with ground truth |
| **NMI** | Yes | Normalized mutual information with ground truth |

### Time Series

- **MASE** (Mean Absolute Scaled Error): scale-free, compared to naive forecast
- **SMAPE** (Symmetric MAPE): handles near-zero better than MAPE
- **Coverage**: for prediction intervals, % of actuals within bounds
- **Winkler score**: sharpness + coverage of prediction intervals

---

## Validation Strategies

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified for classification (preserves class distribution)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Which K-Fold to use:**
| Variant | When |
|---------|------|
| `KFold` | Regression, balanced classification |
| `StratifiedKFold` | Imbalanced classification |
| `GroupKFold` | Data has groups (e.g., same patient, same user) |
| `RepeatedStratifiedKFold` | Want tighter confidence intervals |
| `TimeSeriesSplit` | Temporal data |

### Time Series Validation

Never use random splits for time series. Always respect temporal order.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # Train always comes before test in time
```

**Expanding window**: training set grows each fold (default TimeSeriesSplit)
**Sliding window**: fixed-size training window moves forward

### Nested Cross-Validation

For unbiased estimation when tuning hyperparameters:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = GridSearchCV(estimator=model, param_grid=params, cv=inner_cv, scoring='f1')
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='f1')
print(f"Nested CV F1: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

### Bootstrap Confidence Intervals

```python
from sklearn.utils import resample

def bootstrap_metric(y_true, y_pred, metric_fn, n_iter=1000, ci=0.95):
    scores = []
    for _ in range(n_iter):
        idx = resample(range(len(y_true)), n_samples=len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper
```

---

## Diagnostic Visualizations

### Learning Curves

Diagnose underfitting vs overfitting by varying training set size:

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel('Training Size')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Learning Curve')
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
```

**Interpreting:**
- Gap between train and val → overfitting (more data or regularization)
- Both curves plateau low → underfitting (more features or complex model)
- Converging at high score → good fit

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
```

### ROC and Precision-Recall Curves

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0])
axes[0].set_title('ROC Curve')
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
axes[1].set_title('Precision-Recall Curve')
plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
```

### Residual Plots (Regression)

```python
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# Residual distribution
axes[1].hist(residuals, bins=30, edgecolor='black')
axes[1].set_title('Residual Distribution')

# Residuals vs Predicted (check for patterns)
axes[2].scatter(y_pred, residuals, alpha=0.5)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Residual')
axes[2].set_title('Residuals vs Predicted')

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=150, bbox_inches='tight')
```

### Calibration Plot

```python
from sklearn.calibration import CalibrationDisplay

CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10)
plt.title('Calibration Curve')
plt.savefig('calibration.png', dpi=150, bbox_inches='tight')
```

---

## Explainability (XAI)

### SHAP — Universal Explainability

**TreeExplainer** (fast, exact for tree models):
```python
import shap

explainer = shap.TreeExplainer(model)  # XGBoost, LightGBM, RF
shap_values = explainer.shap_values(X_test)

# Summary plot: feature importance + direction
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')

# Dependence plot: single feature effect
shap.dependence_plot('feature_name', shap_values, X_test)

# Force plot: explain single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**KernelExplainer** (model-agnostic, slower):
```python
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50])  # Subsample for speed
```

**DeepExplainer** (neural networks):
```python
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:50])
```

### Permutation Feature Importance

Model-agnostic, reliable (preferred over impurity-based for tree models):

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 8))
plt.boxplot(result.importances[sorted_idx[:20]].T, vert=False,
            labels=feature_names[sorted_idx[:20]])
plt.title('Permutation Feature Importance')
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
```

### Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

features_to_plot = [0, 1, (0, 1)]  # individual + interaction
PartialDependenceDisplay.from_estimator(model, X_train, features_to_plot,
    feature_names=feature_names)
plt.savefig('pdp.png', dpi=150, bbox_inches='tight')
```

### Deep Learning Explainability

**Grad-CAM** (for CNN image models):
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
grayscale_cam = cam(input_tensor=input_tensor)
visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
```

**Attention Visualization** (for transformers):
```python
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # List of attention matrices per layer
# Visualize last layer attention
import seaborn as sns
sns.heatmap(attentions[-1][0].mean(dim=0).detach().numpy())
```

---

## Fairness Audit

### When to Audit

Audit whenever the model's predictions affect people and protected attributes (gender, race, age, disability) are present or inferable from the data.

### Fairness Metrics

```python
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, f1_score

metric_frame = MetricFrame(
    metrics={'accuracy': accuracy_score, 'f1': f1_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_test
)

print(metric_frame.by_group)
print(f"Demographic parity diff: {demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test):.4f}")
print(f"Equalized odds diff: {equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test):.4f}")
```

### Key Fairness Concepts

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| **Demographic Parity** | P(ŷ=1\|A=a) = P(ŷ=1\|A=b) | Diff < 0.1 (80% rule) |
| **Equalized Odds** | Same TPR and FPR across groups | Diff < 0.1 |
| **Calibration** | P(y=1\|ŷ=p, A=a) = p for all groups | |

### Mitigation Strategies

1. **Pre-processing**: resampling, reweighting training data
2. **In-processing**: adversarial debiasing, fairness constraints
3. **Post-processing**: threshold adjustment per group

```python
from fairlearn.postprocessing import ThresholdOptimizer

postprocessed = ThresholdOptimizer(
    estimator=model, constraints="equalized_odds", objective="balanced_accuracy_score"
)
postprocessed.fit(X_train, y_train, sensitive_features=sensitive_train)
y_pred_fair = postprocessed.predict(X_test, sensitive_features=sensitive_test)
```

---

## Error Analysis

### Systematic Error Analysis Framework

1. **Slice the data** by feature values and compute per-slice metrics
2. **Find worst slices** — where does the model fail most?
3. **Analyze patterns** — what do misclassified examples have in common?
4. **Iterate** — use insights to improve features or training

```python
def slice_analysis(model, X_test, y_test, feature_name, bins=5):
    """Compute metrics per data slice."""
    if X_test[feature_name].dtype in ['float64', 'int64']:
        slices = pd.qcut(X_test[feature_name], bins, duplicates='drop')
    else:
        slices = X_test[feature_name]

    y_pred = model.predict(X_test)
    results = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'slice': slices})

    for name, group in results.groupby('slice'):
        acc = (group['y_true'] == group['y_pred']).mean()
        print(f"Slice {name}: accuracy={acc:.3f}, n={len(group)}")
```

### Worst Predictions Analysis

```python
# For regression: find largest errors
errors = np.abs(y_test - y_pred)
worst_idx = errors.argsort()[-20:][::-1]
print("Worst predictions:")
print(X_test.iloc[worst_idx])

# For classification: find confident wrong predictions
wrong_mask = y_pred != y_test
confident_wrong = y_prob[wrong_mask].argsort()[-20:][::-1]
```

---

## Overfitting/Underfitting Diagnosis

### Quick Diagnosis Table

| Symptom | Diagnosis | Solutions |
|---------|-----------|-----------|
| Train high, Val high | Good fit | Ship it |
| Train high, Val low | Overfitting | More data, regularization, simpler model, dropout, early stopping |
| Train low, Val low | Underfitting | More features, complex model, less regularization, more training |
| Train low, Val high | Bug or data issue | Check data pipeline, look for leakage |

### Regularization Toolkit

| Technique | Works For | How |
|-----------|----------|-----|
| L1 (Lasso) | Linear models | Feature selection, sparsity |
| L2 (Ridge) | Linear models, NN | Weight shrinkage |
| Dropout | Neural networks | Random neuron deactivation |
| Early stopping | Any iterative model | Stop when val loss stops improving |
| Data augmentation | CV, NLP | Increase effective dataset size |
| Ensemble | Any | Reduce variance via averaging |
| Max depth / min samples | Trees | Limit tree complexity |
| Learning rate decay | Neural networks | Gradual convergence |
