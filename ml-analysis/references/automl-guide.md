# AutoML Integration Guide

Reference for automated machine learning frameworks — when to use each, setup, and best practices.

## Framework Selection

| Framework | Best For | Speed | Ease | Interpretability |
|-----------|---------|-------|------|-----------------|
| **AutoGluon** | Tabular, general-purpose | Medium | Very easy | Good (built-in) |
| **H2O AutoML** | Enterprise, large data | Fast | Easy | Good (built-in) |
| **Optuna** | Custom tuning, any model | Varies | Medium | N/A (tuner only) |
| **Auto-sklearn** | sklearn ecosystem | Slow | Easy | Limited |
| **FLAML** | Fast, resource-constrained | Very fast | Easy | Limited |

**Decision guide:**
- "Just find me the best model" → AutoGluon
- "Need production-ready with monitoring" → H2O
- "I have a specific model, tune it" → Optuna
- "Only sklearn models" → Auto-sklearn
- "Fast results, limited compute" → FLAML

---

## AutoGluon

Best general-purpose AutoML. Stacks multiple models automatically.

### Basic Usage

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label='target_column',
    eval_metric='f1',  # or 'rmse', 'roc_auc', etc.
    path='./ag_models'
)

predictor.fit(
    train_data=train_df,
    time_limit=600,  # 10 minutes
    presets='best_quality'  # or 'medium_quality', 'good_quality'
)

# Evaluate
results = predictor.evaluate(test_df)
leaderboard = predictor.leaderboard(test_df)
print(leaderboard)
```

### Presets

| Preset | Quality | Speed | Use When |
|--------|---------|-------|----------|
| `best_quality` | Highest | Slow | Final model, competition |
| `high_quality` | High | Medium | Good default |
| `good_quality` | Good | Fast | Quick iteration |
| `medium_quality` | Decent | Very fast | Exploration |

### Feature Importance

```python
importance = predictor.feature_importance(test_df)
print(importance)
```

### Custom Model List

```python
predictor.fit(
    train_data=train_df,
    hyperparameters={
        'GBM': {},           # LightGBM
        'XGB': {},           # XGBoost
        'RF': {},            # Random Forest
        'NN_TORCH': {},      # Neural network
    },
    time_limit=300
)
```

---

## H2O AutoML

Enterprise-grade, handles large datasets well, built-in explainability.

### Setup and Usage

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Load data
train = h2o.H2OFrame(train_df)
test = h2o.H2OFrame(test_df)

# Identify columns
target = 'label'
features = [c for c in train.columns if c != target]
train[target] = train[target].asfactor()  # For classification

# Run AutoML
aml = H2OAutoML(
    max_models=20,
    max_runtime_secs=600,
    seed=42,
    sort_metric='AUC'
)
aml.train(x=features, y=target, training_frame=train)

# Leaderboard
lb = aml.leaderboard
print(lb.head(10))

# Best model
best = aml.leader
preds = best.predict(test)
perf = best.model_performance(test)
print(perf)
```

### Model Explanation

```python
# Variable importance
best.varimp_plot()

# SHAP
contrib = best.predict_contributions(test)

# Partial dependence
best.partial_plot(train, cols=['feature1', 'feature2'])
```

### Cleanup

```python
h2o.remove_all()
h2o.cluster().shutdown()
```

---

## Optuna

Most flexible — optimize any objective function. Best for custom models.

### Basic Hyperparameter Tuning

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### LightGBM with Optuna

```python
import lightgbm as lgb

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    cv_results = lgb.cv(params, dtrain, num_boost_round=1000,
                        nfold=5, callbacks=[lgb.early_stopping(50)])
    return cv_results['valid auc-mean'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### PyTorch with Optuna

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = build_model(hidden_size, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = eval_epoch(model, val_loader)

        # Pruning: stop bad trials early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study.optimize(objective, n_trials=50)
```

### Samplers

| Sampler | When to Use |
|---------|-------------|
| `TPESampler` (default) | Most cases, good balance |
| `CmaEsSampler` | Continuous parameters, smooth objective |
| `RandomSampler` | Baseline comparison |
| `GridSampler` | Small parameter space, exhaustive search |

### Visualization

```python
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_parallel_coordinate(study)
optuna.visualization.plot_slice(study)
```

---

## FLAML

Extremely fast, resource-aware. Good when compute is limited.

```python
from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train, y_train,
    task='classification',
    time_budget=120,  # 2 minutes
    metric='f1',
    estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree'],
)

print(f"Best model: {automl.best_estimator}")
print(f"Best config: {automl.best_config}")
y_pred = automl.predict(X_test)
```

---

## Best Practices

### Time Budget Guidelines

| Data Size | Exploration | Production |
|-----------|------------|------------|
| <10K rows | 2-5 min | 10-30 min |
| 10K-100K | 5-15 min | 30-60 min |
| 100K-1M | 15-60 min | 1-4 hours |
| >1M | 1+ hours | 4+ hours |

### Imbalanced Data in AutoML

- AutoGluon: handles automatically with `eval_metric='f1'`
- H2O: set `balance_classes=True` or use `weights_column`
- Optuna: handle in objective function (class weights, SMOTE)

### Interpreting AutoML Results

1. Check leaderboard — is the best model much better than others?
2. Look at ensemble composition — which base models contribute?
3. Check for overfitting: compare train vs validation scores
4. Run the winning model through the full evaluation guide
5. Consider the simplest model within 1-2% of the best — often worth the interpretability tradeoff
