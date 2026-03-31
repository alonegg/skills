# ML/DL Model Catalog and Decision Tree

A practical reference for selecting machine learning and deep learning models by task type. Organized as a decision tree: identify the task, assess constraints, pick the model.

---

## 1. Tabular Data Models

### 1.1 Classification

| Model | When to Use | Strengths | Weaknesses | Key Hyperparameters | Cost |
|---|---|---|---|---|---|
| **LogisticRegression** | Baseline; linearly separable data; need interpretability | Fast, interpretable coefficients, well-calibrated probabilities | Cannot capture nonlinear relationships | `C`, `penalty` (l1/l2/elasticnet), `solver` | Very low |
| **SVM (SVC)** | Small-to-medium datasets; high-dimensional features | Effective in high dimensions; kernel trick for nonlinearity | Slow on large datasets (O(n^2)-O(n^3)); poor probability estimates | `C`, `kernel`, `gamma`, `class_weight` | Medium-high |
| **RandomForest** | General-purpose; noisy data; want feature importances | Robust to outliers, handles mixed types, low tuning effort | Can overfit on noisy data; large memory footprint | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | Medium |
| **XGBoost** | Structured/tabular competitions; need high accuracy | State-of-the-art tabular performance; built-in regularization | Sensitive to hyperparameters; slower than LightGBM on large data | `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` | Medium |
| **LightGBM** | Large datasets; high cardinality categoricals | Fastest gradient boosting; native categorical support; low memory | Can overfit on small datasets; leaf-wise growth needs tuning | `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree` | Low-medium |
| **CatBoost** | Datasets with many categoricals; want minimal preprocessing | Best native categorical handling; ordered boosting reduces overfitting | Slower training than LightGBM; large model size | `depth`, `learning_rate`, `iterations`, `l2_leaf_reg`, `border_count` | Medium |
| **TabNet** | Need interpretability from a neural net; medium-large data | Built-in attention for feature selection; end-to-end differentiable | Harder to tune than GBMs; requires GPU for reasonable speed | `n_steps`, `n_a`, `n_shared`, `learning_rate`, `batch_size`, `virtual_batch_size` | High |
| **EBM (InterpretML)** | Regulated domains; need glass-box model with strong performance | Fully interpretable; near-GBM accuracy; automatic interaction detection | Slower training; less community support | `max_bins`, `interactions`, `learning_rate`, `max_rounds`, `min_samples_leaf` | Medium |

**Classification decision flow:**

1. Need interpretability above all? --> LogisticRegression or EBM
2. Small dataset (<1K rows)? --> LogisticRegression, SVM, or RandomForest
3. Many categorical features? --> CatBoost or LightGBM
4. Large dataset (>100K rows) and need speed? --> LightGBM
5. Default strong choice? --> XGBoost or LightGBM

### 1.2 Regression

| Model | When to Use | Strengths | Weaknesses | Key Hyperparameters | Cost |
|---|---|---|---|---|---|
| **Ridge** | Baseline; multicollinearity present; many features | Closed-form solution; stable with correlated features | Linear only | `alpha` | Very low |
| **Lasso** | Feature selection needed; sparse solutions | Drives coefficients to zero; built-in feature selection | Unstable with correlated features; selects one from a group | `alpha` | Very low |
| **ElasticNet** | Correlated features + want sparsity | Combines Ridge and Lasso benefits; groups correlated features | Two hyperparameters to tune | `alpha`, `l1_ratio` | Very low |
| **SVR** | Small datasets; nonlinear relationships | Kernel trick; robust to outliers with epsilon-insensitive loss | Scales poorly; sensitive to feature scaling | `C`, `epsilon`, `kernel`, `gamma` | Medium-high |
| **RandomForest** | General-purpose; heterogeneous features | Robust; handles nonlinearity; feature importance | Cannot extrapolate beyond training range; high memory | `n_estimators`, `max_depth`, `min_samples_leaf` | Medium |
| **XGBoost** | High accuracy on structured data | Regularized; handles missing values; custom objectives | Cannot extrapolate; hyperparameter-sensitive | Same as classification variant | Medium |
| **LightGBM** | Large datasets; fast iteration needed | Speed; memory efficiency; histogram-based splits | Leaf-wise growth can overfit small data | Same as classification variant | Low-medium |
| **Neural Networks (MLP)** | Very large datasets; complex interactions; part of larger pipeline | Universal approximation; flexible architecture | Needs lots of data; hard to tune; hard to interpret | `hidden_layers`, `learning_rate`, `batch_size`, `dropout`, `weight_decay` | High |

**Regression decision flow:**

1. Linear relationship suspected? --> Ridge/Lasso/ElasticNet
2. Need feature selection? --> Lasso or ElasticNet
3. Large dataset, structured? --> LightGBM or XGBoost
4. Part of an end-to-end deep learning pipeline? --> MLP

---

## 2. NLP Models

### 2.1 Classical NLP

| Model | When to Use | Strengths | Weaknesses |
|---|---|---|---|
| **Bag of Words + Classifier** | Quick baseline; small data; limited compute | Simple, fast, interpretable | Ignores word order; sparse representations |
| **TF-IDF + Classifier** | Document classification baseline; information retrieval | Weights important terms; good with SVM or logistic regression | Still ignores order; vocabulary-dependent |
| **Word2Vec / GloVe** | Need dense word representations; analogy tasks | Captures semantic similarity; pretrained vectors available | Static embeddings (no polysemy); requires aggregation for documents |

### 2.2 Deep Learning NLP

| Model | When to Use | Strengths | Weaknesses | Cost |
|---|---|---|---|---|
| **LSTM / GRU** | Sequence modeling; moderate data; need to process variable-length text | Handles sequences naturally; GRU is faster than LSTM | Sequential processing (slow); vanishing gradients on long sequences | Medium |
| **CNN for Text** | Text classification; fixed-length patterns (n-grams) | Fast; captures local patterns well | Misses long-range dependencies; fixed receptive field | Low-medium |
| **BERT** | Classification, NER, QA; need contextual embeddings | Bidirectional context; strong pretrained representations | Large (110M-340M params); expensive fine-tuning | High |
| **RoBERTa** | Same tasks as BERT; want better performance | Better pretraining procedure than BERT | Even more expensive to pretrain; marginal gains for some tasks | High |
| **DistilBERT** | Need BERT-like quality with lower latency/memory | 60% size of BERT; 97% performance; 60% faster | Still needs GPU for reasonable speed; some quality loss | Medium |
| **GPT-series** | Text generation, completion, few-shot tasks | Strong generative ability; in-context learning | Autoregressive (left-to-right only); very large; expensive | Very high |

### 2.3 Task-Specific Guidance

**Named Entity Recognition (NER):**
- Small data: spaCy rule-based or CRF
- Medium data: fine-tune DistilBERT/BERT with token classification head
- Production: fine-tuned BERT or domain-specific model (BioBERT, SciBERT)

**Sentiment Analysis:**
- Quick baseline: TF-IDF + LogisticRegression
- Better accuracy: fine-tune DistilBERT or BERT
- Zero-shot: use an LLM with prompting

**Text Generation:**
- Short/structured: fine-tuned GPT-2
- High quality: GPT-3.5/4 via API or fine-tuned LLaMA
- Domain-specific: fine-tune on domain corpus

**Summarization:**
- Extractive: TextRank, BertSum
- Abstractive: BART, T5, Pegasus
- Long documents: LED (Longformer Encoder-Decoder), or chunk + summarize

### 2.4 Pretrained vs. Train from Scratch

Use pretrained when:
- Dataset is small to medium (<100K labeled examples)
- Task is similar to what the model was pretrained on
- Compute budget is limited
- Domain is general English (or a language with good pretrained models)

Train from scratch when:
- Domain has highly specialized vocabulary (e.g., molecular SMILES, code)
- Massive domain corpus available (>1B tokens)
- Pretrained tokenizer is poorly suited to the domain
- Privacy requirements prevent using external pretrained weights

---

## 3. Computer Vision Models

### 3.1 Classification

| Model | When to Use | Strengths | Weaknesses | Params | Cost |
|---|---|---|---|---|---|
| **ResNet-50** | General baseline; transfer learning starting point | Well-understood; skip connections; widely supported | Older architecture; not parameter-efficient | 25M | Medium |
| **EfficientNet-B0 to B7** | Need accuracy/efficiency tradeoff; mobile to server | Compound scaling; strong accuracy per FLOP | B5+ are very large; sensitivity to input resolution | 5M-66M | Low-high |
| **ViT (Vision Transformer)** | Large datasets; want attention-based features | State-of-the-art on large data; good for multimodal | Needs lots of data or strong augmentation; less inductive bias | 86M-632M | High |
| **MobileNet v3** | Edge/mobile deployment; real-time inference | Tiny; fast; optimized for mobile hardware | Lower accuracy ceiling; limited capacity | 2-5M | Very low |

### 3.2 Object Detection

| Model | When to Use | Strengths | Weaknesses | Cost |
|---|---|---|---|---|
| **YOLOv8 / YOLOv9** | Real-time detection; production deployment | Fast; good accuracy; easy to deploy; single-stage | Less accurate on small objects than two-stage | Medium |
| **Faster R-CNN** | High accuracy needed; two-stage acceptable | Strong accuracy; well-studied; good for small objects | Slower than YOLO; complex pipeline | High |
| **DETR** | Want end-to-end detection without anchors | No anchor boxes or NMS; clean design; good with transformers | Slow convergence; needs long training | High |

### 3.3 Segmentation

| Model | When to Use | Strengths | Weaknesses | Cost |
|---|---|---|---|---|
| **U-Net** | Medical imaging; semantic segmentation; small datasets | Excellent with limited data; skip connections preserve detail | Fixed input size; basic for modern standards | Medium |
| **Mask R-CNN** | Instance segmentation; need per-object masks | Combines detection + segmentation; mature ecosystem | Heavy; slow inference | High |
| **SAM (Segment Anything)** | Zero-shot segmentation; interactive segmentation | Works on any image without training; prompt-based | Large model; needs prompts (points/boxes); not specialized | Very high |

### 3.4 Transfer Learning Strategies

**Full fine-tuning:** Unfreeze all layers. Use when you have 10K+ labeled images and domain differs significantly from ImageNet.

**Feature extraction:** Freeze backbone, train only classification head. Use when you have <1K images or domain is similar to ImageNet.

**Gradual unfreezing:** Start with head only, progressively unfreeze deeper layers. Best general strategy for medium datasets (1K-10K images).

**Learning rate scheduling:** Use lower learning rates for pretrained layers (1e-5 to 1e-4) and higher for new layers (1e-3 to 1e-2).

### 3.5 Data Augmentation Checklist

- **Always useful:** Random horizontal flip, random crop, color jitter, normalization
- **Often useful:** Random rotation (small angles), Gaussian blur, Cutout/CutMix, MixUp
- **Domain-specific:** Elastic deformation (medical), random erasing (re-identification), mosaic (detection)
- **Advanced:** AutoAugment, RandAugment, TrivialAugment (search-based policies)
- **Caution:** Vertical flip (only if semantically valid), extreme color shifts, heavy geometric transforms

---

## 4. Time Series Models

### 4.1 Statistical Models

| Model | When to Use | Strengths | Weaknesses | Key Parameters |
|---|---|---|---|---|
| **ARIMA** | Univariate; stationary (after differencing); short-term forecasts | Well-understood; good for short horizons; confidence intervals | Manual order selection; no exogenous handling (use ARIMAX); assumes linearity | `p`, `d`, `q` |
| **SARIMA** | Univariate with seasonal patterns | Captures seasonality explicitly; well-established theory | Slow to fit; fixed seasonal period; struggles with multiple seasonalities | `p`, `d`, `q`, `P`, `D`, `Q`, `s` |
| **Prophet** | Business time series; holidays; multiple seasonalities | Handles missing data; automatic changepoints; holiday effects; interpretable | Less accurate than ML on complex patterns; slow with many regressors | `changepoint_prior_scale`, `seasonality_prior_scale`, `holidays_prior_scale` |
| **ETS** | Univariate; need error/trend/seasonal decomposition | Automatic model selection; well-calibrated prediction intervals | Linear only; single seasonality; no exogenous variables | `error` (A/M), `trend` (N/A/Ad), `seasonal` (N/A/M) |

### 4.2 ML-Based Time Series

| Model | When to Use | Strengths | Weaknesses |
|---|---|---|---|
| **XGBoost + Lag Features** | Multivariate; complex nonlinear relationships; engineered features | Handles exogenous variables; captures nonlinearity; fast | Requires feature engineering (lags, rolling stats); no native sequence awareness |
| **LightGBM + Lag Features** | Same as XGBoost; larger datasets | Faster; lower memory; native categorical support | Same feature engineering burden as XGBoost |

Feature engineering for ML time series:
- Lag features: `y(t-1)`, `y(t-2)`, ..., `y(t-n)`
- Rolling statistics: rolling mean, std, min, max over windows
- Calendar features: day of week, month, quarter, holiday flags
- Fourier features: sin/cos terms for cyclical patterns
- Differenced features: `y(t) - y(t-1)` for stationarity

### 4.3 Deep Learning Time Series

| Model | When to Use | Strengths | Weaknesses | Cost |
|---|---|---|---|---|
| **LSTM** | Sequence prediction; moderate-length series; established baseline | Handles variable-length sequences; captures long-range patterns | Slow training; hard to parallelize; vanishing gradients still possible | High |
| **Temporal Fusion Transformer** | Multi-horizon forecasting; mixed static/dynamic features | Interpretable attention; handles known future inputs; state-of-the-art | Complex architecture; needs substantial data; slow to train | Very high |
| **N-BEATS** | Univariate forecasting; no exogenous variables needed | Pure time series focus; interpretable decomposition; strong accuracy | Univariate only (N-BEATSx for exogenous); needs long history | High |
| **PatchTST** | Long-horizon forecasting; channel-independent modeling | Efficient patching strategy; transformer-based; handles long context | Newer (less battle-tested); may underperform on short series | High |

**Time series decision flow:**

1. Single variable, short horizon, clear seasonality? --> SARIMA or ETS
2. Business metrics with holidays? --> Prophet
3. Multiple exogenous features, structured data? --> LightGBM/XGBoost with lag features
4. Long sequences, complex temporal patterns, large data? --> Temporal Fusion Transformer or PatchTST
5. Need interpretable decomposition? --> N-BEATS or Prophet

---

## 5. Clustering and Anomaly Detection

### 5.1 Clustering

| Model | When to Use | Strengths | Weaknesses | Key Parameters |
|---|---|---|---|---|
| **K-Means** | Spherical clusters; known K; large datasets | Fast (O(nK)); simple; scales well | Assumes spherical equal-size clusters; sensitive to initialization; must specify K | `n_clusters`, `init`, `n_init` |
| **DBSCAN** | Arbitrary-shape clusters; noise detection; unknown K | Finds arbitrary shapes; identifies outliers; no need to specify K | Sensitive to `eps` and `min_samples`; struggles with varying densities | `eps`, `min_samples` |
| **HDBSCAN** | Varying-density clusters; robust clustering | Handles varying density; soft clustering; robust to parameters | Slower than DBSCAN; memory-intensive on large data | `min_cluster_size`, `min_samples`, `cluster_selection_method` |
| **Gaussian Mixture (GMM)** | Soft assignments needed; elliptical clusters; probabilistic framework | Soft cluster assignments; models cluster shape; BIC for model selection | Sensitive to initialization; assumes Gaussian components; can overfit | `n_components`, `covariance_type`, `init_params` |
| **Spectral Clustering** | Non-convex clusters; graph-structured data | Handles complex shapes; uses graph Laplacian | Expensive (O(n^3) eigendecomposition); must specify K; does not scale | `n_clusters`, `affinity`, `gamma` |

**Clustering decision flow:**

1. Large dataset, roughly spherical clusters? --> K-Means
2. Unknown number of clusters, arbitrary shapes? --> HDBSCAN
3. Need probabilistic soft assignments? --> GMM
4. Graph-structured data or non-convex shapes, small-to-medium data? --> Spectral Clustering
5. Quick noise filtering + clustering? --> DBSCAN

### 5.2 Anomaly Detection

| Model | When to Use | Strengths | Weaknesses | Key Parameters |
|---|---|---|---|---|
| **Isolation Forest** | General-purpose tabular anomaly detection; high dimensions | Fast; handles high dimensions; no distribution assumptions | Struggles with local anomalies; contamination must be estimated | `n_estimators`, `contamination`, `max_features` |
| **Local Outlier Factor** | Local density-based anomalies; known normal neighborhoods | Detects local outliers; adapts to varying densities | Slow on large data; sensitive to `n_neighbors`; not great for novelty detection | `n_neighbors`, `contamination`, `metric` |
| **One-Class SVM** | Small clean training sets; novelty detection | Works well with kernels; solid theoretical foundation | Slow on large data; sensitive to kernel and nu; needs scaled features | `nu`, `kernel`, `gamma` |
| **Autoencoders** | High-dimensional data (images, sequences); complex normal patterns | Learns complex representations; flexible architecture | Needs careful architecture tuning; threshold selection is manual; can memorize anomalies | `encoding_dim`, `layers`, `learning_rate`, `threshold` |

### 5.3 Dimensionality Reduction

| Model | When to Use | Strengths | Weaknesses |
|---|---|---|---|
| **PCA** | Linear reduction; preprocessing for ML; variance analysis | Fast; deterministic; interpretable loadings; preserves global structure | Linear only; sensitive to feature scaling |
| **t-SNE** | 2D/3D visualization; exploring cluster structure | Excellent local structure preservation; reveals clusters visually | Non-deterministic; slow on large data; cannot transform new points; distorts global distances |
| **UMAP** | Visualization and general-purpose reduction; large datasets | Preserves global and local structure; fast; supports transform of new data | Stochastic; hyperparameter-sensitive; less interpretable than PCA |

**Dimensionality reduction guidance:**
- Preprocessing for ML pipeline: PCA (keep 95% variance) or UMAP
- Visualization only: UMAP (preferred) or t-SNE
- Need deterministic results: PCA
- Need to transform new unseen data: PCA or UMAP (not t-SNE)

---

## 6. Model Selection Quick Reference Table

| Scenario | Recommended Models | Notes |
|---|---|---|
| Tabular classification, small data (<1K) | LogisticRegression, RandomForest, SVM | Start simple; cross-validate |
| Tabular classification, medium data (1K-100K) | XGBoost, LightGBM, CatBoost | Tune with Optuna or similar |
| Tabular classification, large data (>100K) | LightGBM, XGBoost | LightGBM for speed |
| Tabular classification, need interpretability | EBM, LogisticRegression, single decision tree | Use SHAP for post-hoc if using GBMs |
| Tabular regression, linear relationship | Ridge, Lasso, ElasticNet | Check residuals for linearity |
| Tabular regression, nonlinear | XGBoost, LightGBM, RandomForest | Feature engineering matters |
| Text classification, small data | TF-IDF + LogisticRegression/SVM | Fast baseline |
| Text classification, medium-large data | Fine-tune DistilBERT or BERT | Use DistilBERT to save compute |
| Text generation | Fine-tune GPT-2 or use LLM API | Consider cost vs quality tradeoff |
| NER | Fine-tune BERT with token classification | spaCy for quick prototyping |
| Image classification, small data (<1K) | Transfer learning (ResNet-50 or EfficientNet) | Freeze backbone, train head |
| Image classification, medium data | Fine-tune EfficientNet or ViT | Gradual unfreezing |
| Image classification, edge/mobile | MobileNet v3 | Quantization for further speedup |
| Object detection, real-time | YOLOv8 | Single-stage, fast inference |
| Object detection, high accuracy | Faster R-CNN | Two-stage, slower but more accurate |
| Image segmentation, medical | U-Net with pretrained encoder | Data augmentation is critical |
| Instance segmentation | Mask R-CNN | Mature; well-supported |
| Time series, univariate, seasonal | SARIMA, Prophet | Prophet for business series |
| Time series, multivariate, structured | LightGBM/XGBoost with lag features | Feature engineering is key |
| Time series, long-horizon, complex | Temporal Fusion Transformer, PatchTST | Needs GPU and substantial data |
| Clustering, large data, spherical | K-Means | Fast and scalable |
| Clustering, arbitrary shapes | HDBSCAN | Robust to parameter choices |
| Anomaly detection, tabular | Isolation Forest | Good default; fast |
| Anomaly detection, high-dimensional | Autoencoder | Threshold tuning required |
| Dimensionality reduction for ML | PCA | Deterministic; preserves variance |
| Dimensionality reduction for visualization | UMAP | Preserves global + local structure |

---

## 7. Resource Requirements

### 7.1 CPU vs GPU Decision

| Model Category | CPU Viable? | GPU Recommended? | GPU Required? |
|---|---|---|---|
| Linear models (Ridge, Lasso, LogReg) | Yes, always | No | No |
| Tree ensembles (RF, XGBoost, LightGBM) | Yes, up to ~10M rows | Helpful for XGBoost (gpu_hist) | No |
| SVM | Yes, up to ~50K rows | No standard GPU support | No |
| Small neural nets (MLP, TabNet) | Yes, for small data | Yes, for medium+ data | No |
| BERT/Transformers fine-tuning | Possible but very slow | Yes | Practically yes |
| CNN training (ResNet, EfficientNet) | Possible but impractical | Yes | Yes for any real workload |
| ViT, large transformers | No | Yes | Yes |
| LSTM/GRU training | Slow but possible | Yes | For sequences >1000 tokens |
| K-Means, DBSCAN, PCA | Yes | Optional (cuML for speedup) | No |

### 7.2 Memory Estimates

| Model | Training Memory Estimate |
|---|---|
| LogisticRegression / Ridge | ~2-5x dataset size |
| RandomForest (100 trees) | ~5-10x dataset size |
| XGBoost / LightGBM | ~2-4x dataset size |
| BERT fine-tune (base) | ~8-12 GB GPU RAM (batch size 16) |
| BERT fine-tune (large) | ~16-24 GB GPU RAM (batch size 8) |
| ResNet-50 fine-tune | ~4-8 GB GPU RAM (batch size 32) |
| EfficientNet-B4 fine-tune | ~8-12 GB GPU RAM (batch size 16) |
| ViT-Base fine-tune | ~8-16 GB GPU RAM (batch size 16) |
| U-Net (256x256 images) | ~4-8 GB GPU RAM |
| YOLOv8 training | ~6-12 GB GPU RAM |
| LSTM (sequence models) | ~2-8 GB GPU RAM depending on sequence length |

### 7.3 Training Time Heuristics

| Model | 10K Samples | 100K Samples | 1M Samples |
|---|---|---|---|
| LogisticRegression | Seconds | Seconds | Minutes |
| RandomForest (100 trees) | Seconds | Minutes | 10-30 min |
| XGBoost (500 rounds) | Seconds | 1-5 min | 10-60 min |
| LightGBM (500 rounds) | Seconds | 30s-2min | 5-30 min |
| SVM (RBF kernel) | Seconds | 10-60 min | Impractical |
| MLP (small) | Seconds | Minutes | 10-30 min (GPU) |
| BERT fine-tune (3 epochs) | 5-15 min (GPU) | 1-3 hours (GPU) | 10-30 hours (GPU) |
| ResNet-50 fine-tune (10 epochs) | 5-10 min (GPU) | 30-60 min (GPU) | 5-10 hours (GPU) |
| K-Means (K=10) | Seconds | Seconds | 1-5 min |
| HDBSCAN | Seconds | 1-10 min | 30-120 min |

All estimates assume reasonable hardware (modern CPU, single mid-range GPU like RTX 3080/A10). Actual times vary with feature count, hyperparameters, and implementation.

### 7.4 Reducing Resource Usage

- **Gradient boosting:** Use `subsample` (0.7-0.9) and `colsample_bytree` (0.7-0.9) to reduce memory and time
- **Neural nets:** Use mixed precision (fp16) to halve GPU memory; gradient accumulation for effective larger batches
- **Transformers:** Use DistilBERT instead of BERT; use LoRA/QLoRA for parameter-efficient fine-tuning
- **Large datasets:** Sample for hyperparameter search, train final model on full data
- **General:** Use early stopping to avoid wasted compute; profile before optimizing

---

## Appendix: Library Quick Reference

| Task | Primary Libraries |
|---|---|
| Tabular ML | scikit-learn, xgboost, lightgbm, catboost, interpret (EBM) |
| NLP | transformers (HuggingFace), spaCy, nltk, sentence-transformers |
| Computer Vision | torchvision, timm, ultralytics (YOLO), segmentation-models-pytorch |
| Time Series | statsmodels, prophet, darts, neuralforecast, sktime |
| Clustering / Anomaly | scikit-learn, hdbscan, pyod |
| Dimensionality Reduction | scikit-learn, umap-learn, openTSNE |
| Hyperparameter Tuning | optuna, ray[tune], scikit-optimize |
| Experiment Tracking | mlflow, wandb, tensorboard |
| Model Interpretation | shap, lime, interpret, captum |
