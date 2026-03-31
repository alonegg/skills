# Unstructured Data Analysis Guide

Reference for processing text, images, and audio data in ML pipelines.

## Table of Contents
1. [NLP Pipeline](#nlp-pipeline)
2. [Computer Vision Pipeline](#computer-vision-pipeline)
3. [Audio Pipeline](#audio-pipeline)
4. [Multi-Modal Data](#multi-modal-data)

---

## NLP Pipeline

### Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)        # Remove HTML
    text = re.sub(r'http\S+', '', text)         # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Remove non-alpha
    text = re.sub(r'\s+', ' ', text).strip()    # Normalize whitespace
    return text

def preprocess(text, remove_stopwords=True, lemmatize=True):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stops]
    if lemmatize:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return tokens
```

### Feature Extraction: Classical

**TF-IDF** — best baseline for most text classification:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.95)
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf = tfidf.transform(test_texts)  # transform only, no fit
```

**Word Embeddings** — when semantic meaning matters:
```python
import gensim.downloader as api
w2v = api.load('word2vec-google-news-300')

def text_to_embedding(text, model, dim=300):
    tokens = text.lower().split()
    vectors = [model[w] for w in tokens if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)
```

### Feature Extraction: Deep Learning

**HuggingFace Transformers** — for state-of-the-art NLP:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = "distilbert-base-uncased"  # Good speed/quality tradeoff
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

tokenized_ds = dataset.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

### Model Selection by Data Size

| Data Size | Recommended Approach | Why |
|-----------|---------------------|-----|
| <500 samples | TF-IDF + LogisticRegression | Not enough data for deep learning |
| 500-5K | TF-IDF + SVM/LR, or few-shot with SetFit | SetFit works well with few examples |
| 5K-50K | Fine-tune DistilBERT | Good balance of speed and quality |
| 50K+ | Fine-tune BERT/RoBERTa | Enough data to leverage large models |
| Very large | Fine-tune with LoRA/QLoRA | Efficient parameter fine-tuning |

### When TF-IDF Beats Transformers

- Very small datasets (<500 samples)
- High-cardinality classification (many classes, few examples each)
- When latency matters and accuracy gap is small
- When computational resources are limited
- Domain-specific jargon not in pretrained vocabulary

### NLP Task Patterns

**Text Classification**:
```python
# Baseline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])
baseline.fit(X_train, y_train)
```

**Named Entity Recognition** (with HuggingFace):
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
results = ner("Hugging Face is based in New York City.")
```

**Semantic Similarity / Embeddings**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
```

**Text Summarization**:
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_text, max_length=130, min_length=30)
```

### NLP Evaluation

- **Classification**: precision, recall, F1 per class. Use `classification_report()`
- **Macro F1** for imbalanced classes (treats all classes equally)
- **Weighted F1** when class frequency matters
- **Confusion matrix** to spot systematic misclassifications between similar classes

### Handling Class Imbalance in Text

1. Class weights: `LogisticRegression(class_weight='balanced')`
2. Oversampling minority with paraphrase generation
3. Focal loss for deep learning
4. Stratified sampling in train/test split

---

## Computer Vision Pipeline

### Image Loading and Preprocessing

```python
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = ImageFolder('data/train', transform=train_transforms)
val_ds = ImageFolder('data/val', transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
```

### Transfer Learning Workflow (fast.ai Style)

The proven approach: freeze pretrained backbone → train head → gradually unfreeze.

```python
import torch
import torch.nn as nn
from torchvision import models

# Step 1: Load pretrained model, replace head
model = models.resnet50(weights='IMAGENET1K_V2')
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)

# Step 2: Freeze backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Step 3: Train head with higher LR (3-5 epochs)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# Step 4: Unfreeze and fine-tune with discriminative LRs
for param in model.parameters():
    param.requires_grad = True

param_groups = [
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
]
optimizer = torch.optim.Adam(param_groups)
```

### Learning Rate Finder

```python
def lr_find(model, train_loader, criterion, start_lr=1e-7, end_lr=10, num_steps=100):
    lrs, losses = [], []
    lr = start_lr
    factor = (end_lr / start_lr) ** (1 / num_steps)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_steps:
            break
        optimizer.param_groups[0]['lr'] = lr
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lrs.append(lr)
        losses.append(loss.item())
        lr *= factor

    # Plot and pick lr where loss is steepest descent (typically 1/10 of minimum)
    return lrs, losses
```

### Pretrained Model Selection

| Model | Params | ImageNet Top-1 | Speed | When to Use |
|-------|--------|----------------|-------|-------------|
| MobileNetV3 | 5.4M | 75.2% | Very fast | Mobile/edge, real-time |
| ResNet50 | 25.6M | 80.4% | Fast | General purpose, proven |
| EfficientNet-B3 | 12M | 82.0% | Medium | Best accuracy/compute |
| ViT-B/16 | 86M | 81.8% | Slow | Large datasets, attention needed |
| ConvNeXt-T | 28.6M | 82.1% | Medium | Modern CNN, ViT-competitive |

### Data Augmentation Strategies

**Standard** (always use):
```python
T.RandomHorizontalFlip()
T.RandomRotation(15)
T.ColorJitter(brightness=0.2, contrast=0.2)
```

**Advanced** (for small datasets, use albumentations):
```python
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50)),
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # Cutout
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

**Mixup** (regularization for small datasets):
```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
```

### Small Dataset Strategies (<1000 images)

1. **Transfer learning** is mandatory — never train from scratch
2. Heavy augmentation (albumentations)
3. Start with a smaller model (ResNet18, EfficientNet-B0)
4. Use mixup/cutmix regularization
5. Consider few-shot learning approaches
6. Increase effective dataset size with test-time augmentation (TTA)

### Object Detection (YOLO)

```python
from ultralytics import YOLO

# Load pretrained YOLOv8
model = YOLO('yolov8n.pt')  # nano for speed, yolov8x for accuracy

# Fine-tune on custom data
results = model.train(
    data='data.yaml',  # YOLO format dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,  # early stopping
)

# Inference
results = model.predict('image.jpg', conf=0.25)
```

### CV Evaluation

- **Classification**: accuracy, top-5 accuracy, per-class precision/recall
- **Detection**: mAP@0.5, mAP@0.5:0.95
- **Segmentation**: IoU (Intersection over Union), Dice coefficient

---

## Audio Pipeline

### Loading and Preprocessing

```python
import librosa
import numpy as np

def load_audio(path, sr=16000, duration=None):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y, sr

def extract_features(y, sr):
    features = {}
    features['mel_spec'] = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
    return features
```

### Audio Classification with CNN on Spectrograms

```python
# Convert audio to mel spectrogram image → treat as image classification
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
# Resize to fixed size, normalize, feed to CNN (ResNet/EfficientNet)
```

### Speech Recognition with Whisper

```python
import whisper
model = whisper.load_model("base")  # tiny/base/small/medium/large
result = model.transcribe("audio.mp3")
print(result["text"])
```

### Audio with torchaudio

```python
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC

waveform, sr = torchaudio.load("audio.wav")
mel_transform = MelSpectrogram(sample_rate=sr, n_mels=128)
mel_spec = mel_transform(waveform)
```

### Audio Task Guide

| Task | Approach | Model |
|------|----------|-------|
| Classification | Mel spectrogram + CNN | ResNet on spectrograms |
| Speech recognition | End-to-end | Whisper |
| Speaker ID | Speaker embeddings | ECAPA-TDNN, wav2vec2 |
| Music analysis | Chroma + MFCC features | CNN or classical ML |
| Emotion detection | Mel features + fine-tune | wav2vec2 fine-tuned |

---

## Multi-Modal Data

### Combining Tabular + Text

**Late Fusion** (simpler, often sufficient):
```python
# Extract features separately, concatenate
text_features = tfidf.fit_transform(df['text_column'])
numeric_features = scaler.fit_transform(df[numeric_cols])
X_combined = np.hstack([text_features.toarray(), numeric_features])
# Train a single model on combined features
```

**Early Fusion with Embeddings**:
```python
# Use sentence-transformers for text → dense vectors
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = encoder.encode(df['text_column'].tolist())
X_combined = np.hstack([text_embeddings, numeric_features])
```

### Combining Tabular + Image

**Late Fusion**:
```python
# Extract image features from pretrained CNN
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Identity()  # Remove classification head
model.eval()

with torch.no_grad():
    image_features = model(image_batch).numpy()

# Concatenate with tabular features
X_combined = np.hstack([image_features, tabular_features])
```

**Dual-Branch Neural Network** (for larger datasets):
```python
class MultiModalModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super().__init__()
        self.image_branch = models.resnet18(weights='IMAGENET1K_V1')
        self.image_branch.fc = nn.Linear(512, 128)
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, tabular):
        img_feat = self.image_branch(image)
        tab_feat = self.tabular_branch(tabular)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(combined)
```

### Multi-Modal Best Practices

1. **Start with late fusion** — it's simpler, easier to debug, and often competitive
2. **Normalize feature scales** — embeddings and tabular features may be on very different scales
3. **Validate each modality separately first** — understand what each contributes
4. **Attention-based fusion** for complex interactions (requires more data)
5. **Missing modalities**: design for graceful degradation when one input is missing
