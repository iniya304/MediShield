# MediShield — Full Implementation Guide

This document is a build-ready, end-to-end implementation reference for MediShield. It covers environment setup, data pipeline, model training, adversarial attacks, the reliability detector, explainability scoring, the abstention decision layer, evaluation, and the demo app — with concrete code for each piece. Treat this as the engineering companion to the improvement plan; it tells you *how* to build what was proposed there.

---

## Table of Contents

1. Environment Setup
2. Project Structure
3. Data Pipeline
4. Model 1: EfficientNet-B0 (Main Classifier)
5. Model 2: ResNet18 (Benchmark)
6. Adversarial Attacks (FGSM / PGD)
7. Realistic Stress-Test Transforms
8. Reliability Feature Engineering
9. Reliability Detector (XGBoost)
10. Reliability Baselines (MSP, Temperature Scaling, Entropy, MC-Dropout)
11. Explainability: Grad-CAM
12. Explainability Quality Scoring (RR, Robustness, Consistency, Superpixel Similarity)
13. Decision Logic / Abstention Layer
14. Evaluation Suite (Coverage-Risk, Calibration, Bootstrap CI, k-Fold, SHAP)
15. External / Out-of-Distribution Validation
16. Streamlit Demo Application
17. Testing & Validation Checklist
18. Run Order (End to End)

---

## 1. Environment Setup

```bash
# Create environment
python -m venv medishield-env
source medishield-env/bin/activate   # (Windows: medishield-env\Scripts\activate)

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost scikit-learn pandas numpy pillow opencv-python
pip install grad-cam                     # pytorch-grad-cam: Grad-CAM/Grad-CAM++/Eigen-CAM implementations
pip install scikit-image                 # SLIC superpixels, SSIM
pip install shap
pip install streamlit
pip install matplotlib seaborn
pip install captum                       # optional: alternative XAI/attribution library, useful for cross-checking Grad-CAM

# Reproducibility
pip freeze > requirements.txt
```

Recommended hardware: GPU (Colab T4 or better) for EfficientNet-B0/ResNet18 training and adversarial example generation. CPU is sufficient for XGBoost training/inference and the Streamlit demo.

Set global seeds everywhere:

```python
# src/utils/seed.py
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 2. Project Structure

```
MediShield/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml                 # central hyperparameters, paths, thresholds
├── data/
│   ├── metadata/                   # HAM10000_metadata.csv, ISIC/BCN20000 metadata (OOD)
│   └── processed/                  # lesion-aware train/val/test splits + augmented cache
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_efficientnet_training.ipynb
│   ├── 03_resnet_training.ipynb
│   ├── 04_adversarial_attacks.ipynb
│   ├── 05_reliability_detector.ipynb
│   ├── 06_explainability_scoring.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_external_validation.ipynb
├── src/
│   ├── utils/
│   │   ├── seed.py
│   │   └── config.py
│   ├── data/
│   │   ├── splitting.py             # lesion-level split logic
│   │   ├── dataset.py               # PyTorch Dataset/DataLoader
│   │   └── transforms.py            # train/val/test transforms
│   ├── models/
│   │   ├── efficientnet.py
│   │   └── resnet.py
│   ├── attacks/
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   └── stress_transforms.py     # noise, blur, brightness, compression
│   ├── reliability/
│   │   ├── features.py              # margin, entropy, perturbation-delta features
│   │   ├── baselines.py             # MSP, temperature scaling, entropy, MC-dropout
│   │   ├── xgb_detector.py          # training + inference
│   │   └── shap_analysis.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   ├── deletion_insertion.py    # region-reduction / faithfulness
│   │   ├── stability_metrics.py     # robustness + consistency (SSIM-based)
│   │   ├── superpixel_similarity.py
│   │   └── quality_score.py         # aggregates into Q
│   ├── decision/
│   │   └── abstention.py
│   └── evaluation/
│       ├── coverage_risk.py
│       ├── calibration.py
│       ├── bootstrap_ci.py
│       ├── kfold.py
│       └── subgroup_analysis.py
├── models/                          # saved weights
│   ├── efficientnet.pth
│   ├── resnet18.pth
│   └── reliability_xgb.json
├── results/
│   ├── metrics/
│   └── figures/
└── app/
    └── streamlit_app.py
```

---

## 3. Data Pipeline

### 3.1 Lesion-level splitting (leakage prevention)

```python
# src/data/splitting.py
import pandas as pd
from sklearn.model_selection import train_test_split

def lesion_level_split(metadata_csv: str, seed: int = 42,
                        train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """
    Splits HAM10000 at the lesion_id level so no lesion appears
    in more than one of train/val/test.
    """
    df = pd.read_csv(metadata_csv)
    df = df[df["dx"].isin(["bkl", "mel", "nv"])].copy()

    # One row per unique lesion, tagged with its class (assume consistent per lesion)
    lesion_df = df.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]]

    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=(1 - train_frac), stratify=lesion_df["dx"], random_state=seed
    )
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=test_frac / (val_frac + test_frac),
        stratify=temp_lesions["dx"], random_state=seed
    )

    def subset(lesion_ids):
        return df[df["lesion_id"].isin(lesion_ids)].reset_index(drop=True)

    train_df = subset(train_lesions["lesion_id"])
    val_df   = subset(val_lesions["lesion_id"])
    test_df  = subset(test_lesions["lesion_id"])

    # Sanity check: zero lesion overlap
    assert set(train_df.lesion_id) & set(val_df.lesion_id) == set()
    assert set(train_df.lesion_id) & set(test_df.lesion_id) == set()
    assert set(val_df.lesion_id) & set(test_df.lesion_id) == set()

    return train_df, val_df, test_df
```

### 3.2 Class balancing to the ~600/class target

```python
def build_balanced_subset(df: pd.DataFrame, per_class_target: int = 600, seed: int = 42):
    frames = []
    for cls, group in df.groupby("dx"):
        n = min(per_class_target, len(group))
        frames.append(group.sample(n=n, random_state=seed))
    return pd.concat(frames).reset_index(drop=True)
```

### 3.3 Dataset class + transforms

```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_TO_IDX = {"bkl": 0, "mel": 1, "nv": 2}

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label
```

```python
# src/data/transforms.py
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

---

## 4. Model 1: EfficientNet-B0 (Main Classifier)

```python
# src/models/efficientnet.py
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_efficientnet(num_classes: int = 3, pretrained: bool = True):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
```

### 4.1 Training loop

```python
# notebooks/02_efficientnet_training.ipynb (core loop, also mirror in a src/train.py script)
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4, patience=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/efficientnet.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    return history
```

### 4.2 Extracting probabilities, confidence, deep features (needed downstream)

```python
# src/models/inference_utils.py
import torch
import torch.nn.functional as F

def get_prediction_bundle(model, image_tensor, device, feature_layer_name="avgpool"):
    """
    Returns: probs (np.array), pred_class (int), confidence (float),
             deep_features (np.array) — extracted via forward hook.
    """
    model.eval()
    features = {}
    def hook(module, inp, out):
        features["feat"] = out.detach().flatten(1)

    layer = dict(model.named_modules())[feature_layer_name]
    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    handle.remove()
    pred_class = int(probs.argmax())
    confidence = float(probs.max())
    deep_feat = features["feat"].cpu().numpy()[0]
    return probs, pred_class, confidence, deep_feat
```

---

## 5. Model 2: ResNet18 (Benchmark)

```python
# src/models/resnet.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes: int = 3, pretrained: bool = True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

Reuse the same `train_model()` loop from Section 4.1 — swap the model and change the save path to `models/resnet18.pth`. No extensive re-tuning needed; it's a comparison point, not the centerpiece.

---

## 6. Adversarial Attacks (FGSM / PGD)

```python
# src/attacks/fgsm.py
import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon, device):
    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * images.grad.sign()
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, images.min().item(), images.max().item())  # stay in valid normalized range
    return adv_images.detach()
```

```python
# src/attacks/pgd.py
import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)  # random start
    adv_images = torch.clamp(adv_images, original_images.min().item(), original_images.max().item())

    for _ in range(num_steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = torch.clamp(original_images + perturbation,
                                  original_images.min().item(), original_images.max().item())

    return adv_images.detach()
```

**Suggested hyperparameters** (tune on validation set, report the sweep):
- FGSM: ε ∈ {0.01, 0.03, 0.05} (in normalized pixel space)
- PGD: ε = 0.03, α = 0.005, steps = 10–20

### 6.1 Attack evaluation harness

```python
# src/attacks/evaluate_attacks.py
def evaluate_under_attack(model, loader, attack_fn, device, **attack_kwargs):
    model.eval()
    correct_clean, correct_adv, changed_preds = 0, 0, 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_preds = model(images).argmax(1)

        adv_images = attack_fn(model, images, labels, device=device, **attack_kwargs)
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(1)

        correct_clean += (clean_preds == labels).sum().item()
        correct_adv += (adv_preds == labels).sum().item()
        changed_preds += (clean_preds != adv_preds).sum().item()
        total += labels.size(0)

    return {
        "clean_accuracy": correct_clean / total,
        "adv_accuracy": correct_adv / total,
        "prediction_change_rate": changed_preds / total,
        "attack_success_rate": 1 - (correct_adv / total),
    }
```

---

## 7. Realistic Stress-Test Transforms

```python
# src/attacks/stress_transforms.py
import numpy as np
import torch
import cv2
from PIL import Image
import io

def add_gaussian_noise(image_tensor, std=0.05):
    noise = torch.randn_like(image_tensor) * std
    return torch.clamp(image_tensor + noise, image_tensor.min(), image_tensor.max())

def apply_blur(pil_image, kernel_size=5):
    arr = np.array(pil_image)
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred)

def adjust_brightness(pil_image, factor=1.3):
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(pil_image).enhance(factor)

def jpeg_compress(pil_image, quality=30):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)
```

Wrap these into a **severity-graded suite** (mirrors ImageNet-C style) for E-series experiments:

```python
STRESS_SEVERITIES = {
    "noise":       [0.02, 0.05, 0.08, 0.12, 0.16],   # std
    "blur":        [3, 5, 7, 9, 11],                  # kernel size
    "brightness":  [0.7, 0.85, 1.15, 1.3, 1.5],       # factor
    "compression": [80, 60, 40, 20, 10],              # jpeg quality (lower = worse)
}
```

---

## 8. Reliability Feature Engineering

```python
# src/reliability/features.py
import numpy as np
from scipy.stats import entropy as scipy_entropy

def compute_base_features(probs: np.ndarray):
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]
    ent = scipy_entropy(probs + 1e-12)  # avoid log(0)
    confidence = sorted_probs[0]
    return {
        "confidence": confidence,
        "margin": margin,
        "entropy": ent,
        "p_bkl": probs[0], "p_mel": probs[1], "p_nv": probs[2],
    }

def compute_perturbation_features(clean_probs, perturbed_probs, clean_feat, perturbed_feat):
    conf_change = clean_probs.max() - perturbed_probs.max()
    prob_shift = np.linalg.norm(clean_probs - perturbed_probs)
    pred_changed = int(clean_probs.argmax() != perturbed_probs.argmax())
    feature_shift = np.linalg.norm(clean_feat - perturbed_feat)
    return {
        "confidence_change": conf_change,
        "probability_shift": prob_shift,
        "prediction_changed": pred_changed,
        "deep_feature_shift": feature_shift,
    }

def assemble_feature_vector(base_feats: dict, perturb_feats: dict = None):
    vec = dict(base_feats)
    if perturb_feats:
        vec.update(perturb_feats)
    return vec
```

### 8.1 Building the training table for XGBoost

```python
# src/reliability/build_training_table.py
import pandas as pd

def build_reliability_dataset(model, loader, device, attack_fn=None, **attack_kwargs):
    rows = []
    for images, labels in loader:
        for i in range(images.size(0)):
            img = images[i]
            label = labels[i].item()

            probs, pred, conf, feat = get_prediction_bundle(model, img, device)
            base_feats = compute_base_features(probs)

            row = dict(base_feats)
            if attack_fn is not None:
                adv_img = attack_fn(model, img.unsqueeze(0), labels[i:i+1], device=device, **attack_kwargs)[0]
                adv_probs, adv_pred, adv_conf, adv_feat = get_prediction_bundle(model, adv_img, device)
                perturb_feats = compute_perturbation_features(probs, adv_probs, feat, adv_feat)
                row.update(perturb_feats)

            row["correct"] = int(pred == label)   # <- ground-truth-derived label, NOT confidence > 0.5
            rows.append(row)

    return pd.DataFrame(rows)
```

**Critical:** the `correct` label comes from comparing prediction to ground truth — never from thresholding confidence. This is what makes the detector learn a genuine signal instead of reproducing an arbitrary cutoff.

---

## 9. Reliability Detector (XGBoost)

```python
# src/reliability/xgb_detector.py
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURE_COLS = [
    "confidence", "margin", "entropy", "p_bkl", "p_mel", "p_nv",
    "confidence_change", "probability_shift", "prediction_changed", "deep_feature_shift",
]

def train_reliability_detector(df, feature_cols=FEATURE_COLS, label_col="correct", seed=42):
    X = df[feature_cols]
    y = df[label_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, val_preds),
        "precision": precision_score(y_val, val_preds),
        "recall": recall_score(y_val, val_preds),
        "f1": f1_score(y_val, val_preds),
        "roc_auc": roc_auc_score(y_val, val_probs),
    }
    model.save_model("models/reliability_xgb.json")
    return model, metrics
```

---

## 10. Reliability Baselines (MSP, Temperature Scaling, Entropy, MC-Dropout)

```python
# src/reliability/baselines.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def msp_score(probs: np.ndarray) -> float:
    """Max Softmax Probability — simplest baseline."""
    return float(probs.max())

def entropy_score(probs: np.ndarray) -> float:
    """Negative entropy, rescaled to [0,1]-ish — lower entropy = higher reliability."""
    ent = -np.sum(probs * np.log(probs + 1e-12))
    max_ent = np.log(len(probs))
    return 1 - (ent / max_ent)

# --- Temperature Scaling ---
class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

def fit_temperature(model, val_loader, device, max_iter=50):
    model.eval()
    scaler = TemperatureScaler(model).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits_list.append(model(images))
            labels_list.append(labels.to(device))
    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)

    def closure():
        optimizer.zero_grad()
        loss = criterion(all_logits / scaler.temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature.item()

# --- MC-Dropout uncertainty ---
def mc_dropout_predict(model, image_tensor, device, n_passes=20):
    """Requires model to have Dropout layers active — force train() mode for dropout only."""
    model.train()  # enables dropout; ensure BatchNorm layers are frozen/eval if present
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = F.softmax(model(image_tensor.unsqueeze(0).to(device)), dim=1)
            preds.append(out.cpu().numpy()[0])
    preds = np.stack(preds)
    mean_probs = preds.mean(axis=0)
    predictive_variance = preds.var(axis=0).sum()  # higher = less reliable
    model.eval()
    return mean_probs, predictive_variance
```

### 10.1 Overlaying all reliability strategies

```python
# src/evaluation/coverage_risk.py (used again in Section 14, defined once)
import numpy as np

def coverage_risk_curve(reliability_scores, correctness, thresholds=None):
    """
    reliability_scores: array of scores where HIGHER = more reliable
    correctness: binary array, 1 = correct prediction
    Returns list of (threshold, coverage, selective_risk)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 50)

    results = []
    for t in thresholds:
        accepted = reliability_scores >= t
        coverage = accepted.mean()
        if accepted.sum() == 0:
            risk = 0.0
        else:
            risk = 1 - correctness[accepted].mean()
        results.append((t, coverage, risk))
    return results
```

Run this once per strategy (MSP, temp-scaled MSP, entropy, XGBoost, MC-dropout variance inverted) and plot all curves on one chart — this is the single most important comparison plot in the whole project.

---

## 11. Explainability: Grad-CAM

```python
# src/explainability/gradcam.py
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np

def get_gradcam_map(model, image_tensor, target_class, target_layer):
    """
    target_layer for EfficientNet-B0: model.features[-1]
    target_layer for ResNet18: model.layer4[-1]
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)[0]  # HxW, normalized [0,1]
    return grayscale_cam
```

---

## 12. Explainability Quality Scoring

### 12.1 Region Reduction / Deletion-Insertion (faithfulness)

```python
# src/explainability/deletion_insertion.py
import numpy as np
import torch
import torch.nn.functional as F

def region_reduction_score(model, image_tensor, saliency_map, target_class, device, k_steps=(10, 20, 30, 40, 50)):
    """
    Progressively masks top-k% activated pixels and measures confidence drop.
    Higher RR => saliency map highlights causally important regions.
    """
    model.eval()
    flat_saliency = saliency_map.flatten()
    sorted_idx = np.argsort(flat_saliency)[::-1]  # descending activation

    with torch.no_grad():
        base_probs = F.softmax(model(image_tensor.unsqueeze(0).to(device)), dim=1)[0]
        base_conf = base_probs[target_class].item()

    drops = []
    h, w = saliency_map.shape
    for k in k_steps:
        n_pixels = int(len(flat_saliency) * (k / 100))
        mask = np.ones_like(flat_saliency)
        mask[sorted_idx[:n_pixels]] = 0
        mask = mask.reshape(h, w)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=image_tensor.shape[1:], mode="nearest")[0]

        masked_image = image_tensor.to(device) * mask_tensor
        with torch.no_grad():
            masked_probs = F.softmax(model(masked_image.unsqueeze(0)), dim=1)[0]
            masked_conf = masked_probs[target_class].item()

        drop = (base_conf - masked_conf) / (base_conf + 1e-8)
        drops.append(drop)

    return float(np.mean(drops))
```

### 12.2 Robustness + Consistency (SSIM-based stability)

```python
# src/explainability/stability_metrics.py
from skimage.metrics import structural_similarity as ssim
import numpy as np

def robustness_score(clean_saliency, perturbed_saliency_list):
    """Average SSIM between clean Grad-CAM and Grad-CAM under a set of perturbations."""
    scores = [ssim(clean_saliency, p, data_range=1.0) for p in perturbed_saliency_list]
    return float(np.mean(scores))

def consistency_score(saliency_maps: list):
    """Average pairwise SSIM across saliency maps from augmented copies of the same image."""
    n = len(saliency_maps)
    if n < 2:
        return 1.0
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(ssim(saliency_maps[i], saliency_maps[j], data_range=1.0))
    return float(np.mean(pairwise))
```

### 12.3 Superpixel Similarity

```python
# src/explainability/superpixel_similarity.py
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float

def superpixel_similarity_score(image_rgb: np.ndarray, saliency_map: np.ndarray, reference_map: np.ndarray,
                                 n_segments=100, compactness=10):
    """
    image_rgb: HxWx3 float image in [0,1]
    saliency_map, reference_map: HxW in [0,1], same shape as image
    """
    segments = slic(img_as_float(image_rgb), n_segments=n_segments, compactness=compactness, start_label=0)

    saliency_vec, reference_vec = [], []
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        saliency_vec.append(saliency_map[mask].mean())
        reference_vec.append(reference_map[mask].mean())

    saliency_vec = np.array(saliency_vec)
    reference_vec = np.array(reference_vec)

    cosine_sim = np.dot(saliency_vec, reference_vec) / (
        np.linalg.norm(saliency_vec) * np.linalg.norm(reference_vec) + 1e-8
    )
    return float(cosine_sim)
```

**Building the reference map** (since HAM10000 lacks pixel-level lesion masks):

```python
def build_reference_map_from_mask(lesion_mask: np.ndarray) -> np.ndarray:
    """If a derived/approximate lesion segmentation mask is available."""
    return lesion_mask.astype(float)  # 1 inside lesion, 0 outside

def build_reference_map_centroid(image_shape, sigma_frac=0.3) -> np.ndarray:
    """Fallback: Gaussian centered on the image, for lesion-centered dermoscopic crops."""
    h, w = image_shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    sigma = sigma_frac * min(h, w)
    gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return gaussian / gaussian.max()
```

### 12.4 Aggregate Quality Score

```python
# src/explainability/quality_score.py
def aggregate_quality_score(rr, robustness, superpixel_sim, consistency,
                              w_rr=0.25, w_r=0.15, w_ss=0.35, w_c=0.25):
    """
    Default weights are a starting point — run a sensitivity analysis
    (see evaluation suite) and justify final weights empirically.
    """
    q = w_rr * rr + w_r * robustness + w_ss * superpixel_sim + w_c * consistency
    return float(np.clip(q, 0, 1))
```

### 12.5 Threshold sensitivity sweep

```python
def threshold_sensitivity(q_scores, correctness, thresholds=(0.4, 0.5, 0.6, 0.7, 0.8)):
    results = []
    for t in thresholds:
        accepted = q_scores >= t
        coverage = accepted.mean()
        error_among_accepted = 1 - correctness[accepted].mean() if accepted.sum() > 0 else 0
        results.append({"threshold": t, "coverage": coverage, "error_rate": error_among_accepted})
    return results
```

---

## 13. Decision Logic / Abstention Layer

```python
# src/decision/abstention.py
def make_decision(reliability_score: float, explanation_quality: float,
                   reliability_threshold: float = 0.6, quality_threshold: float = 0.6,
                   predicted_class_valid: bool = True):
    """
    Two-signal abstention: flags for human review if EITHER
    the reliability score OR the explanation quality is below threshold,
    or the prediction itself is invalid/absent.
    """
    if not predicted_class_valid:
        return "ABSTAIN", "Predicted class invalid/absent"
    if reliability_score < reliability_threshold:
        return "ABSTAIN", f"Low reliability score ({reliability_score:.2f})"
    if explanation_quality < quality_threshold:
        return "ABSTAIN", f"Low explanation quality ({explanation_quality:.2f})"
    return "ACCEPT", "Prediction and explanation both pass thresholds"
```

---

## 14. Evaluation Suite

### 14.1 Calibration (ECE + reliability diagram)

```python
# src/evaluation/calibration.py
import numpy as np
import matplotlib.pyplot as plt

def expected_calibration_error(confidences, correctness, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append((lo, hi, 0, 0, 0))
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_acc - bin_conf)
        bin_data.append((lo, hi, bin_acc, bin_conf, mask.sum()))
    return ece, bin_data

def plot_reliability_diagram(bin_data, save_path="results/figures/reliability_diagram.png"):
    accs = [b[2] for b in bin_data]
    confs = [(b[0] + b[1]) / 2 for b in bin_data]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.bar(confs, accs, width=0.08, alpha=0.7, label="Model")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(save_path)
```

### 14.2 Bootstrap confidence intervals

```python
# src/evaluation/bootstrap_ci.py
import numpy as np

def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper
```

### 14.3 Lesion-level k-fold cross-validation

```python
# src/evaluation/kfold.py
from sklearn.model_selection import StratifiedKFold
import numpy as np

def lesion_level_kfold(lesion_df, k=5, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(lesion_df, lesion_df["dx"]):
        folds.append((lesion_df.iloc[train_idx], lesion_df.iloc[val_idx]))
    return folds
```

### 14.4 SHAP on the reliability detector

```python
# src/reliability/shap_analysis.py
import shap

def explain_reliability_model(xgb_model, X_val):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values, X_val, show=False)
    return shap_values
```

### 14.5 Subgroup analysis

```python
# src/evaluation/subgroup_analysis.py
def subgroup_breakdown(df, group_col, correctness_col, reliability_col):
    return df.groupby(group_col).agg(
        n=(correctness_col, "count"),
        accuracy=(correctness_col, "mean"),
        avg_reliability=(reliability_col, "mean"),
    ).reset_index()
```

---

## 15. External / Out-of-Distribution Validation

```python
# notebooks/08_external_validation.ipynb (core logic)
def evaluate_on_ood(model, xgb_reliability_model, ood_loader, device):
    """
    ood_loader should yield images preprocessed identically to HAM10000
    (224x224, ImageNet normalization) but drawn from ISIC/BCN20000.
    Never train on this data — evaluation only.
    """
    results = []
    for images, labels in ood_loader:
        for i in range(images.size(0)):
            probs, pred, conf, feat = get_prediction_bundle(model, images[i], device)
            feats = compute_base_features(probs)
            reliability = xgb_reliability_model.predict_proba([list(feats.values())])[0][1]
            results.append({
                "correct": int(pred == labels[i].item()),
                "confidence": conf,
                "reliability": reliability,
            })
    df = pd.DataFrame(results)
    return {
        "ood_accuracy": df["correct"].mean(),
        "avg_reliability_on_ood": df["reliability"].mean(),
        "pct_flagged_low_reliability": (df["reliability"] < 0.6).mean(),
    }
```

Report this side-by-side with in-distribution numbers — the key claim to support is that reliability scores drop *and* flagging rate rises on OOD data, even without ever training on it.

---

## 16. Streamlit Demo Application

```python
# app/streamlit_app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np

# --- Load models once (cache) ---
@st.cache_resource
def load_models():
    effnet = build_efficientnet()
    effnet.load_state_dict(torch.load("models/efficientnet.pth", map_location="cpu"))
    effnet.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/reliability_xgb.json")

    return effnet, xgb_model

effnet, xgb_model = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("MediShield — Medical AI Reliability Monitor")
st.caption("Research prototype. Not a diagnostic tool.")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    image_tensor = eval_transform(image)
    probs, pred_class, confidence, deep_feat = get_prediction_bundle(effnet, image_tensor, device)
    class_names = ["bkl", "mel", "nv"]

    base_feats = compute_base_features(probs)
    reliability_score = xgb_model.predict_proba([list(base_feats.values())])[0][1]

    # Grad-CAM
    saliency = get_gradcam_map(effnet, image_tensor, pred_class, target_layer=effnet.features[-1])

    st.subheader("Prediction")
    st.write(f"**{class_names[pred_class]}** — confidence: {confidence:.1%}")
    st.write(f"Reliability score: {reliability_score:.1%}")

    decision, reason = make_decision(reliability_score, explanation_quality=0.75)  # placeholder Q until wired in
    if decision == "ACCEPT":
        st.success(f"✅ ACCEPT — {reason}")
    else:
        st.warning(f"⚠ SUSPICIOUS — HUMAN REVIEW RECOMMENDED ({reason})")

    st.subheader("Grad-CAM")
    st.image(saliency, caption="Activation heatmap", width=300, clamp=True)

    if st.button("Run FGSM Stress Test"):
        adv_tensor = fgsm_attack(effnet, image_tensor.unsqueeze(0), torch.tensor([pred_class]),
                                  epsilon=0.03, device=device)[0]
        adv_probs, adv_pred, adv_conf, _ = get_prediction_bundle(effnet, adv_tensor, device)
        st.write(f"Perturbed prediction: **{class_names[adv_pred]}** — confidence: {adv_conf:.1%}")
        if adv_pred != pred_class:
            st.error("⚠ Prediction changed under adversarial perturbation!")
```

Run with:
```bash
streamlit run app/streamlit_app.py
```

---

## 17. Testing & Validation Checklist

Before treating any result as final, verify:

- [ ] `assert` checks confirm zero lesion_id overlap across train/val/test
- [ ] Reliability labels (`correct`) are derived from ground truth, not from `confidence > 0.5`
- [ ] Val/test transforms contain no random augmentation (deterministic only)
- [ ] Grad-CAM target layer is correct for each architecture (`features[-1]` for EfficientNet-B0, `layer4[-1]` for ResNet18)
- [ ] Superpixel reference map is documented as an approximation (mask-derived or centroid heuristic) in the write-up
- [ ] Coverage-risk curves are compared against at least MSP and temperature-scaled baselines, not XGBoost alone
- [ ] OOD evaluation set was never used in any training or threshold-tuning step
- [ ] All headline metrics are reported with bootstrap CIs, not single point estimates
- [ ] Random seeds are fixed and logged for every experiment
- [ ] Streamlit demo runs end-to-end on a fresh environment (`pip install -r requirements.txt` → `streamlit run`)

---

## 18. Run Order (End to End)

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Data preparation
jupyter notebook notebooks/01_data_preparation.ipynb
# -> produces data/processed/{train,val,test}.csv with lesion-level splits

# 3. Train classifiers
jupyter notebook notebooks/02_efficientnet_training.ipynb
jupyter notebook notebooks/03_resnet_training.ipynb

# 4. Adversarial evaluation
jupyter notebook notebooks/04_adversarial_attacks.ipynb
# -> logs E3/E4 results, builds perturbation-behavior features for reliability table

# 5. Reliability detector
jupyter notebook notebooks/05_reliability_detector.ipynb
# -> trains XGBoost, computes MSP/temp-scaled/entropy baselines, saves coverage-risk plot

# 6. Explainability scoring
jupyter notebook notebooks/06_explainability_scoring.ipynb
# -> RR, robustness, consistency, superpixel similarity, aggregate Q, threshold sweep

# 7. Full evaluation
jupyter notebook notebooks/07_evaluation.ipynb
# -> k-fold CV, bootstrap CIs, calibration/ECE, SHAP, subgroup analysis

# 8. External validation
jupyter notebook notebooks/08_external_validation.ipynb
# -> OOD accuracy + reliability flagging rate on ISIC/BCN20000 subset

# 9. Launch demo
streamlit run app/streamlit_app.py
```
