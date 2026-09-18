# MediShield

**An AI Safety & Reliability Layer for Medical Vision Models**

> MediShield does not merely ask *"What does the medical image show?"* — it asks *"Can we trust the model's prediction?"*

MediShield is a **research prototype**, not a clinical diagnostic system. It should never be presented, marketed, or used as a replacement for a dermatologist or a clinical diagnostic workflow.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Core Problem](#2-the-core-problem)
3. [Research Question](#3-research-question)
4. [Why This Is More Than Classification](#4-why-this-is-more-than-classification)
5. [Architecture](#5-architecture)
6. [Dataset](#6-dataset)
7. [Data Splitting Rules (Leakage Prevention)](#7-data-splitting-rules-leakage-prevention)
8. [Image Preprocessing](#8-image-preprocessing)
9. [Models](#9-models)
10. [Reliability Detector (XGBoost)](#10-reliability-detector-xgboost)
11. [Adversarial Machine Learning](#11-adversarial-machine-learning)
12. [Realistic Stress Testing](#12-realistic-stress-testing)
13. [Explainable AI (Grad-CAM)](#13-explainable-ai-grad-cam)
14. [Abstention Mechanism](#14-abstention-mechanism)
15. [Coverage vs. Risk](#15-coverage-vs-risk)
16. [Experiments](#16-experiments)
17. [Results (Template)](#17-results-template)
18. [Demo Application](#18-demo-application)
19. [Technology Stack](#19-technology-stack)
20. [Project Structure](#20-project-structure)
21. [How to Run](#21-how-to-run)
22. [Scientific Limitations](#22-scientific-limitations)
23. [Final Project Definition](#23-final-project-definition)

---

## 1. Overview

MediShield is a research-oriented prototype for testing and improving the reliability of a medical image classification model. It uses a deep-learning skin-lesion classifier as the **prediction engine**, then wraps it in a **reliability layer** that evaluates whether each individual prediction actually deserves to be trusted.

The system combines several fields into one closed evaluation loop:

- Medical computer vision
- Deep learning (transfer learning with CNNs)
- Adversarial machine learning
- Reliability / uncertainty analysis
- Gradient-boosted trees (XGBoost)
- Explainable AI (Grad-CAM)
- Failure detection
- Selective prediction / abstention
- Robustness stress testing

**MediShield should be pitched as:**
> "We built a reliability and safety layer around a medical vision model and tested whether it can recognize when its own prediction becomes unreliable."

**Not as:**
> "We trained a skin-cancer classifier."

---

## 2. The Core Problem

A conventional medical-image AI pipeline looks like this:

```text
Medical Image
     ↓
CNN
     ↓
Disease Prediction
```

For example:

```text
Skin lesion
     ↓
EfficientNet-B0
     ↓
Melanoma — 94%
```

The problem is that **a high-confidence prediction is not automatically a trustworthy prediction**. A model can:

- Make an incorrect prediction with high confidence
- Become unstable after small input changes
- React badly to adversarial perturbations
- Change its prediction after image degradation (noise, blur, compression, brightness shifts)
- Produce explanations that shift substantially under perturbation
- Encounter an input outside the distribution it learned from

MediShield adds a reliability layer on top of the raw classifier:

```text
Image
  ↓
Medical Vision Model
  ↓
Prediction + Confidence + Features
  ↓
Reliability Analysis
  ↓
Trust / Suspicious
  ↓
Accept prediction OR Abstain
```

---

## 3. Research Question

**Primary question:**
> Can we detect when a medical vision model's prediction has become unreliable, especially under adversarial or realistic image perturbations?

**Secondary questions:**

1. How much does model performance degrade under FGSM and PGD attacks?
2. Does model confidence change when the input is perturbed?
3. Can behavioral features help detect incorrect predictions?
4. Can an XGBoost reliability model distinguish trustworthy from suspicious predictions?
5. Can an abstention mechanism reduce error among predictions the system chooses to accept?
6. Do model explanations (Grad-CAM) change when the input is manipulated?

---

## 4. Why This Is More Than Classification

Classification is only the first layer. The actual contribution stacks several layers on top of it:

```text
Medical Classification
        +
Adversarial Stress Testing
        +
Reliability Detection
        +
Failure Detection
        +
Abstention
        +
Explainability
```

The strongest technical narrative is:

```text
Classification → Attack → Observe behavior → Detect unreliability → Abstain → Explain
```

That six-step chain is the identity of MediShield — not the number of models it contains.

---

## 5. Architecture

### 5.1 High-level system architecture

```text
                         MEDISHIELD
                              │
                              ▼
                     ┌────────────────┐
                     │   Input Image  │
                     └───────┬────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ EfficientNet-B0  │
                    │ Main Classifier  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Probabilities     Confidence    Deep Features
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ Reliability Feature│
                  │      Engine        │
                  └─────────┬──────────┘
                             │
                             ▼
                        ┌─────────┐
                        │ XGBoost │
                        └────┬────┘
                             │
                             ▼
                    Reliability Score
                       /          \
                      /            \
                   HIGH            LOW
                    │                │
                    ▼                ▼
                 ACCEPT           ABSTAIN
                    │                │
                    ▼                ▼
               Prediction       Human Review
                    │
                    ▼
                 Grad-CAM
```

Running alongside the main pipeline is the **stress-testing / robustness loop**, which feeds perturbed images back through the same pipeline to observe how behavior changes under pressure:

```text
Clean image
    ↓
FGSM / PGD / Noise / Blur / Brightness / Compression
    ↓
Stress testing
    ↓
Model behavior
    ↓
Reliability analysis
```

### 5.2 Component responsibilities

| Layer | Component | Responsibility |
|---|---|---|
| Prediction engine | EfficientNet-B0 | Classify the lesion; expose probabilities, confidence, and deep features |
| Benchmark | ResNet18 | Independent architecture used to check whether reliability issues are model-specific |
| Stress layer | FGSM, PGD, noise, blur, brightness, compression | Perturb the input to probe robustness |
| Feature engine | Custom feature extraction code | Turn raw model outputs + perturbation behavior into a tabular feature vector |
| Reliability detector | XGBoost (binary classifier) | Predict whether the classifier's prediction is likely correct/reliable |
| Decision layer | Threshold-based abstention logic | Convert a reliability score into an Accept / Abstain decision |
| Explainability | Grad-CAM | Visualize which image regions drove the CNN's decision, clean vs. perturbed |
| Interface | Streamlit dashboard | Upload an image, run the full pipeline, and visualize every stage interactively |

### 5.3 Data flow, end to end

```text
1. User uploads a dermoscopic image
2. Image is preprocessed (resize → 224x224, ImageNet normalization)
3. EfficientNet-B0 produces:
     - class probabilities (bkl, mel, nv)
     - predicted class
     - confidence (max probability)
     - deep feature embedding
4. (Optional) The image is perturbed via FGSM/PGD/stress transforms
     and re-run through step 3, producing a second set of outputs
5. The Reliability Feature Engine assembles a feature vector from:
     - probabilities, confidence, margin, entropy
     - perturbation-derived behavioral features (if attack was run)
6. XGBoost consumes the feature vector and outputs a reliability
   probability (0-1)
7. The Abstention Layer compares the reliability score to a
   validation-tuned threshold:
     - score >= threshold  → ACCEPT  → show prediction + Grad-CAM
     - score <  threshold  → ABSTAIN → flag for human review
8. Grad-CAM renders a heatmap for the accepted prediction
   (and, in stress-test mode, a side-by-side clean vs. perturbed map)
```

### 5.4 Why two separately-trained models are used

EfficientNet-B0 is **not** trained to know whether it is right or wrong — it only outputs a softmax distribution over classes. XGBoost is trained **separately**, on a different signal entirely: not "which disease is this?" but "does this specific prediction look trustworthy, given how the classifier behaved?" This separation is intentional — it lets the reliability detector learn patterns (e.g., low margin, high entropy, large confidence swings under perturbation) that correlate with classifier error, independent of what the underlying disease actually is.

---

## 6. Dataset

### HAM10000

MediShield uses the **HAM10000** ("Human Against Machine with 10000 training images") skin-lesion dataset.

The full dataset contains:

- 10,015 dermoscopic images
- 7 diagnostic categories
- Metadata: image ID, lesion ID, diagnosis, age, sex, localization, dataset source

MediShield restricts itself to **three classes**:

| Code | Class |
|---|---|
| `bkl` | Benign keratosis-like lesions |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |

Inspected class counts in the full dataset:

| Class | Images |
|---|---:|
| bkl | 1,099 |
| mel | 1,113 |
| nv | 6,705 |

These classes are highly imbalanced (`nv` dominates), so the project uses an **approximately balanced subset** rather than the full raw distribution.

**Target working subset:**

```text
~600 images/class
~1,800 total images
```

This is a target, not a hard requirement — the exact number is determined after lesion-level splitting, once it's clear how many usable images remain per class.

**Split ratios:**

```text
70% Training
15% Validation
15% Test
```

Class balance is preserved as closely as possible while keeping all images of a given lesion together (see below).

---

## 7. Data Splitting Rules (Leakage Prevention)

HAM10000 contains **multiple images of the same physical lesion** (different angles/zoom/lighting of the same spot on the same patient). This makes naive random splitting dangerous.

**Bad split (causes leakage):**

```text
Lesion A
 ├── Image A1 → Training
 └── Image A2 → Test
```

Splitting at the image level lets lesion-specific visual cues leak from training into the test set, inflating apparent performance.

**Correct approach — split at the `lesion_id` level:**

```text
Lesion A
 └── ALL images → Training      (or)

Lesion A
 └── ALL images → Validation    (or)

Lesion A
 └── ALL images → Test
```

**Rule:** the same `lesion_id` must never appear in more than one of Train / Validation / Test.

---

## 8. Image Preprocessing

**Target input tensor:** `224 × 224 × 3`, RGB.

**Normalization:** ImageNet mean/std, because both EfficientNet-B0 and ResNet18 are ImageNet-pretrained.

**Training augmentation** (kept mild, to avoid producing unrealistic medical images):

- Resize
- Crop
- Horizontal flip
- Small rotation

**Validation/Test preprocessing:** deterministic only (resize + normalize) — no random augmentation, so evaluation is reproducible.

---

## 9. Models

### 9.1 EfficientNet-B0 — Main Classifier

**Role:** predict the lesion class and expose internal signals (probabilities, confidence, deep features) that the reliability layer consumes downstream.

Trained via **transfer learning** (ImageNet-pretrained backbone, classification head replaced for 3 classes) rather than from scratch.

```text
Image
 ↓
EfficientNet-B0
 ↓
Class probabilities
 ↓
Predicted class
 ↓
Deep features
```

### 9.2 ResNet18 — Benchmark Model

**Role:** determine whether the reliability problem is specific to one CNN architecture, or generalizes across architectures.

ResNet18 is a comparison point, not the centerpiece — EfficientNet-B0 remains the primary model, and ResNet18 does not need extensive tuning.

---

## 10. Reliability Detector (XGBoost)

XGBoost is **not** a disease classifier. Its job is to predict whether the deep-learning model's prediction is likely to be correct/reliable.

```text
EfficientNet
     ↓
Prediction, Confidence, Probabilities,
Feature statistics, Perturbation behavior
     ↓
XGBoost
     ↓
Reliability probability
```

Example outputs:

```text
Reliability score = 0.94  →  prediction appears reliable
Reliability score = 0.18  →  prediction appears suspicious
```

The acceptance threshold is selected **experimentally**, using validation data (see [Abstention](#14-abstention-mechanism)).

### 10.1 Reliability Features

**Prediction probabilities:** `P(bkl)`, `P(mel)`, `P(nv)`

**Confidence:** `max_probability`

**Prediction margin:** gap between the top-1 and top-2 class probabilities.
Example: Melanoma = 0.94, Nevus = 0.04 → margin = 0.90

**Entropy:** how spread out the probability distribution is.
`[0.94, 0.04, 0.02]` is far more decisive than `[0.36, 0.34, 0.30]`, even though both could have the same top-1 class.

**Perturbation behavior** (when an attack/stress transform is applied):

- Confidence change (clean vs. perturbed)
- Probability shift
- Whether the predicted class changed
- Deep feature representation shift
- Prediction consistency across perturbations

### 10.2 Training the Reliability Detector

Reliability labels are derived from ground-truth correctness, **not** from an arbitrary confidence cutoff:

```text
Ground-truth label
        ↓
Compare with EfficientNet prediction
        ↓
     Correct?
   /        \
 Yes         No
  ↓           ↓
Reliable    Unreliable
   1            0
```

Important: reliability is **not** defined as `confidence > 0.5`. Doing so would make the detector just reproduce an arbitrary threshold instead of learning a genuine signal. Instead, XGBoost learns the relationship:

```text
Model behavior → Prediction reliability
```

---

## 11. Adversarial Machine Learning

One of MediShield's main differentiators: testing how the classifier behaves under **intentional** perturbation.

### 11.1 FGSM (Fast Gradient Sign Method)

```text
Clean image
   ↓
Gradient-based perturbation
   ↓
Adversarial image
   ↓
Classifier
```

Measured: clean accuracy, adversarial accuracy, prediction changes, confidence changes, attack success rate.

### 11.2 PGD (Projected Gradient Descent)

A stronger, iterative attack:

```text
Clean image
     ↓
Small perturbation
     ↓
Model gradient
     ↓
Update
     ↓
Project perturbation
     ↓
Repeat
     ↓
Adversarial image
```

FGSM and PGD are **attack methods**, not separate trained models — they are transformations applied at evaluation time.

---

## 12. Realistic Stress Testing

Adversarial attacks alone aren't representative of everyday failure modes, so MediShield also runs a stress-test suite of realistic image degradations:

```text
Clean
FGSM
PGD
Gaussian noise
Blur
Brightness change
Compression
```

For each condition, the pipeline measures accuracy, confidence, whether the prediction changed, the reliability score, and the resulting abstention rate — giving a broader robustness picture than adversarial attacks alone.

---

## 13. Explainable AI (Grad-CAM)

Grad-CAM is **not another model** — it's a visualization method layered on top of the trained CNN.

**Purpose:** show which image regions influenced the CNN's prediction.

```text
Original image
      ↓
Grad-CAM
      ↓
Highlighted important region
```

The pipeline compares:

```text
Clean image Grad-CAM   vs.   Perturbed image Grad-CAM
```

A significant shift in highlighted regions is presented as evidence of **explanation instability** — but Grad-CAM output is never claimed to prove medical correctness.

---

## 14. Abstention Mechanism

A conventional classifier always outputs a prediction. MediShield instead adds a decision gate:

```text
Input
 ↓
Prediction
 ↓
Reliability score
 ↓
 ┌───────────────┐
 │               │
High            Low
 │               │
 ↓               ↓
Accept         Abstain
                 ↓
           Human review
```

Example UI state:

```text
Prediction:        Melanoma
Model confidence:  91%
Reliability:       23%
Decision:          ⚠ SUSPICIOUS — HUMAN REVIEW RECOMMENDED
```

This is a prototype **safety behavior**, not a clinical recommendation engine.

---

## 15. Coverage vs. Risk

A key evaluation concept for selective prediction systems.

Suppose the system receives 100 images and accepts 80 while abstaining on 20:

```text
Coverage = 80 / 100 = 80%
```

If 2 of the 80 accepted predictions are wrong:

```text
Selective risk = 2 / 80 = 2.5%
```

Plotting selective risk against coverage as the reliability threshold varies is a much stronger evaluation result than reporting plain accuracy alone — it directly shows the value of abstention.

---

## 16. Experiments

| ID | Experiment | Input | Metrics |
|---|---|---|---|
| E1 | Baseline EfficientNet | Clean images | Accuracy, Precision, Recall, F1, Confusion matrix |
| E2 | ResNet18 Benchmark | Clean images | Accuracy, F1, Confusion matrix |
| E3 | FGSM Attack | FGSM-perturbed images | Clean vs. FGSM accuracy, accuracy drop, confidence change, attack success rate |
| E4 | PGD Attack | PGD-perturbed images | Clean vs. PGD accuracy, accuracy drop, confidence change, attack success rate |
| E5 | Reliability Detector | Reliability feature vectors | Accuracy, Precision, Recall, F1, ROC-AUC |
| E6 | Stress Test Matrix | Clean, FGSM, PGD, Noise, Blur, Brightness, Compression | Accuracy, confidence, prediction consistency, reliability score |
| E7 | Abstention | Reliability scores + threshold | Coverage, selective risk, accepted-prediction error, abstention rate |
| E8 | Grad-CAM | Clean vs. perturbed images | Qualitative comparison (+ similarity metric if time permits) |

**Required minimum outputs:** baseline metrics, confusion matrix, FGSM results, PGD results, reliability detector metrics, stress-test comparison, coverage-risk plot, Grad-CAM examples, and a working demo.

---

## 17. Results (Template)

*(Fill in after running the experiments above — all values below are placeholders.)*

**Classification**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | TBD | TBD | TBD | TBD |
| ResNet18 | TBD | TBD | TBD | TBD |

**Adversarial Robustness**

| Condition | Accuracy | Avg Confidence | Prediction Change |
|---|---:|---:|---:|
| Clean | TBD | TBD | — |
| FGSM | TBD | TBD | TBD |
| PGD | TBD | TBD | TBD |

**Stress Testing**

| Condition | Accuracy | Confidence | Reliability |
|---|---:|---:|---:|
| Clean | TBD | TBD | TBD |
| FGSM | TBD | TBD | TBD |
| PGD | TBD | TBD | TBD |
| Noise | TBD | TBD | TBD |
| Blur | TBD | TBD | TBD |
| Brightness | TBD | TBD | TBD |
| Compression | TBD | TBD | TBD |

**Reliability Detector**

| Metric | XGBoost |
|---|---:|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |
| ROC-AUC | TBD |

---

## 18. Demo Application

The Streamlit dashboard follows this recommended interface:

```text
┌──────────────────────────────────────────┐
│              MEDISHIELD                  │
│     Medical AI Reliability Monitor       │
├──────────────────────────────────────────┤
│                                          │
│       [ Upload Skin Lesion Image ]       │
│                                          │
│ Prediction:       Melanoma               │
│ Confidence:       91%                    │
│ Reliability:      24%                    │
│                                          │
│ ⚠ SUSPICIOUS PREDICTION                 │
│ Human review recommended                 │
│                                          │
│ [ Show Grad-CAM ]                        │
│ [ Run Stress Test ]                      │
└──────────────────────────────────────────┘
```

**Expected demo story:**

1. **Upload a clean test image** → `Melanoma — 94% confidence, 96% reliability → TRUSTED`
2. **Click "Run FGSM"** → the system generates a perturbed version of the same image
3. **Compare predictions** → Original: `Melanoma — 94%` vs. Perturbed: `Nevus — 82%`
4. **Reliability layer reacts** → `Reliability: 18% → ⚠ SUSPICIOUS / ABSTAIN`
5. **Show Grad-CAM** → clean explanation vs. perturbed explanation, side by side

Key message conveyed by the demo:
> The system is not only making a prediction. It is checking whether the prediction remains trustworthy under stress.

---

## 19. Technology Stack

| Category | Tools |
|---|---|
| Machine learning | Python, PyTorch, Torchvision, XGBoost, Scikit-learn |
| Computer vision | PIL, OpenCV (as needed) |
| Explainability | Grad-CAM implementation/library |
| Dashboard | Streamlit |
| Development | Google Colab (T4 GPU) for training/attacks/evaluation; VS Code for local dev; Git/GitHub for version control |

| Component | Technology | Purpose |
|---|---|---|
| Main classifier | EfficientNet-B0 | Medical image classification |
| Benchmark | ResNet18 | Architecture comparison |
| Reliability model | XGBoost | Predict prediction reliability |
| Attack 1 | FGSM | Adversarial robustness |
| Attack 2 | PGD | Stronger adversarial robustness |
| XAI | Grad-CAM | Visual explanation |
| Dashboard | Streamlit | Interactive demo |

---

## 20. Project Structure

```text
MediShield/
│
├── README.md
│
├── data/
│   ├── metadata/            # HAM10000 metadata (image_id, lesion_id, dx, etc.)
│   └── processed/           # Balanced, lesion-aware train/val/test splits
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_efficientnet_training.ipynb
│   ├── 03_resnet_training.ipynb
│   ├── 04_adversarial_attacks.ipynb
│   ├── 05_reliability_detector.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── data/                # Dataset loading, lesion-level splitting, transforms
│   ├── models/               # EfficientNet-B0 / ResNet18 definitions and training
│   ├── attacks/              # FGSM, PGD, and stress-test transforms
│   ├── reliability/           # Feature engineering + XGBoost training/inference
│   ├── explainability/        # Grad-CAM implementation
│   └── evaluation/            # Metrics, coverage-risk analysis, plotting
│
├── models/
│   ├── efficientnet.pth
│   ├── resnet18.pth
│   └── reliability_xgb.json
│
├── results/
│   ├── metrics/               # Saved JSON/CSV metrics per experiment
│   └── figures/               # Confusion matrices, robustness plots, Grad-CAM outputs
│
└── app/
    └── streamlit_app.py       # Interactive demo
```

---

## 21. How to Run

> Exact commands depend on the final implementation produced during development — update this section once `src/` and `app/streamlit_app.py` exist.

**1. Environment setup**

```bash
git clone <repository-url>
cd MediShield
pip install torch torchvision xgboost scikit-learn pillow streamlit
```

**2. Prepare the dataset**

```bash
# Download HAM10000 and place metadata/images under data/
jupyter notebook notebooks/01_data_preparation.ipynb
```

**3. Train the models**

```bash
jupyter notebook notebooks/02_efficientnet_training.ipynb
jupyter notebook notebooks/03_resnet_training.ipynb
```

**4. Run adversarial evaluation and train the reliability detector**

```bash
jupyter notebook notebooks/04_adversarial_attacks.ipynb
jupyter notebook notebooks/05_reliability_detector.ipynb
jupyter notebook notebooks/06_evaluation.ipynb
```

**5. Launch the interactive demo**

```bash
streamlit run app/streamlit_app.py
```

**Recommended hardware:** GPU (e.g., Google Colab T4) for training EfficientNet-B0/ResNet18 and for generating adversarial examples. CPU is sufficient for the Streamlit demo and for XGBoost training/inference.

---

## 22. Scientific Limitations

Stated plainly, for transparency:

- HAM10000 is a benchmark dataset, not a complete representation of real-world clinical populations.
- The three-class setup (`bkl`, `mel`, `nv`) is a narrowed research scope, not full dermatological coverage.
- Adversarial attacks (FGSM, PGD) are controlled experiments; they do not simulate every real-world failure mode.
- Reliability detection based on prediction correctness does **not** prove clinical safety.
- Grad-CAM explanations are not guaranteed to represent true medical reasoning — they highlight what the CNN attended to, not necessarily clinically meaningful features.
- The system is **not** a replacement for a dermatologist or a clinical diagnostic workflow.

**Preferred language:** "research prototype," "model reliability," "prediction-level safety mechanism," "human-review escalation."

**Avoid:** "clinically safe," "prevents misdiagnosis," "doctor replacement."

---

## 23. Final Project Definition

**MediShield** is a research prototype that adds an AI safety and reliability layer to a medical vision classifier. It combines EfficientNet-based skin-lesion classification with adversarial stress testing, XGBoost-based reliability detection, explainability, and selective abstention. The system evaluates not only *what* the model predicts, but *whether the model's behavior provides evidence that the prediction can be trusted*.

> "MediShield challenges the assumption that a confident medical AI prediction is automatically trustworthy. We stress-test the model, learn its reliability patterns, detect suspicious predictions, and allow the system to abstain when confidence alone is not enough."