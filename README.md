<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A0E14,50:0D1B2A,100:0A0E14&height=210&section=header&text=MediShield&fontSize=58&fontColor=E6EDF3&animation=fadeIn&fontAlignY=36&desc=A%20Reliability%20%26%20Safety%20Layer%20for%20Medical%20Vision%20Models&descAlignY=58&descSize=17" width="100%"/>

<a href="https://github.com/iniya304/MediShield">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=20&duration=3200&pause=900&color=4FD1FF&center=true&vCenter=true&multiline=true&repeat=true&width=720&height=54&lines=Can+we+trust+the+model's+prediction%3F;Not+classification+alone+%E2%80%94+reliability.;Attack+%E2%80%A2+Detect+%E2%80%A2+Abstain+%E2%80%A2+Explain." alt="MediShield tagline" />
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/status-research%20prototype-8B5CF6?style=for-the-badge&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/python-3.10+-4FD1FF?style=for-the-badge&logo=python&logoColor=0D1117&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/PyTorch-EfficientNet%20%7C%20ResNet18-4FD1FF?style=for-the-badge&logo=pytorch&logoColor=0D1117&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/XGBoost-Reliability%20Detector-8B5CF6?style=for-the-badge&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/Streamlit-Demo%20App-4FD1FF?style=for-the-badge&logo=streamlit&logoColor=0D1117&labelColor=0D1117" />
</p>

<p>
  <img src="https://img.shields.io/github/last-commit/iniya304/MediShield?style=flat-square&color=4FD1FF&labelColor=0D1117" />
  <img src="https://img.shields.io/github/languages/top/iniya304/MediShield?style=flat-square&color=8B5CF6&labelColor=0D1117" />
  <img src="https://img.shields.io/github/stars/iniya304/MediShield?style=flat-square&color=4FD1FF&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/dataset-HAM10000-8B5CF6?style=flat-square&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/license-MIT-4FD1FF?style=flat-square&labelColor=0D1117" />
</p>

</div>

<br/>

<table>
<tr><td>

**MediShield** does not simply ask *"What does the medical image show?"* — it asks *"Can we trust the model's prediction?"*

> ⚠️ **Research prototype, not a clinical tool.** MediShield should never be presented, marketed, or used as a substitute for a dermatologist or a clinical diagnostic workflow.

</td></tr>
</table>

<br/>

<div align="center">

```
Image ──▶ EfficientNet-B0 ──▶ Prediction ──▶ Stress-Test ──▶ XGBoost Reliability ──▶ Accept / Abstain ──▶ Grad-CAM
```

</div>

<br/>

## Contents

<table>
<tr valign="top">
<td width="33%">

**Concept**
- [Overview](#overview)
- [The Core Problem](#the-core-problem)
- [Research Question](#research-question)
- [Beyond Classification](#beyond-classification)

</td>
<td width="33%">

**System**
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Data Splitting Rules](#data-splitting-rules)
- [Preprocessing](#image-preprocessing)
- [Models](#models)
- [Reliability Detector](#reliability-detector-xgboost)

</td>
<td width="33%">

**Evaluation**
- [Adversarial ML](#adversarial-machine-learning)
- [Stress Testing](#realistic-stress-testing)
- [Explainability](#explainable-ai-grad-cam)
- [Abstention](#abstention-mechanism)
- [Coverage vs. Risk](#coverage-vs-risk)
- [Experiments](#experiments)
- [Results](#results-template)

</td>
</tr>
</table>

**Reference:** [Demo Application](#demo-application) · [Tech Stack](#technology-stack) · [Project Structure](#project-structure) · [How to Run](#how-to-run) · [Limitations](#scientific-limitations) · [Final Definition](#final-project-definition)

---

## Overview

MediShield is a research-oriented prototype for testing and improving the reliability of a medical image classification model. It uses a deep-learning skin-lesion classifier as the **prediction engine**, then wraps it in a **reliability layer** that evaluates whether each individual prediction actually deserves to be trusted.

The system combines several fields into one closed evaluation loop:

<table>
<tr>
<td width="50%" valign="top">

- Medical computer vision
- Deep learning (transfer learning with CNNs)
- Adversarial machine learning
- Reliability / uncertainty analysis

</td>
<td width="50%" valign="top">

- Gradient-boosted trees (XGBoost)
- Explainable AI (Grad-CAM)
- Failure detection
- Selective prediction / abstention & robustness stress testing

</td>
</tr>
</table>

> **Pitch it as:** *"We built a reliability and safety layer around a medical vision model and tested whether it can recognize when its own prediction becomes unreliable."*
> **Not as:** *"We trained a skin-cancer classifier."*

---

## The Core Problem

A conventional medical-image AI pipeline looks like this:

```
Medical Image ──▶ CNN ──▶ Disease Prediction
```

For example:

```
Skin lesion ──▶ EfficientNet-B0 ──▶ Melanoma — 94%
```

The problem: **a high-confidence prediction is not automatically a trustworthy prediction.** A model can —

| Failure mode | What it looks like |
|---|---|
| Confident but wrong | High-confidence output that is factually incorrect |
| Unstable | Prediction flips after small, imperceptible input changes |
| Adversarially fragile | Reacts badly to deliberately crafted perturbations |
| Degradation-sensitive | Prediction shifts under noise, blur, compression, or brightness changes |
| Explanation drift | Grad-CAM attention shifts substantially under perturbation |
| Out-of-distribution | Encounters an input unlike anything in its training distribution |

MediShield adds a reliability layer on top of the raw classifier:

```
Image
  │
  ▼
Medical Vision Model
  │
  ▼
Prediction + Confidence + Features
  │
  ▼
Reliability Analysis
  │
  ▼
Trust / Suspicious
  │
  ▼
Accept prediction  OR  Abstain
```

---

## Research Question

**Primary question**

> Can we detect when a medical vision model's prediction has become unreliable — especially under adversarial or realistic image perturbations?

**Secondary questions**

1. How much does model performance degrade under FGSM and PGD attacks?
2. Does model confidence change when the input is perturbed?
3. Can behavioral features help detect incorrect predictions?
4. Can an XGBoost reliability model distinguish trustworthy from suspicious predictions?
5. Can an abstention mechanism reduce error among predictions the system chooses to accept?
6. Do model explanations (Grad-CAM) change when the input is manipulated?

---

## Beyond Classification

Classification is only the first layer. The actual contribution stacks several layers on top of it:

```
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

The strongest technical narrative is a six-step chain — not the number of models involved:

<div align="center">

**Classify → Attack → Observe behavior → Detect unreliability → Abstain → Explain**

</div>

---

## Architecture

<details open>
<summary><b>System overview</b></summary>

```
                              MEDISHIELD
                                  │
                                  ▼
                         ┌────────────────┐
                         │   Input Image  │
                         └───────┬────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  EfficientNet-B0 │
                        │  Main Classifier │
                        └────────┬─────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
            Probabilities    Confidence    Deep Features
                  │              │              │
                  └──────────────┼──────────────┘
                                 │
                                 ▼
                      ┌────────────────────┐
                      │ Reliability Feature│
                      │       Engine       │
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

Running alongside the main pipeline is a **stress-testing / robustness loop**, which feeds perturbed images back through the same pipeline to observe how behavior changes under pressure:

```
Clean image
    │
    ▼
FGSM / PGD / Noise / Blur / Brightness / Compression
    │
    ▼
Stress testing
    │
    ▼
Model behavior
    │
    ▼
Reliability analysis
```

</details>

### Component responsibilities

| Layer | Component | Responsibility |
|---|---|---|
| Prediction engine | EfficientNet-B0 | Classify the lesion; expose probabilities, confidence, and deep features |
| Benchmark | ResNet18 | Independent architecture used to check whether reliability issues are model-specific |
| Stress layer | FGSM, PGD, noise, blur, brightness, compression | Perturb the input to probe robustness |
| Feature engine | Custom feature extraction code | Turn raw model outputs + perturbation behavior into a tabular feature vector |
| Reliability detector | XGBoost (binary classifier) | Predict whether the classifier's prediction is likely correct/reliable |
| Decision layer | Threshold-based abstention logic | Convert a reliability score into an Accept / Abstain decision |
| Explainability | Grad-CAM | Visualize which image regions drove the CNN's decision, clean vs. perturbed |
| Interface | Streamlit dashboard | Upload an image, run the full pipeline, visualize every stage interactively |

<details>
<summary><b>End-to-end data flow</b></summary>

```
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

</details>

### Why two separately-trained models

EfficientNet-B0 is **not** trained to know whether it is right or wrong — it only outputs a softmax distribution over classes. XGBoost is trained **separately**, on a different signal entirely: not *"which disease is this?"* but *"does this specific prediction look trustworthy, given how the classifier behaved?"* This separation is intentional — it lets the reliability detector learn patterns (e.g. low margin, high entropy, large confidence swings under perturbation) that correlate with classifier error, independent of what the underlying disease actually is.

---

## Dataset

### HAM10000

MediShield uses the **HAM10000** ("Human Against Machine with 10000 training images") skin-lesion dataset.

The full dataset contains **10,015** dermoscopic images across **7** diagnostic categories, with metadata for image ID, lesion ID, diagnosis, age, sex, localization, and dataset source.

MediShield restricts itself to **three classes**:

| Code | Class | Images (full dataset) |
|---|---|---:|
| `bkl` | Benign keratosis-like lesions | 1,099 |
| `mel` | Melanoma | 1,113 |
| `nv` | Melanocytic nevi | 6,705 |

These classes are highly imbalanced (`nv` dominates), so the project uses an **approximately balanced subset** rather than the full raw distribution.

<table>
<tr>
<td width="50%" valign="top">

**Target working subset**
```
~600 images / class
~1,800 total images
```
This is a target, not a hard requirement — the exact number is determined after lesion-level splitting.

</td>
<td width="50%" valign="top">

**Split ratios**
```
70%  Training
15%  Validation
15%  Test
```
Class balance is preserved as closely as possible while keeping all images of a given lesion together.

</td>
</tr>
</table>

---

## Data Splitting Rules

> Leakage prevention

HAM10000 contains **multiple images of the same physical lesion** (different angles/zoom/lighting of the same spot on the same patient). This makes naive random splitting dangerous.

**Bad split — causes leakage**

```
Lesion A
 ├── Image A1 → Training
 └── Image A2 → Test
```

Splitting at the image level lets lesion-specific visual cues leak from training into the test set, inflating apparent performance.

**Correct approach — split at the `lesion_id` level**

```
Lesion A → ALL images → Training      (or)
Lesion A → ALL images → Validation    (or)
Lesion A → ALL images → Test
```

> **Rule:** the same `lesion_id` must never appear in more than one of Train / Validation / Test.

---

## Image Preprocessing

| Step | Detail |
|---|---|
| Target tensor | `224 × 224 × 3`, RGB |
| Normalization | ImageNet mean/std — both EfficientNet-B0 and ResNet18 are ImageNet-pretrained |
| Training augmentation | Resize, crop, horizontal flip, small rotation — kept mild to avoid unrealistic medical images |
| Validation / test preprocessing | Deterministic only (resize + normalize) — no random augmentation, so evaluation is reproducible |

---

## Models

### EfficientNet-B0 — Main Classifier

**Role:** predict the lesion class and expose internal signals (probabilities, confidence, deep features) that the reliability layer consumes downstream. Trained via **transfer learning** (ImageNet-pretrained backbone, classification head replaced for 3 classes) rather than from scratch.

```
Image ──▶ EfficientNet-B0 ──▶ Class probabilities ──▶ Predicted class ──▶ Deep features
```

### ResNet18 — Benchmark Model

**Role:** determine whether the reliability problem is specific to one CNN architecture, or generalizes across architectures. ResNet18 is a comparison point, not the centerpiece — EfficientNet-B0 remains the primary model, and ResNet18 does not need extensive tuning.

---

## Reliability Detector (XGBoost)

XGBoost is **not** a disease classifier. Its job is to predict whether the deep-learning model's prediction is likely to be correct/reliable.

```
EfficientNet ──▶ Prediction, Confidence, Probabilities,
                  Feature statistics, Perturbation behavior
                        │
                        ▼
                    XGBoost
                        │
                        ▼
              Reliability probability
```

```
Reliability score = 0.94  →  prediction appears reliable
Reliability score = 0.18  →  prediction appears suspicious
```

The acceptance threshold is selected **experimentally**, using validation data (see [Abstention](#abstention-mechanism)).

### Reliability features

| Feature | Description |
|---|---|
| Prediction probabilities | `P(bkl)`, `P(mel)`, `P(nv)` |
| Confidence | `max_probability` |
| Prediction margin | Gap between top-1 and top-2 class probabilities (e.g. Melanoma 0.94 vs. Nevus 0.04 → margin = 0.90) |
| Entropy | How spread out the probability distribution is — `[0.94, 0.04, 0.02]` is far more decisive than `[0.36, 0.34, 0.30]`, even with the same top-1 class |
| Perturbation behavior | Confidence change, probability shift, class-flip flag, deep-feature shift, cross-perturbation consistency (when an attack/stress transform is applied) |

### Training the reliability detector

Reliability labels are derived from ground-truth correctness, **not** from an arbitrary confidence cutoff:

```
Ground-truth label
        │
        ▼
Compare with EfficientNet prediction
        │
     Correct?
   /        \
 Yes         No
  │           │
  ▼           ▼
Reliable    Unreliable
   1            0
```

> Reliability is **not** defined as `confidence > 0.5`. Doing so would make the detector reproduce an arbitrary threshold instead of learning a genuine signal. XGBoost instead learns the relationship **model behavior → prediction reliability**.

---

## Adversarial Machine Learning

One of MediShield's main differentiators: testing how the classifier behaves under **intentional** perturbation.

<table>
<tr>
<td width="50%" valign="top">

**FGSM** — Fast Gradient Sign Method

```
Clean image
    │
    ▼
Gradient-based perturbation
    │
    ▼
Adversarial image
    │
    ▼
Classifier
```

Measured: clean accuracy, adversarial accuracy, prediction changes, confidence changes, attack success rate.

</td>
<td width="50%" valign="top">

**PGD** — Projected Gradient Descent (stronger, iterative)

```
Clean image → Small perturbation
    │
    ▼
Model gradient → Update
    │
    ▼
Project perturbation → Repeat
    │
    ▼
Adversarial image
```

</td>
</tr>
</table>

> FGSM and PGD are **attack methods**, not separate trained models — they are transformations applied at evaluation time.

---

## Realistic Stress Testing

Adversarial attacks alone aren't representative of everyday failure modes, so MediShield also runs a stress-test suite of realistic image degradations:

```
Clean → FGSM → PGD → Gaussian noise → Blur → Brightness change → Compression
```

For each condition, the pipeline measures accuracy, confidence, whether the prediction changed, the reliability score, and the resulting abstention rate — giving a broader robustness picture than adversarial attacks alone.

---

## Explainable AI (Grad-CAM)

Grad-CAM is **not another model** — it's a visualization method layered on top of the trained CNN.

**Purpose:** show which image regions influenced the CNN's prediction.

```
Original image ──▶ Grad-CAM ──▶ Highlighted important region
```

The pipeline compares:

```
Clean image Grad-CAM   vs.   Perturbed image Grad-CAM
```

A significant shift in highlighted regions is presented as evidence of **explanation instability** — but Grad-CAM output is never claimed to prove medical correctness.

---

## Abstention Mechanism

A conventional classifier always outputs a prediction. MediShield instead adds a decision gate:

```
Input ──▶ Prediction ──▶ Reliability score
                              │
                    ┌─────────┴─────────┐
                   High                Low
                    │                   │
                    ▼                   ▼
                 Accept              Abstain
                                        │
                                        ▼
                                  Human review
```

**Example UI state**

```
Prediction:        Melanoma
Model confidence:  91%
Reliability:       23%
Decision:          ⚠ SUSPICIOUS — HUMAN REVIEW RECOMMENDED
```

> This is a prototype **safety behavior**, not a clinical recommendation engine.

---

## Coverage vs. Risk

A key evaluation concept for selective prediction systems.

Suppose the system receives 100 images and accepts 80 while abstaining on 20:

```
Coverage = 80 / 100 = 80%
```

If 2 of the 80 accepted predictions are wrong:

```
Selective risk = 2 / 80 = 2.5%
```

Plotting selective risk against coverage as the reliability threshold varies is a much stronger evaluation result than reporting plain accuracy alone — it directly shows the value of abstention.

---

## Experiments

| ID | Experiment | Input | Metrics | Status |
|---|---|---|---|---|
| E1 | Baseline EfficientNet | Clean images | Accuracy, Precision, Recall, F1, Confusion matrix | ![done](https://img.shields.io/badge/-done-4FD1FF?style=flat-square&labelColor=0D1117) |
| E2 | ResNet18 Benchmark | Clean images | Accuracy, F1, Confusion matrix | ![done](https://img.shields.io/badge/-done-4FD1FF?style=flat-square&labelColor=0D1117) |
| E3 | FGSM Attack | FGSM-perturbed images | Clean vs. FGSM accuracy, accuracy drop, confidence change, attack success rate | ![done](https://img.shields.io/badge/-done-4FD1FF?style=flat-square&labelColor=0D1117) |
| E4 | PGD Attack | PGD-perturbed images | Clean vs. PGD accuracy, accuracy drop, confidence change, attack success rate | ![done](https://img.shields.io/badge/-done-4FD1FF?style=flat-square&labelColor=0D1117) |
| E5 | Reliability Detector | Reliability feature vectors | Accuracy, Precision, Recall, F1, ROC-AUC | ![done](https://img.shields.io/badge/-done-4FD1FF?style=flat-square&labelColor=0D1117) |
| E6 | Stress Test Matrix | Clean, FGSM, PGD, Noise, Blur, Brightness, Compression | Accuracy, confidence, prediction consistency, reliability score | ![in progress](https://img.shields.io/badge/-in%20progress-8B5CF6?style=flat-square&labelColor=0D1117) |
| E7 | Abstention | Reliability scores + threshold | Coverage, selective risk, accepted-prediction error, abstention rate | ![in progress](https://img.shields.io/badge/-in%20progress-8B5CF6?style=flat-square&labelColor=0D1117) |
| E8 | Grad-CAM | Clean vs. perturbed images | Qualitative comparison (+ similarity metric if time permits) | ![in progress](https://img.shields.io/badge/-in%20progress-8B5CF6?style=flat-square&labelColor=0D1117) |

> Status badges are illustrative — update per-row to reflect actual progress (`done`, `in progress`, `todo`).

**Required minimum outputs:** baseline metrics, confusion matrix, FGSM results, PGD results, reliability detector metrics, stress-test comparison, coverage-risk plot, Grad-CAM examples, and a working demo.

---

## Results (Template)

<details>
<summary><b>Expand results tables — fill in after running experiments</b></summary>

*(All values below are placeholders — replace with real numbers.)*

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

</details>

---

## Demo Application

The Streamlit dashboard follows this recommended interface:

```
┌──────────────────────────────────────────┐
│               MEDISHIELD                 │
│     Medical AI Reliability Monitor        │
├──────────────────────────────────────────┤
│                                            │
│       [ Upload Skin Lesion Image ]        │
│                                            │
│ Prediction:       Melanoma                │
│ Confidence:       91%                     │
│ Reliability:      24%                     │
│                                            │
│ ⚠ SUSPICIOUS PREDICTION                   │
│ Human review recommended                  │
│                                            │
│ [ Show Grad-CAM ]                         │
│ [ Run Stress Test ]                       │
└──────────────────────────────────────────┘
```

**Expected demo story**

1. **Upload a clean test image** → `Melanoma — 94% confidence, 96% reliability → TRUSTED`
2. **Click "Run FGSM"** → the system generates a perturbed version of the same image
3. **Compare predictions** → Original: `Melanoma — 94%` vs. Perturbed: `Nevus — 82%`
4. **Reliability layer reacts** → `Reliability: 18% → ⚠ SUSPICIOUS / ABSTAIN`
5. **Show Grad-CAM** → clean explanation vs. perturbed explanation, side by side

> The system is not only making a prediction. It is checking whether the prediction remains trustworthy under stress.

> 💡 **Tip:** record a short screen-capture GIF of this exact flow and drop it here — `![demo](assets/demo.gif)` — a real animated demo GIF is the single biggest visual upgrade you can make to this README.

---

## Technology Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-4FD1FF?style=for-the-badge&logo=python&logoColor=0D1117&labelColor=0D1117)
![PyTorch](https://img.shields.io/badge/PyTorch-4FD1FF?style=for-the-badge&logo=pytorch&logoColor=0D1117&labelColor=0D1117)
![XGBoost](https://img.shields.io/badge/XGBoost-8B5CF6?style=for-the-badge&labelColor=0D1117)
![OpenCV](https://img.shields.io/badge/OpenCV-4FD1FF?style=for-the-badge&logo=opencv&logoColor=0D1117&labelColor=0D1117)
![Streamlit](https://img.shields.io/badge/Streamlit-8B5CF6?style=for-the-badge&logo=streamlit&logoColor=0D1117&labelColor=0D1117)
![scikit-learn](https://img.shields.io/badge/scikit--learn-4FD1FF?style=for-the-badge&logo=scikitlearn&logoColor=0D1117&labelColor=0D1117)
![Jupyter](https://img.shields.io/badge/Jupyter-8B5CF6?style=for-the-badge&logo=jupyter&logoColor=0D1117&labelColor=0D1117)
![Google Colab](https://img.shields.io/badge/Colab-4FD1FF?style=for-the-badge&logo=googlecolab&logoColor=0D1117&labelColor=0D1117)

</div>

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

## Project Structure

<details>
<summary><b>Expand full folder tree</b></summary>

```
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
│   ├── data/                  # Dataset loading, lesion-level splitting, transforms
│   ├── models/                # EfficientNet-B0 / ResNet18 definitions and training
│   ├── attacks/                # FGSM, PGD, and stress-test transforms
│   ├── reliability/             # Feature engineering + XGBoost training/inference
│   ├── explainability/          # Grad-CAM implementation
│   └── evaluation/              # Metrics, coverage-risk analysis, plotting
│
├── models/
│   ├── efficientnet.pth
│   ├── resnet18.pth
│   └── reliability_xgb.json
│
├── results/
│   ├── metrics/                 # Saved JSON/CSV metrics per experiment
│   └── figures/                 # Confusion matrices, robustness plots, Grad-CAM outputs
│
└── app/
    └── streamlit_app.py         # Interactive demo
```

</details>

---

## How to Run

> Exact commands depend on the final implementation produced during development — update this section once `src/` and `app/streamlit_app.py` exist.

**1. Environment setup**

```bash
git clone https://github.com/iniya304/MediShield.git
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

**Recommended hardware:** GPU (e.g. Google Colab T4) for training EfficientNet-B0/ResNet18 and for generating adversarial examples. CPU is sufficient for the Streamlit demo and for XGBoost training/inference.

---

## Scientific Limitations

Stated plainly, for transparency:

- HAM10000 is a benchmark dataset, not a complete representation of real-world clinical populations.
- The three-class setup (`bkl`, `mel`, `nv`) is a narrowed research scope, not full dermatological coverage.
- Adversarial attacks (FGSM, PGD) are controlled experiments; they do not simulate every real-world failure mode.
- Reliability detection based on prediction correctness does **not** prove clinical safety.
- Grad-CAM explanations are not guaranteed to represent true medical reasoning — they highlight what the CNN attended to, not necessarily clinically meaningful features.
- The system is **not** a replacement for a dermatologist or a clinical diagnostic workflow.

| Use this language | Avoid this language |
|---|---|
| "research prototype" | "clinically safe" |
| "model reliability" | "prevents misdiagnosis" |
| "prediction-level safety mechanism" | "doctor replacement" |
| "human-review escalation" | — |

---

## Final Project Definition

**MediShield** is a research prototype that adds an AI safety and reliability layer to a medical vision classifier. It combines EfficientNet-based skin-lesion classification with adversarial stress testing, XGBoost-based reliability detection, explainability, and selective abstention. The system evaluates not only *what* the model predicts, but *whether the model's behavior provides evidence that the prediction can be trusted*.

<div align="center">

<table>
<tr><td>

*"MediShield challenges the assumption that a confident medical AI prediction is automatically trustworthy. We stress-test the model, learn its reliability patterns, detect suspicious predictions, and allow the system to abstain when confidence alone is not enough."*

</td></tr>
</table>

<br/>

⭐ **If this project is useful to you, consider starring the repo.**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A0E14,50:0D1B2A,100:0A0E14&height=120&section=footer" width="100%"/>

</div>
