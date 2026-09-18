# MediShield — Progress & Implementation Plan

This document tracks the repository structure, file responsibilities, phased implementation plan, and the current progress of the MediShield project. It synthesizes the goals from the Implementation Guide, Improvement Plan, and recent architectural pivots (FastAPI integration & Custom XAI Metrics).

---

## 1. Repository Structure & File Dictionary

This section outlines the target repository structure and what each file is used for in the production pipeline.

```text
MediShield/
├── README.md                      # High-level project overview, scientific context, and results template
├── requirements.txt               # Core dependencies (PyTorch, FastAPI, Quantus, XGBoost, etc.)
├── progress.md                    # THIS FILE: Tracks implementation phases, features, and progress
├── config/
│   └── config.yaml                # Central hyperparameters, paths, and safety thresholds
├── data/
│   ├── metadata/                  # HAM10000 metadata & OOD metadata
│   └── processed/                 # Balanced, lesion-aware train/val/test splits + cache
├── models/                        # Saved model weights (Downloaded from Kaggle)
│   ├── efficientnet_b0.pth
│   ├── resnet18.pth
│   └── reliability_xgb.json

├── api/                           # Production FastAPI Backend Service
│   ├── main.py                    # Application entry point & POST /predict route
│   ├── dependencies.py            # Singleton pattern loaders to keep CNN & XGBoost in memory
│   ├── schemas.py                 # Pydantic schemas (PredictionResponse structure)
│   └── services/
│       ├── pipeline.py            # Orchestrates the end-to-end inference flow (Preprocess -> Classify -> XAI -> Decide)
│       └── gemini_service.py      # Translates numerical metrics into a natural language clinical summary via LLM

├── src/
│   ├── utils/
│   │   ├── seed.py                # Global seed setting for reproducibility
│   │   └── config.py              # Configuration loader
│   ├── data/
│   │   ├── splitting.py           # Lesion-level split logic to prevent leakage
│   │   ├── dataset.py             # PyTorch Dataset and DataLoader wrappers
│   │   └── transforms.py          # Image augmentation and preprocessing (train/val/test)
│   ├── models/
│   │   ├── efficientnet.py        # EfficientNet-B0 definition (Main Classifier)
│   │   └── resnet.py              # ResNet18 definition (Benchmark Model)
│   ├── attacks/
│   │   ├── fgsm.py                # Fast Gradient Sign Method attack
│   │   ├── pgd.py                 # Projected Gradient Descent attack
│   │   └── stress_transforms.py   # Realistic corruptions (noise, blur, brightness, compression)
│   ├── reliability/
│   │   ├── features.py            # Feature engineering: margin, entropy
│   │   ├── baselines.py           # MSP, temperature scaling, entropy baselines
│   │   ├── xgb_detector.py        # XGBoost training, evaluation, and inference wrappers
│   │   └── shap_analysis.py       # SHAP explanations for XGBoost reliability features
│   ├── explainability/
│   │   ├── gradcam.py             # Wrapper for pytorch-grad-cam library
│   │   ├── deletion_insertion.py  # Faithfulness Area-Under-Curve (via Quantus)
│   │   ├── stability_metrics.py   # Custom SSIM-based Robustness & Consistency under perturbations
│   │   └── quality_score.py       # Aggregation into single quality score (Q = 50% Faithfulness + 50% Stability)
│   ├── decision/
│   │   └── abstention.py          # Logic to Accept/Abstain based on Reliability & Explainability
│   └── evaluation/
│       ├── coverage_risk.py       # Coverage vs. selective-risk analysis
│       ├── calibration.py         # Expected Calibration Error (ECE) and reliability diagrams
│       ├── bootstrap_ci.py        # Confidence intervals for headline metrics
│       ├── kfold.py               # K-fold cross validation harness
│       └── subgroup_analysis.py   # Accuracy/Reliability breakdown by age, sex, localization
└── results/                       # Saved outputs (metrics and figures)
```

---

## 2. Phase-by-Phase Implementation Plan

### **Phase 1: Environment & Data Pipeline** (Status: 🟢 Completed)
*Setting up the reproducible foundation and avoiding data leakage.*
- [x] Identify dataset (HAM10000) and classes (`bkl`, `mel`, `nv`).
- [x] Define global seed setting for reproducibility (`src/utils/seed.py`).
- [x] Implement lesion-level splitting to prevent train/test leakage (`src/data/splitting.py`).
- [x] Implement class balancing (target ~600/class).
- [x] Implement PyTorch Datasets and data augmentation (`src/data/dataset.py`, `src/data/transforms.py`).

### **Phase 2: Base Classification Models** (Status: 🟢 Completed)
*Training the primary model and the benchmark.*
- [x] Implement and train EfficientNet-B0 Main Classifier (`src/models/efficientnet.py`).
- [x] Implement and train ResNet18 Benchmark Classifier (`src/models/resnet.py`).
- [x] Implement inference hook to extract probabilities, confidence, and deep features (`get_prediction_bundle`).

### **Phase 3: Adversarial & Stress Testing** (Status: 🟢 Completed)
*Applying pressure to evaluate model robustness.*
- [x] Implement FGSM attack (`src/attacks/fgsm.py`).
- [x] Implement PGD attack (`src/attacks/pgd.py`).
- [x] Implement severity-graded realistic stress transforms: noise, blur, brightness (`src/attacks/stress_transforms.py`).
- [x] Build evaluation harness to measure accuracy drops and confidence changes under attack.

### **Phase 4: Reliability Detector & Baselines** (Status: 🟢 Completed)
*Evaluating if the model knows when it's wrong.*
- [x] Implement reliability feature extraction (margin, entropy, perturbation-delta) (`src/reliability/features.py`).
- [x] Build the tabular dataset mapping CNN behavior to ground-truth correctness.
- [x] Train XGBoost Reliability Detector and evaluate AUROC (`src/reliability/xgb_detector.py`).
- [x] Implement Reliability Baselines: MSP, Temperature Scaling, Entropy (`src/reliability/baselines.py`).

### **Phase 5: Explainability & Quality Scoring** (Status: 🟢 Completed)
*Visualizing decisions and rigorously quantifying explanation quality using custom math.*
- [x] Implement Grad-CAM wrapping the `pytorch-grad-cam` library (`src/explainability/gradcam.py`).
- [x] Implement Faithfulness metric using Deletion/Insertion AUC via `quantus` (`src/explainability/deletion_insertion.py`).
- [x] Implement custom **Robustness Score (R)** using SSIM over noise perturbations (`src/explainability/stability_metrics.py`).
- [x] Implement custom **Consistency Score (C)** using pairwise SSIM across geometric augmentations (`src/explainability/stability_metrics.py`).
- [x] Aggregate Faithfulness, Robustness, and Consistency into a unified Explainability Quality Score `Q` (`src/explainability/quality_score.py`).

### **Phase 6: Abstention Logic & Advanced Rigor** (Status: 🟢 Completed)
*Making the safety decision and ensuring statistical validity.*
- [x] Implement dual-gate Abstention Mechanism (Reject if Reliability < T_R OR Quality < T_Q) (`src/decision/abstention.py`).
- [x] Add Expected Calibration Error (ECE) diagnostics (`src/evaluation/calibration.py`).
- [x] Run SHAP analysis on XGBoost features to explain the safety layer (`src/reliability/shap_analysis.py`).
- [x] Implement Bootstrap Confidence Intervals and K-Fold CV (`src/evaluation/bootstrap_ci.py`, `src/evaluation/kfold.py`).
- [x] Conduct Subgroup Analysis (Age, Sex, Localization) (`src/evaluation/subgroup_analysis.py`).

### **Phase 7: Demo Application & Reporting** (Status: 🟢 Completed)
*Showcasing the pipeline as a production microservice.*
- [x] Build FastAPI REST API architecture (`api/main.py`).
- [x] Orchestrate end-to-end pipeline in a single inference call (`api/services/pipeline.py`).
- [x] Integrate Gemini LLM for natural-language clinical reporting (`api/services/gemini_service.py`).
- [x] Encode Grad-CAM overlay as Base64 for seamless frontend rendering.
- [x] Finalize `requirements.txt` dependencies.

---

## 3. Feature Tracking

| Feature | Subsystem | Status | Priority |
| :--- | :--- | :--- | :--- |
| Lesion-Level Split | Data Pipeline | 🟢 Completed | High |
| Data Augmentation | Data Pipeline | 🟢 Completed | High |
| EfficientNet-B0 Classifier | Models | 🟢 Completed | High |
| ResNet18 Benchmark | Models | 🟢 Completed | High |
| Feature Extraction Hook | Models | 🟢 Completed | High |
| FGSM Attack | Stress Testing | 🟢 Completed | High |
| PGD Attack | Stress Testing | 🟢 Completed | High |
| Image Corruptions | Stress Testing | 🟢 Completed | High |
| XGBoost Reliability Model | Reliability | 🟢 Completed | High |
| Reliability Baselines (MSP, Temp) | Reliability | 🟢 Completed | High |
| Grad-CAM Output | Explainability | 🟢 Completed | High |
| Faithfulness (Quantus) | Explainability | 🟢 Completed | High |
| Custom Robustness (SSIM) | Explainability | 🟢 Completed | High |
| Custom Consistency (SSIM) | Explainability | 🟢 Completed | High |
| Explainability Quality Score (Q) | Explainability | 🟢 Completed | High |
| Dual-Gate Abstention Layer | Decision | 🟢 Completed | High |
| Coverage-Risk Analysis | Evaluation | 🟢 Completed | High |
| Calibration Diagnostics (ECE) | Evaluation | 🟢 Completed | Medium |
| SHAP on XGBoost | Evaluation | 🟢 Completed | Medium |
| Bootstrap CIs | Evaluation | 🟢 Completed | High |
| Subgroup Analysis | Evaluation | 🟢 Completed | Medium |
| FastAPI REST Backend | Architecture | 🟢 Completed | Critical |
| Gemini Text Summary | Architecture | 🟢 Completed | High |
