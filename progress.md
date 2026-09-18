# MediShield — Progress & Implementation Plan

This document tracks the repository structure, file responsibilities, phased implementation plan, and the current progress of the MediShield project. It synthesizes the goals from the Implementation Guide and Improvement Plan.

---

## 1. Repository Structure & File Dictionary

This section outlines the target repository structure and what each file is used for.

```text
MediShield/
├── README.md                      # High-level project overview, scientific context, and results template
├── requirements.txt               # Core dependencies (PyTorch, XGBoost, Streamlit, etc.)
├── progress.md                    # THIS FILE: Tracks implementation phases, features, and progress
├── config/
│   └── config.yaml                # Central hyperparameters, paths, and thresholds
├── data/
│   ├── metadata/                  # HAM10000 metadata (e.g., image_id, lesion_id) & OOD metadata
│   └── processed/                 # Balanced, lesion-aware train/val/test splits + cache
├── notebooks/                     # Jupyter notebooks for prototyping and running experiments
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
│   │   ├── features.py            # Feature engineering: margin, entropy, perturbation-deltas
│   │   ├── baselines.py           # MSP, temperature scaling, entropy, MC-dropout baselines
│   │   ├── xgb_detector.py        # XGBoost training, evaluation, and inference
│   │   └── shap_analysis.py       # SHAP explanations for XGBoost reliability features
│   ├── explainability/
│   │   ├── gradcam.py             # Grad-CAM map generation
│   │   ├── deletion_insertion.py  # Faithfulness metric (region reduction)
│   │   ├── stability_metrics.py   # Robustness + consistency (SSIM-based)
│   │   ├── superpixel_similarity.py # SLIC superpixel alignment scoring
│   │   └── quality_score.py       # Aggregation into single quality score (Q)
│   ├── decision/
│   │   └── abstention.py          # Logic to Accept/Abstain based on Reliability & Explainability
│   └── evaluation/
│       ├── coverage_risk.py       # Coverage vs. selective-risk analysis
│       ├── calibration.py         # Expected Calibration Error (ECE) and reliability diagrams
│       ├── bootstrap_ci.py        # Confidence intervals for headline metrics
│       ├── kfold.py               # K-fold cross validation harness
│       └── subgroup_analysis.py   # Accuracy/Reliability breakdown by age, sex, localization
├── models/                        # Saved model weights
│   ├── efficientnet.pth
│   ├── resnet18.pth
│   └── reliability_xgb.json
├── results/                       # Saved outputs
│   ├── metrics/                   # Saved JSON/CSV metrics per experiment
│   └── figures/                   # Plots (Coverage-Risk, SHAP, Grad-CAM overlays)
└── app/
    └── streamlit_app.py           # Interactive Streamlit Demo Dashboard
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

### **Phase 3: Adversarial & Stress Testing** (Status: 🔴 Pending)
*Applying pressure to evaluate model robustness.*
- [ ] Implement FGSM attack (`src/attacks/fgsm.py`).
- [ ] Implement PGD attack (`src/attacks/pgd.py`).
- [ ] Implement severity-graded realistic stress transforms: noise, blur, brightness, compression (`src/attacks/stress_transforms.py`).
- [ ] Build evaluation harness to measure accuracy drops and confidence changes under attack.

### **Phase 4: Reliability Detector & Baselines** (Status: 🔴 Pending)
*Evaluating if the model knows when it's wrong.*
- [ ] Implement reliability feature extraction (margin, entropy, perturbation-delta) (`src/reliability/features.py`).
- [ ] Build the tabular dataset mapping CNN behavior to ground-truth correctness (`build_training_table.py`).
- [ ] Train XGBoost Reliability Detector and evaluate AUROC (`src/reliability/xgb_detector.py`).
- [ ] Implement Reliability Baselines: MSP, Temperature Scaling, Entropy (`src/reliability/baselines.py`).
- [ ] Compare XGBoost vs. Baselines via Coverage-Risk curves (Experiment E9).

### **Phase 5: Explainability & Quality Scoring** (Status: 🔴 Pending)
*Visualizing decisions and quantifying explanation quality.*
- [ ] Implement Grad-CAM (`src/explainability/gradcam.py`).
- [ ] Implement Deletion/Insertion curves for faithfulness (`src/explainability/deletion_insertion.py`).
- [ ] Implement SSIM-based Stability Metrics (Robustness and Consistency) (`src/explainability/stability_metrics.py`).
- [ ] *(Optional/Medium)* Implement Superpixel alignment score (`src/explainability/superpixel_similarity.py`).
- [ ] Aggregate into a unified Explainability Quality Score `Q` (`src/explainability/quality_score.py`).

### **Phase 6: Abstention Logic & Advanced Rigor** (Status: 🔴 Pending)
*Making the safety decision and ensuring statistical validity.*
- [ ] Implement dual-gate Abstention Mechanism (Reject if Reliability < T_R OR Quality < T_Q) (`src/decision/abstention.py`).
- [ ] Add Expected Calibration Error (ECE) diagnostics (`src/evaluation/calibration.py`).
- [ ] Run SHAP analysis on XGBoost features to explain the safety layer (`src/reliability/shap_analysis.py`).
- [ ] Implement Bootstrap Confidence Intervals and K-Fold CV (`src/evaluation/bootstrap_ci.py`, `src/evaluation/kfold.py`).
- [ ] Conduct Subgroup Analysis (Age, Sex, Localization) (`src/evaluation/subgroup_analysis.py`).
- [ ] *(Optional/Medium)* Evaluate on External OOD Dataset (ISIC 2019/BCN20000).

### **Phase 7: Demo Application & Reporting** (Status: 🔴 Pending)
*Showcasing the pipeline.*
- [ ] Build interactive Streamlit Dashboard (`app/streamlit_app.py`).
- [ ] Integrate full pipeline in Demo (Upload -> Classify -> Stress Test -> Reliability Score -> Grad-CAM).
- [ ] Finalize README with metrics populated in results template.

---

## 3. Feature Tracking

| Feature | Subsystem | Status | Priority |
| :--- | :--- | :--- | :--- |
| Lesion-Level Split | Data Pipeline | 🟢 Completed | High |
| Data Augmentation | Data Pipeline | 🟢 Completed | High |
| EfficientNet-B0 Classifier | Models | 🟢 Completed | High |
| ResNet18 Benchmark | Models | 🟢 Completed | High |
| Feature Extraction Hook | Models | 🟢 Completed | High |
| FGSM Attack | Stress Testing | 🔴 Pending | High |
| PGD Attack | Stress Testing | 🔴 Pending | High |
| Image Corruptions | Stress Testing | 🔴 Pending | High |
| XGBoost Reliability Model | Reliability | 🔴 Pending | High |
| Reliability Baselines (MSP, Temp) | Reliability | 🔴 Pending | High |
| Grad-CAM | Explainability | 🔴 Pending | High |
| Explainability Stability (SSIM) | Explainability | 🔴 Pending | High |
| Faithfulness (Deletion/Insertion) | Explainability | 🔴 Pending | High |
| Explainability Quality Score (Q) | Explainability | 🔴 Pending | Medium |
| Dual-Gate Abstention Layer | Decision | 🔴 Pending | High |
| Coverage-Risk Analysis | Evaluation | 🔴 Pending | High |
| Calibration Diagnostics (ECE) | Evaluation | 🔴 Pending | Medium |
| SHAP on XGBoost | Evaluation | 🔴 Pending | Medium |
| Bootstrap CIs | Evaluation | 🔴 Pending | High |
| Subgroup Analysis | Evaluation | 🔴 Pending | Medium |
| OOD Validation | Generalization | 🔴 Pending | Medium-High |
| Cross-Arch Transfer Test | Generalization | 🔴 Pending | Medium |
| Streamlit Demo App | Demo | 🔴 Pending | High |
