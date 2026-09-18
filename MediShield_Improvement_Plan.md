# MediShield — Improvement & Extension Plan

**Purpose:** a consolidated, prioritized list of changes/additions to strengthen MediShield across every subsystem — reliability detection, adversarial robustness, explainability, dataset/generalization, and evaluation rigor. Each item is tagged with priority (🔴 High / 🟡 Medium / 🟢 Nice-to-have) and rough effort (S/M/L).

One item (Section 4) adapts a quantitative XAI scoring framework from a related paper (Varrier et al., *"Quantitative Evaluation of Explanation Quality in Fetal Ultrasound Quality Criterion Detection Using Saliency Maps,"* IEEE ISBI 2026) — but this is one input among many below, not the basis for the whole project.

---

## 1. Reliability Detector — Prove It Adds Value

The core untested claim in MediShield right now is "XGBoost learns something more than raw confidence." Nothing currently proves that. Add baselines:

| Baseline | Description | Priority | Effort |
|---|---|---|---|
| Max Softmax Probability (MSP) | Use raw top-1 confidence directly as the reliability score | 🔴 High | S |
| Temperature-scaled confidence | Calibrate softmax outputs (temperature scaling on validation set) before using as reliability score | 🔴 High | S |
| Entropy-based score | Use prediction entropy directly as the reliability signal | 🟡 Medium | S |
| MC-Dropout / Deep Ensemble uncertainty | Predictive variance across stochastic forward passes or ensemble members, as a third uncertainty family | 🟢 Nice-to-have | M |

**Deliverable:** overlay all strategies (MSP, temp-scaled, entropy, MC-dropout, XGBoost) on one **coverage-vs-selective-risk plot**, plus AUROC of each as a "predict correctness" signal. If XGBoost doesn't clearly beat the simple baselines, report that honestly — it's still a valid finding and strengthens the paper's credibility.

### 1.1 Calibration diagnostics
Add Expected Calibration Error (ECE) and a reliability diagram for the base EfficientNet-B0 classifier, before and after temperature scaling.
🟡 Medium | S

### 1.2 SHAP on the reliability detector
Run SHAP on the trained XGBoost model to see which features (margin, entropy, perturbation-delta, confidence swing, etc.) actually drive its reliability predictions. Turns the reliability layer into something explainable in its own right, not a second black box on top of the first.
🟡 Medium | S

### 1.3 Feature ablation
Drop-one-feature-at-a-time (or use SHAP importance) to identify which engineered features matter most; report which are redundant.
🟢 Nice-to-have | S

---

## 2. Architecture & Generalization

### 2.1 External out-of-distribution validation
Pull a second dataset with overlapping classes (e.g., ISIC 2019/2020, BCN20000) for **evaluation only, never training**.
- Does raw accuracy degrade on genuinely OOD images (different devices/populations)?
- Does the reliability detector correctly flag these as lower-reliability, despite never training on that distribution?

This is a much stronger "does this generalize" story than perturbation-only stress testing.
🔴 High | M–L (dataset acquisition + preprocessing alignment)

### 2.2 Cross-architecture transfer test
Train the XGBoost reliability model on EfficientNet-B0 behavior; test whether it still separates reliable/unreliable predictions when scoring ResNet18 outputs (and vice versa). Cheap since ResNet18 is already a benchmark model.
🟡 Medium | S

### 2.3 Subgroup analysis
HAM10000 metadata includes age, sex, and lesion localization. Break down accuracy/reliability by subgroup — a reliability system that quietly fails worse on one subgroup is exactly what a "safety layer" project should be checking for.
🟡 Medium | S (metadata already available)

---

## 3. Adversarial & Stress-Testing Improvements

### 3.1 Formalized corruption benchmark
Replace ad hoc noise/blur/brightness/compression transforms with a structured, severity-graded suite (ImageNet-C-style: multiple severity levels per corruption type). Makes stress-test results comparable and reproducible rather than one-off.
🟢 Nice-to-have | M

### 3.2 Explanation stability under adversarial (not just benign) perturbation
Most explanation-robustness literature only tests benign perturbations (rotation, occlusion, blur). Testing how Grad-CAM degrades specifically under FGSM/PGD — which you already generate — is a genuine, not-commonly-done extension.
🔴 High | S (reuses existing attack pipeline)

---

## 4. Explainability Quantification (New Subsystem)

Currently Grad-CAM is treated qualitatively ("clean vs. perturbed side-by-side"). Add a proper quantitative scoring module: `src/explainability/quality_score.py`.

### 4.1 Faithfulness — Deletion/Insertion curves
Progressively mask (or reveal) the top-k% Grad-CAM-activated pixels and track how classification confidence changes. This directly tests whether highlighted pixels are *causally* driving the decision, rather than just looking plausible.
🔴 High | M

### 4.2 Stability metrics
- **Robustness score:** SSIM between Grad-CAM(clean) and Grad-CAM(perturbed) across a perturbation set — reuses your existing stress-test images, cheap to add.
- **Consistency score:** average pairwise SSIM across Grad-CAM maps from several augmented copies of the same image.
🔴 High | S

### 4.3 Superpixel/region-alignment score (adapted from Varrier et al., ISBI 2026)
SLIC-segment the image, compute per-superpixel mean activation, compare against a reference ("ideal") distribution via cosine similarity. Measures whether activation clusters in semantically coherent, lesion-relevant regions rather than scattering across background/artifacts.

**Adaptation note:** unlike the source paper (spatially anchored acquisition criteria), HAM10000 labels are diagnostic, not spatial, so the "ideal" reference pattern needs its own justification — e.g., a derived lesion mask, or a lesion-centroid heuristic as a fallback. Document this as an approximation.
🟡 Medium | M

### 4.4 Aggregate quality score
Combine the above into a single weighted score `Q` for reporting purposes, but:
- justify your own weights via a sensitivity analysis (sweep an acceptance threshold, e.g. 0.4–0.8, and report accepted/rejected rates and error among accepted) rather than copying another paper's weights
- always log sub-scores individually, not just the aggregate, so failure modes stay diagnosable
🟡 Medium | S once components exist

### 4.5 CAM-method comparison
Compare Grad-CAM vs. Grad-CAM++ vs. Eigen-CAM using the same quality-scoring framework — tests whether the scoring approach is CAM-method-agnostic and whether it ranks methods sensibly.
🟢 Nice-to-have | M

### 4.6 Failure-case gallery
A handful of examples where the reliability detector *disagreed* with raw confidence (high confidence but flagged unreliable, or vice versa), and/or where explanation quality was low despite a correct prediction. Often the most convincing qualitative evidence in a report/demo.
🟡 Medium | S

---

## 5. Decision Logic / Abstention Mechanism Updates

### 5.1 Merge explanation quality into the abstention gate
Currently abstention is driven only by the XGBoost reliability score. Add explanation quality as a second, independent trigger:

```
ABSTAIN if:
   reliability_score < threshold_R
   OR
   explanation_quality_score < threshold_Q
```

A prediction can be confident and reliability-score-high, but still get flagged if its explanation is spatially incoherent (e.g., activating background/artifacts instead of the lesion). This is a stronger safety narrative than either signal alone.
🔴 High | S (logic change, depends on Section 4)

### 5.2 Feed explanation-quality features back into XGBoost
Add the sub-scores from Section 4 as additional input features to the reliability detector itself, and test via ablation whether they improve its own predictive power.
🟡 Medium | S

---

## 6. Evaluation Rigor

| Addition | Detail | Priority | Effort |
|---|---|---|---|
| Bootstrap confidence intervals | On every headline metric (accuracy, AUC, coverage-risk curve) — not just point estimates | 🔴 High | S |
| Lesion-level k-fold cross-validation | Instead of a single static 70/15/15 split, for classification metrics | 🔴 High | M |
| Multiple random seeds for splitting | Repeat lesion-level splitting with a few seeds to confirm results aren't an artifact of one split | 🟡 Medium | S |
| Statistical significance testing | McNemar's test (or similar) comparing EfficientNet-B0 vs. ResNet18, and clean vs. perturbed accuracy | 🟡 Medium | S |
| Per-class metric breakdown | Accuracy/precision/recall/F1/AUC per class (bkl/mel/nv), not just aggregate | 🔴 High | S |

---

## 7. Documentation / Reporting Updates

### 7.1 Limitations section (Section 22 of README) — additions
- "Explanation-quality thresholds are empirically chosen, not clinician-validated."
- "The 'ideal' reference pattern used for superpixel similarity is an approximation in the absence of pixel-level lesion masks."
- Keep existing HAM10000/three-class/adversarial-scope caveats.

### 7.2 New experiment rows (Section 16 of README)

| ID | Experiment | Input | Metrics |
|---|---|---|---|
| E9 | Reliability Baseline Comparison | MSP, temp-scaled, entropy, MC-dropout, XGBoost reliability scores | Coverage-risk curves overlay, AUROC of each as a reliability signal |
| E10 | Calibration Analysis | Clean test set | ECE, reliability diagram, before/after temperature scaling |
| E11 | Explanation Quality Scoring | Clean + perturbed Grad-CAM maps | Deletion/insertion curves, robustness, consistency, superpixel alignment, aggregate Q, threshold sensitivity |
| E12 | External Validation (OOD) | ISIC/BCN20000 subset | Accuracy drop, reliability-flagging rate on OOD samples |
| E13 | Architecture Transfer | XGBoost trained on EfficientNet, scored on ResNet18 outputs | Reliability-detection AUROC (cross-architecture) |
| E14 | SHAP Feature Importance | Trained XGBoost reliability model | Feature importance ranking, dependency plots |
| E15 | Subgroup Analysis | Test set broken down by age/sex/localization metadata | Accuracy, reliability score distribution per subgroup |

### 7.3 Results template — add placeholder tables for E9–E15, matching the existing style in Section 17.

---

## 8. Suggested Priority Roadmap

If time-boxed, tackle in this order:

1. 🔴 **Reliability baseline comparison** (MSP / temp-scaled / entropy vs. XGBoost, one coverage-risk plot) — proves the core value proposition (M)
2. 🔴 **Robustness + Consistency Grad-CAM scores** — cheap, reuses existing perturbation pipeline (S)
3. 🔴 **Deletion/Insertion (faithfulness) curves** — standalone, high-value explainability metric (M)
4. 🔴 **Bootstrap CIs + per-class metrics + k-fold CV** — rigor pass that touches every existing result (M)
5. 🔴 **Merge explanation quality into abstention decision logic** — ties the reliability and explainability layers together (S)
6. 🟡 **Calibration diagnostics (ECE) + SHAP on XGBoost** — interpretability of the reliability layer itself (S each)
7. 🟡 **Superpixel/region-alignment score** — needs reference-pattern design decision first (M)
8. 🟡 **Subgroup analysis + cross-architecture transfer test** — cheap generalization checks using data you already have (S each)
9. 🟡 **External OOD validation set** — strongest generalization claim, but highest effort (M–L)
10. 🟢 **Corruption benchmark formalization, CAM-method comparison, MC-dropout/ensemble uncertainty** — polish, if time remains

---

## 9. One-line Summary for the README

> "MediShield combines a reliability detector (validated against confidence and calibration baselines), adversarial and corruption stress-testing, a quantitative explanation-quality score, and a two-signal abstention policy — evaluated with cross-validation, confidence intervals, subgroup breakdowns, and out-of-distribution testing to demonstrate genuine, not just apparent, added value over a raw classifier."
