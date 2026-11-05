# AI-Based Diabetes Prediction System

An end-to-end, interpretable clinical decision-support web application for diabetes risk screening, calibrated risk scoring, SHAP explainability, and counterfactual lifestyle simulation.

Built according to the approved **[Software Requirements & Design Document (SRDD)](srdd.md)**.

---

## 🌟 Key Features

1. **4-Layer Decoupled Architecture:** Clean separation across Presentation (Streamlit), Service, ML Pipeline, and Storage layers.
2. **Clinically Calibrated Risk Probability:** Estimates true population risk scores (0–100%) using `CalibratedClassifierCV` (Sigmoid calibration).
3. **Safety-Optimized Recall ($\ge 80\%$):** Clinically tuned decision threshold prioritizing recall on positive diabetic cases to minimize life-threatening false negatives (NFR-02).
4. **Explainable AI (SHAP):** Local sample-level waterfall attribution decomposes each patient's risk into driving factors (red) and protective factors (green).
5. **Interactive "What-If" Counterfactual Simulator:** Allows patients and clinicians to adjust modifiable risk factors (BMI, exercise, blood sugar targets) to observe real-time risk reduction ($\Delta\text{Risk}$).
6. **Clinical Batch Screening:** Bulk CSV screening with automatic schema validation, risk tier distribution metrics, and processed data export.
7. **Clinical Assessment PDF Export:** One-click generation of professional patient risk cards and dietary/lifestyle guidance via ReportLab.
8. **Algorithmic Transparency:** Full benchmarking comparisons between Logistic Regression, Random Forest, and XGBoost with ROC-AUC curves and confusion matrices.

---

## 🏗️ Repository Architecture

```text
diabetes-prediction-system/
├── data/
│   ├── raw/
│   │   ├── diabetes_data.csv         # Clinically grounded benchmark dataset (5,000 records)
│   │   └── sample_batch_test.csv     # 50-patient screening batch test file
│   └── processed/
│       └── test_split.csv            # Unseen holdout evaluation split
├── models/
│   ├── best_model.joblib             # Serialized end-to-end calibrated pipeline
│   ├── best_base_model.joblib        # Tree estimator for SHAP TreeExplainer
│   ├── shap_background.joblib        # Background reference distribution
│   └── model_metadata.json           # Evaluation metrics, ROC curve coordinates, and threshold
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   └── 02_model_training.ipynb       # Model training & benchmarking
├── scripts/
│   ├── generate_data.py              # Clinical dataset synthesizer
│   └── create_notebooks.py           # Notebook generator script
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Schema validation, constraints, and data splitting
│   ├── preprocessing.py              # ColumnTransformer and SMOTE oversampling
│   ├── pipeline.py                   # DiabetesRiskPipeline wrapper class
│   ├── train.py                      # Multi-model cross-validation, tuning, and calibration
│   ├── explainability.py             # SHAP local waterfall & force visualizer
│   └── report_generator.py           # Clinical PDF assessment report builder
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py         # Boundary & schema validation unit tests
│   ├── test_model.py                 # Pipeline reproducibility & recall verification
│   └── test_safety.py                # False negative audit & high-risk clinical safety tests
├── app.py                            # 3-Tab interactive Streamlit web application
├── requirements.txt                  # Pinned Python dependencies
├── srdd.md                           # Complete Software Requirements & Design Document
└── README.md                         # Project documentation
```

---

## 🩺 Clinical Feature Schema & Reference Baselines

| Feature Name | Clinical Concept | Data Type | Expected Range | Normal Baseline |
| :--- | :--- | :--- | :--- | :--- |
| `Age` | Patient age in years | Integer | $18 - 100$ | $N/A$ |
| `BMI` | Body Mass Index ($\text{kg/m}^2$) | Float | $12.0 - 60.0$ | $18.5 - 24.9$ |
| `Glucose` | Fasting blood glucose ($\text{mg/dL}$) | Float | $50.0 - 300.0$ | $70.0 - 99.0$ |
| `BloodPressure` | Diastolic blood pressure ($\text{mmHg}$) | Integer | $40 - 140$ | $\le 80$ |
| `Cholesterol` | Total cholesterol ($\text{mg/dL}$) | Float | $90.0 - 400.0$ | $< 200$ |
| `PhysActivity` | Physical activity in past 30 days | Binary | $0$ (No), $1$ (Yes) | $1$ (Active) |
| `GenHlth` | Self-reported general health scale | Ordinal | $1$ (Excellent) - $5$ (Poor) | $1 - 2$ |

---

## ⚡ Quick Start Guide

### 1. Installation
Clone or navigate to the repository directory and install the pinned dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Quality Assurance & Safety Tests
Verify pipeline integrity and clinical safety constraints:

```bash
python -m pytest -v tests/
```

### 3. Re-train & Benchmark Models (Optional)
To regenerate data and retrain all candidate algorithms:

```bash
python scripts/generate_data.py
python src/train.py
```

### 4. Launch the Streamlit Web Application
Run the interactive interface in your browser:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🖥️ Streamlit Interface Overview

* **Tab 1: Single Patient Assessment**
  * Input biometric parameters via sidebar sliders and numeric inputs.
  * Load one-click clinical presets (*Healthy Baseline*, *Prediabetic*, *High Risk Case*).
  * View immediate calibrated risk scores and dynamic risk categories (`Low Risk <30%`, `Moderate Risk 30-60%`, `High Risk >60%`).
  * Inspect SHAP waterfall attribution of positive and protective risk drivers.
  * Test lifestyle interventions in real time with the **"What-If" Counterfactual Simulator**.
  * Download an executive **Clinical Assessment Report (PDF)**.
* **Tab 2: Batch Screening (Clinical View)**
  * Bulk upload patient records via CSV (or click "Load Bundled Sample Batch" for instant testing).
  * Filter cohort by risk categories and export predictions to CSV.
* **Tab 3: Model Performance & Insights**
  * View cross-validation and holdout test metrics for Logistic Regression, Random Forest, and XGBoost.
  * Interactive ROC curves and Confusion Matrix heatmap.
  * Evidence-based clinical reference ranges and medical disclaimers.

---

## 🛡️ Clinical Disclaimer
This system is intended as an artificial intelligence educational and assistive screening tool. It does not replace professional medical diagnosis, clinical judgment, or standard laboratory diagnostic tests (such as HbA1c or oral glucose tolerance tests).
