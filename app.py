"""
AI-Based Diabetes Prediction System - Streamlit Web Application
Complies with SRDD Specification v1.0.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    FEATURE_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    NORMAL_BASELINES, validate_input_data
)
from src.pipeline import DiabetesRiskPipeline
from src.explainability import ShapExplanationService
from src.report_generator import generate_pdf_report

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Diabetes Risk Assessment System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .risk-card-moderate {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    pipeline = joblib.load('models/best_model.joblib')
    explainer_service = ShapExplanationService(
        base_model_path='models/best_base_model.joblib',
        pipeline_path='models/best_model.joblib'
    )
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return pipeline, explainer_service, metadata


pipeline, explainer_service, metadata = load_assets()

# Sidebar: Inputs for Single Patient Assessment
st.sidebar.image("https://img.icons8.com/fluency/96/stethoscope.png", width=64)
st.sidebar.title("Clinical Parameters")
st.sidebar.markdown("Enter patient biometric data or load a clinical preset.")

# Preset Buttons
preset = st.sidebar.radio(
    "Quick Test Profile Preset:",
    ("Custom Inputs", "Healthy Baseline", "Prediabetic / Moderate", "High Risk Clinical Case"),
    index=0
)

# Preset Values
defaults = {
    'Custom Inputs': {'Age': 45, 'BMI': 26.5, 'Glucose': 95.0, 'BloodPressure': 75, 'Cholesterol': 190.0, 'PhysActivity': 1, 'GenHlth': 2},
    'Healthy Baseline': {'Age': 28, 'BMI': 21.8, 'Glucose': 84.0, 'BloodPressure': 68, 'Cholesterol': 165.0, 'PhysActivity': 1, 'GenHlth': 1},
    'Prediabetic / Moderate': {'Age': 50, 'BMI': 28.5, 'Glucose': 114.0, 'BloodPressure': 82, 'Cholesterol': 215.0, 'PhysActivity': 0, 'GenHlth': 3},
    'High Risk Clinical Case': {'Age': 62, 'BMI': 36.2, 'Glucose': 185.0, 'BloodPressure': 92, 'Cholesterol': 268.0, 'PhysActivity': 0, 'GenHlth': 4}
}
current_vals = defaults[preset]

with st.sidebar.form("patient_form"):
    st.subheader("Biometric Metrics")
    age = st.slider("Age (years)", min_value=18, max_value=95, value=current_vals['Age'], step=1,
                    help="Patient's current age in years.")
    bmi = st.number_input("Body Mass Index (BMI, kg/m²)", min_value=12.0, max_value=60.0,
                           value=float(current_vals['BMI']), step=0.1,
                           help="Normal range: 18.5 - 24.9 kg/m²")
    glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=50.0, max_value=320.0,
                              value=float(current_vals['Glucose']), step=1.0,
                              help="Normal: 70-99 mg/dL. Prediabetes: 100-125 mg/dL. Diabetes: ≥126 mg/dL.")
    bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=140,
                         value=int(current_vals['BloodPressure']), step=1,
                         help="Normal diastolic pressure: ≤ 80 mmHg.")
    chol = st.number_input("Total Cholesterol (mg/dL)", min_value=90.0, max_value=400.0,
                           value=float(current_vals['Cholesterol']), step=1.0,
                           help="Desirable: < 200 mg/dL. Borderline: 200-239 mg/dL. High: ≥240 mg/dL.")
    
    st.subheader("Lifestyle & Self-Perception")
    phys_act = st.selectbox("Physical Activity (Past 30 Days)", options=[1, 0],
                            format_func=lambda x: "Yes (Active)" if x == 1 else "No (Sedentary)",
                            index=0 if current_vals['PhysActivity'] == 1 else 1)
    
    gen_hlth_labels = {
        1: "1 - Excellent",
        2: "2 - Very Good",
        3: "3 - Good",
        4: "4 - Fair",
        5: "5 - Poor"
    }
    gen_hlth = st.selectbox("Self-Reported General Health", options=[1, 2, 3, 4, 5],
                            format_func=lambda x: gen_hlth_labels[x],
                            index=current_vals['GenHlth'] - 1)
    
    submit_button = st.form_submit_button("🩺 Assess Patient Risk", use_container_width=True)

# Build current patient DataFrame
patient_record = pd.DataFrame([{
    'Age': age,
    'BMI': bmi,
    'Glucose': glucose,
    'BloodPressure': bp,
    'Cholesterol': chol,
    'PhysActivity': phys_act,
    'GenHlth': gen_hlth
}])

# Application Title and Navigation Tabs
st.markdown('<div class="main-header">AI-Based Diabetes Risk Prediction & Clinical Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Calibrated Machine Learning Risk Scoring, Explainable AI (SHAP), and Population Screening</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "👤 Tab 1: Single Patient Assessment",
    "📋 Tab 2: Batch Screening (Clinical View)",
    "📈 Tab 3: Model Performance & Insights"
])

# ==========================================
# TAB 1: SINGLE PATIENT ASSESSMENT
# ==========================================
with tab1:
    proba = float(pipeline.predict_proba(patient_record)[0])
    risk_pct = proba * 100
    risk_category = pipeline.get_risk_category(proba)
    is_diabetic = int(pipeline.predict(patient_record)[0])
    
    # Risk Badge & Gauge display
    card_class = "risk-card-low" if risk_category == "Low Risk" else (
        "risk-card-moderate" if risk_category == "Moderate Risk" else "risk-card-high"
    )
    badge_color = "#10B981" if risk_category == "Low Risk" else ("#F59E0B" if risk_category == "Moderate Risk" else "#EF4444")
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="metric-label" style="color: {badge_color};">Risk Assessment Level</div>
                <div class="metric-value" style="color: {badge_color};">{risk_category.upper()}</div>
                <div style="font-size: 1.05rem; color: #374151; margin-top: 0.3rem;">
                    Status: <b>{'Diabetic (Screening Positive)' if is_diabetic else 'Non-Diabetic (Screening Negative)'}</b>
                    (Clinical Decision Threshold: {pipeline.optimal_threshold:.2f})
                </div>
            </div>
            <div style="text-align: right;">
                <div class="metric-label" style="color: {badge_color};">Calibrated Risk Probability</div>
                <div class="metric-value" style="color: {badge_color};">{risk_pct:.1f}%</div>
                <div style="font-size: 0.9rem; color: #4B5563;">Calibrated with Empirical Incidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress visual bar
    st.progress(min(proba, 1.0))

    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.subheader("🔍 Explainable AI: Feature Risk Contribution (SHAP)")
        st.caption("SHAP waterfall decomposition showing individual factors that push risk higher (red) or lower (green).")
        
        explanation = explainer_service.explain_instance(patient_record)
        fig_shap = explainer_service.generate_waterfall_plot(explanation)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

    with col_right:
        st.subheader("💡 Clinical Guidance & Key Observations")
        
        # Clinical Alerts based on thresholds
        if glucose >= 126:
            st.error(f"⚠️ **Fasting Hyperglycemia ({glucose:.1f} mg/dL):** Exceeds ADA threshold for diabetes (≥126 mg/dL). Diagnostic confirmation recommended.")
        elif glucose >= 100:
            st.warning(f"⚠️ **Impaired Fasting Glucose ({glucose:.1f} mg/dL):** In the prediabetic range (100-125 mg/dL).")
        else:
            st.success(f"✅ **Normal Fasting Glucose ({glucose:.1f} mg/dL):** Well within non-diabetic range (70-99 mg/dL).")

        if bmi >= 30:
            st.error(f"⚠️ **Obesity Tier (BMI {bmi:.1f}):** Significantly increases insulin resistance.")
        elif bmi >= 25:
            st.warning(f"⚠️ **Overweight (BMI {bmi:.1f}):** Modest weight reduction can lower diabetes risk.")
        else:
            st.success(f"✅ **Normal BMI ({bmi:.1f}):** Healthy body mass ratio.")

        if chol >= 240:
            st.error(f"⚠️ **High Total Cholesterol ({chol:.1f} mg/dL):** Consider comprehensive lipid panel.")
        elif chol >= 200:
            st.warning(f"⚠️ **Borderline Cholesterol ({chol:.1f} mg/dL):** Dietary modification advised.")
            
        st.markdown("---")
        # PDF Report Download
        pdf_buffer = generate_pdf_report(
            patient_data=patient_record.iloc[0].to_dict(),
            risk_score=proba,
            risk_category=risk_category
        )
        st.download_button(
            label="📄 Download Clinical Assessment Report (PDF)",
            data=pdf_buffer,
            file_name=f"diabetes_risk_report_{int(age)}yo.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # -------------------------------------------------------------
    # Counterfactual "What-If" Simulator (FR-07)
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔄 Interactive 'What-If' Lifestyle Counterfactual Simulator")
    st.markdown("Simulate how modifying lifestyle factors (weight loss, blood sugar management, or physical activity) dynamically alters the patient's predicted risk score.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        target_bmi = st.slider("Target BMI (kg/m²)", min_value=15.0, max_value=50.0, value=float(bmi), step=0.5)
    with sim_col2:
        target_glucose = st.slider("Target Fasting Glucose (mg/dL)", min_value=60.0, max_value=250.0, value=float(glucose), step=1.0)
    with sim_col3:
        target_phys = st.selectbox("Simulate Regular Exercise", options=[1, 0],
                                   format_func=lambda x: "Yes (Active)" if x == 1 else "No (Sedentary)",
                                   index=0 if phys_act == 1 else 1)

    # Compute counterfactual
    counterfactual_record = patient_record.copy()
    counterfactual_record['BMI'] = target_bmi
    counterfactual_record['Glucose'] = target_glucose
    counterfactual_record['PhysActivity'] = target_phys

    new_proba = float(pipeline.predict_proba(counterfactual_record)[0])
    delta_proba = (new_proba - proba) * 100

    col_sim_res1, col_sim_res2, col_sim_res3 = st.columns(3)
    with col_sim_res1:
        st.metric("Baseline Risk Score", f"{risk_pct:.1f}%")
    with col_sim_res2:
        st.metric("Simulated Risk Score", f"{new_proba * 100:.1f}%", delta=f"{delta_proba:+.1f}%", delta_color="inverse")
    with col_sim_res3:
        sim_cat = pipeline.get_risk_category(new_proba)
        st.metric("Simulated Risk Category", sim_cat)

# ==========================================
# TAB 2: BATCH SCREENING (CLINICAL VIEW)
# ==========================================
with tab2:
    st.subheader("📋 Population & Cohort Screening (Batch Mode)")
    st.markdown("Upload a CSV file containing patient records to process automated multi-patient risk screening.")

    # Option to load bundled sample batch
    col_upload, col_sample = st.columns([0.7, 0.3])
    with col_sample:
        st.markdown("**Test with Sample Dataset:**")
        if st.button("Load Bundled Sample Batch (50 Patients)", use_container_width=True):
            st.session_state['batch_df'] = pd.read_csv('data/raw/sample_batch_test.csv')

    with col_upload:
        uploaded_file = st.file_uploader("Upload Batch CSV File", type=["csv"])
        if uploaded_file is not None:
            st.session_state['batch_df'] = pd.read_csv(uploaded_file)

    if 'batch_df' in st.session_state:
        batch_df = st.session_state['batch_df']
        st.markdown(f"**Loaded Cohort:** `{len(batch_df)} patient records`")

        # Validate columns
        is_valid, validation_errors = validate_input_data(batch_df, is_inference=True)

        if not is_valid:
            st.error("Validation Error in uploaded file:\n" + "\n".join(validation_errors))
        else:
            with st.spinner("Processing batch inference..."):
                batch_probs = pipeline.predict_proba(batch_df)
                batch_preds = pipeline.predict(batch_df)
                batch_categories = [pipeline.get_risk_category(p) for p in batch_probs]

                results_df = batch_df.copy()
                results_df['Predicted_Risk_Score'] = (batch_probs * 100).round(1)
                results_df['Risk_Category'] = batch_categories
                results_df['Predicted_Diagnosis'] = np.where(batch_preds == 1, 'Diabetic', 'Non-Diabetic')

            # Aggregate Metrics
            high_count = sum(1 for c in batch_categories if c == 'High Risk')
            mod_count = sum(1 for c in batch_categories if c == 'Moderate Risk')
            low_count = sum(1 for c in batch_categories if c == 'Low Risk')

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Screened", len(batch_df))
            b_col2.metric("High Risk Patients", high_count, f"{(high_count/len(batch_df)*100):.1f}%")
            b_col3.metric("Moderate Risk", mod_count, f"{(mod_count/len(batch_df)*100):.1f}%")
            b_col4.metric("Low Risk", low_count, f"{(low_count/len(batch_df)*100):.1f}%")

            # Filter options
            st.markdown("#### Patient Cohort Results")
            filter_cat = st.multiselect(
                "Filter by Risk Category:",
                options=['High Risk', 'Moderate Risk', 'Low Risk'],
                default=['High Risk', 'Moderate Risk', 'Low Risk']
            )
            filtered_df = results_df[results_df['Risk_Category'].isin(filter_cat)]

            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=350
            )

            # Download screened CSV
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Screened Cohort Predictions (CSV)",
                data=csv_data,
                file_name="cohort_diabetes_screening_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==========================================
# TAB 3: MODEL PERFORMANCE & TRANSPARENCY
# ==========================================
with tab3:
    st.subheader("📈 Model Architecture, Evaluation & Algorithmic Transparency")
    st.markdown("Verification metrics from the benchmark evaluation of candidate algorithms on unseen stratified holdout test data.")

    # Benchmark Comparison Table
    bench = metadata['benchmark_results']
    table_rows = []
    for model_name, m_data in bench.items():
        table_rows.append({
            'Model Name': model_name,
            'CV ROC-AUC': f"{m_data['cv_mean_roc_auc']:.4f}",
            'Holdout ROC-AUC': f"{m_data['test_roc_auc']:.4f}",
            'Recall (Safety Target ≥ 80%)': f"{m_data['test_recall']*100:.1f}%",
            'Precision': f"{m_data['test_precision']*100:.1f}%",
            'F1-Score': f"{m_data['test_f1']:.4f}",
            'Decision Threshold': f"{m_data['optimal_threshold']:.3f}"
        })
    bench_df = pd.DataFrame(table_rows)
    st.markdown("#### 🏆 Algorithmic Benchmark Comparison")
    st.dataframe(bench_df, use_container_width=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 📊 ROC-AUC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.2), dpi=120)
        for model_name, m_data in bench.items():
            fpr = m_data['roc_curve']['fpr']
            tpr = m_data['roc_curve']['tpr']
            auc_val = m_data['test_roc_auc']
            ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})", linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
        ax_roc.set_xlabel("False Positive Rate", fontweight='bold')
        ax_roc.set_ylabel("True Positive Rate (Recall)", fontweight='bold')
        ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=9)
        ax_roc.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    with col_m2:
        st.markdown(f"#### 🎯 Confusion Matrix: {metadata['best_model']}")
        best_cm = np.array(bench[metadata['best_model']]['confusion_matrix'])
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2), dpi=120)
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Non-Diabetic', 'Diabetic'],
                    yticklabels=['Non-Diabetic', 'Diabetic'],
                    annot_kws={"size": 13, "weight": "bold"})
        ax_cm.set_xlabel("Predicted Diagnosis", fontweight='bold')
        ax_cm.set_ylabel("Actual Ground Truth", fontweight='bold')
        ax_cm.set_title("Holdout Test Confusion Matrix", fontweight='bold')
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    # Reference Baselines Table
    st.markdown("---")
    st.markdown("#### 📖 Clinical Reference Ranges & Evidence Baselines")
    ref_rows = []
    for feat, baseline in NORMAL_BASELINES.items():
        ref_rows.append({
            'Feature': feat,
            'Clinical Reference Range': baseline,
            'Clinical Association': {
                'Age': 'Aging correlates with pancreatic beta-cell dysfunction and insulin resistance.',
                'BMI': 'Excess adiposity promotes chronic low-grade inflammation and cellular insulin resistance.',
                'Glucose': 'Fasting blood glucose directly reflects glycemic regulation and ADA diagnostic thresholds.',
                'BloodPressure': 'Essential hypertension frequently clusters with metabolic syndrome and vascular damage.',
                'Cholesterol': 'Dyslipidemia exacerbates cardiovascular risk and atherogenic progression.',
                'PhysActivity': 'Skeletal muscle contraction drives non-insulin dependent glucose uptake.',
                'GenHlth': 'Self-rated overall physical condition and quality-of-life proxy.'
            }.get(feat, '')
        })
    st.table(pd.DataFrame(ref_rows))

    st.info(
        "**Clinical Disclaimer:** This software is designed as an assistive clinical decision-support tool. "
        "It does not replace professional clinical evaluation, standard diagnostic laboratory tests (such as oral glucose tolerance testing or HbA1c), "
        "or physician judgment."
    )
