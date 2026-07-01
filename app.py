"""
Streamlit app: HR inputs employee data -> get flight risk score,
risk tier, live SHAP-driven recommendations, and cost-benefit ROI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from utils import MODELS_DIR

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS for clean professional look ─────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1f2937; margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem;
    }
    .result-card {
        background: #f9fafb; border-radius: 12px;
        padding: 1.5rem; border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #1f2937;
    }
    .action-card {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem; border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .action-card-critical {
        background: #fef2f2; border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem; border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.25rem; font-weight: 600;
        color: #1f2937; margin: 1.5rem 0 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))


pipeline = load_model()

# ── HEADER ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 Employee Flight Risk Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter employee details to predict attrition risk, '
            'understand key drivers, and calculate retention ROI.</div>',
            unsafe_allow_html=True)
st.divider()

# ── INPUT FORM ─────────────────────────────────────────────────────────
st.markdown("### 👤 Employee Details")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Compensation**")
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status",
                                       ["Single", "Married", "Divorced"])

    with col2:
        st.markdown("**Job & Work Patterns**")
        department = st.selectbox("Department",
                                   ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        business_travel = st.selectbox("Business Travel",
                                        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance_from_home = st.slider("Distance From Home (km)", 1, 29, 5)

    with col3:
        st.markdown("**Satisfaction & Tenure**")
        job_satisfaction = st.selectbox(
            "Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
        env_satisfaction = st.selectbox(
            "Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox(
            "Work Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4], index=2)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_manager = st.slider("Years With Current Manager", 0, 17, 3)

st.divider()

# ── PREDICT BUTTON ─────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Analyse Flight Risk", type="primary", use_container_width=True)

if predict_clicked:

    input_data = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": monthly_income,
        "JobLevel": job_level,
        "YearsAtCompany": years_at_company,
        "TotalWorkingYears": max(years_at_company, 1),
        "OverTime": overtime,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": env_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "RelationshipSatisfaction": 3,
        "Department": department,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
        "Gender": gender,
        "BusinessTravel": business_travel,
        "DailyRate": 800,
        "DistanceFromHome": distance_from_home,
        "Education": 3,
        "EducationField": "Life Sciences",
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 13,
        "PerformanceRating": 3,
        "StockOptionLevel": 1,
        "TrainingTimesLastYear": 2,
        "YearsInCurrentRole": min(years_at_company, 4)
    }])

    THRESHOLD = 0.30
    proba = pipeline.predict_proba(input_data)[0][1]
    will_leave = proba >= THRESHOLD

    if proba < 0.25:
        tier, tier_color, tier_bg = "LOW", "#16a34a", "#f0fdf4"
    elif proba < 0.50:
        tier, tier_color, tier_bg = "MEDIUM", "#d97706", "#fffbeb"
    elif proba < 0.75:
        tier, tier_color, tier_bg = "HIGH", "#dc2626", "#fef2f2"
    else:
        tier, tier_color, tier_bg = "CRITICAL", "#7f1d1d", "#fef2f2"

    # ── RESULT SUMMARY ─────────────────────────────────────────────────
    st.markdown("## 📊 Prediction Results")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        verdict = "🚨 Will Leave" if will_leave else "✅ Will Stay"
        verdict_color = "#dc2626" if will_leave else "#16a34a"
        st.markdown(
            f'<div class="result-card">'
            f'<div class="metric-label">Prediction</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:{verdict_color}">'
            f'{verdict}</div></div>',
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="metric-label">Flight Risk Probability</div>'
            f'<div class="metric-value">{proba:.1%}</div></div>',
            unsafe_allow_html=True
        )

    with r3:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="metric-label">Risk Tier</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:{tier_color}">'
            f'{tier}</div></div>',
            unsafe_allow_html=True
        )

    with r4:
        annual_salary = monthly_income * 12
        replacement = annual_salary * 2.0
        st.markdown(
            f'<div class="result-card">'
            f'<div class="metric-label">Replacement Cost Risk</div>'
            f'<div class="metric-value">${replacement:,.0f}</div>'
            f'<div style="font-size:0.75rem;color:#6b7280">if not retained</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Risk progress bar
    st.progress(float(proba))
    st.divider()

    # ── SHAP EXPLANATIONS ───────────────────────────────────────────────
    st.markdown("### 🔍 Why the Model Made This Prediction")
    st.caption("SHAP values show exactly which factors pushed this employee's "
               "risk score up (red) or down (blue).")

    shap_success = False
    shap_feature_impacts = {}

    try:
        import shap
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        feature_eng  = pipeline.named_steps["feature_engineering"]
        encode_step  = pipeline.named_steps["encode"]
        model_step   = pipeline.named_steps["model"]

        # Transform input employee through preprocessing
        X_fe  = feature_eng.transform(input_data)
        X_enc = encode_step.transform(X_fe)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()

        raw_names   = encode_step.get_feature_names_out()
        X_enc_df    = pd.DataFrame(X_enc, columns=raw_names)

        # Load background sample saved during training
        bg_path = os.path.join(MODELS_DIR, "shap_background.joblib")
        bg_sample = joblib.load(bg_path)

        # Use background sample — this is what gives SHAP meaningful contrast
        explainer   = shap.Explainer(model_step, bg_sample)
        shap_values = explainer(X_enc_df)

        if len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        sv_arr = sv.values[0]

        # Clean feature names — remove ColumnTransformer prefixes
        def clean_name(name):
            name = name.replace("remainder__", "").replace("onehot__", "")
            name = name.replace("_", " ").title()
            return name

        clean_names = [clean_name(n) for n in raw_names]
        shap_feature_impacts = dict(zip(raw_names, sv_arr))
        shap_success = True

        # Top N by absolute impact
        top_n       = 12
        abs_impacts = np.abs(sv_arr)
        top_indices = np.argsort(abs_impacts)[-top_n:][::-1]  # highest first

        top_features = [clean_names[i] for i in top_indices]
        top_values   = [float(sv_arr[i]) for i in top_indices]
        colors       = ["#ef4444" if v > 0 else "#3b82f6" for v in top_values]

        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(top_features))
        bars  = ax.barh(list(y_pos), top_values, color=colors, height=0.6)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.invert_yaxis()  # highest impact at top
        ax.axvline(0, color="#374151", linewidth=1.0)
        ax.set_xlabel("SHAP Value  (positive = increases attrition risk)", fontsize=10)
        ax.set_title(f"Top {top_n} Risk Drivers — SHAP Feature Impact", fontsize=12, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Value labels
        for bar, val in zip(bars, top_values):
            if abs(val) > 0.0005:
                xpos  = bar.get_width() + 0.001 if val >= 0 else bar.get_width() - 0.001
                align = "left" if val >= 0 else "right"
                ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha=align, fontsize=9,
                        color="#374151")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close("all")

        st.caption("🔴 Red = increases flight risk   |   🔵 Blue = reduces flight risk")

    except Exception as e:
        st.warning(f"SHAP computation unavailable: {e}")
        shap_success = False

    st.divider()

    # ── HR ACTION RECOMMENDATIONS ───────────────────────────────────────
    st.markdown("### 📋 HR Action Recommendations")

    RECOMMENDATION_MAP = {
        "OverTime":                 ("🔴 Critical", "High overtime detected",
                                     "Review workload — enforce overtime limits or introduce comp time"),
        "MonthlyIncome":            ("🟠 High", "Compensation below expectation",
                                     "Benchmark salary against market; schedule compensation review"),
        "YearsSinceLastPromotion":  ("🟠 High", "Promotion stagnation",
                                     "Discuss career progression; check promotion eligibility immediately"),
        "JobSatisfaction":          ("🟡 Medium", "Low job satisfaction",
                                     "Schedule one-on-one; identify specific role dissatisfiers"),
        "EnvironmentSatisfaction":  ("🟡 Medium", "Low environment satisfaction",
                                     "Review team dynamics, workspace, and management approach"),
        "WorkLifeBalance":          ("🟠 High", "Poor work-life balance",
                                     "Assess workload; explore flexible or hybrid arrangements"),
        "StockOptionLevel":         ("🟡 Medium", "Low equity compensation",
                                     "Review stock option eligibility for this job level"),
        "Age":                      ("🟡 Medium", "Early-career risk profile",
                                     "Implement mentorship programme and structured development path"),
        "DistanceFromHome":         ("🟡 Medium", "Long commute flagged",
                                     "Explore remote or hybrid working options"),
        "RelationshipSatisfaction": ("🟡 Medium", "Low relationship satisfaction",
                                     "Address team health; consider team-building intervention"),
        "PromotionStagnationIndex": ("🔴 Critical", "High promotion stagnation index",
                                     "Flag for immediate promotion review — stagnation is severe"),
        "SatisfactionComposite":    ("🟠 High", "Low composite satisfaction",
                                     "Conduct full satisfaction survey; prioritise retention conversation"),
        "OvertimeBurdenFlag":       ("🔴 Critical", "Overtime + poor work-life balance combined",
                                     "Immediate intervention — this combination is the strongest attrition signal"),
        "IncomePerExperience":      ("🟠 High", "Underpaid relative to experience",
                                     "Conduct compensation equity review vs experience level"),
        "CareerVelocity":           ("🟡 Medium", "Slow career advancement",
                                     "Set concrete milestones for next level; discuss career roadmap"),
        "LoyaltyScore":             ("🟡 Medium", "Low manager loyalty score",
                                     "Strengthen manager relationship — early check-in recommended"),
    }

    if shap_success and shap_feature_impacts:
        # Sort by SHAP impact — show only features actively increasing risk
        risk_drivers = {k: v for k, v in shap_feature_impacts.items() if v > 0.01}
        sorted_drivers = sorted(risk_drivers.items(),
                                 key=lambda x: x[1], reverse=True)

        actions_shown = 0
        seen_keys = set()

        for feature, shap_val in sorted_drivers:
            if actions_shown >= 5:
                break
            matched_key = None
            for key in RECOMMENDATION_MAP:
                if key not in seen_keys and (feature == key or
                   feature.startswith(key) or key in feature):
                    matched_key = key
                    break

            if matched_key:
                seen_keys.add(matched_key)
                severity, label, action = RECOMMENDATION_MAP[matched_key]
                impact_pct = shap_val * 100
                card_class = "action-card-critical" if "🔴" in severity else "action-card"
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<strong>{severity} — {label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem">'
                    f'(+{impact_pct:.1f}% risk contribution)</span><br>'
                    f'→ {action}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                actions_shown += 1

        if actions_shown == 0:
            st.success("✅ No major risk drivers detected. Employee profile appears stable.")

    else:
        # Clean rule-based fallback (no error message shown to user)
        fallback_actions = []
        if overtime == "Yes":
            fallback_actions.append(("🔴 Critical", "High overtime detected",
                                      "Review workload distribution"))
        if job_satisfaction <= 2:
            fallback_actions.append(("🟠 High", "Low job satisfaction",
                                      "Schedule career discussion immediately"))
        if years_since_promotion >= 4:
            fallback_actions.append(("🟠 High", "Promotion stagnation",
                                      "Review career progression plan"))
        if monthly_income < 4000:
            fallback_actions.append(("🟡 Medium", "Below-market compensation",
                                      "Benchmark salary against market rate"))

        if fallback_actions:
            for severity, label, action in fallback_actions:
                card_class = "action-card-critical" if "🔴" in severity else "action-card"
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<strong>{severity} — {label}</strong><br>→ {action}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ Employee appears stable based on available inputs.")

    st.divider()

    # ── COST-BENEFIT ANALYSIS ───────────────────────────────────────────
    st.markdown("### 💰 Retention Cost-Benefit Analysis")
    st.caption("Business case for taking retention action on this employee.")

    annual_salary        = monthly_income * 12
    replacement_low      = annual_salary * 0.50
    replacement_high     = annual_salary * 2.00

    cb1, cb2 = st.columns([1, 1])

    with cb1:
        st.markdown("**Replacement cost if employee leaves:**")
        st.markdown(
            f"| Scenario | Cost |\n|---|---|\n"
            f"| Annual Salary | ${annual_salary:,} |\n"
            f"| Conservative (50%) | ${replacement_low:,.0f} |\n"
            f"| Realistic (200%) | ${replacement_high:,.0f} |"
        )

    with cb2:
        retention_cost = st.number_input(
            "Estimated retention intervention cost ($)",
            min_value=0, max_value=100000, value=3000, step=500,
            help="Salary raise, bonus, training budget, etc."
        )

    net_low  = replacement_low  - retention_cost
    net_high = replacement_high - retention_cost
    roi      = (net_high / retention_cost * 100) if retention_cost > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Conservative Net Saving", f"${net_low:,.0f}")
    with m2:
        st.metric("Realistic Net Saving", f"${net_high:,.0f}")
    with m3:
        st.metric("ROI on Retention", f"{roi:.0f}%")


    st.markdown("---")
    if will_leave:
        risk_pct   = f"{proba:.1%}"
        cost_str   = f"{retention_cost:,}"
        saving_str = f"{net_high:,.0f}"
        roi_str    = f"{roi:.0f}"
        st.markdown(
            f'<div style="background:#fef2f2;border:1px solid #fca5a5;'
            f'border-radius:8px;padding:1rem 1.25rem;color:#991b1b;">'
            f'🚨 <strong>Action Required</strong> — {risk_pct} flight risk detected. '
            f'A USD {cost_str} retention intervention could save up to '
            f'USD {saving_str}. Estimated ROI: {roi_str}%.'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        risk_pct = f"{proba:.1%}"
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #86efac;'
            f'border-radius:8px;padding:1rem 1.25rem;color:#166534;">'
            f'✅ <strong>Low flight risk ({risk_pct})</strong> — '
            f'No immediate action needed. Continue monitoring quarterly.'
            f'</div>',
            unsafe_allow_html=True
        )



