"""
╔══════════════════════════════════════════════════════════════════╗
║          CREDITWISE — STREAMLIT INTERACTIVE DASHBOARD           ║
║          Run: streamlit run streamlit_app.py                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CreditWise — Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Sora', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #00d2ff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        color: #a8dadc;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    .metric-card {
        background: #1a2332;
        border: 1px solid #2d3f55;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #8892a4;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #00d2ff;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .metric-card .sub {
        color: #5f7a8a;
        font-size: 0.75rem;
    }

    .approved-banner {
        background: linear-gradient(135deg, #0d4c2b, #1a7a45);
        border: 1px solid #2ecc71;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: #2ecc71;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .rejected-banner {
        background: linear-gradient(135deg, #4c0d0d, #7a1a1a);
        border: 1px solid #e74c3c;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: #e74c3c;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .section-title {
        color: #00d2ff;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #2d3f55;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d2ff, #0099cc);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #33ddff, #00aadd);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 210, 255, 0.35);
    }

    .risk-flag {
        background: #3d1515;
        border-left: 3px solid #e74c3c;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #f1948a;
    }
    .ok-flag {
        background: #0d3d22;
        border-left: 3px solid #2ecc71;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #82e0aa;
    }

    .stSlider > div > div { color: #00d2ff; }
    [data-testid="stSidebar"] { background: #0d1b2a; }
    [data-testid="stSidebar"] .css-1d391kg { color: #a8dadc; }

    div[data-testid="stMetric"] {
        background: #1a2332;
        border: 1px solid #2d3f55;
        border-radius: 10px;
        padding: 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #8892a4 !important; }
    div[data-testid="stMetric"] div { color: #00d2ff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏦 CreditWise Loan Approval System</h1>
    <p>Intelligent AI-Powered Credit Risk Assessment | SecureTrust Bank</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Upload & Settings
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    threshold = st.slider("🎯 Decision Threshold", 0.30, 0.90, 0.50, 0.05,
                          help="Higher = More conservative (fewer approvals, higher precision)")
    n_neighbors = st.slider("KNN — Neighbors (k)", 3, 15, 7, 2)
    show_eda = st.checkbox("Show EDA Section", value=True)

    st.markdown("---")
    st.markdown("**📌 About Threshold**")
    st.caption(
        "Raising the threshold means the model approves a loan only when "
        "it's MORE confident. This reduces false approvals (Type I Error) "
        "at the cost of some false rejections."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_and_preprocess(file_bytes, _n_neighbors):
    """Load data, preprocess, and train all models."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    raw_df = df.copy()

    # Impute
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes("object").columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # Drop ID
    if "Applicant_ID" in df.columns:
        df.drop("Applicant_ID", axis=1, inplace=True)

    # Encode
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])

    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    oh = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_df = pd.DataFrame(oh.fit_transform(df[ohe_cols]),
                          columns=oh.get_feature_names_out(ohe_cols),
                          index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), enc_df], axis=1)

    # Feature engineering
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=100000, random_state=42),
        "Naive Bayes"        : GaussianNB(),
        f"KNN (k={_n_neighbors})": KNeighborsClassifier(n_neighbors=_n_neighbors)
    }

    results = []
    trained = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        yp = m.predict(X_test_s)
        yprob = m.predict_proba(X_test_s)[:, 1] if hasattr(m, "predict_proba") else None
        results.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, yp, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, yp, zero_division=0), 4),
            "AUC"      : round(roc_auc_score(y_test, yprob), 4) if yprob is not None else None,
        })
        trained[name] = (m, yp)

    results_df  = pd.DataFrame(results).set_index("Model")
    best_name   = results_df["Precision"].idxmax()
    best_model  = trained[best_name][0]

    return (raw_df, X_train_s, X_test_s, y_train, y_test,
            scaler, oh, X.columns.tolist(), ohe_cols,
            results_df, best_model, best_name, trained)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is None:
    st.info("👈 **Please upload your `loan_approval_data.csv` file** using the sidebar to begin.")
    st.markdown("""
    #### What this app does:
    - 🔍 Automatically trains **3 ML models** on your loan data
    - 📊 Shows **EDA charts** with business insights
    - 🏆 Selects the **best model** based on Precision (minimizes bad loan approvals)
    - 🎯 Lets you **predict loan approval** for any applicant in real-time
    - ⚙️ Supports **threshold tuning** for precision control
    """)
    st.stop()

# ── Load & Train ─────────────────────────────────────────────────────────────
with st.spinner("Training models... please wait"):
    (raw_df, X_train_s, X_test_s, y_train, y_test,
     scaler, oh, feature_cols, ohe_cols,
     results_df, best_model, best_name, trained) = load_and_preprocess(
         uploaded_file.getvalue(), n_neighbors
     )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Insights",
    "🤖 Model Results",
    "🎯 Predict Loan",
    "💼 Interview Guide"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: EDA
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not show_eda:
        st.info("EDA is hidden. Enable it in the sidebar.")
    else:
        st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",  f"{raw_df.shape[0]:,}")
        c2.metric("Features",       f"{raw_df.shape[1]-1}")
        c3.metric("Approval Rate",  f"{(raw_df['Loan_Approved'].value_counts(normalize=True).get(1,0)*100):.1f}%")
        c4.metric("Missing Values", f"{raw_df.isnull().sum().sum()}")

        st.markdown('<div class="section-title">Target Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1b2a")
            ax.set_facecolor("#0d1b2a")
            counts = raw_df["Loan_Approved"].value_counts()
            ax.pie(counts, labels=["Rejected", "Approved"], autopct="%1.1f%%",
                   colors=["#e74c3c", "#2ecc71"], startangle=90,
                   wedgeprops=dict(edgecolor="#0d1b2a", linewidth=2),
                   textprops={"color": "white"})
            ax.set_title("Loan Approval Split", color="white", fontsize=12)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1b2a")
            ax.set_facecolor("#1a2332")
            sns.countplot(data=raw_df, x="Loan_Approved", palette=["#e74c3c", "#2ecc71"], ax=ax)
            ax.set_xticklabels(["Rejected", "Approved"], color="white")
            ax.tick_params(colors="white")
            ax.set_title("Count Plot", color="white")
            ax.set_xlabel("", color="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3f55")
            st.pyplot(fig)
            plt.close()

        # Income & Credit Score
        st.markdown('<div class="section-title">Financial Features Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1b2a")
            ax.set_facecolor("#1a2332")
            sns.boxplot(data=raw_df, x="Loan_Approved", y="Applicant_Income",
                        palette=["#e74c3c", "#2ecc71"], ax=ax)
            ax.set_xticklabels(["Rejected", "Approved"], color="white")
            ax.tick_params(colors="white")
            ax.set_title("Income by Approval", color="white")
            ax.set_xlabel("")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3f55")
            st.pyplot(fig)
            plt.close()
            st.caption("📌 Higher income applicants tend to get approved more often.")

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1b2a")
            ax.set_facecolor("#1a2332")
            sns.histplot(data=raw_df, x="Credit_Score", hue="Loan_Approved",
                         bins=25, palette=["#e74c3c", "#2ecc71"],
                         element="step", ax=ax)
            ax.tick_params(colors="white")
            ax.set_title("Credit Score Distribution", color="white")
            ax.set_xlabel("Credit Score", color="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3f55")
            st.pyplot(fig)
            plt.close()
            st.caption("📌 Approved applicants cluster at higher credit scores.")

        # Correlation Heatmap
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        num_df = raw_df.select_dtypes("number")
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0d1b2a")
        ax.set_facecolor("#0d1b2a")
        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, linewidths=0.5, ax=ax,
                    annot_kws={"size": 8, "color": "white"})
        ax.set_title("Feature Correlations", color="white", fontsize=13)
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Correlated features (>0.8) may cause multicollinearity. "
                   "DTI_Ratio and Credit_Score are squared to capture non-linear effects.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-title">🤖 Model Performance Comparison</div>',
                unsafe_allow_html=True)

    # Best model highlight
    st.success(f"🏆 **Best Model: {best_name}** (selected by highest Precision)")

    # Metrics table
    st.dataframe(
        results_df.style
        .highlight_max(axis=0, color="#1a4d2e")
        .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("""
    **Metric Explanations:**
    - **Accuracy** — % of total correct predictions
    - **Precision** ⭐ — Of all loans predicted Approved, how many were actually good? (KEY: minimizes Type I Error)
    - **Recall** — Of all actual good loans, how many did we catch?
    - **F1 Score** — Balance between Precision and Recall
    - **AUC** — Overall ability to distinguish approved vs rejected
    """)

    # Bar chart
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0d1b2a")
    ax.set_facecolor("#1a2332")
    x = np.arange(len(results_df))
    w = 0.2
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * w, results_df[m], w, label=m, color=c, edgecolor="#0d1b2a")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="white")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(results_df.index, color="white", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="white")
    ax.legend(fontsize=9, facecolor="#1a2332", labelcolor="white")
    ax.axhline(0.8, linestyle="--", color="#5f7a8a", linewidth=1, alpha=0.7)
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3f55")
    ax.set_title("Model Metrics Comparison", color="white", fontsize=13)
    st.pyplot(fig)
    plt.close()

    # Confusion matrices
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1b2a")
    for ax, (name, (model, yp)) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"],
                    linewidths=0.5)
        ax.set_title(name, fontsize=10, color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("📌 FP (top-right) = Bad loans approved → financial risk. "
               "We minimize FP by maximizing Precision.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-title">🎯 Loan Application Form</div>',
                unsafe_allow_html=True)
    st.caption(f"Using **{best_name}** | Threshold: **{threshold}**")

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Personal Info**")
            age         = st.number_input("Age", 18, 75, 32)
            gender      = st.selectbox("Gender", ["Male", "Female"])
            marital     = st.selectbox("Marital Status", ["Married", "Single"])
            dependents  = st.number_input("Dependents", 0, 10, 1)
            education   = st.selectbox("Education Level",
                                       ["Graduate", "Postgraduate", "Undergraduate"])

        with c2:
            st.markdown("**💼 Employment & Income**")
            emp_status  = st.selectbox("Employment Status",
                                       ["Salaried", "Self-Employed", "Business"])
            employer    = st.selectbox("Employer Category", ["Govt", "Private", "Self"])
            income      = st.number_input("Monthly Income (₹)", 5000, 500000, 55000, 1000)
            co_income   = st.number_input("Co-applicant Income (₹)", 0, 300000, 15000, 1000)
            savings     = st.number_input("Savings (₹)", 0, 1000000, 25000, 1000)

        with c3:
            st.markdown("**🏦 Loan Details**")
            loan_amount = st.number_input("Loan Amount (₹)", 10000, 5000000, 200000, 5000)
            loan_term   = st.number_input("Loan Term (months)", 6, 360, 60)
            loan_purpose= st.selectbox("Loan Purpose",
                                       ["Home", "Education", "Personal", "Business"])
            prop_area   = st.selectbox("Property Area",
                                       ["Urban", "Semi-Urban", "Rural"])
            collateral  = st.number_input("Collateral Value (₹)", 0, 5000000, 60000, 5000)
            credit_score= st.slider("Credit Score", 300, 900, 720)
            dti_ratio   = st.slider("DTI Ratio", 0.05, 1.0, 0.28, 0.01)
            existing_loans = st.number_input("Existing Loans", 0, 10, 1)

        submitted = st.form_submit_button("🔍 Predict Loan Approval")

    if submitted:
        # Education encoding map
        edu_map = {"Graduate": 1, "Postgraduate": 2, "Undergraduate": 0}

        input_data = {
            "Age"               : age,
            "Applicant_Income"  : income,
            "Coapplicant_Income": co_income,
            "Loan_Amount"       : loan_amount,
            "Loan_Term"         : loan_term,
            "Credit_Score"      : credit_score,
            "DTI_Ratio"         : dti_ratio,
            "Savings"           : savings,
            "Collateral_Value"  : collateral,
            "Existing_Loans"    : existing_loans,
            "Dependents"        : dependents,
            "Education_Level"   : edu_map[education],
            "Employment_Status" : emp_status,
            "Marital_Status"    : marital,
            "Loan_Purpose"      : loan_purpose,
            "Property_Area"     : prop_area,
            "Gender"            : gender,
            "Employer_Category" : employer
        }

        # Preprocess & predict
        input_df = pd.DataFrame([input_data])
        input_df["DTI_Ratio_sq"]    = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
        input_df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

        enc = oh.transform(input_df[ohe_cols])
        enc_df = pd.DataFrame(enc, columns=oh.get_feature_names_out(ohe_cols))
        input_final = pd.concat([input_df.drop(columns=ohe_cols), enc_df], axis=1)
        input_final = input_final.reindex(columns=feature_cols, fill_value=0)
        input_scaled = scaler.transform(input_final)

        prob     = best_model.predict_proba(input_scaled)[0][1]
        decision = 1 if prob >= threshold else 0

        st.markdown("---")
        st.markdown("### 📋 Decision")

        if decision == 1:
            st.markdown(f"""
            <div class="approved-banner">
                ✅ LOAN APPROVED<br>
                <span style="font-size:1rem;font-weight:400">
                Approval Probability: {prob*100:.1f}% &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rejected-banner">
                ❌ LOAN REJECTED<br>
                <span style="font-size:1rem;font-weight:400">
                Approval Probability: {prob*100:.1f}% &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk analysis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Risk Factor Analysis:**")
            flags = []
            if credit_score < 600: flags.append("⚠️ Low credit score (< 600)")
            if dti_ratio > 0.5:    flags.append("⚠️ High DTI ratio (> 50%)")
            if existing_loans > 2: flags.append("⚠️ Multiple existing loans")
            if income < 20000:     flags.append("⚠️ Below-average income")

            if flags:
                for f in flags:
                    st.markdown(f'<div class="risk-flag">{f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-flag">✅ No major risk flags detected</div>',
                            unsafe_allow_html=True)

        with col2:
            # Probability gauge
            fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0d1b2a")
            ax.set_facecolor("#0d1b2a")
            color = "#2ecc71" if prob >= threshold else "#e74c3c"
            ax.barh(["Approval Probability"], [prob], color=color, height=0.4)
            ax.barh(["Approval Probability"], [1 - prob], left=[prob],
                    color="#2d3f55", height=0.4)
            ax.axvline(x=threshold, color="#f39c12", linestyle="--", linewidth=2,
                       label=f"Threshold ({threshold})")
            ax.set_xlim(0, 1)
            ax.set_title(f"Probability: {prob*100:.1f}%", color="white", fontsize=12)
            ax.tick_params(colors="white")
            ax.legend(fontsize=9, facecolor="#1a2332", labelcolor="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3f55")
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: INTERVIEW GUIDE
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    ## 💼 How to Explain This Project in an Interview

    ---

    ### 🎤 30-Second Elevator Pitch
    > *"I built CreditWise, an intelligent loan approval prediction system for a bank. It uses
    historical applicant data to predict whether a loan should be approved or rejected.
    I trained three models — Logistic Regression, Naive Bayes, and KNN — and selected the best
    based on Precision, because in banking, approving a bad loan causes real financial loss.
    I also built a Streamlit UI so anyone can input applicant details and get a real-time decision."*

    ---

    ### ❓ Common HR/Technical Questions

    **Q1: Why did you choose Precision as your main metric?**
    > In loan approval, a False Positive (approving a bad loan) directly causes financial loss
    to the bank. This is a Type I Error. Precision = TP / (TP + FP) directly measures how many
    of our "approved" predictions were actually safe. By maximizing Precision, we minimize bad loans.

    **Q2: What is a Type I Error in this project?**
    > Type I Error = False Positive = Approving a loan that should be rejected.
    This is worse than a Type II Error (rejecting a good loan) because it causes
    actual monetary loss, not just missed business.

    **Q3: Why did you square DTI_Ratio and Credit_Score?**
    > Squared features capture non-linear relationships. A very high DTI doesn't
    just linearly increase risk — it exponentially signals danger. Squaring amplifies
    the difference between borderline and extreme values, giving the model stronger signals.

    **Q4: Why StandardScaler?**
    > KNN and Logistic Regression are distance/gradient-based. Without scaling,
    features with large ranges (like Income at 50,000) dominate over small-range
    features (like Dependents = 2). StandardScaler normalizes all to zero mean,
    unit variance.

    **Q5: What is stratified train-test split?**
    > It ensures the proportion of approved and rejected loans is the same in both
    train and test sets. Without this, the model might train on mostly rejected loans
    and be biased.

    **Q6: Why OneHotEncoder with drop='first'?**
    > To avoid the dummy variable trap (multicollinearity). If we one-hot encode
    "Gender" into Male and Female, both together give the same info. Dropping one
    makes them independent.

    **Q7: What improvements would you make with more time?**
    > 1. XGBoost / Random Forest for better accuracy
    > 2. SMOTE for handling class imbalance
    > 3. SHAP values for model explainability
    > 4. Model deployment on AWS/GCP
    > 5. A/B testing in production

    ---



# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center style='color:#5f7a8a;font-size:0.8rem'>"
    "CreditWise Loan Approval System · Built with Scikit-Learn + Streamlit · "
    "SecureTrust Bank · ML Minor Project"
    "</center>",
    unsafe_allow_html=True
)
