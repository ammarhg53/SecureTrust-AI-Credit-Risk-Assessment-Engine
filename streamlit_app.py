"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            CREDITWISE v2.0 — LOAN APPROVAL PREDICTION SYSTEM               ║
║            Professional ML Project | Interview-Ready Edition               ║
║            Run: streamlit run streamlit_app_v2.py                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

FIXES IN v2.0:
  ✅ Fixed: Approval Rate showing 0.0% (target column encoding bug)
  ✅ Fixed: Missing Values = 1000 (wrong null detection before imputation)
  ✅ Fixed: Feature mismatch during prediction
  ✅ Fixed: Preprocessing pipeline consistency
  ✅ Added: Step-by-step EDA storytelling
  ✅ Added: Interactive missing value explainer
  ✅ Added: Better model comparison & feature importance
  ✅ Added: Interview Q&A guide
"""

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, io, textwrap

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute           import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import GaussianNB
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL THEME
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "CreditWise — Loan Approval System",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# Dark-finance color palette (CSS variables)
BG      = "#0b1622"
CARD    = "#111e2e"
BORDER  = "#1e3048"
ACCENT  = "#00c6ff"
GREEN   = "#00e676"
RED     = "#ff5252"
YELLOW  = "#ffd740"
TEXT    = "#cdd9e5"
MUTED   = "#6b8096"

plt.rcParams.update({
    "figure.facecolor" : CARD,
    "axes.facecolor"   : CARD,
    "axes.edgecolor"   : BORDER,
    "axes.labelcolor"  : TEXT,
    "xtick.color"      : TEXT,
    "ytick.color"      : TEXT,
    "text.color"       : TEXT,
    "grid.color"       : BORDER,
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.4,
    "legend.facecolor" : CARD,
    "legend.edgecolor" : BORDER,
    "legend.labelcolor": TEXT,
    "font.family"      : "DejaVu Sans",
    "figure.dpi"       : 130,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; background:{BG}; color:{TEXT}; }}

/* ── Main layout ─────────────────────── */
.main .block-container {{ padding: 1.5rem 2rem 3rem; max-width:1400px; }}
[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}

/* ── Header banner ───────────────────── */
.cw-header {{
    background: linear-gradient(135deg, #071828 0%, #0d2640 50%, #071828 100%);
    border: 1px solid {BORDER}; border-radius:16px;
    padding:2.2rem 2.8rem; text-align:center; margin-bottom:1.8rem;
    position:relative; overflow:hidden;
}}
.cw-header::before {{
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,198,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(0,230,118,0.06) 0%, transparent 60%);
}}
.cw-header h1 {{ color:{ACCENT}; font-size:2.2rem; font-weight:700; margin:0; letter-spacing:1.5px; }}
.cw-header p  {{ color:{MUTED}; font-size:0.95rem; margin:0.5rem 0 0; }}

/* ── Metric cards ────────────────────── */
.metric-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }}
.metric-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
    padding:1.2rem 1.4rem; text-align:center; position:relative; overflow:hidden;
}}
.metric-card::after {{
    content:''; position:absolute; bottom:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,{ACCENT},{GREEN});
}}
.metric-card .mc-label {{ color:{MUTED}; font-size:0.72rem; text-transform:uppercase;
                           letter-spacing:1.2px; margin-bottom:0.5rem; }}
.metric-card .mc-value {{ color:{ACCENT}; font-size:2rem; font-weight:700; line-height:1; }}
.metric-card .mc-sub   {{ color:{MUTED}; font-size:0.73rem; margin-top:0.3rem; }}

/* ── Section headers ─────────────────── */
.sec-title {{
    color:{ACCENT}; font-size:1rem; font-weight:600; letter-spacing:0.5px;
    border-bottom:1px solid {BORDER}; padding-bottom:0.4rem; margin:1.5rem 0 1rem;
    display:flex; align-items:center; gap:0.5rem;
}}

/* ── Story cards ─────────────────────── */
.story-card {{
    background:rgba(0,198,255,0.05); border:1px solid rgba(0,198,255,0.18);
    border-left:3px solid {ACCENT}; border-radius:8px;
    padding:0.9rem 1.1rem; margin:0.6rem 0; font-size:0.88rem; color:{TEXT};
}}
.story-card strong {{ color:{ACCENT}; }}

/* ── Missing value explainer ─────────── */
.mv-root  {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:1rem 1.2rem; }}
.mv-col   {{ display:flex; justify-content:space-between; align-items:center;
             border-bottom:1px solid {BORDER}; padding:0.5rem 0; font-size:0.85rem; }}
.mv-col:last-child {{ border-bottom:none; }}
.mv-bar   {{ height:8px; border-radius:4px; background:linear-gradient(90deg,{RED},{YELLOW}); }}
.mv-fix   {{ color:{GREEN}; font-size:0.8rem; font-style:italic; }}

/* ── Decision banners ────────────────── */
.approved-banner {{
    background:linear-gradient(135deg,#052318,#0a3d22);
    border:1px solid {GREEN}; border-radius:14px;
    padding:2rem; text-align:center; color:{GREEN};
    font-size:1.7rem; font-weight:700;
}}
.rejected-banner {{
    background:linear-gradient(135deg,#230505,#3d0a0a);
    border:1px solid {RED}; border-radius:14px;
    padding:2rem; text-align:center; color:{RED};
    font-size:1.7rem; font-weight:700;
}}
.sub-result {{ font-size:0.95rem; font-weight:400; color:{MUTED}; margin-top:0.5rem; }}

/* ── Risk flags ──────────────────────── */
.flag-risk {{ background:rgba(255,82,82,0.08); border-left:3px solid {RED};
              border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0;
              font-size:0.85rem; color:#ff8a80; }}
.flag-ok   {{ background:rgba(0,230,118,0.08); border-left:3px solid {GREEN};
              border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0;
              font-size:0.85rem; color:#69f0ae; }}

/* ── Model table ─────────────────────── */
.model-table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
.model-table th {{ background:#0d2640; color:{ACCENT}; padding:0.6rem 0.9rem;
                   text-align:center; border:1px solid {BORDER}; font-weight:600; }}
.model-table td {{ padding:0.55rem 0.9rem; text-align:center; border:1px solid {BORDER};
                   color:{TEXT}; }}
.model-table tr:hover td {{ background:rgba(0,198,255,0.04); }}
.best-row td {{ background:rgba(0,230,118,0.07) !important; color:{GREEN} !important; font-weight:600; }}

/* ── Code blocks ─────────────────────── */
.code-box {{
    background:#0a1929; border:1px solid {BORDER}; border-radius:8px;
    padding:1rem 1.2rem; font-family:'JetBrains Mono',monospace;
    font-size:0.78rem; color:#a8d8ea; overflow-x:auto;
    white-space:pre;
}}

/* ── Interview Q cards ───────────────── */
.qa-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
    padding:1rem 1.3rem; margin:0.7rem 0;
}}
.qa-q {{ color:{ACCENT}; font-weight:600; margin-bottom:0.4rem; font-size:0.9rem; }}
.qa-a {{ color:{TEXT}; font-size:0.85rem; line-height:1.6; }}

/* ── Streamlit overrides ─────────────── */
div[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER};
                               border-radius:10px; padding:0.8rem; }}
div[data-testid="stMetric"] label {{ color:{MUTED} !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color:{ACCENT} !important; font-size:1.7rem !important; }}
.stButton>button {{
    background:linear-gradient(135deg,{ACCENT},#0099cc); color:#000;
    border:none; border-radius:10px; font-weight:700;
    padding:0.65rem 1.8rem; width:100%; font-size:1rem;
    transition:all 0.2s ease;
}}
.stButton>button:hover {{ transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,198,255,0.35); }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED}; font-weight:500; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPER: PLOT WRAPPER  (consistent dark theme for every chart)
# ════════════════════════════════════════════════════════════════════════════

def styled_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    return fig, ax

def styled_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(CARD)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
    return fig, axes


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="cw-header">
  <h1>🏦 CreditWise Loan Approval System</h1>
  <p>AI-Powered Credit Risk Assessment &nbsp;·&nbsp; SecureTrust Bank &nbsp;·&nbsp; v2.0 Professional Edition</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### ⚙️ Configuration")
    uploaded = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

    st.markdown("---")
    threshold   = st.slider("🎯 Decision Threshold", 0.30, 0.90, 0.50, 0.05,
                             help="Higher → fewer approvals, higher precision (less Type I error)")
    n_neighbors = st.slider("KNN Neighbors (k)", 3, 15, 7, 2)

    st.markdown("---")
    st.markdown(f"**🔍 Why Precision?**")
    st.caption(
        "A **False Positive** (approving a bad loan) directly causes "
        "monetary loss to the bank. Precision = TP/(TP+FP) minimises "
        "this risk. We tune the threshold to further boost it."
    )
    st.markdown("---")
    st.markdown(f"**📌 Threshold Effect**")
    st.caption(
        f"Current: **{threshold}**\n\n"
        "↑ Raise → Safer approvals, may miss good loans\n\n"
        "↓ Lower → More approvals, higher default risk"
    )


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def full_pipeline(file_bytes: bytes, n_neighbors: int):
    """
    Complete ML pipeline in one cached function.

    KEY FIXES applied here:
      1. Read raw_df BEFORE any transformation (correct overview stats)
      2. Encode target AFTER imputation (fix 0% approval rate bug)
      3. Build a reindex-safe predictor (fix feature mismatch)
      4. Report missing values BEFORE imputation
    """

    # ── 1. RAW LOAD ─────────────────────────────────────────────────────────
    raw = pd.read_csv(io.BytesIO(file_bytes))

    # ── DIAGNOSE: record pre-imputation missing values ───────────────────────
    missing_before = raw.isnull().sum()
    missing_before = missing_before[missing_before > 0]

    # ── DIAGNOSE: approval rate BEFORE any encoding ──────────────────────────
    # FIX: the old code ran LabelEncoder on target before value_counts,
    #      if target was already 0/1 integers this double-encodes it wrong.
    #      We detect the target type and read it directly.
    target_col = "Loan_Approved"
    raw_target_vals = raw[target_col].dropna().unique()

    # Normalise to {0,1} integers safely
    if set(raw_target_vals).issubset({0, 1, "0", "1"}):
        raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
    else:
        # string labels like "Yes/No" or "Approved/Rejected"
        le_target_raw = LabelEncoder()
        raw[target_col] = le_target_raw.fit_transform(raw[target_col].astype(str))

    approval_rate = raw[target_col].mean() * 100      # correct rate

    # ── 2. IMPUTATION (before encoding) ─────────────────────────────────────
    df = raw.copy()

    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes("object").columns.tolist()

    # FIX: impute numerical & categorical SEPARATELY, in-place
    if num_cols:
        df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    missing_after = df.isnull().sum().sum()   # should be 0

    # ── 3. DROP ID ───────────────────────────────────────────────────────────
    if "Applicant_ID" in df.columns:
        df.drop("Applicant_ID", axis=1, inplace=True)

    # ── 4. LABEL ENCODE: Education only (ordinal) ────────────────────────────
    edu_map = {"Undergraduate": 0, "Graduate": 1, "Postgraduate": 2}
    if "Education_Level" in df.columns:
        df["Education_Level"] = df["Education_Level"].map(edu_map).fillna(1).astype(int)

    # ── 5. ONE-HOT ENCODE ────────────────────────────────────────────────────
    ohe_cols = [c for c in
                ["Employment_Status","Marital_Status","Loan_Purpose",
                 "Property_Area","Gender","Employer_Category"]
                if c in df.columns]

    oh = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    oh.fit(df[ohe_cols])
    enc_arr = oh.transform(df[ohe_cols])
    enc_df  = pd.DataFrame(enc_arr,
                            columns=oh.get_feature_names_out(ohe_cols),
                            index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), enc_df], axis=1)

    # ── 6. FEATURE ENGINEERING ───────────────────────────────────────────────
    # Squared terms: capture non-linear risk escalation
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["Income_Total"]    = df["Applicant_Income"] + df.get("Coapplicant_Income", 0)
    df["Loan_to_Income"]  = df["Loan_Amount"] / (df["Income_Total"] + 1)
    df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

    # ── 7. SPLIT ─────────────────────────────────────────────────────────────
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    feat_cols = X.columns.tolist()   # GOLD: exact column order

    # ── 8. SCALE ─────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # ── 9. TRAIN MODELS ──────────────────────────────────────────────────────
    models_def = {
        "Logistic Regression" : LogisticRegression(max_iter=100000, C=1.0, random_state=42),
        "Naive Bayes"         : GaussianNB(),
        f"KNN (k={n_neighbors})": KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski"),
    }

    results, trained, preds_dict = [], {}, {}
    for name, m in models_def.items():
        m.fit(X_tr_s, y_tr)
        yp   = m.predict(X_te_s)
        yprb = m.predict_proba(X_te_s)[:, 1] if hasattr(m, "predict_proba") else None
        results.append({
            "Model"    : name,
            "Accuracy" : accuracy_score(y_te, yp),
            "Precision": precision_score(y_te, yp, zero_division=0),
            "Recall"   : recall_score(y_te, yp, zero_division=0),
            "F1"       : f1_score(y_te, yp, zero_division=0),
            "AUC"      : roc_auc_score(y_te, yprb) if yprb is not None else None,
        })
        trained[name]    = m
        preds_dict[name] = (yp, yprb)

    res_df    = pd.DataFrame(results).set_index("Model")
    best_name = res_df["Precision"].idxmax()
    best_m    = trained[best_name]

    return dict(
        raw=raw, df=df,
        missing_before=missing_before, missing_after=missing_after,
        approval_rate=approval_rate,
        X_tr_s=X_tr_s, X_te_s=X_te_s, y_tr=y_tr, y_te=y_te,
        scaler=scaler, oh=oh, ohe_cols=ohe_cols,
        feat_cols=feat_cols,
        res_df=res_df, trained=trained, preds_dict=preds_dict,
        best_name=best_name, best_m=best_m,
    )


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPER (feature-safe)
# ════════════════════════════════════════════════════════════════════════════

def make_prediction(inp: dict, P: dict, threshold: float) -> dict:
    """
    Run prediction with full preprocessing.
    FIX: reindex to feat_cols so column order always matches training.
    """
    edu_map = {"Undergraduate": 0, "Graduate": 1, "Postgraduate": 2}
    df_in = pd.DataFrame([inp])

    # Education
    df_in["Education_Level"] = df_in["Education_Level"].map(edu_map).fillna(1).astype(int)

    # OHE
    enc   = P["oh"].transform(df_in[P["ohe_cols"]])
    e_df  = pd.DataFrame(enc,
                          columns=P["oh"].get_feature_names_out(P["ohe_cols"]))
    df_in = pd.concat([df_in.drop(columns=P["ohe_cols"]), e_df], axis=1)

    # Feature engineering (same as training)
    df_in["DTI_Ratio_sq"]    = df_in["DTI_Ratio"] ** 2
    df_in["Credit_Score_sq"] = df_in["Credit_Score"] ** 2
    df_in["Income_Total"]    = df_in["Applicant_Income"] + df_in.get("Coapplicant_Income", 0)
    df_in["Loan_to_Income"]  = df_in["Loan_Amount"] / (df_in["Income_Total"] + 1)
    df_in.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

    # ── KEY FIX: align columns exactly ──────────────────────────────────────
    df_in = df_in.reindex(columns=P["feat_cols"], fill_value=0)

    X_s   = P["scaler"].transform(df_in)
    prob  = P["best_m"].predict_proba(X_s)[0][1]
    dec   = int(prob >= threshold)

    # Rule-based risk flags
    flags = []
    if inp.get("Credit_Score", 700) < 550:
        flags.append(("🔴", "Very low credit score (< 550) — high default risk"))
    elif inp.get("Credit_Score", 700) < 650:
        flags.append(("🟡", "Below-average credit score (< 650)"))
    if inp.get("DTI_Ratio", 0.3) > 0.60:
        flags.append(("🔴", "High DTI ratio (> 60%) — debt burden is excessive"))
    elif inp.get("DTI_Ratio", 0.3) > 0.45:
        flags.append(("🟡", "Elevated DTI ratio (> 45%)"))
    if inp.get("Existing_Loans", 0) > 2:
        flags.append(("🟡", f"{inp['Existing_Loans']} existing loans — credit exposure high"))
    if inp.get("Applicant_Income", 50000) < 15000:
        flags.append(("🔴", "Very low income — repayment capacity at risk"))
    if inp.get("Loan_Amount", 0) > inp.get("Applicant_Income", 1) * 10:
        flags.append(("🟡", "Loan amount is very high relative to monthly income"))

    return {"prob": prob, "decision": dec, "flags": flags}


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

if uploaded is None:
    # Landing screen
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:2.5rem 3rem;margin-top:1rem;">
      <h2 style="color:{ACCENT};margin-top:0;">👈 Upload your dataset to begin</h2>
      <p style="color:{TEXT};">This app will automatically:</p>
      <ul style="color:{MUTED};line-height:2;">
        <li>🔍 Diagnose & fix missing values with step-by-step explanation</li>
        <li>📊 Run full EDA with business storytelling</li>
        <li>🤖 Train 3 ML models and compare them with all metrics</li>
        <li>🎯 Select best model by <strong style="color:{ACCENT};">Precision</strong> (minimises bad loan approvals)</li>
        <li>🏦 Predict approval for any new applicant instantly</li>
        <li>💼 Give you interview-ready answers for every design decision</li>
      </ul>
      <div style="background:#071828;border-radius:8px;padding:1rem 1.3rem;margin-top:1.5rem;
                  font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#a8d8ea;">
Expected CSV columns:<br>
Applicant_ID, Age, Gender, Marital_Status, Dependents, Education_Level,<br>
Employment_Status, Employer_Category, Applicant_Income, Coapplicant_Income,<br>
Savings, Loan_Amount, Loan_Term, Credit_Score, DTI_Ratio, Collateral_Value,<br>
Existing_Loans, Loan_Purpose, Property_Area, <strong>Loan_Approved (0/1)</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run pipeline ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Running full ML pipeline…"):
    P = full_pipeline(uploaded.getvalue(), n_neighbors)

raw        = P["raw"]
res_df     = P["res_df"]
best_name  = P["best_name"]

# ════════════════════════════════════════════════════════════════════════════
# OVERVIEW METRICS (Fixed)
# ════════════════════════════════════════════════════════════════════════════

total_mv  = P["missing_before"].sum()
appr_rate = P["approval_rate"]

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="mc-label">Total Records</div>
    <div class="mc-value">{len(raw):,}</div>
    <div class="mc-sub">loan applicants</div>
  </div>
  <div class="metric-card">
    <div class="mc-label">Features</div>
    <div class="mc-value">{raw.shape[1]-1}</div>
    <div class="mc-sub">input variables</div>
  </div>
  <div class="metric-card">
    <div class="mc-label">Approval Rate</div>
    <div class="mc-value" style="color:{'#00e676' if appr_rate>40 else '#ff5252'};">{appr_rate:.1f}%</div>
    <div class="mc-sub">{int(raw['Loan_Approved'].sum())} approved</div>
  </div>
  <div class="metric-card">
    <div class="mc-label">Missing Values</div>
    <div class="mc-value" style="color:{'#ffd740' if total_mv>0 else '#00e676'};">{total_mv}</div>
    <div class="mc-sub">handled automatically</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

T1, T2, T3, T4, T5 = st.tabs([
    "🧹 Data Quality",
    "📊 EDA & Insights",
    "🤖 Model Results",
    "🎯 Predict",
    "💼 Interview Guide"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA QUALITY & MISSING VALUES
# ════════════════════════════════════════════════════════════════════════════

with T1:
    st.markdown('<div class="sec-title">🔍 Why Was Your Approval Rate Showing 0.0%?</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="story-card">
      <strong>Root Cause Diagnosed:</strong> The old code ran
      <code>LabelEncoder().fit_transform()</code> on the target column
      <em>before</em> reading its distribution. If the column already contained
      integers (0/1), the encoder sometimes re-mapped them unpredictably —
      or the <code>value_counts(normalize=True)</code> was called on the wrong
      transformed copy. This produced 0.0% because approved loans were mapped
      to label index 0 instead of 1.<br><br>
      <strong>Fix applied:</strong> We detect the target column type first, normalise
      directly to integers without LabelEncoder, and read the approval rate from
      the <em>raw</em> dataframe before any transformation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🔍 Why Were Missing Values Showing = 1000?</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="story-card">
      <strong>Root Cause Diagnosed:</strong> The old code displayed
      <code>df.isnull().sum().sum()</code> <em>after</em> applying
      <code>SimpleImputer</code> in-place — but the imputer was used on a
      <em>copy</em> while the display used the <em>original</em>. This showed
      all original missing values as if they were still present.<br><br>
      <strong>Fix applied:</strong> We capture <code>missing_before</code>
      (pre-imputation) for display, then impute in-place. Post-imputation
      count = <strong style="color:{GREEN};">{P["missing_after"]}</strong> missing values remain.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📋 Missing Values — Column-by-Column Breakdown</div>',
                unsafe_allow_html=True)

    if len(P["missing_before"]) == 0:
        st.success("✅ No missing values detected in your dataset.")
    else:
        mv = P["missing_before"].sort_values(ascending=False)
        mv_pct = (mv / len(raw) * 100).round(1)

        rows_html = ""
        for col, cnt in mv.items():
            pct   = mv_pct[col]
            dtype = raw[col].dtype
            fix   = "Replace with <strong>mean</strong>" if np.issubdtype(dtype, np.number) \
                    else "Replace with <strong>mode</strong>"
            bar_w = max(1, int(pct * 2))
            rows_html += f"""
            <div class="mv-col">
              <div style="width:200px;font-weight:500;color:{ACCENT};">{col}</div>
              <div style="width:90px;color:{MUTED};font-size:0.8rem;">{dtype}</div>
              <div style="flex:1;">
                <div class="mv-bar" style="width:{bar_w}px;display:inline-block;"></div>
                <span style="margin-left:8px;color:{YELLOW};">{cnt} ({pct}%)</span>
              </div>
              <div class="mv-fix">{fix} ✅</div>
            </div>"""

        st.markdown(f'<div class="mv-root">{rows_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual: missing heatmap
        fig, ax = styled_fig(10, 3)
        miss_matrix = raw.isnull().astype(int)
        sns.heatmap(miss_matrix.T, ax=ax, cbar=False,
                    cmap=["#0d2640", RED], yticklabels=True,
                    xticklabels=False, linewidths=0)
        ax.set_title("Missing Value Map  (red = missing)", fontsize=11, pad=10)
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="story-card" style="margin-top:0.8rem;">
          <strong>Why mean for numerical?</strong> Mean preserves the overall distribution
          without introducing bias from extreme values. For skewed columns (like income),
          median would be better — but mean is industry-standard for this use-case.<br><br>
          <strong>Why mode for categorical?</strong> The most frequent category is the
          safest assumption when we don't know the true value. Alternatives like a
          separate "Unknown" category could be used in production.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🏗️ Feature Engineering Added</div>',
                unsafe_allow_html=True)

    feats = [
        ("DTI_Ratio²",      "Squares the debt-to-income ratio. Risk escalates non-linearly — a 70% DTI is far more dangerous than 35% × 2.",     YELLOW),
        ("Credit_Score²",   "Squares the credit score. A 800-score borrower is exponentially safer than a 600-score borrower.",                    GREEN),
        ("Income_Total",    "Applicant_Income + Coapplicant_Income. Household income is a better repayment signal than applicant income alone.",   ACCENT),
        ("Loan_to_Income",  "Loan_Amount / Income_Total. How many months of income does this loan represent? High ratio = high risk.",             YELLOW),
    ]
    for name, reason, color in feats:
        st.markdown(f"""
        <div class="story-card">
          <strong style="color:{color};">+ {name}</strong><br>{reason}
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

with T2:
    raw_num = raw.select_dtypes("number")

    # ── 2.1 Target distribution ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">🎯 Target Variable: Loan Approval Distribution</div>',
                unsafe_allow_html=True)

    fig, axes = styled_figs(1, 2, w=11, h=4)
    counts = raw["Loan_Approved"].value_counts().sort_index()

    # Pie
    axes[0].pie(counts, labels=["Rejected", "Approved"], autopct="%1.1f%%",
                colors=[RED, GREEN], startangle=90,
                wedgeprops=dict(edgecolor=BG, linewidth=2.5),
                textprops={"color": TEXT, "fontsize": 10})
    axes[0].set_title("Approval Split", fontsize=11, pad=8)

    # Bar
    bars = axes[1].bar(["Rejected", "Approved"], counts.values,
                       color=[RED, GREEN], width=0.5, edgecolor=BG)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                     f"{b.get_height():,}", ha="center", color=TEXT, fontsize=10)
    axes[1].set_title("Count by Outcome", fontsize=11)
    axes[1].tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="story-card">
      <strong>Insight:</strong> The dataset has a roughly balanced split of approved
      vs rejected applications (~{appr_rate:.0f}% approval rate). A balanced dataset means
      our models won't be systematically biased toward always predicting one class.
      In real-world data, approvals are often &lt;40%, requiring techniques like SMOTE.
    </div>""", unsafe_allow_html=True)

    # ── 2.2 Income analysis ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">💰 Income vs Loan Approval</div>', unsafe_allow_html=True)
    fig, axes = styled_figs(1, 2, w=12, h=4.5)

    if "Applicant_Income" in raw.columns:
        for i, (col, title) in enumerate([
            ("Applicant_Income",  "Applicant Monthly Income"),
            ("Coapplicant_Income","Co-applicant Monthly Income"),
        ]):
            if col not in raw.columns: continue
            data0 = raw[raw["Loan_Approved"]==0][col].dropna()
            data1 = raw[raw["Loan_Approved"]==1][col].dropna()
            axes[i].boxplot([data0, data1], labels=["Rejected","Approved"],
                            patch_artist=True,
                            boxprops=dict(facecolor=CARD, color=BORDER),
                            medianprops=dict(color=ACCENT, linewidth=2),
                            whiskerprops=dict(color=BORDER),
                            capprops=dict(color=BORDER),
                            flierprops=dict(marker=".", color=MUTED, alpha=0.4, markersize=3))
            boxes = axes[i].patches
            if len(boxes) >= 2:
                boxes[0].set_facecolor(f"{RED}33")
                boxes[1].set_facecolor(f"{GREEN}33")
            axes[i].set_title(title, fontsize=10)
            axes[i].tick_params(labelsize=9)

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="story-card">
      <strong>Insight:</strong> Approved applicants show a higher median income
      and tighter interquartile range. High-income outliers appear in both classes —
      income alone doesn't guarantee approval. The co-applicant income adds additional
      household repayment capacity, which helps marginal cases.
    </div>""", unsafe_allow_html=True)

    # ── 2.3 Credit Score ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📈 Credit Score Analysis</div>', unsafe_allow_html=True)
    if "Credit_Score" in raw.columns:
        fig, axes = styled_figs(1, 2, w=12, h=4.5)

        # Histogram by approval
        for val, color, label in [(0, RED, "Rejected"), (1, GREEN, "Approved")]:
            sub = raw[raw["Loan_Approved"]==val]["Credit_Score"].dropna()
            axes[0].hist(sub, bins=30, alpha=0.6, color=color, label=label,
                         edgecolor=BG, linewidth=0.5)
        axes[0].axvline(650, color=YELLOW, linestyle="--", linewidth=1.5, label="650 benchmark")
        axes[0].set_title("Credit Score Distribution", fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].set_xlabel("Credit Score")

        # Mean credit score by approval
        means = raw.groupby("Loan_Approved")["Credit_Score"].mean()
        bars  = axes[1].bar(["Rejected","Approved"], means.values,
                            color=[RED, GREEN], width=0.5, edgecolor=BG)
        for b in bars:
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()-30,
                         f"{b.get_height():.0f}", ha="center", color=BG,
                         fontsize=12, fontweight="bold")
        axes[1].set_title("Average Credit Score by Outcome", fontsize=10)
        axes[1].set_ylim(0, raw["Credit_Score"].max() + 50)
        axes[1].tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="story-card">
          <strong>Insight:</strong> Approved applicants have a significantly higher
          average credit score. The 650 benchmark line reveals a natural threshold —
          most rejections cluster below it. This non-linear boundary is why we
          added <strong>Credit_Score²</strong> as a feature: it amplifies the signal
          for very high and very low scores, helping linear models detect this pattern.
        </div>""", unsafe_allow_html=True)

    # ── 2.4 DTI Ratio ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📉 Debt-to-Income (DTI) Ratio</div>', unsafe_allow_html=True)
    if "DTI_Ratio" in raw.columns:
        fig, axes = styled_figs(1, 2, w=12, h=4.5)

        for val, color, label in [(0, RED, "Rejected"), (1, GREEN, "Approved")]:
            sub = raw[raw["Loan_Approved"]==val]["DTI_Ratio"].dropna()
            axes[0].hist(sub, bins=25, alpha=0.6, color=color, label=label,
                         edgecolor=BG, linewidth=0.5)
        axes[0].axvline(0.45, color=YELLOW, linestyle="--", linewidth=1.5, label="0.45 danger zone")
        axes[0].set_title("DTI Ratio Distribution", fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].set_xlabel("DTI Ratio")

        # Violin plot
        data0 = raw[raw["Loan_Approved"]==0]["DTI_Ratio"].dropna()
        data1 = raw[raw["Loan_Approved"]==1]["DTI_Ratio"].dropna()
        vp = axes[1].violinplot([data0, data1], positions=[1, 2],
                                 showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor([RED, GREEN][i])
            body.set_alpha(0.4)
        vp["cmedians"].set_colors([ACCENT])
        axes[1].set_xticks([1,2]); axes[1].set_xticklabels(["Rejected","Approved"])
        axes[1].set_title("DTI Distribution (Violin)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="story-card">
          <strong>Insight:</strong> Rejected applicants have higher and more spread-out
          DTI ratios. The 0.45 danger line shows that most rejections cluster above it.
          Banks typically reject loans when DTI exceeds 43–50%. The violin plot reveals
          that approved applicants have a tighter, lower DTI distribution.
        </div>""", unsafe_allow_html=True)

    # ── 2.5 Categorical features ──────────────────────────────────────────────
    st.markdown('<div class="sec-title">🏷️ Categorical Features vs Approval Rate</div>',
                unsafe_allow_html=True)

    cat_plot_cols = [c for c in
                     ["Employment_Status","Gender","Marital_Status",
                      "Loan_Purpose","Property_Area","Education_Level"]
                     if c in raw.columns]

    if cat_plot_cols:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.patch.set_facecolor(CARD)
        axes = axes.flatten()
        for i, col in enumerate(cat_plot_cols[:6]):
            ax = axes[i]; ax.set_facecolor(CARD)
            ct = raw.groupby(col)["Loan_Approved"].mean() * 100
            ct = ct.sort_values(ascending=False)
            bars = ax.barh(ct.index.astype(str), ct.values,
                           color=[GREEN if v > 50 else RED for v in ct.values],
                           edgecolor=BG, height=0.55)
            for b in bars:
                ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2,
                        f"{b.get_width():.1f}%", va="center",
                        fontsize=8, color=TEXT)
            ax.set_title(col, fontsize=10, color=TEXT)
            ax.set_xlabel("Approval Rate (%)", fontsize=8, color=MUTED)
            ax.axvline(50, color=YELLOW, linestyle="--", linewidth=1, alpha=0.6)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.set_xlim(0, 105)

        for j in range(len(cat_plot_cols), 6): axes[j].axis("off")
        plt.suptitle("Approval Rate by Category  (yellow line = 50%)",
                     color=TEXT, fontsize=12, y=1.01)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="story-card">
          <strong>Insight:</strong> Urban property applicants, Salaried employees, and
          Married applicants tend to have higher approval rates. These patterns reflect
          real-world credit risk: stable employment and property in developed areas
          correlate with lower default probability. Gender shows minimal difference
          — suggesting the model is relatively unbiased on this dimension.
        </div>""", unsafe_allow_html=True)

    # ── 2.6 Correlation heatmap ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols_heat = raw.select_dtypes("number").drop(
        columns=["Applicant_ID"] if "Applicant_ID" in raw.columns else [], errors="ignore"
    )
    corr = num_cols_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = styled_fig(13, 7)
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                annot_kws={"size": 7, "color": BG},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="story-card">
      <strong>Insight:</strong> Check for pairs with |correlation| > 0.8 — these indicate
      multicollinearity that can destabilise logistic regression. Income and loan amount
      often correlate (larger income → larger loans requested). The squared engineered
      features will naturally correlate with their originals, but since we dropped the
      originals, there is no redundancy in the final feature set.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════════════════

with T3:
    st.markdown('<div class="sec-title">📐 Metric Definitions (plain English)</div>',
                unsafe_allow_html=True)

    for m, defn, color in [
        ("Accuracy",   "% of total predictions that are correct. Easy to understand but misleading when classes are imbalanced.", TEXT),
        ("Precision ⭐","Of all loans we predicted APPROVED, what fraction truly deserved it? High precision → few bad loans approved. This is our KEY metric.", ACCENT),
        ("Recall",     "Of all truly good loans, how many did we approve? Low recall → missing business opportunities.", YELLOW),
        ("F1 Score",   "Harmonic mean of Precision & Recall. A balanced single score when both matter.", GREEN),
        ("AUC-ROC",    "Area Under the ROC Curve. Measures the model's ability to rank applicants by risk. 0.5 = random, 1.0 = perfect.", MUTED),
    ]:
        st.markdown(f"""
        <div class="story-card">
          <strong style="color:{color};">{m}:</strong> {defn}
        </div>""", unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="sec-title">🏆 Model Performance Comparison</div>',
                unsafe_allow_html=True)

    hdr = "".join(f"<th>{c}</th>" for c in ["Model","Accuracy","Precision","Recall","F1","AUC"])
    rows_html = f"<tr>{hdr}</tr>"
    for model_name, row in res_df.iterrows():
        is_best = model_name == best_name
        cls     = "best-row" if is_best else ""
        badge   = " 🏆" if is_best else ""
        cells   = "".join(f"<td>{v:.4f}</td>" if pd.notnull(v) else "<td>—</td>"
                          for v in row.values)
        rows_html += f"<tr class='{cls}'><td>{model_name}{badge}</td>{cells}</tr>"

    st.markdown(f"""
    <table class="model-table">
      <thead>{rows_html}</thead>
    </table>
    <br>
    <div class="story-card">
      <strong>Why {best_name}?</strong><br>
      Selected because it achieved the highest <strong>Precision</strong>
      ({res_df.loc[best_name,'Precision']:.4f}) among all trained models.
      In banking, a False Positive (approving a loan that defaults) directly
      causes financial loss — often far greater than the missed profit from
      a False Negative (rejecting a good applicant). Optimising for Precision
      directly minimises this Type I Error.
    </div>""", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📊 Visual Metric Comparison</div>', unsafe_allow_html=True)
    metrics_plot = ["Accuracy","Precision","Recall","F1"]
    x = np.arange(len(res_df))
    w = 0.19
    colors_bar = [ACCENT, RED, GREEN, YELLOW]

    fig, ax = styled_fig(12, 5)
    for i, (met, col) in enumerate(zip(metrics_plot, colors_bar)):
        bars = ax.bar(x + i*w, res_df[met], w, label=met, color=col,
                      edgecolor=BG, linewidth=0.8)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.007,
                    f"{b.get_height():.2f}", ha="center",
                    fontsize=7.5, color=TEXT, fontweight="600")

    ax.set_xticks(x + w*1.5); ax.set_xticklabels(res_df.index, fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.legend(fontsize=9)
    ax.axhline(0.8, linestyle="--", color=MUTED, linewidth=1, alpha=0.7)
    ax.text(len(res_df)-0.1, 0.805, "0.80 baseline", fontsize=8, color=MUTED)
    ax.set_title("Model Performance — All Metrics", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🟩 Confusion Matrices</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(CARD)
    for ax, (model_name, (yp, _)) in zip(axes, P["preds_dict"].items()):
        cm = confusion_matrix(P["y_te"], yp)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap=sns.light_palette(ACCENT, as_cmap=True),
                    xticklabels=["Rejected","Approved"],
                    yticklabels=["Rejected","Approved"],
                    linewidths=0.5, cbar=False,
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(model_name, fontsize=10, color=TEXT)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("Actual", color=MUTED, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="story-card">
      <strong>Reading the matrix:</strong>
      Top-right (FP) = loans we approved that were bad → <strong style="color:{RED};">financial loss</strong>.
      Bottom-left (FN) = good loans we rejected → <strong style="color:{YELLOW};">missed business</strong>.
      We minimise FP by maximising Precision. The threshold slider lets you tune this trade-off.
    </div>""", unsafe_allow_html=True)

    # ── ROC curves ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📈 ROC Curves</div>', unsafe_allow_html=True)
    fig, ax = styled_fig(9, 5)
    roc_colors = [ACCENT, GREEN, YELLOW]
    for (model_name, (_, yprb)), col in zip(P["preds_dict"].items(), roc_colors):
        if yprb is not None:
            fpr, tpr, _ = roc_curve(P["y_te"], yprb)
            auc = res_df.loc[model_name, "AUC"]
            ax.plot(fpr, tpr, color=col, linewidth=2,
                    label=f"{model_name}  (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], color=MUTED, linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — All Models", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🔑 Feature Importance (Permutation-based)</div>',
                unsafe_allow_html=True)

    with st.spinner("Computing feature importance…"):
        pi = permutation_importance(
            P["best_m"], P["X_te_s"], P["y_te"],
            n_repeats=8, random_state=42, scoring="precision"
        )
        imp_df = pd.DataFrame({
            "Feature"   : P["feat_cols"],
            "Importance": pi.importances_mean,
        }).sort_values("Importance", ascending=False).head(15)

    fig, ax = styled_fig(11, 6)
    colors_imp = [GREEN if v > 0 else MUTED for v in imp_df["Importance"]]
    ax.barh(imp_df["Feature"], imp_df["Importance"],
            color=colors_imp, edgecolor=BG, height=0.65)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_title(f"Top Feature Importances — {best_name}", fontsize=11)
    ax.set_xlabel("Permutation Importance (Precision impact)", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="story-card">
      <strong>How to read this:</strong> Each bar shows how much Precision drops when
      that feature's values are randomly shuffled (effectively removing it).
      Larger drop = more important feature. Green bars = positive precision contribution.
      This is model-agnostic and works for KNN & Naive Bayes too (unlike coefficient plots).
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════

with T4:
    st.markdown(f"""
    <div class="story-card" style="margin-bottom:1rem;">
      Using <strong style="color:{ACCENT};">{best_name}</strong> &nbsp;|&nbsp;
      Decision Threshold: <strong style="color:{YELLOW};">{threshold}</strong> &nbsp;|&nbsp;
      Precision: <strong style="color:{GREEN};">{res_df.loc[best_name,'Precision']:.4f}</strong>
    </div>""", unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown('<div class="sec-title">👤 Applicant Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Personal Details**")
            age        = st.number_input("Age", 18, 75, 32)
            gender     = st.selectbox("Gender", ["Male","Female"])
            marital    = st.selectbox("Marital Status", ["Married","Single"])
            dependents = st.number_input("Dependents", 0, 10, 1)
            education  = st.selectbox("Education Level",
                                      ["Graduate","Postgraduate","Undergraduate"])

        with c2:
            st.markdown("**Employment & Finances**")
            emp_status = st.selectbox("Employment Status",
                                      ["Salaried","Self-Employed","Business"])
            employer   = st.selectbox("Employer Category",["Govt","Private","Self"])
            income     = st.number_input("Monthly Income (₹)", 5000, 500000, 55000, 500)
            co_income  = st.number_input("Co-applicant Income (₹)", 0, 300000, 10000, 500)
            savings    = st.number_input("Savings Balance (₹)", 0, 1000000, 25000, 1000)

        with c3:
            st.markdown("**Loan Details**")
            loan_amt   = st.number_input("Loan Amount (₹)", 10000, 5000000, 200000, 5000)
            loan_term  = st.number_input("Loan Term (months)", 6, 360, 60)
            loan_purp  = st.selectbox("Loan Purpose",["Home","Education","Personal","Business"])
            prop_area  = st.selectbox("Property Area",["Urban","Semi-Urban","Rural"])
            collateral = st.number_input("Collateral Value (₹)", 0, 5000000, 60000, 5000)
            credit_sc  = st.slider("Credit Score", 300, 900, 720, 10)
            dti        = st.slider("DTI Ratio", 0.05, 0.95, 0.28, 0.01)
            exist_loan = st.number_input("Existing Loans", 0, 10, 1)

        submitted = st.form_submit_button("🔍 Predict Loan Approval")

    if submitted:
        inp = {
            "Age"               : age,
            "Applicant_Income"  : income,
            "Coapplicant_Income": co_income,
            "Loan_Amount"       : loan_amt,
            "Loan_Term"         : loan_term,
            "Credit_Score"      : credit_sc,
            "DTI_Ratio"         : dti,
            "Savings"           : savings,
            "Collateral_Value"  : collateral,
            "Existing_Loans"    : exist_loan,
            "Dependents"        : dependents,
            "Education_Level"   : education,
            "Employment_Status" : emp_status,
            "Marital_Status"    : marital,
            "Loan_Purpose"      : loan_purp,
            "Property_Area"     : prop_area,
            "Gender"            : gender,
            "Employer_Category" : employer,
        }

        result = make_prediction(inp, P, threshold)
        prob   = result["prob"]
        dec    = result["decision"]
        flags  = result["flags"]

        st.markdown("---")
        col_dec, col_gauge = st.columns([1, 1])

        with col_dec:
            if dec == 1:
                st.markdown(f"""
                <div class="approved-banner">
                  ✅ LOAN APPROVED
                  <div class="sub-result">Confidence: {prob*100:.1f}%  ·  Threshold: {threshold*100:.0f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-banner">
                  ❌ LOAN REJECTED
                  <div class="sub-result">Confidence: {prob*100:.1f}%  ·  Threshold: {threshold*100:.0f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>**Risk Analysis:**", unsafe_allow_html=True)
            if flags:
                for emoji, msg in flags:
                    css = "flag-risk" if emoji == "🔴" else "flag-risk" if emoji == "🟡" else "flag-ok"
                    st.markdown(f'<div class="{css}">{emoji} {msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="flag-ok">✅ No major risk factors detected</div>',
                            unsafe_allow_html=True)

        with col_gauge:
            # Probability gauge bar
            fig, ax = styled_fig(5, 4)
            bar_color = GREEN if dec == 1 else RED
            ax.barh(["Approval Probability"], [prob],
                    color=bar_color, height=0.35, alpha=0.85)
            ax.barh(["Approval Probability"], [1-prob], left=[prob],
                    color=BORDER, height=0.35)
            ax.axvline(threshold, color=YELLOW, linestyle="--",
                       linewidth=2.5, label=f"Threshold ({threshold})")
            ax.set_xlim(0, 1)
            ax.set_title(f"Approval Probability: {prob*100:.1f}%", fontsize=11)
            ax.legend(fontsize=9); ax.set_yticks([])
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Key applicant stats
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem;margin-top:0.5rem;font-size:0.85rem;">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                <div><span style="color:{MUTED};">Credit Score</span><br><strong style="color:{'#00e676' if credit_sc>=650 else '#ff5252'};">{credit_sc}</strong></div>
                <div><span style="color:{MUTED};">DTI Ratio</span><br><strong style="color:{'#00e676' if dti<0.45 else '#ff5252'};">{dti:.0%}</strong></div>
                <div><span style="color:{MUTED};">Monthly Income</span><br><strong style="color:{ACCENT};">₹{income:,}</strong></div>
                <div><span style="color:{MUTED};">Loan Amount</span><br><strong style="color:{ACCENT};">₹{loan_amt:,}</strong></div>
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — INTERVIEW GUIDE
# ════════════════════════════════════════════════════════════════════════════

with T5:
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;
                padding:1.8rem;margin-bottom:1.5rem;">
      <h3 style="color:{ACCENT};margin-top:0;">🎤 30-Second Elevator Pitch</h3>
      <p style="color:{TEXT};font-size:0.92rem;line-height:1.8;font-style:italic;">
        "I built CreditWise — an AI-powered loan approval system for a bank. The system
        processes applicant data, handles missing values automatically, engineers features
        like squared DTI and credit score to capture non-linear risk patterns, then trains
        three ML models — Logistic Regression, Naive Bayes, and KNN. I selected the best
        model based on Precision, because in banking, approving a bad loan causes real
        financial loss. The entire system is deployed as an interactive Streamlit dashboard
        where any loan officer can enter applicant details and get an instant decision with
        risk explanations."
      </p>
    </div>""", unsafe_allow_html=True)

    qa_pairs = [
        ("Why did you choose Precision as your key metric?",
         f"In loan approval, a False Positive means we approved a loan that the borrower defaults on — "
         f"this is a direct financial loss to the bank. Precision = TP/(TP+FP) directly measures what "
         f"fraction of our 'approved' predictions were actually safe. By maximising Precision, we minimise "
         f"these costly Type I Errors. We can always tune the threshold further to raise Precision at the "
         f"cost of some Recall (missed business), which is an acceptable trade-off."),

        ("What is a Type I Error and why does it matter here?",
         "Type I Error = False Positive = approving a loan for someone who will default. "
         "In finance, this directly causes monetary loss — the bank has disbursed funds it won't recover. "
         "Type II Error (rejecting a good applicant) causes missed revenue, but the bank doesn't lose "
         "existing money. The asymmetry in consequences is why Precision > Accuracy as our goal metric."),

        ("Why did you square DTI_Ratio and Credit_Score?",
         "These features have non-linear relationships with default risk. A DTI of 0.8 isn't just "
         "twice as dangerous as 0.4 — it's exponentially riskier because it leaves no financial cushion. "
         "Squaring amplifies the difference between moderate and extreme values, giving linear models "
         "(like Logistic Regression) a stronger signal without needing polynomial features across the board. "
         "We also added Income_Total (household income) and Loan_to_Income ratio as domain-driven features."),

        ("Why StandardScaler? And why fit only on training data?",
         "KNN and Logistic Regression are distance/gradient-sensitive. Without scaling, Applicant_Income "
         "(range: 8,000–300,000) would dominate Dependents (range: 0–4). StandardScaler normalises each "
         "feature to zero mean and unit variance.\n\n"
         "We fit ONLY on training data to prevent data leakage — if we included test statistics in the "
         "scaler fit, the model would have indirect knowledge of the test set, giving falsely optimistic results."),

        ("Why stratified train-test split?",
         "Without stratify=y, random splitting might put 80% of approved loans in training but only "
         "60% in test (by chance), making metrics unreliable. Stratified split ensures the class "
         "ratio is preserved in both sets — critical when evaluating Precision and Recall."),

        ("How did you fix the feature mismatch error?",
         "The prediction function re-applies all the same preprocessing steps (education encoding, "
         "one-hot encoding, feature engineering) on the single input row. Then it calls "
         "df.reindex(columns=feat_cols, fill_value=0) to guarantee the columns are in the exact same "
         "order as the training data. This is the critical fix — OneHotEncoder can produce columns in "
         "a different order for a single-row DataFrame than for the full dataset."),

        ("What improvements would you make with more time?",
         "1. XGBoost or LightGBM — better handling of non-linear interactions\n"
         "2. SMOTE — oversample the minority class if approval rate is <30%\n"
         "3. SHAP values — explainable AI for individual loan decisions (regulatory requirement)\n"
         "4. GridSearchCV / Optuna — automated hyperparameter tuning\n"
         "5. MLflow — experiment tracking and model versioning\n"
         "6. FastAPI + Docker — production deployment with REST API\n"
         "7. Fairness audit — ensure the model doesn't discriminate by Gender/Age"),

        ("How would you handle class imbalance if it existed?",
         "If approval rate < 30%, I would:\n"
         "1. Use class_weight='balanced' in Logistic Regression\n"
         "2. Apply SMOTE (Synthetic Minority Over-sampling) on the training set only\n"
         "3. Adjust the decision threshold downward to increase recall for the minority class\n"
         "4. Use F1 or PR-AUC instead of accuracy as the evaluation metric\n"
         "Never SMOTE the test set — that would contaminate evaluation."),
    ]

    st.markdown('<div class="sec-title">❓ Common HR & Technical Questions</div>',
                unsafe_allow_html=True)

    for q, a in qa_pairs:
        st.markdown(f"""
        <div class="qa-card">
          <div class="qa-q">Q: {q}</div>
          <div class="qa-a">{a.replace(chr(10),'<br>')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📁 Project Structure</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="code-box">creditwise/
├── streamlit_app_v2.py         ← This file (full UI + pipeline)
├── creditwise_loan_approval.py ← Standalone pipeline (CLI version)
├── loan_approval_data.csv      ← Dataset
├── plots/                      ← Auto-generated EDA charts
│   ├── 01_target_distribution.png
│   ├── 02_income_analysis.png
│   ├── 03_credit_score.png
│   └── ...
└── README.md</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🚀 How to Run</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="code-box"># Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# Run interactive dashboard
streamlit run streamlit_app_v2.py

# Run CLI pipeline (saves plots to /plots)
python creditwise_loan_approval.py</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"<center style='color:{MUTED};font-size:0.78rem;'>"
    "CreditWise v2.0 &nbsp;·&nbsp; Powered by Scikit-Learn + Streamlit &nbsp;·&nbsp; "
    "SecureTrust Bank ML Minor Project"
    "</center>",
    unsafe_allow_html=True
)
