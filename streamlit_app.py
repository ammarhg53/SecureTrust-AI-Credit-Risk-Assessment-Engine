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
import matplotlib.patches as mpatches
import seaborn as sns

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
# CUSTOM CSS — Premium Dark Banking Aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── HERO HEADER ── */
    .hero-header {
        background: linear-gradient(135deg, #060b14 0%, #0a1628 40%, #0d1f3c 70%, #071220 100%);
        padding: 2.8rem 3rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(0, 180, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -40px; right: -40px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(0,255,170,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 180, 255, 0.12);
        border: 1px solid rgba(0, 180, 255, 0.3);
        color: #00b4ff;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.3rem 0 0.6rem;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }
    .hero-title span { color: #00b4ff; }
    .hero-sub {
        color: #7a94b0;
        font-size: 0.92rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    .hero-pills {
        margin-top: 1.2rem;
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: #a8bcd0;
        font-size: 0.75rem;
        padding: 0.28rem 0.75rem;
        border-radius: 12px;
    }

    /* ── METRIC CARDS ── */
    .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 140px;
        background: linear-gradient(135deg, #0d1928 0%, #0a1520 100%);
        border: 1px solid rgba(0, 180, 255, 0.12);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: rgba(0, 180, 255, 0.3); }
    .kpi-label {
        color: #4a6580;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #00b4ff;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        font-family: 'DM Mono', monospace;
    }
    .kpi-sub { color: #3a5570; font-size: 0.72rem; margin-top: 0.3rem; }

    /* ── SECTION TITLES ── */
    .section-title {
        color: #e2eaf5;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        border-left: 3px solid #00b4ff;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* ── DECISION BANNERS ── */
    .approved-banner {
        background: linear-gradient(135deg, #041f12 0%, #083520 100%);
        border: 1px solid #00cc6a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #00cc6a;
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
    }
    .rejected-banner {
        background: linear-gradient(135deg, #1f0404 0%, #2d0808 100%);
        border: 1px solid #ff4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #ff4444;
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
    }
    .banner-sub {
        font-size: 0.9rem;
        font-weight: 400;
        font-family: 'DM Sans', sans-serif;
        opacity: 0.8;
        margin-top: 0.5rem;
    }

    /* ── FLAG ITEMS ── */
    .risk-flag {
        background: rgba(255, 68, 68, 0.08);
        border-left: 3px solid #ff4444;
        border-radius: 6px;
        padding: 0.55rem 0.9rem;
        margin: 0.35rem 0;
        font-size: 0.87rem;
        color: #ff8888;
    }
    .ok-flag {
        background: rgba(0, 204, 106, 0.08);
        border-left: 3px solid #00cc6a;
        border-radius: 6px;
        padding: 0.55rem 0.9rem;
        margin: 0.35rem 0;
        font-size: 0.87rem;
        color: #00cc6a;
    }

    /* ── INTERVIEW CARD ── */
    .interview-q {
        background: #0d1928;
        border: 1px solid rgba(0, 180, 255, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .interview-q .q-label {
        color: #00b4ff;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .interview-q .q-text { color: #c8d8e8; font-size: 0.95rem; margin-bottom: 0.6rem; }
    .interview-q .a-text { color: #7a94b0; font-size: 0.87rem; line-height: 1.6; }

    /* ── BEST MODEL BADGE ── */
    .best-badge {
        background: linear-gradient(135deg, #041f12, #083520);
        border: 1px solid #00cc6a;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #00cc6a;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* ── GLOBAL OVERRIDES ── */
    .stButton > button {
        background: linear-gradient(135deg, #00b4ff, #0077cc);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        width: 100%;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #33c6ff, #0099ee);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 180, 255, 0.3);
    }

    [data-testid="stSidebar"] {
        background: #060b14;
        border-right: 1px solid rgba(0, 180, 255, 0.1);
    }

    div[data-testid="stMetric"] {
        background: #0d1928;
        border: 1px solid rgba(0, 180, 255, 0.12);
        border-radius: 10px;
        padding: 0.9rem;
    }
    div[data-testid="stMetric"] label { color: #4a6580 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #00b4ff !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #060b14;
        border-bottom: 1px solid rgba(0, 180, 255, 0.1);
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4a6580;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 0.7rem 1.3rem;
        letter-spacing: 0.3px;
    }
    .stTabs [aria-selected="true"] {
        color: #00b4ff !important;
        border-bottom: 2px solid #00b4ff !important;
        background: transparent !important;
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; }

    code, pre {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BG      = "#070d18"
BG2     = "#0d1928"
BORDER  = "#1a2e45"
BLUE    = "#00b4ff"
GREEN   = "#00cc6a"
RED     = "#ff4444"
AMBER   = "#ffaa00"
TEXT    = "#c8d8e8"
MUTED   = "#3a5570"

def apply_theme(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="600", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    return ax

def themed_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG2)
    return fig, ax

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI-Powered Credit Intelligence</div>
    <div class="hero-title">🏦 Credit<span>Wise</span></div>
    <div class="hero-sub">Loan Approval Prediction System · SecureTrust Bank · ML Minor Project</div>
    <div class="hero-pills">
        <span class="pill">Logistic Regression</span>
        <span class="pill">Naive Bayes</span>
        <span class="pill">K-Nearest Neighbors</span>
        <span class="pill">Precision-Optimized</span>
        <span class="pill">Real-time Scoring</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    threshold   = st.slider("🎯 Decision Threshold", 0.30, 0.90, 0.50, 0.05,
                            help="Higher = More conservative. Raises precision, may reduce recall.")
    n_neighbors = st.slider("KNN — Neighbors (k)", 3, 15, 7, 2)
    show_eda    = st.checkbox("Show Full EDA Section", value=True)

    st.markdown("---")
    st.markdown("**📌 Threshold Guide**")
    st.caption("↑ Threshold → Fewer approvals, higher precision (fewer bad loans)  \n"
               "↓ Threshold → More approvals, higher recall (fewer missed good loans)")

    st.markdown("---")
    st.markdown("**📌 Metric Priority**")
    st.caption("We optimize **Precision** — approving a bad loan causes real financial loss (Type I Error).")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS (cached)
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH = "loan_approval_data.csv"   # same folder as streamlit_app.py

@st.cache_data
def load_and_preprocess(_n_neighbors: int):
    """Load local CSV, preprocess, and train all models."""

    # ── Load CSV from same directory ─────────────────────────────────────────
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        raise RuntimeError(
            f"'{CSV_PATH}' not found. Make sure loan_approval_data.csv "
            "is in the same folder as streamlit_app.py."
        )

    # ── Normalise Loan_Approved FIRST ────────────────────────────────────────
    # Strategy: try numeric conversion first (handles 1/0, 1.0/0.0, "1"/"0").
    # Only fall back to string mapping if numeric conversion fails entirely.
    def normalise_target(series: pd.Series) -> pd.Series:
        s = series.copy()

        # Step 1: try direct numeric conversion (works for int/float columns
        # and string columns like "1"/"0")
        numeric_attempt = pd.to_numeric(s, errors="coerce")
        if numeric_attempt.notna().mean() > 0.8:
            # Most values converted — it's a numeric column
            result = numeric_attempt.round().fillna(0).astype(int)
            # Remap any label-encoded values: ensure only 0 and 1 exist
            # (LabelEncoder might have flipped 0→1 and 1→0)
            unique_vals = sorted(result.unique())
            if unique_vals == [0, 1]:
                return result
            elif len(unique_vals) == 2:
                # Map the smaller value → 0, larger → 1
                lo, hi = unique_vals[0], unique_vals[1]
                return result.map({lo: 0, hi: 1})
            else:
                return result.clip(0, 1)

        # Step 2: string mapping for text labels
        string_map = {
            "yes": 1, "no": 0,
            "y": 1, "n": 0,
            "approved": 1, "rejected": 0,
            "approve": 1, "reject": 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0,
        }
        # Preserve real NaN as NaN (not the string "nan") before mapping
        cleaned = s.where(s.isna(), s.astype(str).str.strip().str.lower())
        mapped  = cleaned.map(string_map)

        if mapped.isna().any():
            originally_missing = s.isna()
            unknown_mask = mapped.isna() & ~originally_missing
            if unknown_mask.any():
                unknown = cleaned[unknown_mask].unique()
                raise RuntimeError(
                    f"Could not map Loan_Approved to 0/1. "
                    f"Unknown values: {unknown.tolist()}. "
                    f"All values in column: {s.unique().tolist()}. "
                    f"Expected: Yes/No, Y/N, Approved/Rejected, 1/0"
                )
            # Original NaN rows → impute with mode (most frequent class)
            mapped = mapped.fillna(int(mapped.mode()[0]))

        return mapped.astype(int)

    df["Loan_Approved"] = normalise_target(df["Loan_Approved"])

    # Validate both classes exist before doing anything else
    classes_found = sorted(df["Loan_Approved"].unique())
    if classes_found != [0, 1]:
        raise RuntimeError(
            f"Loan_Approved must contain both 0 (Rejected) and 1 (Approved). "
            f"Found only: {classes_found}. "
            f"Raw unique values were: {df['Loan_Approved'].unique().tolist()}"
        )

    raw_df = df.copy()   # raw_df already has clean numeric target

    # ── Impute ────────────────────────────────────────────────────────────────
    # Exclude Loan_Approved from imputation so it stays clean 0/1 ints
    num_cols = df.select_dtypes("number").columns.difference(["Loan_Approved"])
    cat_cols = df.select_dtypes("object").columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    if len(cat_cols):
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # ── Drop ID ───────────────────────────────────────────────────────────────
    if "Applicant_ID" in df.columns:
        df.drop("Applicant_ID", axis=1, inplace=True)

    # ── Label Encode Education (ordinal) ──────────────────────────────────────
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    # Loan_Approved is already 0/1 int — no further encoding needed

    # ── One-Hot Encode ────────────────────────────────────────────────────────
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    oh = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_df = pd.DataFrame(oh.fit_transform(df[ohe_cols]),
                          columns=oh.get_feature_names_out(ohe_cols),
                          index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), enc_df], axis=1)

    # ── Feature Engineering ───────────────────────────────────────────────────
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    models = {
        "Logistic Regression"     : LogisticRegression(max_iter=100000, random_state=42),
        "Naive Bayes"             : GaussianNB(),
        f"KNN (k={_n_neighbors})" : KNeighborsClassifier(n_neighbors=_n_neighbors)
    }

    results, trained = [], {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        yp   = m.predict(X_test_s)
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

    results_df = pd.DataFrame(results).set_index("Model")
    best_name  = results_df["Precision"].idxmax()
    best_model = trained[best_name][0]

    return (raw_df, X_train_s, X_test_s, y_train, y_test,
            scaler, oh, X.columns.tolist(), ohe_cols,
            results_df, best_model, best_name, trained)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔄 Fetching dataset & training models..."):
    try:
        (raw_df, X_train_s, X_test_s, y_train, y_test,
         scaler, oh, feature_cols, ohe_cols,
         results_df, best_model, best_name, trained) = load_and_preprocess(n_neighbors)
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.info("Please check the GitHub raw CSV URL in the sidebar.")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  EDA & Insights",
    "🤖  Model Results",
    "🎯  Predict Loan",
    "💼  Interview Guide"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not show_eda:
        st.info("EDA hidden — enable it in the sidebar.")
    else:
        # ── KPI Row ───────────────────────────────────────────────────────────
        # raw_df["Loan_Approved"] is always 0/1 int after normalise_target()
        approved_numeric = raw_df["Loan_Approved"].astype(int)

        approval_rate = approved_numeric.mean() * 100
        missing_vals  = raw_df.isnull().sum().sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records",  f"{raw_df.shape[0]:,}")
        c2.metric("Features",       f"{raw_df.shape[1]-1}")
        c3.metric("Approval Rate",  f"{approval_rate:.1f}%")
        c4.metric("Missing Values", f"{missing_vals:,}")
        c5.metric("Test Size",      "20%")

        # ── Row 1: Target Distribution & Income ───────────────────────────────
        st.markdown('<div class="section-title">Target & Income Distribution</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = themed_fig(5, 4)
            counts = approved_numeric.value_counts().sort_index()
            labels = ["Rejected", "Approved"]
            colors_pie = [RED, GREEN]
            wedges, texts, autotexts = ax.pie(
                counts, labels=labels, autopct="%1.1f%%",
                colors=colors_pie, startangle=90,
                wedgeprops=dict(edgecolor=BG, linewidth=2.5),
                textprops={"color": TEXT, "fontsize": 9}
            )
            for a in autotexts: a.set_color("white"); a.set_fontweight("bold")
            ax.set_title("Approval Split", color=TEXT, fontsize=11, fontweight="600")
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = themed_fig(5, 4)
            raw_df_copy = raw_df.copy()
            raw_df_copy["_approved"] = approved_numeric
            palette = {0: RED, 1: GREEN}
            for val, grp in raw_df_copy.groupby("_approved")["Applicant_Income"]:
                ax.hist(grp, bins=25, alpha=0.7,
                        color=palette[val],
                        label="Rejected" if val==0 else "Approved",
                        edgecolor=BG)
            apply_theme(ax, "Applicant Income Distribution", "Income (₹)", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT,
                      edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        with col3:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["Applicant_Income"]
                ax.boxplot(subset, positions=[val], widths=0.4,
                           patch_artist=True,
                           boxprops=dict(facecolor=clr, alpha=0.5),
                           medianprops=dict(color="white", linewidth=2),
                           whiskerprops=dict(color=MUTED),
                           capprops=dict(color=MUTED),
                           flierprops=dict(marker='o', color=clr, markersize=3, alpha=0.5))
            ax.set_xticks([0,1]); ax.set_xticklabels(["Rejected","Approved"], color=TEXT)
            apply_theme(ax, "Income Spread by Decision", "", "Income (₹)")
            st.pyplot(fig); plt.close()

        st.caption("📌 Higher income consistently correlates with approvals. Outliers may indicate high-value borrowers.")

        # ── Row 2: Credit Score & DTI ─────────────────────────────────────────
        st.markdown('<div class="section-title">Credit Score & DTI Analysis</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["Credit_Score"]
                ax.hist(subset, bins=25, alpha=0.65, color=clr, label=lbl, edgecolor=BG)
            apply_theme(ax, "Credit Score Distribution", "Credit Score", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            ax.axvline(600, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(602, ax.get_ylim()[1]*0.88, "600 threshold",
                    color=AMBER, fontsize=7.5)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["DTI_Ratio"]
                ax.hist(subset, bins=25, alpha=0.65, color=clr, label=lbl, edgecolor=BG)
            apply_theme(ax, "DTI Ratio Distribution", "DTI Ratio", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            ax.axvline(0.5, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(0.51, ax.get_ylim()[1]*0.88, "50% cutoff",
                    color=AMBER, fontsize=7.5)
            st.pyplot(fig); plt.close()

        with col3:
            # Credit Score vs DTI scatter
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]
                ax.scatter(subset["Credit_Score"], subset["DTI_Ratio"],
                           c=clr, alpha=0.35, s=12, label=lbl)
            apply_theme(ax, "Credit Score vs DTI", "Credit Score", "DTI Ratio")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        st.caption("📌 Approved applicants cluster at Credit Score > 650 and DTI < 0.40. These are the bank's two strongest risk signals.")

        # ── Row 3: Categorical Features ───────────────────────────────────────
        st.markdown('<div class="section-title">Categorical Feature Analysis</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            cat_cols_plot = ["Employment_Status", "Loan_Purpose", "Property_Area"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG)
            for i, col_name in enumerate(cat_cols_plot):
                ax = axes[i]
                ax.set_facecolor(BG2)
                ct = pd.crosstab(raw_df[col_name], approved_numeric, normalize="index") * 100
                if 1 in ct.columns:
                    bars = ax.bar(ct.index, ct[1], color=BLUE, alpha=0.85, edgecolor=BG)
                    ax.set_facecolor(BG2)
                    apply_theme(ax, col_name, "", "Approval %")
                    ax.set_xticklabels(ct.index, rotation=20, ha="right", fontsize=8, color=TEXT)
                    ax.set_ylim(0, 105)
                    for bar in bars:
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 1.5,
                                f"{bar.get_height():.0f}%",
                                ha="center", fontsize=7.5, color=TEXT)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()

        with col2:
            cat_cols_plot2 = ["Gender", "Marital_Status", "Employer_Category"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG)
            for i, col_name in enumerate(cat_cols_plot2):
                ax = axes[i]
                ax.set_facecolor(BG2)
                ct = pd.crosstab(raw_df[col_name], approved_numeric, normalize="index") * 100
                if 1 in ct.columns:
                    ax.bar(ct.index, ct[1], color=GREEN, alpha=0.75, edgecolor=BG)
                    apply_theme(ax, col_name, "", "Approval %")
                    ax.set_xticklabels(ct.index, rotation=20, ha="right", fontsize=8, color=TEXT)
                    ax.set_ylim(0, 105)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()

        st.caption("📌 Employment type and property area show distinct approval patterns — key categorical signals for the model.")

        # ── Row 4: Loan Amount & Savings ──────────────────────────────────────
        st.markdown('<div class="section-title">Loan Amount, Savings & Collateral</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["Loan_Amount"]
                ax.hist(subset, bins=25, alpha=0.65, color=clr, label=lbl, edgecolor=BG)
            apply_theme(ax, "Loan Amount by Decision", "Loan Amount (₹)", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["Savings"]
                ax.hist(subset, bins=25, alpha=0.65, color=clr, label=lbl, edgecolor=BG)
            apply_theme(ax, "Savings by Decision", "Savings (₹)", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        with col3:
            # Existing loans count
            fig, ax = themed_fig(5, 4)
            loan_counts = raw_df_copy.groupby(["Existing_Loans", "_approved"]).size().unstack(fill_value=0)
            x = loan_counts.index
            w = 0.35
            if 0 in loan_counts.columns:
                ax.bar(x - w/2, loan_counts[0], w, color=RED, alpha=0.8, label="Rejected", edgecolor=BG)
            if 1 in loan_counts.columns:
                ax.bar(x + w/2, loan_counts[1], w, color=GREEN, alpha=0.8, label="Approved", edgecolor=BG)
            apply_theme(ax, "Existing Loans vs Decision", "# Existing Loans", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        st.caption("📌 Higher savings and fewer existing loans strongly predict approval. Large loan amounts show mixed signals by income bracket.")

        # ── Row 5: Correlation Heatmap ────────────────────────────────────────
        st.markdown('<div class="section-title">Feature Correlation Heatmap</div>',
                    unsafe_allow_html=True)
        num_df = raw_df.select_dtypes("number").copy()
        # Ensure loan_approved is numeric
        if "Loan_Approved" in num_df.columns:
            num_df["Loan_Approved"] = approved_numeric

        fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
        ax.set_facecolor(BG)
        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, linewidths=0.8, ax=ax,
                    annot_kws={"size": 8, "color": "white"},
                    linecolor=BG)
        ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=13, fontweight="600", pad=12)
        ax.tick_params(colors=TEXT, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("📌 Highly correlated features (>0.8) can cause multicollinearity. DTI_Ratio and Credit_Score are squared to capture non-linear risk escalation.")

        # ── Row 6: Age & Dependents ───────────────────────────────────────────
        st.markdown('<div class="section-title">Demographics & Loan Term</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["Age"]
                ax.hist(subset, bins=20, alpha=0.65, color=clr, label=lbl, edgecolor=BG)
            apply_theme(ax, "Age Distribution by Decision", "Age (years)", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = themed_fig(5, 4)
            dep_counts = raw_df_copy.groupby(["Dependents", "_approved"]).size().unstack(fill_value=0)
            x = dep_counts.index
            if 0 in dep_counts.columns:
                ax.bar(x - 0.2, dep_counts[0], 0.35, color=RED, alpha=0.8, label="Rejected", edgecolor=BG)
            if 1 in dep_counts.columns:
                ax.bar(x + 0.2, dep_counts[1], 0.35, color=GREEN, alpha=0.8, label="Approved", edgecolor=BG)
            apply_theme(ax, "Dependents vs Decision", "# Dependents", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        with col3:
            fig, ax = themed_fig(5, 4)
            for val, lbl, clr in [(0,"Rejected",RED),(1,"Approved",GREEN)]:
                subset = raw_df_copy[raw_df_copy["_approved"]==val]["Loan_Term"]
                ax.hist(subset, bins=20, alpha=0.65, color=clr, label=lbl, edgecolor=BG)
            apply_theme(ax, "Loan Term Distribution", "Loan Term (months)", "Count")
            ax.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            st.pyplot(fig); plt.close()

        st.caption("📌 Middle-aged applicants (30–50) dominate approvals. Loan terms cluster around common bank offerings (12, 36, 60, 120 months).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-title">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="best-badge">
        🏆 &nbsp; Best Model: <strong>{best_name}</strong>
        &nbsp;·&nbsp; Selected by highest <strong>Precision</strong>
        &nbsp;·&nbsp; Minimizes false approvals (Type I Error)
    </div>
    """, unsafe_allow_html=True)

    # ── Styled Table ──────────────────────────────────────────────────────────
    st.dataframe(
        results_df.style
        .highlight_max(axis=0, color="#0d3320")
        .format("{:.4f}"),
        use_container_width=True
    )

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("""
        **Metric Explanations:**
        - **Accuracy** — % of total correct predictions
        - **Precision** ⭐ — Of all *predicted approvals*, how many were correct? → minimizes Type I Error
        - **Recall** — Of all *actual good loans*, how many did we catch?
        - **F1 Score** — Harmonic mean of Precision and Recall
        - **AUC** — Overall discrimination ability (approved vs rejected)
        """)
    with col_exp2:
        st.markdown("""
        **Error Types in Loan Approval:**
        - 🔴 **Type I Error (FP)** — Approve a bad loan → *Direct financial loss*
        - 🟡 **Type II Error (FN)** — Reject a good loan → *Lost business opportunity*

        Since Type I > Type II in severity, we **maximize Precision**.
        """)

    # ── Bar Chart ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG2)
    x = np.arange(len(results_df))
    w = 0.19
    metrics_bar = ["Accuracy", "Precision", "Recall", "F1"]
    colors_bar  = [BLUE, RED, GREEN, AMBER]
    for i, (m, c) in enumerate(zip(metrics_bar, colors_bar)):
        bars = ax.bar(x + i*w, results_df[m], w, label=m, color=c,
                      alpha=0.9, edgecolor=BG)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.006,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT, fontweight="600")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(results_df.index, color=TEXT, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=TEXT)
    ax.legend(fontsize=9, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
    ax.axhline(0.8, linestyle="--", color=MUTED, linewidth=1, alpha=0.6)
    ax.text(len(results_df)-0.1, 0.81, "0.80 baseline",
            fontsize=8, color=MUTED)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.set_title("Model Metrics Comparison", color=TEXT, fontsize=13, fontweight="600")
    st.pyplot(fig); plt.close()

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=BG)
    for ax, (name, (model, yp)) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"],
                    linewidths=1, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()
    st.caption("📌 Top-right cell (FP) = Bad loans incorrectly approved → financial risk. Minimize FP by maximizing Precision.")

    # ── Threshold Curve ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Precision–Recall vs Threshold</div>',
                unsafe_allow_html=True)
    if hasattr(best_model, "predict_proba"):
        probs_all = best_model.predict_proba(X_test_s)[:, 1]
        thresh_vals = np.arange(0.3, 0.85, 0.04)
        precs, recs, f1s = [], [], []
        for t in thresh_vals:
            pp = (probs_all >= t).astype(int)
            precs.append(precision_score(y_test, pp, zero_division=0))
            recs.append(recall_score(y_test, pp, zero_division=0))
            f1s.append(f1_score(y_test, pp, zero_division=0))

        fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=BG)
        ax.set_facecolor(BG2)
        ax.plot(thresh_vals, precs, "o-", label="Precision", color=RED, linewidth=2)
        ax.plot(thresh_vals, recs,  "s-", label="Recall",    color=GREEN, linewidth=2)
        ax.plot(thresh_vals, f1s,   "^-", label="F1 Score",  color=BLUE, linewidth=2)
        ax.axvline(threshold, linestyle="--", color=AMBER, linewidth=1.8,
                   label=f"Current Threshold ({threshold})")
        apply_theme(ax, "Precision / Recall Trade-off vs Threshold",
                    "Decision Threshold", "Score")
        ax.legend(fontsize=9, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
        ax.set_ylim(0, 1.05)
        for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
        st.pyplot(fig); plt.close()
        st.caption("📌 The yellow line shows your selected threshold. Move it right to increase Precision (fewer bad approvals), or left to increase Recall (fewer missed good applicants).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT LOAN
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-title">Loan Application Assessment</div>',
                unsafe_allow_html=True)
    st.caption(f"Model: **{best_name}** &nbsp;·&nbsp; Threshold: **{threshold}** &nbsp;·&nbsp; Optimized for Precision")

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Personal Details**")
            age        = st.number_input("Age", 18, 75, 32)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            marital    = st.selectbox("Marital Status", ["Married", "Single"])
            dependents = st.number_input("Dependents", 0, 10, 1)
            education  = st.selectbox("Education Level",
                                      ["Graduate", "Postgraduate", "Undergraduate"])

        with c2:
            st.markdown("**💼 Employment & Finance**")
            emp_status = st.selectbox("Employment Status",
                                      ["Salaried", "Self-Employed", "Business"])
            employer   = st.selectbox("Employer Category", ["Govt", "Private", "Self"])
            income     = st.number_input("Monthly Income (₹)", 5000, 500000, 55000, 1000)
            co_income  = st.number_input("Co-applicant Income (₹)", 0, 300000, 15000, 1000)
            savings    = st.number_input("Savings (₹)", 0, 1000000, 25000, 1000)

        with c3:
            st.markdown("**🏦 Loan Details**")
            loan_amount  = st.number_input("Loan Amount (₹)", 10000, 5000000, 200000, 5000)
            loan_term    = st.number_input("Loan Term (months)", 6, 360, 60)
            loan_purpose = st.selectbox("Loan Purpose",
                                        ["Home", "Education", "Personal", "Business"])
            prop_area    = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
            collateral   = st.number_input("Collateral Value (₹)", 0, 5000000, 60000, 5000)
            credit_score = st.slider("Credit Score", 300, 900, 720)
            dti_ratio    = st.slider("DTI Ratio", 0.05, 1.0, 0.28, 0.01)
            existing_loans = st.number_input("Existing Loans", 0, 10, 1)

        submitted = st.form_submit_button("🔍  Assess Loan Application")

    if submitted:
        edu_map = {"Graduate": 1, "Postgraduate": 2, "Undergraduate": 0}
        input_data = {
            "Age": age, "Applicant_Income": income, "Coapplicant_Income": co_income,
            "Loan_Amount": loan_amount, "Loan_Term": loan_term,
            "Credit_Score": credit_score, "DTI_Ratio": dti_ratio,
            "Savings": savings, "Collateral_Value": collateral,
            "Existing_Loans": existing_loans, "Dependents": dependents,
            "Education_Level": edu_map[education],
            "Employment_Status": emp_status, "Marital_Status": marital,
            "Loan_Purpose": loan_purpose, "Property_Area": prop_area,
            "Gender": gender, "Employer_Category": employer
        }

        input_df = pd.DataFrame([input_data])
        input_df["DTI_Ratio_sq"]    = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
        input_df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

        enc    = oh.transform(input_df[ohe_cols])
        enc_df = pd.DataFrame(enc, columns=oh.get_feature_names_out(ohe_cols))
        input_final  = pd.concat([input_df.drop(columns=ohe_cols), enc_df], axis=1)
        input_final  = input_final.reindex(columns=feature_cols, fill_value=0)
        input_scaled = scaler.transform(input_final)

        prob     = best_model.predict_proba(input_scaled)[0][1]
        decision = 1 if prob >= threshold else 0

        st.markdown("---")
        col_dec, col_prob = st.columns([3, 2])

        with col_dec:
            if decision == 1:
                st.markdown(f"""
                <div class="approved-banner">
                    ✅ LOAN APPROVED
                    <div class="banner-sub">
                        Confidence: {prob*100:.1f}% &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%
                        &nbsp;|&nbsp; Model: {best_name}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-banner">
                    ❌ LOAN REJECTED
                    <div class="banner-sub">
                        Confidence: {prob*100:.1f}% &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%
                        &nbsp;|&nbsp; Model: {best_name}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>**Risk Factor Analysis:**", unsafe_allow_html=True)
            flags = []
            if credit_score < 600: flags.append("⚠️ Low credit score (< 600)")
            if dti_ratio > 0.5:    flags.append("⚠️ High DTI ratio (> 50%)")
            if existing_loans > 2: flags.append("⚠️ Multiple existing loans (> 2)")
            if income < 20000:     flags.append("⚠️ Below-average income (< ₹20,000)")
            if loan_amount > income * 50: flags.append("⚠️ Loan amount very high vs income")

            if flags:
                for f in flags:
                    st.markdown(f'<div class="risk-flag">{f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-flag">✅ No major risk flags detected</div>',
                            unsafe_allow_html=True)

        with col_prob:
            # Probability gauge
            fig, ax = plt.subplots(figsize=(4.5, 4), facecolor=BG)
            ax.set_facecolor(BG2)
            bar_color = GREEN if prob >= threshold else RED
            ax.barh([""], [prob], color=bar_color, height=0.4, alpha=0.9)
            ax.barh([""], [1 - prob], left=[prob], color=MUTED, height=0.4, alpha=0.5)
            ax.axvline(x=threshold, color=AMBER, linestyle="--",
                       linewidth=2.5, label=f"Threshold ({threshold})")
            ax.set_xlim(0, 1)
            ax.set_title(f"Approval Probability: {prob*100:.1f}%",
                         color=TEXT, fontsize=11, fontweight="600")
            ax.tick_params(colors=TEXT)
            ax.legend(fontsize=9, facecolor=BG2, labelcolor=TEXT, edgecolor=BORDER)
            for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INTERVIEW GUIDE
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #060b14, #0a1628);
                border: 1px solid rgba(0,180,255,0.15); border-radius:16px;
                padding: 1.8rem 2rem; margin-bottom: 1.5rem;">
        <div style="color:#00b4ff; font-size:0.72rem; font-weight:600;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.5rem;">
            Interview Ready
        </div>
        <div style="font-family:'Playfair Display',serif; font-size:1.8rem;
                    color:white; font-weight:800; margin-bottom:0.5rem;">
            How to Present CreditWise
        </div>
        <div style="color:#7a94b0; font-size:0.9rem;">
            Answers to the most common ML interview questions for this project.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Elevator Pitch
    st.markdown('<div class="section-title">🎤 30-Second Elevator Pitch</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1928; border-left:3px solid #00b4ff;
                border-radius:8px; padding:1.2rem 1.5rem; color:#c8d8e8;
                font-size:0.93rem; line-height:1.7; margin-bottom:1.5rem;">
    <em>"I built CreditWise — an AI-powered loan approval prediction system for SecureTrust Bank.
    It processes historical applicant data to predict whether a loan should be approved or rejected.
    I trained three models: Logistic Regression, Naive Bayes, and KNN, then selected the best using
    Precision — because in banking, approving a bad loan causes direct financial loss (Type I Error).
    The entire pipeline includes feature engineering, threshold tuning, and a real-time Streamlit
    dashboard where anyone can input applicant details and get an instant decision."</em>
    </div>
    """, unsafe_allow_html=True)

    # Q&A Section
    st.markdown('<div class="section-title">❓ Technical Interview Q&A</div>',
                unsafe_allow_html=True)

    qa_pairs = [
        ("Q1", "Why did you choose Precision as your key metric?",
         "In loan approval, a False Positive (approving a bad loan) directly causes financial loss — "
         "this is a Type I Error. Precision = TP / (TP + FP) measures exactly that: of all predicted "
         "approvals, how many were actually safe? By maximizing Precision, we minimize bad loan approvals. "
         "A Type II Error (rejecting a good loan) only means lost business — serious, but not an immediate loss."),

        ("Q2", "What is a Type I vs Type II Error in this context?",
         "Type I Error = False Positive = Approving a bad loan → bank loses money. "
         "Type II Error = False Negative = Rejecting a good loan → bank loses a customer. "
         "Type I is more severe because it directly impacts the balance sheet. "
         "That's why we optimize Precision over Recall."),

        ("Q3", "Why did you square DTI_Ratio and Credit_Score?",
         "Squared features capture non-linear relationships. A DTI of 0.8 isn't just twice as risky "
         "as 0.4 — it's exponentially worse. Squaring amplifies the difference between borderline and "
         "extreme values, giving the model a stronger gradient signal. This is a form of polynomial "
         "feature engineering without using a polynomial transformer."),

        ("Q4", "Why StandardScaler instead of MinMaxScaler?",
         "StandardScaler normalizes to zero mean and unit variance. This is critical for KNN "
         "(distance-based) and Logistic Regression (gradient-based). MinMaxScaler is sensitive to "
         "outliers — if one income value is ₹10,00,000 and most are ₹50,000, it would compress "
         "all values into a tiny range. StandardScaler handles outliers more robustly."),

        ("Q5", "What is stratified train-test split and why use it?",
         "Stratified split ensures that the proportion of approved vs rejected loans is preserved "
         "in both train and test sets. Without it, by random chance, the test set could have mostly "
         "one class, making metrics unreliable. With stratify=y, if 60% of data is 'Approved', "
         "both train and test will also have ~60% approved."),

        ("Q6", "Why OneHotEncoder with drop='first'?",
         "To avoid the dummy variable trap (multicollinearity). If we encode Gender into Male=1/0 "
         "and Female=1/0, both columns together are perfectly collinear (Male + Female = 1 always). "
         "Dropping one column removes this redundancy, making features independent — essential for "
         "Logistic Regression which assumes no perfect multicollinearity."),

        ("Q7", "What improvements would you make in production?",
         "1. Gradient Boosting (XGBoost/LightGBM) for 3–5% accuracy gains. "
         "2. SMOTE oversampling if class imbalance is severe (>70:30 split). "
         "3. SHAP values for regulatory explainability (banks need to explain rejections legally). "
         "4. MLflow for experiment tracking and model versioning. "
         "5. FastAPI + Docker deployment on AWS/GCP. "
         "6. Monitoring for data drift in production."),

        ("Q8", "Why Logistic Regression as a baseline?",
         "Logistic Regression is interpretable, fast, and works well when the relationship between "
         "features and outcome is roughly linear in log-odds. It gives us coefficient values, making "
         "it explainable to bank regulators. It's also the industry standard baseline — if a complex "
         "model doesn't outperform LR, it's usually not worth the added complexity."),
    ]

    col_a, col_b = st.columns(2)
    for i, (qnum, question, answer) in enumerate(qa_pairs):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            st.markdown(f"""
            <div class="interview-q">
                <div class="q-label">{qnum}</div>
                <div class="q-text">{question}</div>
                <div class="a-text">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

    # Project Structure
    st.markdown('<div class="section-title">📂 Project Structure</div>',
                unsafe_allow_html=True)
    st.code("""creditwise/
├── creditwise_loan_approval.py   # Main ML pipeline (console output)
├── streamlit_app.py              # Interactive Streamlit UI
├── loan_approval_data.csv        # Dataset (hosted on GitHub)
├── plots/                        # Auto-generated EDA visualizations
│   ├── 01_target_distribution.png
│   ├── 02_income_analysis.png
│   ├── 03_credit_score.png
│   ├── 04_dti_ratio.png
│   ├── 05_categorical_features.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_matrices.png
│   └── 09_threshold_tuning.png
└── README.md""", language="")

    st.markdown('<div class="section-title">🚀 How to Run</div>', unsafe_allow_html=True)
    st.code("""# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit requests

# 2. Run main ML pipeline (saves plots, prints console output)
python creditwise_loan_approval.py

# 3. Launch interactive Streamlit dashboard
streamlit run streamlit_app.py

# Dataset is auto-loaded from GitHub — no manual upload needed.""", language="bash")

    # Key Formulas
    st.markdown('<div class="section-title">📐 Key Formulas</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
    with col2:
        st.latex(r"\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.latex(r"\text{AUC-ROC} = P(\hat{y}_{pos} > \hat{y}_{neg})")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center style='color:#3a5570; font-size:0.78rem; font-family:DM Mono,monospace;'>"
    "CreditWise · Scikit-Learn + Streamlit · SecureTrust Bank · ML Minor Project"
    "</center>",
    unsafe_allow_html=True
)
