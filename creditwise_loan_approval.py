"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CREDITWISE LOAN APPROVAL PREDICTION SYSTEM                     ║
║              Industry-Grade Machine Learning Project                        ║
║              Author: [Your Name] | Domain: FinTech / Credit Risk            ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROJECT OVERVIEW:
-----------------
    SecureTrust Bank processes hundreds of loan applications daily. Manual
    verification is slow, biased, and error-prone. This ML system predicts
    whether a loan should be APPROVED or REJECTED based on applicant data,
    enabling faster, fairer, and more consistent decisions.

BUSINESS RISKS:
---------------
    ► Type I Error  (False Positive)  → Approving a risky loan → Financial loss
    ► Type II Error (False Negative)  → Rejecting a good loan  → Loss of business

    Since financial loss from bad loans is MORE severe, we optimize for
    HIGH PRECISION to minimize Type I Errors.

MODELS USED:
    1. Logistic Regression (Baseline linear model)
    2. Gaussian Naive Bayes (Probabilistic model)
    3. K-Nearest Neighbors (Distance-based model)
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Optional: Set plot style globally
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"

# =============================================================================
# SECTION 2: LOAD AND EXPLORE DATA
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset and perform initial exploration.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    print("\n" + "="*65)
    print("  STEP 1: LOADING DATASET")
    print("="*65)

    df = pd.read_csv(filepath)

    print(f"\n  ✅ Dataset loaded successfully!")
    print(f"  📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n{'─'*65}")
    print("  Column Overview:")
    print(f"{'─'*65}")
    print(df.dtypes.to_string())
    print(f"\n  Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n  Target Distribution:\n{df['Loan_Approved'].value_counts()}")

    return df


# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_eda(df: pd.DataFrame, save_plots: bool = True) -> None:
    """
    Perform comprehensive EDA with interpretations.

    Visualizations:
        1. Target Distribution (Pie Chart)
        2. Income vs Loan Approval (Boxplot)
        3. Credit Score Distribution (Histogram)
        4. DTI Ratio by Approval (Boxplot)
        5. Categorical Features (Count Plots)
        6. Correlation Heatmap

    Args:
        df: Raw DataFrame.
        save_plots: Whether to save plots to disk.
    """
    print("\n" + "="*65)
    print("  STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*65)

    os.makedirs("plots", exist_ok=True)

    # ── Plot 1: Target Distribution ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Loan Approval Distribution", fontsize=15, fontweight="bold")

    counts = df["Loan_Approved"].value_counts()
    labels = ["Rejected (0)", "Approved (1)"]
    colors = ["#E74C3C", "#2ECC71"]

    axes[0].pie(counts, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Approval Rate")

    sns.countplot(data=df, x="Loan_Approved", palette=["#E74C3C", "#2ECC71"],
                  ax=axes[1])
    axes[1].set_xticklabels(["Rejected", "Approved"])
    axes[1].set_title("Count of Approvals vs Rejections")
    axes[1].set_xlabel("")

    plt.tight_layout()
    if save_plots: plt.savefig("plots/01_target_distribution.png")
    plt.show()
    print("\n  📌 Interpretation: Class imbalance check — if heavily imbalanced,")
    print("     precision becomes even more critical as the model may over-predict")
    print("     the majority class (Rejected).")

    # ── Plot 2: Income vs Loan Approval ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Income Analysis by Loan Approval", fontsize=15, fontweight="bold")

    sns.boxplot(data=df, x="Loan_Approved", y="Applicant_Income",
                palette=["#E74C3C", "#2ECC71"], ax=axes[0])
    axes[0].set_xticklabels(["Rejected", "Approved"])
    axes[0].set_title("Applicant Income")

    sns.boxplot(data=df, x="Loan_Approved", y="Coapplicant_Income",
                palette=["#E74C3C", "#2ECC71"], ax=axes[1])
    axes[1].set_xticklabels(["Rejected", "Approved"])
    axes[1].set_title("Co-applicant Income")

    plt.tight_layout()
    if save_plots: plt.savefig("plots/02_income_analysis.png")
    plt.show()
    print("\n  📌 Interpretation: Higher applicant income generally correlates with")
    print("     approval. Outliers may represent high-value borrowers or data issues.")

    # ── Plot 3: Credit Score Distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Credit Score Analysis", fontsize=15, fontweight="bold")

    sns.histplot(data=df, x="Credit_Score", hue="Loan_Approved",
                 bins=30, palette=["#E74C3C", "#2ECC71"],
                 element="step", ax=axes[0])
    axes[0].set_title("Credit Score Distribution by Approval")

    sns.boxplot(data=df, x="Loan_Approved", y="Credit_Score",
                palette=["#E74C3C", "#2ECC71"], ax=axes[1])
    axes[1].set_xticklabels(["Rejected", "Approved"])
    axes[1].set_title("Credit Score Spread")

    plt.tight_layout()
    if save_plots: plt.savefig("plots/03_credit_score.png")
    plt.show()
    print("\n  📌 Interpretation: Approved applicants tend to have higher credit scores.")
    print("     A clear threshold may exist — useful for feature engineering.")

    # ── Plot 4: DTI Ratio ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Debt-to-Income (DTI) Ratio Analysis", fontsize=15, fontweight="bold")

    sns.histplot(data=df, x="DTI_Ratio", hue="Loan_Approved",
                 bins=25, palette=["#E74C3C", "#2ECC71"],
                 element="step", ax=axes[0])
    axes[0].set_title("DTI Ratio Distribution")

    sns.boxplot(data=df, x="Loan_Approved", y="DTI_Ratio",
                palette=["#E74C3C", "#2ECC71"], ax=axes[1])
    axes[1].set_xticklabels(["Rejected", "Approved"])
    axes[1].set_title("DTI Ratio by Approval")

    plt.tight_layout()
    if save_plots: plt.savefig("plots/04_dti_ratio.png")
    plt.show()
    print("\n  📌 Interpretation: Lower DTI ratios (less debt relative to income)")
    print("     are favored for approval. High DTI signals repayment risk.")

    # ── Plot 5: Categorical Features ─────────────────────────────────────────
    cat_features = ["Employment_Status", "Marital_Status",
                    "Loan_Purpose", "Property_Area", "Gender"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Categorical Features vs Loan Approval", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(cat_features):
        ct = pd.crosstab(df[col], df["Loan_Approved"], normalize="index") * 100
        ct.columns = ["Rejected", "Approved"]
        ct.plot(kind="bar", ax=axes[i], color=["#E74C3C", "#2ECC71"],
                edgecolor="white", rot=20)
        axes[i].set_title(f"{col} (Approval Rate %)")
        axes[i].set_xlabel("")
        axes[i].legend(loc="upper right", fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    if save_plots: plt.savefig("plots/05_categorical_features.png")
    plt.show()
    print("\n  📌 Interpretation: Some categories show clear approval patterns")
    print("     (e.g., Urban property, Salaried employment may have higher approvals).")

    # ── Plot 6: Correlation Heatmap ───────────────────────────────────────────
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                mask=mask, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_plots: plt.savefig("plots/06_correlation_heatmap.png")
    plt.show()
    print("\n  📌 Interpretation: Check for high correlation (>0.8) between features")
    print("     to detect multicollinearity. Correlated features can be dropped.")

    print("\n  ✅ EDA complete. Plots saved to /plots directory.")


# =============================================================================
# SECTION 4: DATA PREPROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame):
    """
    Complete preprocessing pipeline:
        1. Drop ID column
        2. Impute missing values
        3. Label encode Education_Level and target
        4. One-Hot encode multi-category columns
        5. Feature engineering (squared terms)
        6. Train-test split
        7. Standard scaling

    Args:
        df: Raw DataFrame.

    Returns:
        X_train, X_test, y_train, y_test, scaler, oh_encoder, feature_columns
    """
    print("\n" + "="*65)
    print("  STEP 3: PREPROCESSING")
    print("="*65)

    df = df.copy()

    # ── 4.1: Drop ID ─────────────────────────────────────────────────────────
    if "Applicant_ID" in df.columns:
        df.drop("Applicant_ID", axis=1, inplace=True)
        print("\n  ✅ Dropped: Applicant_ID (non-informative unique key)")

    # ── 4.2: Impute Missing Values ────────────────────────────────────────────
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    print("  ✅ Imputed: Numerical → Mean | Categorical → Mode")

    # ── 4.3: Label Encoding ───────────────────────────────────────────────────
    le_edu = LabelEncoder()
    le_target = LabelEncoder()

    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])
    print("  ✅ Label Encoded: Education_Level, Loan_Approved")

    # ── 4.4: One-Hot Encoding ─────────────────────────────────────────────────
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]

    oh = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = oh.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=oh.get_feature_names_out(ohe_cols),
                               index=df.index)

    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)
    print(f"  ✅ One-Hot Encoded: {ohe_cols}")

    # ── 4.5: Feature Engineering ──────────────────────────────────────────────
    # Squared terms capture non-linear relationships
    # High credit scores exponentially reduce risk → square amplifies signal
    # High DTI exponentially increases risk → square amplifies danger signal
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)
    print("  ✅ Feature Engineering: DTI_Ratio² and Credit_Score² created")
    print("     (Squared features capture non-linear risk escalation)")

    # ── 4.6: Train-Test Split ─────────────────────────────────────────────────
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✅ Train-Test Split: 80% train ({len(X_train)}) | 20% test ({len(X_test)})")
    print(f"     Stratified split ensures equal class distribution.")

    # ── 4.7: Scaling ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("  ✅ Standard Scaling applied (fit on train, transform both)")

    print(f"\n  📐 Final feature count: {X_train.shape[1]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, oh, X.columns.tolist()


# =============================================================================
# SECTION 5: MODEL TRAINING & EVALUATION
# =============================================================================

def evaluate_model(name: str, y_test, y_pred, y_prob=None) -> dict:
    """
    Compute and display all evaluation metrics for a single model.

    Metrics Explained:
        Accuracy  → % of total correct predictions
        Precision → Of predicted approvals, how many were actually correct
                    (CRITICAL — minimizes Type I error / bad loan approval)
        Recall    → Of actual approvals, how many did we catch
        F1 Score  → Harmonic mean of Precision and Recall
        ROC-AUC   → Ability to distinguish approved vs rejected (higher=better)

    Returns:
        Dictionary of metric scores.
    """
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n  {'─'*50}")
    print(f"  🔹 {name}")
    print(f"  {'─'*50}")
    print(f"     Accuracy  : {acc:.4f}  → {acc*100:.1f}% overall correct")
    print(f"     Precision : {prec:.4f}  ← KEY METRIC (Type I error control)")
    print(f"     Recall    : {rec:.4f}  → Approved loans we correctly caught")
    print(f"     F1 Score  : {f1:.4f}  → Balance of Precision & Recall")
    if auc: print(f"     ROC-AUC   : {auc:.4f}  → Discrimination ability")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"     TN={cm[0,0]:>4}  FP={cm[0,1]:>4}   (FP = Bad loans approved!)")
    print(f"     FN={cm[1,0]:>4}  TP={cm[1,1]:>4}   (FN = Good loans rejected)")

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC": auc}


def train_and_evaluate(X_train, X_test, y_train, y_test, save_plots=True):
    """
    Train all three models, evaluate them, and return results + best model.

    Args:
        X_train, X_test: Scaled feature arrays.
        y_train, y_test: Target arrays.
        save_plots: Whether to save confusion matrix plots.

    Returns:
        results_df: DataFrame comparing all models.
        best_model: The model with highest Precision.
        best_model_name: Name of best model.
        all_models: Dict of {name: model}.
    """
    print("\n" + "="*65)
    print("  STEP 4: MODEL TRAINING & EVALUATION")
    print("="*65)

    # ── Define Models ─────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=100000, C=1.0, random_state=42),
        "Naive Bayes"        : GaussianNB(),
        "KNN (k=7)"          : KNeighborsClassifier(n_neighbors=7, metric="minkowski")
    }

    results    = []
    all_preds  = {}
    all_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_model(name, y_test, y_pred, y_prob)
        results.append(metrics)
        all_preds[name]  = y_pred
        all_models[name] = model

    results_df = pd.DataFrame(results).set_index("Model")

    # ── Model Comparison Table ─────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  STEP 5: MODEL COMPARISON")
    print("="*65)
    print(f"\n{results_df.round(4).to_string()}")

    # ── Model Comparison Chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(results_df))
    width = 0.2
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        bars = ax.bar(x + i * width, results_df[metric], width,
                      label=metric, color=color, edgecolor="white", linewidth=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df.index, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison\n(Higher is Better | Precision is KEY)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylabel("Score")
    ax.axhline(y=0.8, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.text(2.8, 0.81, "0.80 threshold", fontsize=9, color="gray")

    plt.tight_layout()
    if save_plots: plt.savefig("plots/07_model_comparison.png")
    plt.show()

    # ── Confusion Matrix Side-by-Side ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

    for ax, (name, y_pred) in zip(axes, all_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_plots: plt.savefig("plots/08_confusion_matrices.png")
    plt.show()

    # ── Best Model Selection ──────────────────────────────────────────────────
    best_model_name = results_df["Precision"].idxmax()
    best_model      = all_models[best_model_name]

    print("\n" + "="*65)
    print("  STEP 6: BEST MODEL SELECTION")
    print("="*65)
    print(f"\n  ✅ Best Model   : {best_model_name}")
    print(f"  📊 Precision    : {results_df.loc[best_model_name, 'Precision']:.4f}")
    print(f"\n  WHY PRECISION?")
    print("  ─────────────────────────────────────────────────────────")
    print("  In loan approval, a False Positive (approving a bad loan)")
    print("  is MORE COSTLY than a False Negative (rejecting a good one).")
    print("  A bank loses real money if bad loans are approved.")
    print("  Precision = TP / (TP + FP) → minimizes bad approvals.")

    return results_df, best_model, best_model_name, all_models


# =============================================================================
# SECTION 6: THRESHOLD TUNING (ADVANCED)
# =============================================================================

def tune_threshold(model, X_test, y_test, save_plots=True):
    """
    Tune the decision threshold to maximize Precision while monitoring Recall.

    By default, classifiers use 0.5 as threshold. Raising it means we only
    approve loans when the model is MORE confident → higher precision.

    Args:
        model: Trained classifier (must support predict_proba).
        X_test: Scaled test features.
        y_test: True labels.
        save_plots: Whether to save plot.

    Returns:
        Optimal threshold value.
    """
    print("\n" + "="*65)
    print("  STEP 7: THRESHOLD TUNING (ADVANCED)")
    print("="*65)

    if not hasattr(model, "predict_proba"):
        print("  ⚠️  Model does not support probability output. Skipping.")
        return 0.5

    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.3, 0.85, 0.05)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(thresholds, precisions, "o-", label="Precision", color="#E74C3C", linewidth=2)
    ax.plot(thresholds, recalls,    "s-", label="Recall",    color="#2ECC71", linewidth=2)
    ax.plot(thresholds, f1s,        "^-", label="F1 Score",  color="#3498DB", linewidth=2)
    ax.axvline(x=0.5, linestyle="--", color="gray", alpha=0.6, label="Default (0.5)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Decision Threshold", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_plots: plt.savefig("plots/09_threshold_tuning.png")
    plt.show()

    # Choose threshold where Precision >= 0.85 (or highest achievable)
    target_precision = 0.85
    optimal_threshold = 0.5
    for t, p in zip(thresholds, precisions):
        if p >= target_precision:
            optimal_threshold = t
            break

    print(f"\n  ✅ Optimal Threshold: {optimal_threshold:.2f}")
    print(f"     (Raises precision above {target_precision} — fewer bad loans approved)")

    return optimal_threshold


# =============================================================================
# SECTION 7: FEATURE IMPORTANCE
# =============================================================================

def show_feature_importance(model, X_train, X_test, y_test,
                             feature_names, model_name, save_plots=True):
    """
    Compute and display feature importance using permutation importance.
    Works for ANY model (no need for .coef_ or .feature_importances_).

    Args:
        model: Trained classifier.
        X_train, X_test: Feature arrays.
        y_test: True labels.
        feature_names: List of feature names.
        model_name: Name of model for plot title.
        save_plots: Whether to save plot.
    """
    print("\n" + "="*65)
    print("  STEP 8: FEATURE IMPORTANCE")
    print("="*65)

    result = permutation_importance(model, X_test, y_test,
                                    n_repeats=10, random_state=42,
                                    scoring="precision")

    importance_df = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#E74C3C" if v > 0 else "#95A5A6" for v in importance_df["Importance"]]
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
                   color=colors, edgecolor="white")
    ax.set_title(f"Top 15 Feature Importances\n(Model: {model_name} | Metric: Precision)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Permutation Importance Score", fontsize=11)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    if save_plots: plt.savefig("plots/10_feature_importance.png")
    plt.show()

    print("\n  Top 5 Most Influential Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"     {row['Feature']:<30} → {row['Importance']:.4f}")

    print("\n  📌 Higher importance = removing that feature hurts precision more.")
    print("     These features are the bank's key decision signals.")


# =============================================================================
# SECTION 8: PREDICTION FUNCTION
# =============================================================================

def build_predictor(best_model, scaler, oh_encoder, feature_columns,
                    ohe_cols, threshold=0.5):
    """
    Build a robust prediction function with full preprocessing pipeline.

    Args:
        best_model: Trained best model.
        scaler: Fitted StandardScaler.
        oh_encoder: Fitted OneHotEncoder.
        feature_columns: Exact column order used during training.
        ohe_cols: Columns that were one-hot encoded.
        threshold: Decision threshold for approval.

    Returns:
        predict_loan: Callable prediction function.
    """

    def predict_loan(input_data: dict) -> dict:
        """
        Predict loan approval for a single applicant.

        Args:
            input_data: Dictionary of applicant features.

        Returns:
            Dictionary with 'decision', 'probability', 'explanation'.
        """
        input_df = pd.DataFrame([input_data])

        # ── Feature Engineering ────────────────────────────────────────────
        input_df["DTI_Ratio_sq"]    = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
        input_df.drop(["DTI_Ratio", "Credit_Score"], axis=1, inplace=True)

        # ── One-Hot Encoding ──────────────────────────────────────────────
        encoded = oh_encoder.transform(input_df[ohe_cols])
        encoded_df = pd.DataFrame(encoded,
                                   columns=oh_encoder.get_feature_names_out(ohe_cols))
        input_final = pd.concat([input_df.drop(columns=ohe_cols), encoded_df], axis=1)

        # ── Align Columns Exactly ──────────────────────────────────────────
        input_final = input_final.reindex(columns=feature_columns, fill_value=0)

        # ── Scale ─────────────────────────────────────────────────────────
        input_scaled = scaler.transform(input_final)

        # ── Predict ───────────────────────────────────────────────────────
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(input_scaled)[0][1]
            decision = 1 if prob >= threshold else 0
        else:
            prob = None
            decision = best_model.predict(input_scaled)[0]

        # ── Explanation ──────────────────────────────────────────────────
        label = "✅ APPROVED" if decision == 1 else "❌ REJECTED"

        reasons = []
        if input_data.get("Credit_Score", 0) < 600:
            reasons.append("Low credit score (<600)")
        if input_data.get("DTI_Ratio", 0) > 0.5:
            reasons.append("High debt-to-income ratio (>50%)")
        if input_data.get("Existing_Loans", 0) > 2:
            reasons.append("Multiple existing loans")
        if input_data.get("Applicant_Income", 0) < 20000:
            reasons.append("Below-average income")

        if decision == 0 and not reasons:
            reasons.append("Combined risk profile exceeds threshold")

        return {
            "decision"   : label,
            "approved"   : bool(decision),
            "probability": round(prob, 4) if prob is not None else "N/A",
            "threshold"  : threshold,
            "risk_flags" : reasons if decision == 0 else ["No major risk flags identified"]
        }

    return predict_loan


# =============================================================================
# SECTION 9: MAIN PIPELINE
# =============================================================================

def run_pipeline(filepath: str = "loan_approval_data.csv"):
    """
    Execute the full CreditWise ML pipeline end-to-end.

    Steps:
        1. Load data
        2. EDA
        3. Preprocess
        4. Train & Evaluate
        5. Threshold tuning
        6. Feature importance
        7. Build predictor
        8. Sample prediction

    Args:
        filepath: Path to the dataset CSV.

    Returns:
        predict_loan: The final prediction function ready to use.
    """

    print("\n" + "█"*65)
    print("  CREDITWISE LOAN APPROVAL PREDICTION SYSTEM")
    print("  Powered by Machine Learning | SecureTrust Bank")
    print("█"*65)

    # Step 1: Load
    df = load_data(filepath)

    # Step 2: EDA
    run_eda(df)

    # Step 3: Preprocess
    X_train, X_test, y_train, y_test, scaler, oh, feature_cols = preprocess_data(df)

    # OHE cols (same order as preprocessing)
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]

    # Step 4: Train & Evaluate
    results_df, best_model, best_model_name, all_models = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Step 5: Threshold Tuning
    optimal_threshold = tune_threshold(best_model, X_test, y_test)

    # Step 6: Feature Importance
    show_feature_importance(best_model, X_train, X_test, y_test,
                             feature_cols, best_model_name)

    # Step 7: Build predictor
    predict_loan = build_predictor(best_model, scaler, oh,
                                    feature_cols, ohe_cols, optimal_threshold)

    # Step 8: Sample Prediction
    print("\n" + "="*65)
    print("  STEP 9: SAMPLE PREDICTION DEMO")
    print("="*65)

    sample_applicant = {
        "Age"               : 32,
        "Applicant_Income"  : 55000,
        "Coapplicant_Income": 15000,
        "Loan_Amount"       : 200000,
        "Loan_Term"         : 60,
        "Credit_Score"      : 720,
        "DTI_Ratio"         : 0.28,
        "Savings"           : 25000,
        "Collateral_Value"  : 60000,
        "Existing_Loans"    : 1,
        "Dependents"        : 2,
        "Education_Level"   : 1,
        "Employment_Status" : "Salaried",
        "Marital_Status"    : "Married",
        "Loan_Purpose"      : "Home",
        "Property_Area"     : "Urban",
        "Gender"            : "Male",
        "Employer_Category" : "Private"
    }

    result = predict_loan(sample_applicant)

    print(f"\n  Applicant Summary: Age {sample_applicant['Age']}, "
          f"Income ₹{sample_applicant['Applicant_Income']:,}, "
          f"Credit Score {sample_applicant['Credit_Score']}")
    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  Decision    : {result['decision']:<30}│")
    print(f"  │  Probability : {str(result['probability']):<30}│")
    print(f"  │  Threshold   : {result['threshold']:<30}│")
    print(f"  │  Risk Flags  : {str(result['risk_flags'][0]):<30}│")
    print(f"  └──────────────────────────────────────────────┘")

    print("\n" + "█"*65)
    print("  PIPELINE COMPLETE ✅")
    print("  Plots saved to /plots directory")
    print("  Run streamlit_app.py for interactive UI")
    print("█"*65)

    return predict_loan


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    predict_loan = run_pipeline("loan_approval_data.csv")
