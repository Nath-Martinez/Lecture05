import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLE HELPERS
# ─────────────────────────────────────────────
BG      = "#0f1117"
BG2     = "#1a1d27"
ACCENT  = "#7ee8a2"
RED     = "#ff6b9d"
BLUE    = "#4fc3f7"
MUTED   = "#8890b5"
COLORS  = [ACCENT, RED, BLUE]
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
PALETTE = [ACCENT, RED, BLUE, "#ffb347", "#b39ddb", "#80deea", "#ef9a9a"]

MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree":       DecisionTreeClassifier,
    "Random Forest":       RandomForestClassifier,
    "Gradient Boosting":   GradientBoostingClassifier,
    "SVM":                 SVC,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Naive Bayes":         GaussianNB,
}
MODEL_PARAMS = {
    "Logistic Regression": {"max_iter": 1000, "random_state": 42},
    "Decision Tree":       {"random_state": 42},
    "Random Forest":       {"random_state": 42},
    "Gradient Boosting":   {"random_state": 42},
    "SVM":                 {"probability": True, "random_state": 42},
    "K-Nearest Neighbors": {},
    "Naive Bayes":         {},
}

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3154")
    return fig, ax

def dark_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax in (axes.flat if hasattr(axes, "flat") else axes):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3154")
    return fig, axes

# ─────────────────────────────────────────────
# DATA & TRAINING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y

@st.cache_data
def train_all(selected_models, test_size, use_scaling, cv_folds):
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    if use_scaling:
        sc      = StandardScaler()
        X_tr_s  = pd.DataFrame(sc.fit_transform(X_tr), columns=X.columns)
        X_te_s  = pd.DataFrame(sc.transform(X_te),     columns=X.columns)
        X_all_s = pd.DataFrame(sc.transform(X),        columns=X.columns)
    else:
        X_tr_s  = X_tr.reset_index(drop=True)
        X_te_s  = X_te.reset_index(drop=True)
        X_all_s = X.reset_index(drop=True)

    results = {}
    for name in selected_models:
        clf    = MODELS[name](**MODEL_PARAMS[name])
        clf.fit(X_tr_s, y_tr.values)
        y_pred = clf.predict(X_te_s)
        cv     = cross_val_score(
            MODELS[name](**MODEL_PARAMS[name]),
            X_tr_s, y_tr.values,
            cv=StratifiedKFold(cv_folds), scoring="accuracy",
        )
        results[name] = {
            "model":     clf,
            "y_pred":    y_pred,
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_te, y_pred,    average="weighted", zero_division=0),
            "f1":        f1_score(y_te, y_pred,        average="weighted", zero_division=0),
            "cv_mean":   cv.mean(),
            "cv_std":    cv.std(),
            "cm":        confusion_matrix(y_te, y_pred),
        }
    return results, X_tr_s, X_te_s, X_all_s, y_tr.values, y_te.values

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Iris Classifier Studio")
    st.markdown("---")

    st.markdown("**⚙️ Models**")
    selected_models = st.multiselect(
        "Choose models to train", list(MODELS.keys()),
        default=["Logistic Regression", "Random Forest", "SVM"],
    )

    st.markdown("---")
    st.markdown("**🔬 Training**")
    test_size   = st.slider("Test size (%)", 10, 40, 20, 5) / 100
    use_scaling = st.checkbox("Feature Scaling (StandardScaler)", value=True)
    cv_folds    = st.slider("Cross-validation folds", 3, 10, 5)

    st.markdown("---")
    st.markdown("**📊 Visualization**")
    feat_options = [
        "sepal length (cm) vs sepal width (cm)",
        "petal length (cm) vs petal width (cm)",
        "sepal length (cm) vs petal length (cm)",
    ]
    boundary_features = st.selectbox("Decision boundary features", feat_options, index=1)
    show_pca = st.checkbox("Show PCA projection",     value=True)
    show_roc = st.checkbox("Show ROC curves",         value=True)
    show_cm  = st.checkbox("Show confusion matrices", value=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("🌸 Iris Classifier Studio")
st.caption("Interactive multi-model classification — decision boundaries · ROC curves · confusion matrices.")
st.markdown("---")

if not selected_models:
    st.warning("👈 Select at least one model from the sidebar.")
    st.stop()

X, y = load_data()
feat1, feat2 = [f.strip() for f in boundary_features.split(" vs ")]
results, X_tr_s, X_te_s, X_all_s, y_tr, y_te = train_all(
    tuple(selected_models), test_size, use_scaling, cv_folds
)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance",
    "🗺️ Decision Boundaries",
    "📉 ROC Curves",
    "🔢 Confusion Matrices",
    "🔍 Data Explorer",
])

# ════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════
with tab1:
    # ── Metric cards ──
    cols = st.columns(len(selected_models))
    for i, name in enumerate(selected_models):
        r = results[name]
        with cols[i]:
            st.metric(label=f"🎯 {name}", value=f"{r['accuracy']:.1%}", delta=f"F1: {r['f1']:.3f}")
            st.caption(f"Precision: {r['precision']:.3f} | Recall: {r['recall']:.3f} | CV: {r['cv_mean']:.3f} ±{r['cv_std']:.3f}")

    st.markdown("---")

    # ── Grouped bar chart ──
    st.subheader("Comparative Metrics")
    metrics       = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    n  = len(selected_models)
    x  = np.arange(len(metric_labels))
    w  = 0.8 / n

    fig, ax = dark_fig(10, 5)
    for i, name in enumerate(selected_models):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * w - (n - 1) * w / 2, vals, w * 0.9,
                      label=name, color=PALETTE[i % len(PALETTE)], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=MUTED)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.legend(facecolor=BG, edgecolor="#2d3154", labelcolor=MUTED, fontsize=8)
    ax.grid(axis="y", color="#1e2135", linewidth=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── CV bar with error bars ──
    st.subheader("Cross-Validation Scores")
    fig2, ax2 = dark_fig(8, 4)
    for i, name in enumerate(selected_models):
        ax2.bar(name, results[name]["cv_mean"],
                yerr=results[name]["cv_std"],
                color=PALETTE[i % len(PALETTE)], alpha=0.88,
                error_kw=dict(ecolor="#ffffff55", capsize=4))
        ax2.text(i, results[name]["cv_mean"] + results[name]["cv_std"] + 0.01,
                 f"{results[name]['cv_mean']:.3f}", ha="center", fontsize=8, color=MUTED)

    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("CV Accuracy", color=MUTED)
    ax2.tick_params(axis="x", colors=MUTED, rotation=15)
    ax2.grid(axis="y", color="#1e2135", linewidth=0.7)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ════════════════════════════════════════
# TAB 2 — DECISION BOUNDARIES
# ════════════════════════════════════════
with tab2:
    st.info(f"Boundary computed on: **{feat1}** vs **{feat2}**  |  ★ = test points")

    cmap_bg = ListedColormap(["#1a3a2a", "#3a1a2a", "#1a2a3a"])

    X_2d_tr = X_tr_s[[feat1, feat2]].reset_index(drop=True)
    X_2d_te = X_te_s[[feat1, feat2]].reset_index(drop=True)

    n_models = len(selected_models)
    n_cols_b = min(2, n_models)
    n_rows_b = (n_models + 1) // 2

    fig_b, axes_b = plt.subplots(n_rows_b, n_cols_b,
                                 figsize=(7 * n_cols_b, 5 * n_rows_b))
    fig_b.patch.set_facecolor(BG)

    # Flatten axes safely
    if n_models == 1:
        axes_b = [axes_b]
    elif n_rows_b == 1:
        axes_b = list(axes_b)
    else:
        axes_b = [ax for row in axes_b for ax in row]

    # Hide unused axes
    for j in range(n_models, len(axes_b)):
        axes_b[j].set_visible(False)

    for i, name in enumerate(selected_models):
        ax = axes_b[i]
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3154")
        ax.tick_params(colors=MUTED, labelsize=7)

        clf_2d = MODELS[name](**MODEL_PARAMS[name])
        clf_2d.fit(X_2d_tr, y_tr)

        x_min, x_max = X_2d_tr[feat1].min() - .5, X_2d_tr[feat1].max() + .5
        y_min, y_max = X_2d_tr[feat2].min() - .5, X_2d_tr[feat2].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = pd.DataFrame({feat1: xx.ravel(), feat2: yy.ravel()})
        Z = clf_2d.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, Z, colors=["#7ee8a240", "#ff6b9d40", "#4fc3f740"],
                   linewidths=1.2, levels=[-0.5, 0.5, 1.5, 2.5])

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            m_tr = y_tr == cls_id
            m_te = y_te == cls_id
            ax.scatter(X_2d_tr[feat1][m_tr], X_2d_tr[feat2][m_tr],
                       c=COLORS[cls_id], s=40, alpha=0.8, edgecolors="#ffffff30", linewidths=0.4)
            ax.scatter(X_2d_te[feat1][m_te], X_2d_te[feat2][m_te],
                       c=COLORS[cls_id], s=120, marker="*", alpha=1.0,
                       edgecolors="#ffffff70", linewidths=0.6)

        patches = [mpatches.Patch(color=COLORS[j], label=CLASS_NAMES[j]) for j in range(3)]
        ax.legend(handles=patches, facecolor=BG, edgecolor="#2d3154",
                  labelcolor=MUTED, fontsize=7, loc="upper right")
        ax.set_title(name, color="#c5cae9", fontsize=10, pad=8)
        ax.set_xlabel(feat1, color=MUTED, fontsize=8)
        ax.set_ylabel(feat2, color=MUTED, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_b)
    plt.close(fig_b)

    # ── PCA projection ──
    if show_pca:
        st.markdown("---")
        st.subheader("PCA 2D Projection (All Features)")
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all_s)

        fig_p, ax_p = dark_fig(9, 5)
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = y.values == cls_id
            ax_p.scatter(X_pca[mask, 0], X_pca[mask, 1],
                         c=COLORS[cls_id], s=50, alpha=0.8,
                         edgecolors="#ffffff30", linewidths=0.4, label=cls_name)
        ax_p.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", color=MUTED)
        ax_p.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", color=MUTED)
        ax_p.legend(facecolor=BG, edgecolor="#2d3154", labelcolor=MUTED)
        ax_p.grid(color="#1e2135", linewidth=0.6)
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

# ════════════════════════════════════════
# TAB 3 — ROC CURVES
# ════════════════════════════════════════
with tab3:
    if show_roc:
        y_bin = label_binarize(y_te, classes=[0, 1, 2])

        fig_r, axes_r = plt.subplots(1, 3, figsize=(15, 5))
        fig_r.patch.set_facecolor(BG)

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            ax = axes_r[cls_id]
            ax.set_facecolor(BG2)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3154")
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.plot([0, 1], [0, 1], "--", color="#3a3f5c", linewidth=1)

            for i, name in enumerate(selected_models):
                mdl = results[name]["model"]
                if hasattr(mdl, "predict_proba"):
                    y_score = mdl.predict_proba(X_te_s)[:, cls_id]
                elif hasattr(mdl, "decision_function"):
                    df_val  = mdl.decision_function(X_te_s)
                    y_score = df_val[:, cls_id] if df_val.ndim > 1 else df_val
                else:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, cls_id], y_score)
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                        linewidth=2, label=f"{name} ({roc_auc:.3f})")

            ax.set_title(f"ROC — {cls_name}", color="#c5cae9", fontsize=10)
            ax.set_xlabel("False Positive Rate", color=MUTED, fontsize=8)
            ax.set_ylabel("True Positive Rate", color=MUTED, fontsize=8)
            ax.legend(facecolor=BG, edgecolor="#2d3154", labelcolor=MUTED, fontsize=7)
            ax.grid(color="#1e2135", linewidth=0.6)

        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)
    else:
        st.info("Enable ROC Curves in the sidebar ↩")

# ════════════════════════════════════════
# TAB 4 — CONFUSION MATRICES
# ════════════════════════════════════════
with tab4:
    if show_cm:
        n  = len(selected_models)
        nc = min(3, n)
        nr = (n + nc - 1) // nc

        fig_c, axes_c = plt.subplots(nr, nc, figsize=(5 * nc, 4.5 * nr))
        fig_c.patch.set_facecolor(BG)

        if n == 1:
            axes_c = [axes_c]
        elif nr == 1:
            axes_c = list(axes_c)
        else:
            axes_c = [ax for row in axes_c for ax in row]

        for j in range(n, len(axes_c)):
            axes_c[j].set_visible(False)

        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom", [BG, "#1a3a4a", BLUE]
        )

        for i, name in enumerate(selected_models):
            ax = axes_c[i]
            cm = results[name]["cm"]
            im = ax.imshow(cm, cmap=custom_cmap, aspect="auto")
            ax.set_facecolor(BG2)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3154")

            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(CLASS_NAMES, color=MUTED, fontsize=8, rotation=15)
            ax.set_yticklabels(CLASS_NAMES, color=MUTED, fontsize=8)
            ax.set_xlabel("Predicted", color=MUTED, fontsize=8)
            ax.set_ylabel("Actual", color=MUTED, fontsize=8)
            ax.set_title(name, color="#c5cae9", fontsize=10, pad=8)

            for row in range(3):
                for col in range(3):
                    val = cm[row, col]
                    color = "white" if val < cm.max() * 0.6 else BG
                    ax.text(col, row, str(val), ha="center", va="center",
                            fontsize=14, fontweight="bold", color=color)

        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

        st.markdown("---")
        st.subheader("Classification Reports")
        for name in selected_models:
            with st.expander(f"📋 {name}"):
                st.code(
                    classification_report(y_te, results[name]["y_pred"],
                                          target_names=CLASS_NAMES),
                    language=None,
                )
    else:
        st.info("Enable Confusion Matrices in the sidebar ↩")

# ════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ════════════════════════════════════════
with tab5:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Classes", 3)

    st.markdown("---")

    # ── Pairplot-style scatter matrix ──
    st.subheader("Feature Scatter Matrix")
    features = list(X.columns)
    nf = len(features)
    fig_s, axes_s = plt.subplots(nf, nf, figsize=(12, 10))
    fig_s.patch.set_facecolor(BG)

    for r in range(nf):
        for c in range(nf):
            ax = axes_s[r][c]
            ax.set_facecolor(BG2)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2135")
            ax.tick_params(colors=MUTED, labelsize=6)

            if r == c:
                for cls_id in range(3):
                    vals = X[features[r]][y == cls_id]
                    ax.hist(vals, bins=15, alpha=0.55, color=COLORS[cls_id], density=True)
            else:
                for cls_id in range(3):
                    mask = y == cls_id
                    ax.scatter(X[features[c]][mask], X[features[r]][mask],
                               c=COLORS[cls_id], s=8, alpha=0.6)

            if r == nf - 1:
                ax.set_xlabel(features[c], color=MUTED, fontsize=6)
            if c == 0:
                ax.set_ylabel(features[r], color=MUTED, fontsize=6)

    patches = [mpatches.Patch(color=COLORS[j], label=CLASS_NAMES[j]) for j in range(3)]
    fig_s.legend(handles=patches, loc="upper right",
                 facecolor=BG, edgecolor="#2d3154", labelcolor=MUTED, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_s)
    plt.close(fig_s)

    st.markdown("---")

    # ── Violin / box plots ──
    st.subheader("Feature Distributions")
    feat_sel = st.selectbox("Select feature", X.columns.tolist())
    fig_v, ax_v = dark_fig(8, 4)

    data_by_class = [X[feat_sel][y == cls_id].values for cls_id in range(3)]
    parts = ax_v.violinplot(data_by_class, positions=range(3),
                            showmeans=True, showmedians=True)
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[j])
        pc.set_alpha(0.55)
    for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        parts[part].set_color(MUTED)

    ax_v.set_xticks(range(3))
    ax_v.set_xticklabels(CLASS_NAMES, color=MUTED)
    ax_v.set_ylabel(feat_sel, color=MUTED)
    ax_v.grid(axis="y", color="#1e2135", linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig_v)
    plt.close(fig_v)

    st.markdown("---")
    st.subheader("Raw Data")
    df_show = X.copy()
    df_show["species"] = y.map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    st.dataframe(df_show, use_container_width=True)
