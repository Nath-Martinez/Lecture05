import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1f2235 100%);
        border: 1px solid #2d3154; border-radius: 12px;
        padding: 20px; text-align: center; margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem; font-weight: 600; color: #7ee8a2;
    }
    .metric-label {
        font-size: 0.8rem; color: #8890b5;
        text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px;
    }
    .section-title {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        color: #5a6285; text-transform: uppercase; letter-spacing: 3px;
        margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #1e2135;
    }
    div[data-testid="stSidebar"] {
        background-color: #0a0c14; border-right: 1px solid #1a1d27;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
COLORS      = ["#7ee8a2", "#ff6b9d", "#4fc3f7"]
PALETTE     = ["#7ee8a2", "#ff6b9d", "#4fc3f7", "#ffb347", "#b39ddb", "#80deea", "#ef9a9a"]
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

MODELS = {
    "Logistic Regression":  LogisticRegression,
    "Decision Tree":        DecisionTreeClassifier,
    "Random Forest":        RandomForestClassifier,
    "Gradient Boosting":    GradientBoostingClassifier,
    "SVM":                  SVC,
    "K-Nearest Neighbors":  KNeighborsClassifier,
    "Naive Bayes":          GaussianNB,
}
MODEL_PARAMS = {
    "Logistic Regression":  {"max_iter": 1000, "random_state": 42},
    "Decision Tree":        {"random_state": 42},
    "Random Forest":        {"random_state": 42},
    "Gradient Boosting":    {"random_state": 42},
    "SVM":                  {"probability": True, "random_state": 42},
    "K-Nearest Neighbors":  {},
    "Naive Bayes":          {},
}

DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8890b5"))
GRID = dict(gridcolor="#1e2135")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
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
            MODELS[name](**MODEL_PARAMS[name]), X_tr_s, y_tr.values,
            cv=StratifiedKFold(cv_folds), scoring="accuracy"
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

# ─────────────────────────────────────────────
# DECISION BOUNDARY — pure Plotly
# ─────────────────────────────────────────────
def make_boundary_fig(model_name, clf_2d, X_2d_tr, y_tr, X_2d_te, y_te, feat1, feat2):
    x_min, x_max = X_2d_tr[feat1].min() - .5, X_2d_tr[feat1].max() + .5
    y_min, y_max = X_2d_tr[feat2].min() - .5, X_2d_tr[feat2].max() + .5
    res = 150
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    grid = pd.DataFrame({feat1: xx.ravel(), feat2: yy.ravel()})
    Z    = clf_2d.predict(grid).reshape(xx.shape)

    bg      = ["#1a3a2a", "#3a1a2a", "#1a2a3a"]
    cs      = [[0.00, bg[0]], [0.33, bg[0]],
               [0.33, bg[1]], [0.66, bg[1]],
               [0.66, bg[2]], [1.00, bg[2]]]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=np.linspace(x_min, x_max, res),
        y=np.linspace(y_min, y_max, res),
        z=Z, colorscale=cs, zmin=0, zmax=2,
        showscale=False, opacity=0.42, hoverinfo="skip",
    ))
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        m_tr = y_tr == cls_id
        m_te = y_te == cls_id
        fig.add_trace(go.Scatter(
            x=X_2d_tr[feat1][m_tr], y=X_2d_tr[feat2][m_tr],
            mode="markers", name=cls_name, legendgroup=cls_name,
            marker=dict(color=COLORS[cls_id], size=7, opacity=0.85,
                        line=dict(width=0.5, color="rgba(255,255,255,.25)")),
        ))
        fig.add_trace(go.Scatter(
            x=X_2d_te[feat1][m_te], y=X_2d_te[feat2][m_te],
            mode="markers", name=f"{cls_name} (test)",
            legendgroup=cls_name, showlegend=False,
            marker=dict(color=COLORS[cls_id], size=13, symbol="star", opacity=1.0,
                        line=dict(width=1, color="rgba(255,255,255,.55)")),
        ))
    fig.update_layout(
        title=dict(text=model_name, font=dict(color="#c5cae9", size=13)),
        **DARK, plot_bgcolor="#0d0f18",
        xaxis=dict(title=feat1, **GRID, range=[x_min, x_max]),
        yaxis=dict(title=feat2, **GRID, range=[y_min, y_max]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        height=400, margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌸 Iris Classifier Studio")
    st.markdown("---")
    st.markdown('<div class="section-title">⚙️ Models</div>', unsafe_allow_html=True)
    selected_models = st.multiselect(
        "Choose models to train", list(MODELS.keys()),
        default=["Logistic Regression", "Random Forest", "SVM"],
    )
    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Training</div>', unsafe_allow_html=True)
    test_size   = st.slider("Test size (%)", 10, 40, 20, 5) / 100
    use_scaling = st.checkbox("Feature Scaling (StandardScaler)", value=True)
    cv_folds    = st.slider("Cross-validation folds", 3, 10, 5)
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Visualization</div>', unsafe_allow_html=True)
    boundary_features = st.selectbox(
        "Decision boundary features",
        ["sepal length (cm) vs sepal width (cm)",
         "petal length (cm) vs petal width (cm)",
         "sepal length (cm) vs petal length (cm)"],
        index=1,
    )
    show_pca = st.checkbox("Show PCA projection",    value=True)
    show_roc = st.checkbox("Show ROC curves",        value=True)
    show_cm  = st.checkbox("Show confusion matrices", value=True)

# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("# 🌸 Iris Classifier Studio")
st.markdown("Interactive multi-model classification — decision boundaries · ROC curves · confusion matrices.")
st.markdown("---")

if not selected_models:
    st.warning("👈 Please select at least one model from the sidebar.")
    st.stop()

X, y = load_data()
feat1, feat2 = [f.strip() for f in boundary_features.split(" vs ")]
results, X_tr_s, X_te_s, X_all_s, y_tr, y_te = train_all(
    tuple(selected_models), test_size, use_scaling, cv_folds
)

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Overview",
    "🗺️ Decision Boundaries",
    "📉 ROC Curves",
    "🔢 Confusion Matrices",
    "🔍 Data Explorer",
])

# ── TAB 1 ──────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(len(selected_models))
    for i, name in enumerate(selected_models):
        r = results[name]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1rem;font-weight:600;color:#c5cae9;margin-bottom:10px;">{name}</div>
                <div class="metric-value">{r['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
                <hr style="border-color:#2d3154;margin:10px 0;">
                <div style="display:flex;justify-content:space-around;">
                    <div>
                        <div style="color:#7ee8a2;font-family:'IBM Plex Mono',monospace;font-size:.95rem;">{r['f1']:.3f}</div>
                        <div class="metric-label">F1</div>
                    </div>
                    <div>
                        <div style="color:#4fc3f7;font-family:'IBM Plex Mono',monospace;font-size:.95rem;">{r['cv_mean']:.3f}</div>
                        <div class="metric-label">CV</div>
                    </div>
                    <div>
                        <div style="color:#ff6b9d;font-family:'IBM Plex Mono',monospace;font-size:.95rem;">{r['recall']:.3f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Comparative Performance</div>', unsafe_allow_html=True)
    metrics       = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    fig_bar = go.Figure()
    for i, name in enumerate(selected_models):
        fig_bar.add_trace(go.Bar(
            name=name, x=metric_labels,
            y=[results[name][m] for m in metrics],
            marker_color=PALETTE[i % len(PALETTE)], opacity=0.88,
        ))
    fig_bar.update_layout(
        **DARK, barmode="group", height=360,
        yaxis=dict(**GRID, range=[0, 1.05]),
        xaxis=GRID, legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">Cross-Validation (mean ± std)</div>', unsafe_allow_html=True)
    fig_cv = go.Figure()
    for i, name in enumerate(selected_models):
        fig_cv.add_trace(go.Bar(
            x=[name], y=[results[name]["cv_mean"]],
            error_y=dict(type="data", array=[results[name]["cv_std"]],
                         visible=True, color="rgba(255,255,255,.3)"),
            marker_color=PALETTE[i % len(PALETTE)], showlegend=False,
        ))
    fig_cv.update_layout(
        **DARK, height=300,
        yaxis=dict(**GRID, range=[0, 1.05], title="CV Accuracy"),
        xaxis=GRID, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

# ── TAB 2 ──────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Decision Boundary Visualization</div>', unsafe_allow_html=True)
    st.info(f"Boundary computed on: **{feat1}** vs **{feat2}**  |  ⭐ = test points")

    X_2d_tr = X_tr_s[[feat1, feat2]].reset_index(drop=True)
    X_2d_te = X_te_s[[feat1, feat2]].reset_index(drop=True)
    n_cols  = min(2, len(selected_models))
    cols    = st.columns(n_cols)

    for i, name in enumerate(selected_models):
        clf_2d = MODELS[name](**MODEL_PARAMS[name])
        clf_2d.fit(X_2d_tr, y_tr)
        with cols[i % n_cols]:
            st.plotly_chart(
                make_boundary_fig(name, clf_2d, X_2d_tr, y_tr, X_2d_te, y_te, feat1, feat2),
                use_container_width=True,
            )

    if show_pca:
        st.markdown("---")
        st.markdown('<div class="section-title">PCA 2D Projection (All Features)</div>', unsafe_allow_html=True)
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all_s)
        fig_pca = go.Figure()
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = y.values == cls_id
            fig_pca.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers", name=cls_name,
                marker=dict(color=COLORS[cls_id], size=8, opacity=0.8,
                            line=dict(width=.5, color="rgba(255,255,255,.25)")),
            ))
        fig_pca.update_layout(
            **DARK, height=430,
            xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", **GRID),
            yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", **GRID),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

# ── TAB 3 ──────────────────────────────────────
with tab3:
    if show_roc:
        st.markdown('<div class="section-title">ROC Curves — One-vs-Rest</div>', unsafe_allow_html=True)
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            st.markdown(f"**Class: {cls_name}**")
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dot", color="#3a3f5c", width=1))
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
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={auc(fpr,tpr):.3f})",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                ))
            fig_roc.update_layout(
                **DARK, height=360,
                xaxis=dict(title="False Positive Rate", **GRID),
                yaxis=dict(title="True Positive Rate",  **GRID),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Enable ROC Curves in the sidebar ↩")

# ── TAB 4 ──────────────────────────────────────
with tab4:
    if show_cm:
        st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
        n_cols_cm = min(2, len(selected_models))
        cols_cm   = st.columns(n_cols_cm)
        for i, name in enumerate(selected_models):
            fig_cm = px.imshow(
                results[name]["cm"], x=CLASS_NAMES, y=CLASS_NAMES,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale=[[0,"#0f1117"],[.5,"#1a3a4a"],[1,"#4fc3f7"]],
                text_auto=True, title=name,
            )
            fig_cm.update_layout(
                **DARK, title_font=dict(color="#c5cae9"),
                coloraxis_showscale=False,
                height=320, margin=dict(t=40, b=10),
            )
            with cols_cm[i % n_cols_cm]:
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Classification Reports</div>', unsafe_allow_html=True)
        for name in selected_models:
            with st.expander(f"📋 {name}"):
                st.code(classification_report(y_te, results[name]["y_pred"],
                                              target_names=CLASS_NAMES), language=None)
    else:
        st.info("Enable Confusion Matrices in the sidebar ↩")

# ── TAB 5 ──────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Classes", 3)

    st.markdown("---")
    st.markdown('<div class="section-title">Scatter Matrix</div>', unsafe_allow_html=True)
    df_plot = X.copy()
    df_plot["species"] = y.map({0:"Setosa", 1:"Versicolor", 2:"Virginica"})
    fig_pair = px.scatter_matrix(
        df_plot, dimensions=X.columns.tolist(),
        color="species", color_discrete_sequence=COLORS, opacity=0.7,
    )
    fig_pair.update_traces(marker=dict(size=4))
    fig_pair.update_layout(**DARK, height=600, margin=dict(t=20))
    st.plotly_chart(fig_pair, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Feature Distribution</div>', unsafe_allow_html=True)
    feat_sel = st.selectbox("Select feature", X.columns.tolist())
    fig_dist = go.Figure()
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        vals = X[feat_sel][y == cls_id]
        fig_dist.add_trace(go.Violin(
            x=[cls_name]*len(vals), y=vals,
            name=cls_name, fillcolor=COLORS[cls_id],
            line_color=COLORS[cls_id], opacity=0.6,
            box_visible=True, meanline_visible=True,
        ))
    fig_dist.update_layout(
        **DARK, showlegend=False, height=380,
        yaxis=dict(title=feat_sel, **GRID), margin=dict(t=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df_plot, use_container_width=True)
