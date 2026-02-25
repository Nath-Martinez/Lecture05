import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1f2235 100%);
        border: 1px solid #2d3154;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #7ee8a2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8890b5;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }
    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #5a6285;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2135;
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #8890b5 !important;
        font-size: 0.85rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #0a0c14;
        border-right: 1px solid #1a1d27;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODELS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    target_names = iris.target_names
    return X, y, target_names

MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "SVM": SVC,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
}

MODEL_PARAMS = {
    "Logistic Regression": {"max_iter": 1000, "random_state": 42},
    "Decision Tree": {"random_state": 42},
    "Random Forest": {"random_state": 42},
    "Gradient Boosting": {"random_state": 42},
    "SVM": {"probability": True, "random_state": 42},
    "K-Nearest Neighbors": {},
    "Naive Bayes": {},
}

COLORS = ["#7ee8a2", "#ff6b9d", "#4fc3f7"]
CLASS_COLORS = {0: "#7ee8a2", 1: "#ff6b9d", 2: "#4fc3f7"}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Iris Classifier Studio")
    st.markdown("---")

    st.markdown('<div class="section-title">⚙️ Model Selection</div>', unsafe_allow_html=True)
    selected_models = st.multiselect(
        "Choose models to train",
        list(MODELS.keys()),
        default=["Logistic Regression", "Random Forest", "SVM"],
    )

    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Training Options</div>', unsafe_allow_html=True)
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100
    use_scaling = st.checkbox("Feature Scaling (StandardScaler)", value=True)
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Visualization</div>', unsafe_allow_html=True)
    boundary_features = st.selectbox(
        "Decision boundary features",
        ["sepal length (cm) vs sepal width (cm)",
         "petal length (cm) vs petal width (cm)",
         "sepal length (cm) vs petal length (cm)"],
        index=1,
    )
    show_pca = st.checkbox("Show PCA projection", value=True)
    show_roc = st.checkbox("Show ROC curves", value=True)
    show_cm = st.checkbox("Show confusion matrices", value=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🌸 Iris Classifier Studio")
st.markdown("Interactive multi-model classification with decision boundaries, performance metrics, and ROC analysis.")
st.markdown("---")

if not selected_models:
    st.warning("👈 Please select at least one model from the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD & SPLIT
# ─────────────────────────────────────────────
X, y, target_names = load_data()

feat1, feat2 = boundary_features.split(" vs ")
feat1, feat2 = feat1.strip(), feat2.strip()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

if use_scaling:
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    X_all_s   = pd.DataFrame(scaler.transform(X), columns=X.columns)
else:
    X_train_s, X_test_s, X_all_s = X_train, X_test, X

# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
@st.cache_data
def train_all(selected, test_size, use_scaling, cv_folds):
    X, y, target_names = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    if use_scaling:
        sc = StandardScaler()
        X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=X.columns)
        X_te_s  = pd.DataFrame(sc.transform(X_te), columns=X.columns)
        X_all_s  = pd.DataFrame(sc.transform(X), columns=X.columns)
    else:
        X_tr_s, X_te_s, X_all_s, sc = X_tr, X_te, X, None

    results = {}
    for name in selected:
        clf = MODELS[name](**MODEL_PARAMS[name])
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        cv = cross_val_score(clf, X_tr_s, y_tr, cv=StratifiedKFold(cv_folds), scoring="accuracy")
        results[name] = {
            "model": clf,
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_te, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_te, y_pred, average="weighted", zero_division=0),
            "cv_mean": cv.mean(),
            "cv_std": cv.std(),
            "cm": confusion_matrix(y_te, y_pred),
        }
    return results, X_tr_s, X_te_s, X_all_s, y_tr, y_te, sc

results, X_train_s, X_test_s, X_all_s, y_train, y_test, scaler_obj = train_all(
    tuple(selected_models), test_size, use_scaling, cv_folds
)

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Overview",
    "🗺️ Decision Boundaries",
    "📉 ROC Curves",
    "🔢 Confusion Matrices",
    "🔍 Data Explorer",
])

# ══════════════════════════════════════════════
# TAB 1 — PERFORMANCE OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Model Performance Metrics</div>', unsafe_allow_html=True)

    # Metric cards for each model
    cols = st.columns(len(selected_models))
    for i, name in enumerate(selected_models):
        r = results[name]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.1rem;font-weight:600;color:#c5cae9;margin-bottom:12px;">{name}</div>
                <div class="metric-value">{r['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
                <hr style="border-color:#2d3154;margin:12px 0;">
                <div style="display:flex;justify-content:space-around;">
                    <div>
                        <div style="color:#7ee8a2;font-family:'IBM Plex Mono',monospace;font-size:1rem;">{r['f1']:.3f}</div>
                        <div class="metric-label">F1</div>
                    </div>
                    <div>
                        <div style="color:#4fc3f7;font-family:'IBM Plex Mono',monospace;font-size:1rem;">{r['cv_mean']:.3f}</div>
                        <div class="metric-label">CV Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Grouped bar chart
    st.markdown('<div class="section-title">Comparative Performance</div>', unsafe_allow_html=True)
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig = go.Figure()
    palette = ["#7ee8a2", "#ff6b9d", "#4fc3f7", "#ffb347", "#b39ddb", "#80deea", "#ef9a9a"]
    for i, name in enumerate(selected_models):
        fig.add_trace(go.Bar(
            name=name,
            x=metric_labels,
            y=[results[name][m] for m in metrics],
            marker_color=palette[i % len(palette)],
            marker_line_color="rgba(0,0,0,0)",
            opacity=0.9,
        ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8890b5", family="IBM Plex Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c5cae9")),
        xaxis=dict(gridcolor="#1e2135"),
        yaxis=dict(gridcolor="#1e2135", range=[0, 1.05]),
        margin=dict(t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # CV scores
    st.markdown('<div class="section-title">Cross-Validation Results</div>', unsafe_allow_html=True)
    cv_df = pd.DataFrame({
        "Model": selected_models,
        "CV Mean": [results[n]["cv_mean"] for n in selected_models],
        "CV Std": [results[n]["cv_std"] for n in selected_models],
    })
    fig2 = go.Figure()
    for i, row in cv_df.iterrows():
        fig2.add_trace(go.Bar(
            name=row["Model"],
            x=[row["Model"]],
            y=[row["CV Mean"]],
            error_y=dict(type="data", array=[row["CV Std"]], visible=True, color="#ffffff44"),
            marker_color=palette[i % len(palette)],
            showlegend=False,
        ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8890b5"),
        yaxis=dict(gridcolor="#1e2135", range=[0, 1.05], title="CV Accuracy"),
        xaxis=dict(gridcolor="#1e2135"),
        margin=dict(t=20, b=20),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — DECISION BOUNDARIES
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Decision Boundary Visualization</div>', unsafe_allow_html=True)
    st.info(f"Plotting decision boundary on features: **{feat1}** vs **{feat2}**")

    def plot_decision_boundary(model, X_2d_train, y_train, X_2d_test, y_test, title, feat1, feat2):
        x_min = X_2d_train[feat1].min() - 0.5
        x_max = X_2d_train[feat1].max() + 0.5
        y_min = X_2d_train[feat2].min() - 0.5
        y_max = X_2d_train[feat2].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

        # Build full feature vector with zeros for unused features
        grid_full = np.zeros((xx.ravel().shape[0], X_train_s.shape[1]))
        fi1 = list(X_train_s.columns).index(feat1)
        fi2 = list(X_train_s.columns).index(feat2)
        grid_full[:, fi1] = xx.ravel()
        grid_full[:, fi2] = yy.ravel()

        Z = model.predict(grid_full).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        cmap_bg = plt.cm.colors.ListedColormap(["#1a3a2a", "#3a1a2a", "#1a2a3a"])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, Z, colors=["#7ee8a220", "#ff6b9d20", "#4fc3f720"], linewidths=1.5, levels=[-0.5, 0.5, 1.5, 2.5])

        scatter_colors = [COLORS[int(c)] for c in y_train]
        ax.scatter(X_2d_train[feat1], X_2d_train[feat2], c=scatter_colors, s=60, alpha=0.7, edgecolors="#ffffff30", linewidths=0.5, label="Train")
        scatter_colors_te = [COLORS[int(c)] for c in y_test]
        ax.scatter(X_2d_test[feat1], X_2d_test[feat2], c=scatter_colors_te, s=120, marker="*", alpha=1.0, edgecolors="#ffffff60", linewidths=0.8, label="Test")

        ax.set_xlabel(feat1, color="#8890b5", fontsize=9)
        ax.set_ylabel(feat2, color="#8890b5", fontsize=9)
        ax.set_title(title, color="#c5cae9", fontsize=11, pad=10)
        ax.tick_params(colors="#5a6285")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2135")

        patches = [mpatches.Patch(color=COLORS[i], label=n) for i, n in enumerate(["Setosa", "Versicolor", "Virginica"])]
        ax.legend(handles=patches, facecolor="#0a0c14", edgecolor="#2d3154", labelcolor="#8890b5", fontsize=8)
        plt.tight_layout()
        return fig

    n_cols = min(2, len(selected_models))
    cols = st.columns(n_cols)
    X_2d_train = X_train_s[[feat1, feat2]]
    X_2d_test  = X_test_s[[feat1, feat2]]

    for i, name in enumerate(selected_models):
        mdl = results[name]["model"]
        # Retrain on only 2 features for boundary plot
        mdl_2d = MODELS[name](**MODEL_PARAMS[name])
        mdl_2d.fit(X_2d_train, y_train)
        with cols[i % n_cols]:
            fig = plot_decision_boundary(mdl_2d, X_2d_train, y_train, X_2d_test, y_test, name, feat1, feat2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # PCA projection
    if show_pca:
        st.markdown("---")
        st.markdown('<div class="section-title">PCA 2D Projection (All Features)</div>', unsafe_allow_html=True)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all_s)

        fig_pca = go.Figure()
        for cls_id, cls_name in enumerate(["Setosa", "Versicolor", "Virginica"]):
            mask = y.values == cls_id
            fig_pca.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                name=cls_name,
                marker=dict(color=COLORS[cls_id], size=8, opacity=0.8, line=dict(width=0.5, color="#ffffff40")),
            ))
        fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8890b5"),
            xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", gridcolor="#1e2135"),
            yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", gridcolor="#1e2135"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=420,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — ROC CURVES
# ══════════════════════════════════════════════
with tab3:
    if show_roc:
        st.markdown('<div class="section-title">ROC Curves — One-vs-Rest</div>', unsafe_allow_html=True)
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        class_names = ["Setosa", "Versicolor", "Virginica"]

        for cls_id, cls_name in enumerate(class_names):
            st.markdown(f"**Class: {cls_name}**")
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#3a3f5c", width=1))

            for i, name in enumerate(selected_models):
                mdl = results[name]["model"]
                if hasattr(mdl, "predict_proba"):
                    y_score = mdl.predict_proba(X_test_s)[:, cls_id]
                elif hasattr(mdl, "decision_function"):
                    y_score = mdl.decision_function(X_test_s)[:, cls_id] if y_bin.shape[1] > 2 else mdl.decision_function(X_test_s)
                else:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, cls_id], y_score)
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={roc_auc:.3f})",
                    mode="lines",
                    line=dict(color=palette[i % len(palette)], width=2),
                ))

            fig_roc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8890b5"),
                xaxis=dict(title="False Positive Rate", gridcolor="#1e2135"),
                yaxis=dict(title="True Positive Rate", gridcolor="#1e2135"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=360,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Enable ROC Curves in the sidebar ↩")

# ══════════════════════════════════════════════
# TAB 4 — CONFUSION MATRICES
# ══════════════════════════════════════════════
with tab4:
    if show_cm:
        st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
        n_cols_cm = min(2, len(selected_models))
        cols_cm = st.columns(n_cols_cm)
        class_labels = ["Setosa", "Versicolor", "Virginica"]

        for i, name in enumerate(selected_models):
            cm = results[name]["cm"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=class_labels, y=class_labels,
                color_continuous_scale=[[0, "#0f1117"], [0.5, "#1a3a4a"], [1, "#4fc3f7"]],
                text_auto=True,
                title=name,
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8890b5"),
                title_font=dict(color="#c5cae9"),
                coloraxis_showscale=False,
                margin=dict(t=40, b=10),
                height=320,
            )
            with cols_cm[i % n_cols_cm]:
                st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        st.markdown("---")
        st.markdown('<div class="section-title">Classification Reports</div>', unsafe_allow_html=True)
        for name in selected_models:
            with st.expander(f"📋 {name}"):
                report = classification_report(
                    y_test, results[name]["y_pred"],
                    target_names=["Setosa", "Versicolor", "Virginica"]
                )
                st.code(report, language=None)
    else:
        st.info("Enable Confusion Matrices in the sidebar ↩")

# ══════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(X))
    col2.metric("Features", X.shape[1])
    col3.metric("Classes", 3)

    st.markdown("---")
    st.markdown('<div class="section-title">Pairplot (Feature Relationships)</div>', unsafe_allow_html=True)
    df_plot = X.copy()
    df_plot["species"] = y.map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})

    fig_pair = px.scatter_matrix(
        df_plot,
        dimensions=X.columns.tolist(),
        color="species",
        color_discrete_sequence=COLORS,
        opacity=0.7,
    )
    fig_pair.update_traces(marker=dict(size=4))
    fig_pair.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8890b5"),
        height=600,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_pair, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
    feat_sel = st.selectbox("Select feature", X.columns.tolist())
    fig_dist = go.Figure()
    for cls_id, cls_name in enumerate(["Setosa", "Versicolor", "Virginica"]):
        vals = X[feat_sel][y == cls_id]
        fig_dist.add_trace(go.Violin(
            x=[cls_name] * len(vals), y=vals,
            name=cls_name, fillcolor=COLORS[cls_id],
            line_color=COLORS[cls_id], opacity=0.6,
            box_visible=True, meanline_visible=True,
        ))
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8890b5"),
        yaxis=dict(title=feat_sel, gridcolor="#1e2135"),
        showlegend=False, height=380,
        margin=dict(t=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df_plot.style.background_gradient(cmap="Blues"), use_container_width=True)
