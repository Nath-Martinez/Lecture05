import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Configuration de la page
st.set_page_config(page_title="Classification Iris", layout="wide", page_icon="🌸")

# --- 1. CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names, iris.feature_names

df, target_names, feature_names = load_data()

# --- 2. BARRE LATÉRALE (INTERFACE UTILISATEUR) ---
st.sidebar.header("Paramètres de l'application")

# Sélection des 2 caractéristiques pour l'entraînement et la visualisation 2D
st.sidebar.subheader("1. Choix des caractéristiques (2D)")
feature_x = st.sidebar.selectbox("Caractéristique X (Axe horizontal)", feature_names, index=0)
feature_y = st.sidebar.selectbox("Caractéristique Y (Axe vertical)", feature_names, index=1)

if feature_x == feature_y:
    st.sidebar.warning("Veuillez sélectionner deux caractéristiques différentes.")

# Sélection du modèle
st.sidebar.subheader("2. Choix du modèle")
model_choice = st.sidebar.selectbox("Algorithme", ["Régression Logistique", "SVM", "Random Forest", "KNN"])

# Paramètres du modèle
st.sidebar.subheader("Paramètres de l'algorithme")
if model_choice == "Régression Logistique":
    C = st.sidebar.slider("C (Inverse de la force de régularisation)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, max_iter=1000)
elif model_choice == "SVM":
    C = st.sidebar.slider("C (Régularisation)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Noyau (Kernel)", ["linear", "rbf", "poly"])
    model = SVC(C=C, kernel=kernel)
elif model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Nombre d'arbres", 10, 200, 100)
    max_depth = st.sidebar.slider("Profondeur maximale", 1, 20, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else: # KNN
    n_neighbors = st.sidebar.slider("Nombre de voisins (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Sélection des métriques et visuels
st.sidebar.subheader("3. Affichage et Métriques")
show_metrics = st.sidebar.multiselect("Métriques de performance", ["Accuracy", "Precision", "Recall", "F1-Score"], default=["Accuracy", "F1-Score"])
show_decision_boundary = st.sidebar.checkbox("Afficher la frontière de décision", True)
show_confusion_matrix = st.sidebar.checkbox("Afficher la matrice de confusion", True)

# --- 3. ENTRAÎNEMENT DU MODÈLE ---
# Préparation des données (seulement les 2 features sélectionnées)
X = df[[feature_x, feature_y]].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entraînement
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 4. AFFICHAGE PRINCIPAL ---
st.title("🌸 Classificateur Interactif - Dataset Iris")
st.markdown("Cette application entraîne un modèle de Machine Learning sur deux caractéristiques du jeu de données Iris afin de pouvoir visualiser facilement les frontières de décision.")

# Affichage des métriques
if show_metrics:
    st.header("📊 Métriques de Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    if "Accuracy" in show_metrics:
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    if "Precision" in show_metrics:
        col2.metric("Precision (macro)", f"{precision_score(y_test, y_pred, average='macro'):.2f}")
    if "Recall" in show_metrics:
        col3.metric("Recall (macro)", f"{recall_score(y_test, y_pred, average='macro'):.2f}")
    if "F1-Score" in show_metrics:
        col4.metric("F1-Score (macro)", f"{f1_score(y_test, y_pred, average='macro'):.2f}")

st.divider()

# Création des colonnes pour les graphiques
plot_col1, plot_col2 = st.columns(2)

# Graphique de la Frontière de décision
with plot_col1:
    if show_decision_boundary and feature_x != feature_y:
        st.subheader("🗺️ Frontière de Décision")
        
        # Création d'une grille (meshgrid)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        # Prédiction sur toute la grille
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        
        # Afficher les points d'entraînement
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=40, label='Train')
        # Afficher les points de test
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='w', s=60, marker='*', label='Test')
        
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(f"Frontières ({model_choice})")
        
        # Légende propre
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_bold.colors[i], markersize=10) for i in range(3)]
        ax.legend(handles, target_names, loc='best')
        
        st.pyplot(fig)

# Graphique de la Matrice de confusion
with plot_col2:
    if show_confusion_matrix:
        st.subheader("🎯 Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
        ax_cm.set_ylabel("Vraie classe")
        ax_cm.set_xlabel("Classe prédite")
        ax_cm.set_title("Performance sur les données de Test")
        
        st.pyplot(fig_cm)
