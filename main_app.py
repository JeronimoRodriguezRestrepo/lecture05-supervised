import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
st.set_page_config(page_title="Iris ML App", layout="wide")

st.title("🌸 Clasificación del Iris Dataset")
st.markdown("Aplicación interactiva de Machine Learning con múltiples modelos y visualizaciones.")


# -----------------------------
# CARGA DE DATOS
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# -----------------------------
# SIDEBAR - CONFIGURACIÓN
# -----------------------------
st.sidebar.header("⚙️ Configuración del Modelo")

model_option = st.sidebar.selectbox(
    "Selecciona el modelo",
    ("Logistic Regression", "SVM", "KNN", "Random Forest")
)

test_size = st.sidebar.slider("Tamaño del test (%)", 10, 50, 30)

# Parámetros específicos
if model_option == "SVM":
    C = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf"))

elif model_option == "KNN":
    k = st.sidebar.slider("Número de vecinos (k)", 1, 15, 5)

elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100)


# Selección de features para frontera de decisión
st.sidebar.header("📊 Visualización")
feature_x = st.sidebar.selectbox("Feature eje X", feature_names, index=0)
feature_y = st.sidebar.selectbox("Feature eje Y", feature_names, index=1)

# -----------------------------
# PREPROCESAMIENTO
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# SELECCIÓN DEL MODELO
# -----------------------------
if model_option == "Logistic Regression":
    model = LogisticRegression()

elif model_option == "SVM":
    model = SVC(C=C, kernel=kernel, probability=True)

elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators)


# -----------------------------
# ENTRENAMIENTO
# -----------------------------
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# -----------------------------
# MÉTRICAS
# -----------------------------
st.header("📈 Métricas de desempeño")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='macro'):.2f}")


# -----------------------------
# MATRIZ DE CONFUSIÓN
# -----------------------------
st.subheader("Matriz de Confusión")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax_cm)

plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig_cm)


# -----------------------------
# FRONTERA DE DECISIÓN
# -----------------------------
st.subheader("Frontera de Decisión (2 Features)")

# Índices de las features seleccionadas
idx_x = feature_names.index(feature_x)
idx_y = feature_names.index(feature_y)

X_2d = X[:, [idx_x, idx_y]]
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_2d, y, test_size=test_size / 100, random_state=42
)

scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2)
X_test2 = scaler2.transform(X_test2)

model.fit(X_train2, y_train2)

# Crear malla
x_min, x_max = X_train2[:, 0].min() - 1, X_train2[:, 0].max() + 1
y_min, y_max = X_train2[:, 1].min() - 1, X_train2[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = ax.scatter(
    X_train2[:, 0],
    X_train2[:, 1],
    c=y_train2,
    cmap=plt.cm.coolwarm,
    edgecolors="k"
)

ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
st.pyplot(fig)
