import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

st.title("📊 Machine Learning Model Evaluator")

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
model_name = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"])

if model_name == "Logistic Regression":
    model = LogisticRegression(); X_eval = X_scaled
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42); X_eval = X
elif model_name == "kNN":
    model = KNeighborsClassifier(); X_eval = X_scaled
elif model_name == "Naive Bayes":
    model = GaussianNB(); X_eval = X
else:
    model = RandomForestClassifier(random_state=42); X_eval = X
        
model.fit(X_eval, y)
y_pred = model.predict(X_eval)
y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
    
st.subheader(f"📈 Metrics for {model_name}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
try:
    if len(set(y)) > 2:
        auc_val = roc_auc_score(y, y_proba, multi_class='ovr')
    else:
        auc_val = roc_auc_score(y, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
    auc_text = f"{auc_val:.4f}"
except Exception:
    auc_text = "N/A"

c2.metric("AUC", auc_text)
c3.metric("Precision", f"{precision_score(y, y_pred, zero_division=0):.4f}")
c4.metric("Recall", f"{recall_score(y, y_pred, zero_division=0):.4f}")
c5.metric("F1", f"{f1_score(y, y_pred, zero_division=0):.4f}")
c6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")
    
st.subheader("🧩 Confusion Matrix")
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

   

