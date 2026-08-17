import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

st.title("📊 Machine Learning Model Evaluator")

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Churn / HeartDisease target detect karein
    default_idx = 0
    for target_candidate in ['Churn', 'HeartDisease']:
        if target_candidate in df.columns:
            default_idx = list(df.columns).index(target_candidate)
            break

    target_column = st.selectbox("Select Target Column", df.columns, index=default_idx)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Features processing
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.mean(numeric_only=True))

        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_name = st.selectbox(
            "Select Model", 
            ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "Gradient Boosting"]
        )

        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "kNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        model = models[model_name]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        st.subheader(f"📈 Metrics for {model_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        c5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                c6.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
            except Exception:
                c6.metric("AUC", "N/A")

        st.subheader("🧩 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)