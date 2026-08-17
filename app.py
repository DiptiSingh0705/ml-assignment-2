import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.title("📊 Machine Learning Model Evaluator")

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # HeartDisease default target set karein agar present ho
    default_idx = list(df.columns).index('HeartDisease') if 'HeartDisease' in df.columns else 0
    target_column = st.selectbox("Select Target Column", df.columns, index=default_idx)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Categorical features & missing values handle karein
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.mean(numeric_only=True))

        # Target variable ko classification-compatible banayein
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_name = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"])

        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "kNN":
            model = KNeighborsClassifier()
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        else:
            model = RandomForestClassifier(random_state=42)

        # Encoded target model par fit karein
        model.fit(X_scaled, y_encoded)
        y_pred = model.predict(X_scaled)

        st.subheader(f"📈 Metrics for {model_name}")
        c1, c2 = st.columns(2)
        
        acc = accuracy_score(y_encoded, y_pred)
        c1.metric("Accuracy", f"{acc:.4f}")

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_scaled)
                n_classes = len(set(y_encoded))
                if n_classes > 2:
                    auc_val = roc_auc_score(y_encoded, y_proba, multi_class='ovr')
                elif n_classes == 2:
                    auc_val = roc_auc_score(y_encoded, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                else:
                    auc_val = 0.0
                auc_text = f"{auc_val:.4f}"
            except Exception:
                auc_text = "N/A"
            c2.metric("AUC", auc_text)