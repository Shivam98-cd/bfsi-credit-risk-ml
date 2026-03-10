import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_pipeline(file_path):
    # 1. Create directory if it doesn't exist
    if not os.path.exists("model"):
        os.makedirs("model")

    # 2. Load Data
    df = pd.read_excel(file_path)

    # 3. Encoding (Handling IDs and Categoricals)
    # ✅ FIX: Use a separate LabelEncoder per column and save each one.
    # Previously, a single `le` instance was reused in a loop — only the
    # last column's encoder was retained in memory, making consistent
    # encoding of new data impossible.
    exclude = ['Customer_ID', 'Customer_Name', 'Application_Date']
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c not in exclude]

    encoders = {}  # dictionary to hold one encoder per column
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # store with column name as key
        print(f"  Encoded '{col}': {list(le.classes_)}")

    # Save all encoders in one file — keyed by column name
    joblib.dump(encoders, "model/encoders.pkl")
    print(f"✅ Saved: model/encoders.pkl ({len(encoders)} encoders for: {list(encoders.keys())})")

    # 4. Feature/Target Split
    X = df.drop(['Loan_Default'] + exclude, axis=1, errors='ignore')
    y = df['Loan_Default']

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Model Training Loop
    models = {
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "Random_Forest": RandomForestClassifier(n_estimators=100),
        "Decision_Tree": DecisionTreeClassifier(max_depth=5),
        "Logistic_Regression": LogisticRegression(max_iter=1000)
    }

    print("\n🚀 Training models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, f"model/{name}.pkl")
        print(f"✅ Saved: model/{name}.pkl")

    # 8. Save Utility Files (Crucial for app.py)
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(X.columns.tolist(), "model/features.pkl")
    joblib.dump(X_test_scaled, "model/X_test.pkl")
    joblib.dump(y_test, "model/y_test.pkl")

    print("\n✨ All 9 files successfully created in the 'model/' folder!")
    print("   (encoders.pkl added — one LabelEncoder saved per categorical column)")


if __name__ == "__main__":
    build_pipeline("credit_risk_customers.xlsx")