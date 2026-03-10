# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


def preprocess_data(file_path, target='target'):
    df = pd.read_excel(file_path)

    # Encode categorical features
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Separate target
    X = df.drop(columns=[target])
    y = df[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save encoders and scaler
    with open('model/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_scaled, y