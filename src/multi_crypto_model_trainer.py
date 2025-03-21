import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# Define dataset paths
data_files = {
    "BTC": "labeled_BTC_data.csv",
    "ETH": "labeled_ETH_data.csv",
    "DOGE": "labeled_DOGE_data.csv"
}

# Define base features for training
base_features = ['MACD_Diff', 'MACD_Histogram_Change', 'RSI_14', 'Price_Range', 'Price_Momentum',
                 'Stoch_Momentum', 'OBV_Change']

# Create a folder for models if not exists
os.makedirs("models", exist_ok=True)

for crypto, file_path in data_files.items():
    print(f"\nProcessing dataset for {crypto}...")

    df = pd.read_csv(file_path)

    # === BTC Fix: Address Label Imbalance ===
    if crypto == "BTC":
        print("\nChecking BTC Label Distribution Before Balancing:")
        print(df['Label_Multi'].value_counts())  # See original distribution

        # Reduce Sideways Class to Avoid Overwhelming Data
        df_sideways = df[df['Label_Multi'] == 0]
        df_up = df[df['Label_Multi'] == 1]
        df_down = df[df['Label_Multi'] == -1]

        df_sideways_downsampled = df_sideways.sample(n=min(len(df_up), len(df_down)), random_state=42)
        df = pd.concat([df_up, df_down, df_sideways_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        print("\nBTC Label Distribution After Balancing:")
        print(df['Label_Multi'].value_counts())

    # === Feature Engineering Updates ===
    # BTC: Add Volatility Features
    if crypto == "BTC":
        df['Close_STD_7'] = df['Close'].rolling(window=7).std()
        df.dropna(inplace=True)  # Drop NaNs caused by shifting

    # ETH: Add Lagged Close Prices
    if crypto == "ETH":
        df['Close_Lag_1'] = df['Close'].shift(1)
        df['Close_Lag_2'] = df['Close'].shift(2)
        df['Close_Lag_3'] = df['Close'].shift(3)
        df.dropna(inplace=True)  # Drop NaNs caused by shifting

    # Determine Feature Set Based on Crypto
    if crypto == "BTC":
        features = base_features + ['Close_STD_7']
    elif crypto == "ETH":
        features = base_features + ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
    else:
        features = base_features  # DOGE remains unchanged

    X = df[features]
    y = df['Label_Multi']

    # Fix ETH Precision Issue (Handle Class Imbalance)
    class_weight = "balanced" if crypto == "ETH" else None

    # Encode labels for XGBoost
    label_mapping = {-1: 0, 0: 1, 1: 2}
    y_encoded = y.map(label_mapping)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # === Hyperparameter Tuned Models ===
    rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight=class_weight)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', learning_rate=0.005, n_estimators=500, max_depth=8, random_state=42)
    logistic_model = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs', C=10, class_weight=class_weight)

    # Train models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train_encoded, y_train_encoded)
    logistic_model.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb_encoded = xgb_model.predict(X_test_encoded)
    y_pred_xgb = pd.Series(y_pred_xgb_encoded).map({0: -1, 1: 0, 2: 1})
    y_pred_logistic = logistic_model.predict(X_test)

    # Evaluate models
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)
    logistic_report = classification_report(y_test, y_pred_logistic, output_dict=True)

    # Collect accuracy scores
    model_results = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "Logistic Regression"],
        "Accuracy": [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_xgb), accuracy_score(y_test, y_pred_logistic)],
        "Precision (Avg)": [rf_report['weighted avg']['precision'], xgb_report['weighted avg']['precision'], logistic_report['weighted avg']['precision']],
        "Recall (Avg)": [rf_report['weighted avg']['recall'], xgb_report['weighted avg']['recall'], logistic_report['weighted avg']['recall']],
        "F1-Score (Avg)": [rf_report['weighted avg']['f1-score'], xgb_report['weighted avg']['f1-score'], logistic_report['weighted avg']['f1-score']]
    })

    # Save trained models
    joblib.dump(rf_model, f"models/{crypto}_rf_model.pkl")
    joblib.dump(xgb_model, f"models/{crypto}_xgb_model.pkl")
    joblib.dump(logistic_model, f"models/{crypto}_logistic_model.pkl")

    print(f"\nModel training complete for {crypto}. Models saved in 'models/' folder.")
    print(model_results)
