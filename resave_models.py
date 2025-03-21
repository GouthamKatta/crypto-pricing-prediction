import joblib
import xgboost as xgb

# Define models and paths
cryptos = ["BTC", "ETH", "DOGE"]
for crypto in cryptos:
    model_path = f"models/{crypto}_xgb_model.pkl"

    # Load the old model
    model = joblib.load(model_path)

    # Save it in XGBoost's native format
    model.get_booster().save_model(f"models/{crypto}_xgb_model.json")

    print(f"Re-saved {crypto} XGBoost model as JSON format.")

print("✅ All XGBoost models updated successfully!")
