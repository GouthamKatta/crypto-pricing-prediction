from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define available cryptos and models
cryptos = ["BTC", "ETH", "DOGE"]
models = {}

# Load trained models
for crypto in cryptos:
    models[crypto] = {
        "RandomForest": joblib.load(f"models/{crypto}_rf_model.pkl"),
        "XGBoost": joblib.load(f"models/{crypto}_xgb_model.pkl"),
        "LogisticRegression": joblib.load(f"models/{crypto}_logistic_model.pkl")
    }

# Define features (must match those used in training)
features = ['MACD_Diff', 'MACD_Histogram_Change', 'RSI_14', 'Price_Range', 
            'Price_Momentum', 'Stoch_Momentum', 'OBV_Change']

# Add additional features for BTC and ETH
btc_features = features + ['Close_STD_7']
eth_features = features + ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint for predicting stock movement.
    Expects a JSON request with 'crypto' and feature values.
    """
    data = request.get_json()

    # Validate crypto input
    crypto = data.get("crypto")
    if crypto not in cryptos:
        return jsonify({"error": "Invalid crypto ticker. Use BTC, ETH, or DOGE."}), 400

    # Extract feature values
    input_features = data.get("features")
    if not input_features or not isinstance(input_features, dict):
        return jsonify({"error": "Missing or invalid 'features' field in request."}), 400

    # Match features with the right set
    selected_features = features
    if crypto == "BTC":
        selected_features = btc_features
    elif crypto == "ETH":
        selected_features = eth_features

    # Ensure all required features are provided
    missing_features = [f for f in selected_features if f not in input_features]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400

    # Convert features into DataFrame
    try:
        input_df = pd.DataFrame([input_features], columns=selected_features)
    except Exception as e:
        return jsonify({"error": f"Error processing input data: {str(e)}"}), 400

    # Make predictions using all models
    predictions = {}
    for model_name, model in models[crypto].items():
        pred_label = model.predict(input_df)[0]
        movement = "Uptrend" if pred_label == 1 else "Downtrend" if pred_label == -1 else "Sideways"
        predictions[model_name] = movement

    # Return response
    return jsonify({
        "crypto": crypto,
        "predictions": predictions
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
