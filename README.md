# Crypto Price Movement Prediction API

## Project Overview
This project predicts cryptocurrency price movements using Machine Learning models.  
It supports **BTC**, **ETH**, and **DOGE**, and predicts if the price will go **Uptrend**, **Downtrend**, or **Sideways**.

## Features
- Fetch historical crypto data (Alpha Vantage API)
- Clean and preprocess the data
- Generate technical indicators (MA, EMA, RSI, MACD, OBV, etc.)
- Train classification models (Random Forest, XGBoost, Logistic Regression)
- Serve predictions via a Flask API
- Fully Dockerized for easy setup and deployment

## Setup Instructions

### Run Locally with Python
```bash
pip install -r requirements.txt
python api.py
```

### Run with Docker
```bash
docker build -t crypto-api .
docker run -p 5003:5003 crypto-api
```

## API Usage (Postman)
**POST to**: `http://localhost:5003/predict`

### Sample JSON Body:
```json
{
  "crypto": "ETH",
  "features": {
    "MACD_Diff": 0.02,
    "MACD_Histogram_Change": -0.01,
    "RSI_14": 55.3,
    "Price_Range": 120.5,
    "Price_Momentum": 2.3,
    "Stoch_Momentum": 1.5,
    "OBV_Change": 30000,
    "Close_Lag_1": 1875.23,
    "Close_Lag_2": 1868.40,
    "Close_Lag_3": 1855.78
  }
}
```

### Sample Response:
```json
{
  "crypto": "ETH",
  "predictions": {
    "LogisticRegression": "Downtrend",
    "RandomForest": "Sideways",
    "XGBoost": "Uptrend"
  }
}
```

## Folder Structure
- `api.py`: Flask API
- `fetch_cryptodata_py`: Fetches historical data
- `clean_multiple_crypto.py`: Cleans raw CSVs
- `indicators.py`: Computes technical indicators
- `crypto_data_labeler.py`: Adds features & labels
- `multi_crypto_model_trainer.py`: Trains models
- `Dockerfile`: Docker config

## Author
**Goutham Katta**  
Done for Fun

## License
You can use this however you want, just keep my name on it, and don’t sue me if it blows up. 
