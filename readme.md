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
