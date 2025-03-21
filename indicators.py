import pandas as pd
import numpy as np
import os

def compute_technical_indicators(df):
    """
    Computes various technical indicators for crypto price analysis.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    # Ensure numerical columns are of float type
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    
    # Simple Moving Averages
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands (20-day SMA ± 2*std dev)
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(window=20).std()
    
    # MACD Indicator
    df["MACD_Line"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD_Line"] - df["MACD_Signal"]
    
    # RSI (Relative Strength Index, 14-day)
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (%K, %D)
    df["Lowest_Low"] = df["Low"].rolling(window=14).min()
    df["Highest_High"] = df["High"].rolling(window=14).max()
    df["%K"] = ((df["Close"] - df["Lowest_Low"]) / (df["Highest_High"] - df["Lowest_Low"])) * 100
    df["%D"] = df["%K"].rolling(window=3).mean()
    
    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    
    return df

# Process all cleaned crypto data files
crypto_files = ["cleaned_BTC_data.csv", "cleaned_ETH_data.csv", "cleaned_DOGE_data.csv"]

for file_path in crypto_files:
    if os.path.exists(file_path):
        crypto_name = file_path.split("_")[1]  # Extracts BTC, ETH, DOGE
        print(f"Processing {crypto_name} data...")
        df = pd.read_csv(file_path)
        df = compute_technical_indicators(df)
        output_file = f"technical_indicators_{crypto_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Technical indicators computed and saved to {output_file}")
    else:
        print(f"File not found: {file_path}")
