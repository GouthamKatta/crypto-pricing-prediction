import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

# List of cryptocurrencies
cryptos = ["BTC", "ETH", "DOGE"]

# Prediction label parameters
N = 7  # Prediction horizon (e.g., next 7 days)
X = 0.05  # Threshold for defining Uptrend/Downtrend (5%)

for crypto in cryptos:
    input_file = f"technical_indicators_{crypto}.csv"
    output_file = f"labeled_{crypto}_data.csv"

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}. Skipping {crypto}...")
        continue

    # Load dataset
    df = pd.read_csv(input_file)

    # Convert date column if available
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

    # Ensure numeric columns
    numeric_cols = ['MACD_Line', 'MACD_Signal', 'MACD_Histogram', 'RSI_14',
                    'Lowest_Low', 'Highest_High', '%K', '%D', 'OBV', 'Close']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # === Feature Engineering ===
    df['MACD_Diff'] = df['MACD_Line'] - df['MACD_Signal']
    df['MACD_Histogram_Change'] = df['MACD_Histogram'].diff()
    df['RSI_Trend'] = np.where(df['RSI_14'] > 50, 1, -1)
    df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
    df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
    df['Stoch_Momentum'] = df['%K'] - df['%D']
    df['Stoch_Trend'] = np.where(df['%K'] > df['%D'], 1, -1)
    df['Stoch_Overbought'] = (df['%K'] > 80).astype(int)
    df['Stoch_Oversold'] = (df['%K'] < 20).astype(int)
    df['Price_Range'] = df['Highest_High'] - df['Lowest_Low']
    df['Price_Momentum'] = df['Close'].diff()
    df['OBV_Change'] = df['OBV'].diff()
    df['OBV_Trend'] = np.where(df['OBV_Change'] > 0, 1, -1)

    # === Generate Multi-Class Labels (Up, Down, Sideways) ===
    df['Future_Close'] = df['Close'].shift(-N)
    df['Price_Change'] = (df['Future_Close'] - df['Close']) / df['Close']

    df['Label_Multi'] = np.where(df['Price_Change'] > X, 1,  # Uptrend
                                 np.where(df['Price_Change'] < -X, -1, 0))  # Downtrend, Sideways

    # === Normalize Features ===
    feature_cols = ['MACD_Diff', 'MACD_Histogram_Change', 'RSI_14', 'Price_Range', 'Price_Momentum',
                    'Stoch_Momentum', 'OBV_Change']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # === Undersampling for BTC Only ===
    if crypto == "BTC":
        df_sideways = df[df['Label_Multi'] == 0]
        df_up = df[df['Label_Multi'] == 1]
        df_down = df[df['Label_Multi'] == -1]

        # Balance by reducing Sideways samples to match the minority class
        df_sideways_downsampled = resample(df_sideways,
                                           replace=False,  # No oversampling, only undersampling
                                           n_samples=min(len(df_up), len(df_down)),  # Match smallest class
                                           random_state=42)

        # Combine balanced dataset
        df = pd.concat([df_up, df_down, df_sideways_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"BTC class distribution after undersampling: \n{df['Label_Multi'].value_counts()}")

    # === Final Processing & Save Dataset ===
    df.dropna(inplace=True)  # Remove NaN values caused by shifting
    df.to_csv(output_file, index=False)

    print(f"Processed and saved: {output_file}")

print("Feature engineering complete for all datasets! 🚀")
