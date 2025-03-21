import pandas as pd
import matplotlib.pyplot as plt
import os

# List of indicator files
file_paths = [
    "technical_indicators_BTC.csv",
    "technical_indicators_ETH.csv",
    "technical_indicators_DOGE.csv"
]

def compute_crossovers(df):
    # Bullish crossover when MA_50 crosses above MA_200
    df['Bullish_Crossover'] = (df['MA_50'] > df['MA_200']) & (df['MA_50'].shift(1) <= df['MA_200'].shift(1))
    # Bearish crossover when MA_50 crosses below MA_200
    df['Bearish_Crossover'] = (df['MA_50'] < df['MA_200']) & (df['MA_50'].shift(1) >= df['MA_200'].shift(1))
    return df

def plot_price_volume_chart(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping...")
        return

    df = pd.read_csv(file_path)

    required_columns = ["Date", "Close", "MA_50", "MA_200", "Volume"]
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing '{column}' in {file_path}. Skipping...")
            return
    
    df["Date"] = pd.to_datetime(df["Date"])

    # Compute crossover indicators if missing
    df = compute_crossovers(df)

    crypto_name = file_path.split("_")[-1].split(".")[0]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(df["Date"], df["Close"], label="Closing Price", color="blue", linewidth=1.5)
    ax1.plot(df["Date"], df["MA_50"], label="50-Day MA", color="orange", linestyle="dashed")
    ax1.plot(df["Date"], df["MA_200"], label="200-Day MA", color="red", linestyle="dashed")

    bullish_crossovers = df[df["Bullish_Crossover"]]
    bearish_crossovers = df[df["Bearish_Crossover"]]

    ax1.scatter(bullish_crossovers["Date"], bullish_crossovers["Close"],
                label="Bullish Crossover", color="green", marker="^", s=100)
    ax1.scatter(bearish_crossovers["Date"], bearish_crossovers["Close"],
                label="Bearish Crossover", color="red", marker="v", s=100)

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.set_title(f"{crypto_name.upper()} - Daily Price with Technical Indicators", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(df["Date"], df["Volume"], alpha=0.3, label="Trading Volume", color="gray")
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.legend(loc="upper right")

    plt.xticks(rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    output_image = f"chart_{crypto_name}.png"
    plt.savefig(output_image, dpi=300)
    print(f"Saved chart to {output_image}")
    plt.close()

# Loop through each file and generate/save its chart
for file_path in file_paths:
    plot_price_volume_chart(file_path)
 
