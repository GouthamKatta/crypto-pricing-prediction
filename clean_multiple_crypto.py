import pandas as pd
import os

def load_and_clean_crypto_data(file_path):
    """
    Loads a CSV file containing historical cryptocurrency data and cleans it.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A cleaned and structured DataFrame.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Rename columns for clarity
    column_mappings = {
        "Unnamed: 0": "Date",
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "Volume": "Volume"
    }
    df.rename(columns=column_mappings, inplace=True)

    # Convert 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure numerical columns are float type
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Sort by date in ascending order
    df = df.sort_values(by="Date").reset_index(drop=True)

    return df

def process_multiple_csv_files(file_paths):
    """
    Processes multiple cryptocurrency CSV files.

    Args:
        file_paths (list): List of CSV file paths.

    Returns:
        dict: Dictionary containing cleaned DataFrames for each crypto.
    """
    cleaned_data = {}

    for file_path in file_paths:
        # Extract crypto symbol from file name
        crypto_name = os.path.basename(file_path).split("_")[0]  # Extracts BTC, ETH, etc.
        
        print(f"Processing {crypto_name} data...")

        # Load and clean data
        df = load_and_clean_crypto_data(file_path)

        # Save cleaned data to a new CSV file
        cleaned_file_path = f"cleaned_{crypto_name}_data.csv"
        df.to_csv(cleaned_file_path, index=False)

        print(f"Cleaned data saved to {cleaned_file_path}")

        # Store in dictionary
        cleaned_data[crypto_name] = df

    return cleaned_data

# Example usage:
file_paths = [
    "/Users/goutham/Desktop/TH project Fin/BTC_historical_data.csv",
    "/Users/goutham/Desktop/TH project Fin/ETH_historical_data.csv",
    "/Users/goutham/Desktop/TH project Fin/DOGE_historical_data.csv"
]

cleaned_data_dict = process_multiple_csv_files(file_paths)
