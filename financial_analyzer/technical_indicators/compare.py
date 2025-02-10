import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the Tesla dataset
dataset = load_dataset('tsla-historic-prices')

# Convert the dataset into a DataFrame
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column to datetime format
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the index
tesla_df.set_index('Date', inplace=True)

# Calculate the 20-day Simple Moving Average
tesla_df['SMA_20'] = tesla_df['Close'].rolling(window=20).mean()

# Calculate the 20-day Exponential Moving Average
tesla_df['EMA_20'] = tesla_df['Close'].ewm(span=20, adjust=False).mean()

# Using a smaller date range for better visualization
tesla_df_small = tesla_df.loc['2018']

# Plotting
tesla_df_small[['Close', 'SMA_20', 'EMA_20']].plot(figsize=(12, 6), title="TSLA Close Price, SMA 20, and EMA 20 (2018)")
plt.show()