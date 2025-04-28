import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import ccxt
import time
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras

# Initialize exchange
exchange = ccxt.mexc({
    'apiKey': '###',
    'secret': '###', # Replace with your actual secret key
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True,
        'defaultNetwork': 'MAINNET',
        'recvWindow': 60000,
        'createMarketBuyOrderRequiresPrice': False
    },
    'timeout': 30000
})

# Function to prepare data
def prepare_data(df, sequence_length=60):
    # Ensure no NaN values
    df = df.dropna()
    
    X, y = [], []
    features = ['close', 'volume', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist']
    
    # Normalize features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Prepare stop_loss and take_profit
    df['stop_loss'] = df['close'] - 1.5 * df['atr']
    df['take_profit'] = df['close'] + 3 * df['atr']
    
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length][features].values)
        y.append(df.iloc[i+sequence_length][['stop_loss', 'take_profit']].values)
    
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(2, activation='relu'))  # Output layer for stop loss and take profit
    model.compile(optimizer='adam', loss='mse')
    return model

# Fetch data function
def fetch_data(exchange, symbol='BTC/USDT:USDT', timeframe='15m', since='2018-01-01'):
    since_ts = exchange.parse8601(f'{since}T00:00:00Z')
    candles = []
    print("Starting data fetch...")
    end_ts = exchange.milliseconds()  # Current time
    
    while True:
        try:
            new_candles = exchange.fetch_ohlcv(symbol, timeframe, since_ts, limit=1000)
            if not new_candles or new_candles[-1][0] >= end_ts:  # Stop at current time
                print("No new candles fetched, exiting loop.")
                break
            candles.extend(new_candles)
            since_ts = new_candles[-1][0] + 1
            print(f"Fetched {len(new_candles)} new candles, total: {len(candles)}")
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
        except ccxt.NetworkError as e:
            print(f"Network error: {str(e)}")
            break
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {str(e)}")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            break
    if not candles:
        print("No data fetched. Please check your API credentials and network connection.")
    else:
        print(f"Total candles fetched: {len(candles)}")
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Load the dataset from the provided path
df = pd.read_csv('C:/python/btc_usd_historical_data.csv')

# Convert the timestamp to datetime
df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Set the timestamp as the index
df.set_index('timestamp', inplace=True)

# Filter data from January 1st, 2018, to today
df = df[df.index >= '2018-01-01']

# Resample to 15-minute intervals if needed
df = df.resample('15min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Rename columns to match your existing script
df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Recalculate indicators
atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
df['atr'] = atr_indicator.average_true_range()

rsi_indicator = RSIIndicator(df['close'], window=14)
df['rsi'] = rsi_indicator.rsi()

macd_indicator = MACD(df['close'])
df['macd'] = macd_indicator.macd()
df['macd_signal'] = macd_indicator.macd_signal()
df['macd_hist'] = macd_indicator.macd_diff()

stoch_indicator = StochasticOscillator(df['high'], df['low'], df['close'], window=14)
df['stoch_k'] = stoch_indicator.stoch()
df['stoch_d'] = stoch_indicator.stoch_signal()

# Handle NaN values in indicators
df.ffill(inplace=True)
df.bfill(inplace=True)  # Backfill as an additional step

# Check for NaN values in indicators
print("NaN values in indicators after filling:")
print(df[['atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d']].isna().sum())

# Normalize input features
scaler = MinMaxScaler()

# Select features to normalize
features_to_normalize = ['close', 'volume', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d']
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Check for NaN values after normalization
print("NaN values after normalization:")
print(df[features_to_normalize].isna().sum())

# Set stop_loss and take_profit based on ATR
# Example: stop_loss = entry_price - 1.5 * ATR, take_profit = entry_price + 3 * ATR
# For simplicity, assume entry_price is the close price
df['stop_loss'] = df['close'] - 1.5 * df['atr']
df['take_profit'] = df['close'] + 3 * df['atr']

# Function to calculate indicators
def calculate_indicators(df):
    # Check if DataFrame is empty or None
    if df is None or len(df) == 0:
        print("Error: Empty DataFrame passed to calculate_indicators")
        return None

    try:
        # Ensure the DataFrame has the required columns
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing columns {missing_columns}")
            print("Available columns:", df.columns)
            return None

        # Calculate ATR
        atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()

        # Calculate RSI
        rsi_indicator = RSIIndicator(df['close'], window=14)
        df['rsi'] = rsi_indicator.rsi()

        # Calculate MACD
        macd_indicator = MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        # Calculate Bollinger Bands
        bb_indicator = BollingerBands(df['close'], window=20)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()

        return df

    except Exception as e:
        print(f"Error in calculate_indicators: {str(e)}")
        return None

# After fetching data, calculate indicators
# Assuming df is the DataFrame with the fetched data
df = calculate_indicators(df)

# Prepare data for LSTM model
X, y = prepare_data(df)

# Check for NaN values in X and y
print("Checking for NaN values in X and y after preprocessing...")
print("NaN values in X:", np.isnan(X).sum())
print("NaN values in y:", np.isnan(y).sum())

# If NaN values are found, print a sample of the data
if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
    print("Sample X data with NaN:", X[np.isnan(X).any(axis=(1, 2))])
    print("Sample y data with NaN:", y[np.isnan(y).any(axis=1)])

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # 0.33 * 0.3 = 0.1

# Build and train the model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
# Save the model with a new name to include the new features
model.save('C:/python/model_checkpoints/lstm_models/lstm_stop_loss_take_profit_model_with_BBBands.h5') 

# Print system time and timestamp being sent
print(f"System time: {time.time()} | Since timestamp: {exchange.parse8601('2018-01-01T00:00:00Z')}") 

def calculate_buy_hold_return(exchange, start_date):
    # Placeholder implementation
    print("Calculating buy & hold return...")
    # Example: Calculate the return from buying at the start date and holding until now
    # Fetch the starting price
    start_price = exchange.fetch_ohlcv('BTC/USDT:USDT', '1d', exchange.parse8601(f'{start_date}T00:00:00Z'), limit=1)[0][4]
    # Fetch the current price
    current_price = exchange.fetch_ticker('BTC/USDT:USDT')['last']
    # Calculate the return
    buy_hold_return = ((current_price - start_price) / start_price) * 100
    return buy_hold_return

def fetch_multi_timeframe_data(exchange, symbol='BTC/USDT:USDT', since='2018-01-01'):
    print("Fetching multi-timeframe data...")
    
    # Fallback to CSV data if exchange data fetch fails
    try:
        # Try to load data from CSV if no exchange data
        df = pd.read_csv('C:/python/btc_usd_historical_data.csv')
        
        # Convert the timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        # Filter data from January 1st, 2018, to today
        df = df[df.index >= since]

        # Resample to 15-minute intervals if needed
        df = df.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Rename columns to match your existing script
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        print(f"Loaded {len(df)} rows from CSV")
        return df

    except Exception as e:
        print(f"Error fetching multi-timeframe data: {str(e)}")
        return None

def backtest_strategy(data, buy_hold_return):
    # Placeholder implementation
    print("Running backtest strategy...")
    trades = []  # List to store trade details
    final_balance = 10000  # Example starting balance
    # Implement your strategy logic
    return trades, final_balance

def main():
    try:
        # Initialize exchange
        exchange = ccxt.mexc({
            'apiKey': 'mx0vgl6ZLbK69Qf7FZ',
            'secret': 'd81e9bfc0dab489390a110ea1d68a23d',
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
                'defaultNetwork': 'MAINNET',
                'recvWindow': 60000,
                'createMarketBuyOrderRequiresPrice': False
            },
            'timeout': 30000
        })
        
        # Test authentication and load markets
        try:
            print("Testing API connection...")
            exchange.check_required_credentials()
            print("API credentials verified")
            
            # Load markets with specific type
            exchange.load_markets(True)  # Force reload markets
            symbol = 'BTC/USDT:USDT'  # MEXC futures format
            if symbol not in exchange.markets:
                print(f"Market {symbol} not found. Available markets:", list(exchange.markets.keys())[:5])
                return
            print(f"Successfully loaded {symbol} market")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        # Delete old cache files to force fresh data
        for tf in ['15m', '1h', '4h']:
            cache_file = f'btc_usdt_{tf}_data.csv'
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Removed old cache file: {cache_file}")
        
        start_date = '2018-01-01'  # Update start date to 2018
        print(f"\nFetching data from {start_date} to present...")
        
        # Calculate buy & hold return first
        buy_hold_return = calculate_buy_hold_return(exchange, start_date)
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        
        # Fetch trading data and run backtest
        data = fetch_multi_timeframe_data(exchange, symbol='BTC/USDT:USDT', since=start_date)
        
        # Recalculate indicators on the data
        if data is not None:
            data = calculate_indicators(data)
            
            if data is None:
                print("Failed to calculate indicators")
                return
            
            trades, final_balance = backtest_strategy(data, buy_hold_return=buy_hold_return)
        else:
            print("No data available for processing")
            return
        
        # Verify the model path exists
        model_path = r'C:\python\model_checkpoints\lstm_models\lstm_stop_loss_take_profit_model_with_BBBands.h5'
        
        # Add explicit path checking
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Available files in the directory:")
            print(os.listdir(os.path.dirname(model_path)))
            return

        # Use explicit keras import for loading
        print(f"Loading model from: {model_path}")
        lstm_model = keras.models.load_model(model_path)

        # Prepare data for LSTM model
        features = ['close', 'volume', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist']
        X = data[features].values

        # Normalize the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Reshape data for LSTM input to match original model
        X_reshaped = np.array([X_scaled[i:i+60] for i in range(len(X_scaled)-60)])

        # Make predictions
        predictions = lstm_model.predict(X_reshaped)

        # Print the first few predictions for demonstration
        print("Sample predictions:", predictions[:5])

        # Integrate predictions into the backtest strategy
        # This will require modifying the entry/exit logic to use predictions as part of the decision-making process
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()