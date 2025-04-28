import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import ta
import os
import time
from datetime import datetime
from exchange_config import exchange, fetch_with_retry
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy, MeanAbsoluteError

# Define global constants
FEATURES = [
    'close', 'volume', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'asia_session', 'europe_session', 'us_session',
    'close_1h_change', 'close_4h_change', 'close_1d_change',
    'volume_1h_change', 'volume_4h_change'
]

def fetch_historical_data(timeframe='15m', since='2024-01-01'):
    """Fetch historical data with improved error handling"""
    print(f"\nFetching historical data from MEXC for {timeframe} timeframe...")
    
    try:
        # Convert date to timestamp
        since_ts = exchange.parse8601(f'{since}T00:00:00Z')
        end_ts = exchange.milliseconds()
        all_candles = []
        
        while since_ts < end_ts:
            try:
                # Use fetch_with_retry instead of direct fetch
                candles = fetch_with_retry(
                    exchange,
                    'BTC/USDT:USDT',
                    timeframe,
                    since_ts,
                    limit=1000
                )
                
                if not candles:
                    print(f"No more data available for {timeframe}")
                    break
                    
                all_candles.extend(candles)
                since_ts = candles[-1][0] + 1
                
                print(f"Fetched {len(all_candles)} candles up to {datetime.fromtimestamp(candles[-1][0]/1000)}")
                time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
                
            except Exception as e:
                print(f"Error fetching batch: {str(e)}")
                time.sleep(5)  # Wait before retrying
                continue
        
        if not all_candles:
            print(f"No data available for {timeframe}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save to CSV for caching
        filename = f'btc_usdt_{timeframe}_data.csv'
        df.to_csv(filename)
        print(f"Saved {len(df)} candles to {filename}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching {timeframe} data: {str(e)}")
        return None

def add_indicators(df):
    """Add technical indicators and features to the dataframe"""
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Session indicators
    df['asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    df['europe_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['us_session'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)
    
    # Multi-timeframe features
    df['close_1h_change'] = df['close'].pct_change(4)  # 4 15-min periods = 1 hour
    df['close_4h_change'] = df['close'].pct_change(16)  # 16 15-min periods = 4 hours
    df['close_1d_change'] = df['close'].pct_change(96)  # 96 15-min periods = 1 day
    df['volume_1h_change'] = df['volume'].pct_change(4)
    df['volume_4h_change'] = df['volume'].pct_change(16)
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    # Replace deprecated fillna with ffill and bfill
    df = df.ffill().bfill()
    
    return df

def add_labels(df, forward_window=4):
    """Add target labels for model training"""
    # Calculate future price extremes
    df['future_max'] = df['high'].rolling(forward_window).max().shift(-forward_window)
    df['future_min'] = df['low'].rolling(forward_window).min().shift(-forward_window)
    
    # Calculate ATR-based thresholds
    df['tp'] = df['close'] + (df['atr'] * 3.0)  # 3 ATR for take profit
    df['sl'] = df['close'] - (df['atr'] * 1.5)  # 1.5 ATR for stop loss
    
    # Calculate future returns and direction
    df['future_return'] = df['close'].shift(-forward_window) / df['close'] - 1
    
    # Calculate ATR-based thresholds for direction
    atr_threshold = df['atr'] / df['close']
    
    # Determine direction using vectorized operations
    df['direction'] = 'Hold'  # Default value
    df.loc[df['future_return'] > atr_threshold, 'direction'] = 'Buy'
    df.loc[df['future_return'] < -atr_threshold, 'direction'] = 'Sell'
    
    return df

def calculate_targets(df, forward_window=4):
    """Calculate optimal stop loss and take profit levels"""
    # Calculate ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Calculate future price extremes
    df['future_max'] = df['high'].rolling(forward_window).max().shift(-forward_window)
    df['future_min'] = df['low'].rolling(forward_window).min().shift(-forward_window)
    
    # Calculate ATR-based levels
    df['atr_sl'] = df['close'] - (df['atr'] * 1.5)
    df['atr_tp'] = df['close'] + (df['atr'] * 3.0)
    
    # Calculate future returns and direction
    df['future_return'] = df['close'].shift(-forward_window) / df['close'] - 1
    
    # Calculate ATR-based thresholds for direction
    atr_threshold = df['atr'] / df['close']
    
    # Determine direction using vectorized operations
    df['direction'] = 'Hold'  # Default value
    df.loc[df['future_return'] > atr_threshold, 'direction'] = 'Buy'
    df.loc[df['future_return'] < -atr_threshold, 'direction'] = 'Sell'
    
    # Calculate stop loss and take profit based on direction
    df['stop_loss'] = np.where(
        df['direction'] == 'Buy',
        np.minimum(df['future_min'], df['atr_sl']),  # For Buy: SL below current price
        np.maximum(df['future_max'], df['close'] + (df['atr'] * 1.5))  # For Sell: SL above current price
    )
    
    df['take_profit'] = np.where(
        df['direction'] == 'Buy',
        np.maximum(df['future_max'], df['atr_tp']),  # For Buy: TP above current price
        np.minimum(df['future_min'], df['close'] - (df['atr'] * 3.0))  # For Sell: TP below current price
    )
    
    # Validate and clean the data
    df['stop_loss'] = df['stop_loss'].ffill().bfill()
    df['take_profit'] = df['take_profit'].ffill().bfill()
    
    # Ensure stop loss and take profit are in correct positions relative to current price
    df['stop_loss'] = np.where(
        df['direction'] == 'Buy',
        np.minimum(df['stop_loss'], df['close'] * 0.999),  # For Buy: SL below current price
        np.maximum(df['stop_loss'], df['close'] * 1.001)   # For Sell: SL above current price
    )
    
    df['take_profit'] = np.where(
        df['direction'] == 'Buy',
        np.maximum(df['take_profit'], df['close'] * 1.001),  # For Buy: TP above current price
        np.minimum(df['take_profit'], df['close'] * 0.999)   # For Sell: TP below current price
    )
    
    # Calculate relative distances (as percentages)
    df['sl_distance'] = np.where(
        df['direction'] == 'Buy',
        (df['stop_loss'] - df['close']) / df['close'] * 100,  # For Buy: negative percentage
        (df['stop_loss'] - df['close']) / df['close'] * 100   # For Sell: positive percentage
    )
    
    df['tp_distance'] = np.where(
        df['direction'] == 'Buy',
        (df['take_profit'] - df['close']) / df['close'] * 100,  # For Buy: positive percentage
        (df['take_profit'] - df['close']) / df['close'] * 100   # For Sell: negative percentage
    )
    
    # Clip distances to reasonable ranges
    df['sl_distance'] = np.where(
        df['direction'] == 'Buy',
        df['sl_distance'].clip(-5, 0),    # For Buy: -5% to 0%
        df['sl_distance'].clip(0, 5)      # For Sell: 0% to 5%
    )
    
    df['tp_distance'] = np.where(
        df['direction'] == 'Buy',
        df['tp_distance'].clip(0, 10),    # For Buy: 0% to 10%
        df['tp_distance'].clip(-10, 0)    # For Sell: -10% to 0%
    )
    
    # Print validation information
    print("\nTarget Validation:")
    print(f"Current Price Range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print(f"Stop Loss Range: {df['stop_loss'].min():.2f} to {df['stop_loss'].max():.2f}")
    print(f"Take Profit Range: {df['take_profit'].min():.2f} to {df['take_profit'].max():.2f}")
    print(f"SL Distance Range: {df['sl_distance'].min():.2f}% to {df['sl_distance'].max():.2f}%")
    print(f"TP Distance Range: {df['tp_distance'].min():.2f}% to {df['tp_distance'].max():.2f}%")
    
    return df

def prepare_data(df, sequence_length=60):
    """Prepare data for LSTM model training"""
    # Calculate targets first (before any scaling)
    df = calculate_targets(df)
    
    # Separate scalers for features and targets
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Scale features
    features_scaled = feature_scaler.fit_transform(df[FEATURES])
    df[FEATURES] = features_scaled
    
    # Prepare targets as percentage distances
    targets = df[['sl_distance', 'tp_distance']].copy()
    
    # Validate targets before scaling
    print("\nTarget Validation Before Scaling:")
    print(f"SL Distance Range: {targets['sl_distance'].min():.2f}% to {targets['sl_distance'].max():.2f}%")
    print(f"TP Distance Range: {targets['tp_distance'].min():.2f}% to {targets['tp_distance'].max():.2f}%")
    
    # Scale targets
    targets_scaled = target_scaler.fit_transform(targets)
    
    # Print scaled ranges for debugging
    print("\nTarget Ranges After Scaling:")
    print(f"SL Distance (scaled): min={targets_scaled[:, 0].min():.4f}, max={targets_scaled[:, 0].max():.4f}")
    print(f"TP Distance (scaled): min={targets_scaled[:, 1].min():.4f}, max={targets_scaled[:, 1].max():.4f}")
    
    # Prepare sequences
    X, y_dir, y_reg = [], [], []
    le = LabelEncoder()
    df['direction_encoded'] = le.fit_transform(df['direction'])
    y_dir_cat = to_categorical(df['direction_encoded'], num_classes=3)

    for i in range(len(df) - sequence_length):
        X.append(df[FEATURES].iloc[i:i+sequence_length].values)
        y_dir.append(y_dir_cat[i + sequence_length])
        y_reg.append(targets_scaled[i + sequence_length])

    return np.array(X), np.array(y_dir), np.array(y_reg), feature_scaler, target_scaler

def build_multi_output_model(input_shape):
    """Build multi-output LSTM model for direction and price prediction"""
    inputs = Input(shape=input_shape)
    
    # Simplified LSTM architecture
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Direction branch
    direction_branch = Dense(64, activation='relu')(x)
    direction_branch = BatchNormalization()(direction_branch)
    out_dir = Dense(3, activation='softmax', name='direction')(direction_branch)
    
    # Price prediction branch
    price_branch = Dense(64, activation='relu')(x)
    price_branch = BatchNormalization()(price_branch)
    out_tp_sl = Dense(2, activation='tanh', name='tp_sl')(price_branch)
    
    model = Model(inputs=inputs, outputs=[out_dir, out_tp_sl])
    
    # Configure optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss={
            'direction': CategoricalCrossentropy(),
            'tp_sl': MeanSquaredError()
        },
        loss_weights={
            'direction': 0.3,
            'tp_sl': 0.7
        },
        metrics={
            'direction': [CategoricalAccuracy()],
            'tp_sl': [MeanAbsoluteError()]
        }
    )
    
    return model

def main():
    """Main function to train the model"""
    # Fetch and prepare data
    print("Fetching historical data...")
    df = fetch_historical_data()
    if df is None:
        print("Failed to fetch data. Exiting...")
        return
        
    print("Calculating indicators...")
    df = add_indicators(df)
    
    print("Generating labels...")
    df = add_labels(df)
    
    print("Preparing sequences...")
    X, y_dir, y_reg, feature_scaler, target_scaler = prepare_data(df)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_dir_train, y_dir_test = y_dir[:train_size], y_dir[train_size:]
    y_reg_train, y_reg_test = y_reg[:train_size], y_reg[train_size:]
    
    print("Building model...")
    model = build_multi_output_model((X.shape[1], X.shape[2]))
    
    # Create model directory if it doesn't exist
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Set up callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = os.path.join(model_dir, f'bot_model_{timestamp}.h5')
    scaler_path = os.path.join(model_dir, f'scalers_{timestamp}.pkl')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            min_delta=0.001
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    print("Training model...")
    history = model.fit(
        X_train,
        {'direction': y_dir_train, 'tp_sl': y_reg_train},
        validation_data=(X_test, {'direction': y_dir_test, 'tp_sl': y_reg_test}),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    loss = model.evaluate(X_test, {'direction': y_dir_test, 'tp_sl': y_reg_test}, verbose=0)
    print(f"\nTest Loss: {loss}")
    
    # Save scalers with current data ranges
    import joblib
    joblib.dump({
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'data_ranges': {
            'features': {col: (df[col].min(), df[col].max()) for col in FEATURES},
            'targets': {
                'stop_loss': (df['stop_loss'].min(), df['stop_loss'].max()),
                'take_profit': (df['take_profit'].min(), df['take_profit'].max())
            }
        }
    }, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scalers saved to: {scaler_path}")
    print("✅ Model training complete.")

if __name__ == "__main__":
    main() 