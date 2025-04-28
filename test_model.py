import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
from exchange_config import exchange
import ta
import os

# Define global constants
FEATURES = [
    'close', 'volume', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'asia_session', 'europe_session', 'us_session',
    'close_1h_change', 'close_4h_change', 'close_1d_change',
    'volume_1h_change', 'volume_4h_change'
]

def fetch_test_data(exchange, symbol='BTC/USDT:USDT', timeframe='15m', lookback=60):
    """Fetch recent data for testing"""
    print(f"Fetching test data from MEXC for {timeframe} timeframe...")
    try:
        # Calculate start time (need more data for indicators)
        end_ts = exchange.milliseconds()
        # Fetch 200 candles to ensure we have enough data for all indicators
        # (96 for daily changes, 20 for Bollinger Bands, etc.)
        start_ts = end_ts - (200 * 15 * 60 * 1000)  # 15 minutes per candle
        
        candles = exchange.fetch_ohlcv(symbol, timeframe, start_ts, limit=200)
        
        if candles:
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            print(f"Fetched {len(df)} historical candles")
            return df
        else:
            print("No test data available")
            return None
            
    except Exception as e:
        print(f"Error fetching test data: {str(e)}")
        return None

def add_indicators(df):
    """Add technical indicators and features to the dataframe"""
    # Store original volume for later use
    original_volume = df['volume'].copy()
    
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
    
    # Volume changes - use simple percentage change like price changes
    df['volume_1h_change'] = df['volume'].pct_change(4)  # 4 15-min periods = 1 hour
    df['volume_4h_change'] = df['volume'].pct_change(16)  # 16 15-min periods = 4 hours
    
    # Price changes with proper normalization
    df['close_1h_change'] = df['close'].pct_change(4).clip(-0.1, 0.1)  # Max 10% change
    df['close_4h_change'] = df['close'].pct_change(16).clip(-0.2, 0.2)  # Max 20% change
    df['close_1d_change'] = df['close'].pct_change(96).clip(-0.3, 0.3)  # Max 30% change
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate defaults
    df['volume_1h_change'] = df['volume_1h_change'].fillna(0)
    df['volume_4h_change'] = df['volume_4h_change'].fillna(0)
    df['close_1h_change'] = df['close_1h_change'].fillna(0)
    df['close_4h_change'] = df['close_4h_change'].fillna(0)
    df['close_1d_change'] = df['close_1d_change'].fillna(0)
    
    # Fill remaining NaN values
    df = df.ffill().bfill()
    
    # Print validation information
    print("\nIndicator Validation (before scaling):")
    for col in ['volume_1h_change', 'volume_4h_change', 'close_1h_change', 'close_4h_change', 'close_1d_change']:
        if col in df.columns:
            print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
    
    return df

def prepare_test_data(df, feature_scaler, lookback=60):
    """Prepare test data for prediction using the same logic as training"""
    # Create a copy of the dataframe to avoid modifying the original
    df_scaled = df.copy()
    
    # Scale features
    try:
        # First, ensure all features are in the correct order
        df_scaled = df_scaled[FEATURES]
        
        # Scale all features together
        df_scaled[FEATURES] = pd.DataFrame(
            feature_scaler.fit_transform(df_scaled[FEATURES]),
            columns=FEATURES,
            index=df_scaled.index
        )
        
        # Print validation information
        print("\nFeature ranges after scaling:")
        for feature in FEATURES:
            print(f"{feature}: min={df_scaled[feature].min():.4f}, max={df_scaled[feature].max():.4f}, mean={df_scaled[feature].mean():.4f}")
        
    except Exception as e:
        print(f"\nError in feature scaling: {str(e)}")
        return None, df
    
    # Prepare sequence
    X = df_scaled.values[-lookback:].reshape(1, lookback, len(FEATURES))
    
    # Print input data details
    print("\nInput Data Details:")
    print(f"Input shape: {X.shape}")
    print(f"Number of features: {len(FEATURES)}")
    
    print("\nSample of first sequence:")
    print("First 5 timesteps of first 5 features:")
    print(pd.DataFrame(X[0, :5, :5], columns=FEATURES[:5]))
    
    return X, df

def interpret_prediction(direction_pred, tp_sl_pred, target_scaler, current_price, df):
    """Interpret model predictions"""
    try:
        # Debug raw predictions
        print("\nRaw Predictions:")
        print(f"Direction Prediction Shape: {direction_pred.shape}")
        print(f"TP/SL Prediction Shape: {tp_sl_pred.shape}")
        print(f"Direction Prediction: {direction_pred}")
        print(f"TP/SL Prediction: {tp_sl_pred}")
        
        # Get direction probabilities
        direction_probs = direction_pred[0]
        if np.any(np.isnan(direction_probs)):
            print("Warning: NaN values in direction predictions")
            direction_probs = np.array([0.33, 0.33, 0.34])  # Default to equal probabilities
        
        direction = ['Buy', 'Sell', 'Hold'][np.argmax(direction_probs)]
        direction_confidence = float(np.max(direction_probs))
        
        # Ensure direction_confidence is a valid number
        if np.isnan(direction_confidence):
            direction_confidence = 0.5
        
        # Debug information for target scaler
        print("\nTarget Scaler Debug Info:")
        print(f"Target scaler data range: {target_scaler.data_range_}")
        print(f"Target scaler data min: {target_scaler.data_min_}")
        print(f"Target scaler data max: {target_scaler.data_max_}")
        
        # Get actual SL and TP predictions from the model
        try:
            # Validate predictions before inverse transform
            if np.any(np.isnan(tp_sl_pred)):
                raise ValueError("NaN values in predictions before inverse transform")
            
            # Inverse transform the predictions
            sl_tp_pred = target_scaler.inverse_transform(tp_sl_pred)
            
            # Convert percentage distances to absolute prices based on direction
            if direction == 'Buy':
                sl_distance = sl_tp_pred[0][0]  # Negative percentage for Buy
                tp_distance = sl_tp_pred[0][1]  # Positive percentage for Buy
                stop_loss = current_price * (1 + sl_distance/100)  # SL below current price
                take_profit = current_price * (1 + tp_distance/100)  # TP above current price
            elif direction == 'Sell':
                sl_distance = sl_tp_pred[0][0]  # Positive percentage for Sell
                tp_distance = sl_tp_pred[0][1]  # Negative percentage for Sell
                stop_loss = current_price * (1 + sl_distance/100)  # SL above current price
                take_profit = current_price * (1 + tp_distance/100)  # TP below current price
            else:  # Hold
                # Use ATR-based calculations for Hold
                atr = df['atr'].iloc[-1]
                if np.isnan(atr) or atr == 0:
                    atr = current_price * 0.01  # Fallback to 1% of price if ATR is invalid
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 3.0)
            
            # Print prediction details
            print("\nPrediction Details:")
            print(f"SL Distance: {sl_distance:.2f}%")
            print(f"TP Distance: {tp_distance:.2f}%")
            print(f"Calculated SL: {stop_loss:.2f}")
            print(f"Calculated TP: {take_profit:.2f}")
                
        except Exception as e:
            print(f"\nError inverting target scaler: {str(e)}")
            print("Falling back to ATR-based calculations...")
            
            # Fallback to ATR-based calculations
            atr = df['atr'].iloc[-1]
            if np.isnan(atr) or atr == 0:
                atr = current_price * 0.01  # Fallback to 1% of price if ATR is invalid
            
            if direction == 'Buy':
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 3.0)
            elif direction == 'Sell':
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 3.0)
            else:  # Hold
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 3.0)
        
        # Calculate risk/reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk != 0 else 0
        
        return {
            'direction': direction,
            'direction_confidence': direction_confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward
        }
    except Exception as e:
        print(f"\nError in prediction interpretation: {str(e)}")
        return None

def main():
    # Load the latest model and scalers
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('bot_model_') and f.endswith('.h5')]
    scaler_files = [f for f in os.listdir(model_dir) if f.startswith('scalers_') and f.endswith('.pkl')]
    
    if not model_files or not scaler_files:
        print("No model or scaler files found. Please train the model first.")
        return
    
    latest_model = sorted(model_files)[-1]
    latest_scaler = sorted(scaler_files)[-1]
    
    try:
        # Load model with custom objects
        model = load_model(
            os.path.join(model_dir, latest_model),
            custom_objects={
                'CategoricalCrossentropy': CategoricalCrossentropy,
                'MeanSquaredError': MeanSquaredError,
                'CategoricalAccuracy': CategoricalAccuracy,
                'MeanAbsoluteError': MeanAbsoluteError,
                'Adam': Adam
            }
        )
        
        # Print model input shape
        print("\nModel Input Shape:")
        print(f"Expected input shape: {model.input_shape}")
        
        # Load scalers
        scalers = joblib.load(os.path.join(model_dir, latest_scaler))
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        
        # Fetch test data
        df = fetch_test_data(exchange)
        if df is None:
            return
        
        # Add indicators
        df = add_indicators(df)
        
        # Prepare test data
        X, df_original = prepare_test_data(df, feature_scaler)
        if X is None:
            return
        
        # Verify input shape matches model expectations
        if X.shape[1:] != model.input_shape[1:]:
            print(f"\nError: Input shape {X.shape[1:]} does not match model's expected shape {model.input_shape[1:]}")
            return
        
        # Make prediction
        print("\nMaking prediction...")
        direction_pred, tp_sl_pred = model.predict(X)
        
        # Print prediction details
        print("\nPrediction Output Details:")
        print(f"Direction prediction shape: {direction_pred.shape}")
        print(f"TP/SL prediction shape: {tp_sl_pred.shape}")
        print(f"Direction prediction: {direction_pred}")
        print(f"TP/SL prediction: {tp_sl_pred}")
        
        # Interpret prediction
        current_price = df_original['close'].iloc[-1]
        result = interpret_prediction(direction_pred, tp_sl_pred, target_scaler, current_price, df_original)
        
        if result is None:
            return
        
        # Print results
        print("\nModel Prediction Results:")
        print(f"Current Price: {current_price:.2f}")
        print(f"Direction: {result['direction']} (Confidence: {result['direction_confidence']:.2%})")
        print(f"Stop Loss: {result['stop_loss']:.2f}")
        print(f"Take Profit: {result['take_profit']:.2f}")
        print(f"Risk/Reward Ratio: {result['risk_reward_ratio']:.2f}")
        
        # Print additional context
        print("\nMarket Context:")
        print(f"RSI: {df_original['rsi'].iloc[-1]:.2f}")
        print(f"ATR: {df_original['atr'].iloc[-1]:.2f}")
        print(f"MACD Histogram: {df_original['macd_hist'].iloc[-1]:.2f}")
        print(f"Bollinger Width: {df_original['bb_width'].iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"Error loading or using model: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have the latest version of TensorFlow installed")
        print("2. Try retraining the model with the latest version")
        print("3. Check if the model file is not corrupted")
        print("4. Verify that the scalers file matches the model version")

if __name__ == "__main__":
    main() 