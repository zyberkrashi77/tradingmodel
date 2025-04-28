import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import ta
import os
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from btc_futures import backtest_strategy
from exchange_config import exchange, load_config

# Load configuration
config = load_config()

# Constants from config
SYMBOL = config['SYMBOL']
TIMEFRAME = config['TIMEFRAME']
START_DATE = config['START_DATE']
INITIAL_BALANCE = config['INITIAL_BALANCE']
RISK_PER_TRADE = config['RISK_PER_TRADE']
LEVERAGE = config['LEVERAGE']
MAKER_FEE = config['MAKER_FEE']
TAKER_FEE = config['TAKER_FEE']
MODEL_PATH = config['MODEL_PATH']
LOG_LEVEL = config['LOG_LEVEL']
LOG_FILE = config['LOG_FILE']

print("\nInitializing MEXC exchange...")
# Exchange is already initialized from exchange_config.py
print("Exchange instance created")

def fetch_multi_timeframe_data(exchange, symbol='BTC/USDT:USDT', timeframes=['15m', '1h', '4h'], since='2024-01-01'):
    """Fetch data for multiple timeframes from MEXC"""
    data = {}
    
    print("Fetching historical data from MEXC...")
    for tf in timeframes:
        try:
            # Fetch data from MEXC
            since_ts = exchange.parse8601(f'{since}T00:00:00Z')
            candles = []
            end_ts = exchange.milliseconds()  # Current time
            
            while True:
                try:
                    new_candles = exchange.fetch_ohlcv(symbol, tf, since_ts, limit=1000)
                    if not new_candles or new_candles[-1][0] >= end_ts:
                        break
                    candles.extend(new_candles)
                    since_ts = new_candles[-1][0] + 1
                    print(f"\rFetched {len(candles)} {tf} candles", end='')
                    time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
                except Exception as e:
                    print(f"\nError fetching {tf} data: {str(e)}")
                    break
            
            if candles:
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                data[tf] = df
                print(f"\nTotal {tf} candles: {len(df)}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
            else:
                print(f"No data available for {tf}")
                
        except Exception as e:
            print(f"Error processing {tf} data: {str(e)}")
            return None
    
    return data

def add_indicators(df):
    """Add technical indicators"""
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
    
    # Add ADX
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    
    return df

def calculate_running_drawdown(balance, max_balance):
    """Calculate current drawdown"""
    if max_balance > 0:
        return ((max_balance - balance) / max_balance) * 100
    return 0

def calculate_position_size(balance, current_price, stop_distance, risk_per_trade, leverage):
    """Calculate position size with proper risk management"""
    # Calculate risk amount (e.g., 2% of balance)
    risk_amount = balance * risk_per_trade
    
    # Calculate maximum position size based on leverage
    max_position_value = balance * leverage
    max_position_size = max_position_value / current_price
    
    # Calculate position size based on risk (without double leverage)
    risk_based_size = risk_amount / stop_distance  # Removed extra leverage multiplication
    
    # Use the smaller of the two sizes
    position_size = min(risk_based_size, max_position_size)
    
    return position_size

def save_trade_analysis(trades_df, initial_balance, final_balance, filename=None):
    """Save detailed trade analysis to Excel with enhanced tracking"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'trade_analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Trade Details Sheet with enhanced tracking
        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['date'] = trades_df['datetime'].dt.date
        trades_df['time'] = trades_df['datetime'].dt.time
        
        # Initialize tracking columns
        detailed_trades = []
        running_balance = initial_balance
        entry_trade = None
        
        for _, trade in trades_df.iterrows():
            if trade['action'] in ['buy', 'sell']:
                entry_trade = trade
                # Calculate entry costs
                position_value = trade['position_value']
                entry_fee = trade['fees']
                running_balance -= entry_fee
                
                detailed_trades.append({
                    'timestamp': trade['timestamp'],
                    'action': trade['action'],
                    'price': trade['price'],
                    'position_size': trade['size'],
                    'position_value': position_value,
                    'leverage_used': trade['leverage_used'],
                    'entry_fee': entry_fee,
                    'running_balance': running_balance,
                    'trade_type': 'entry',
                    'balance_change': -entry_fee,
                    'cumulative_pnl': None,
                    'trade_pnl': None,
                    'total_fees': entry_fee
                })
                
            elif trade['action'] == 'exit' and entry_trade is not None:
                # Calculate exit details
                exit_fee = trade['total_fees'] - entry_trade['fees']
                price_change_pct = (trade['price'] - entry_trade['price']) / entry_trade['price'] * 100
                position_value = trade['position_value']
                direction = 1 if entry_trade['action'] == 'buy' else -1
                
                # Calculate PnL components
                raw_pnl = position_value * (price_change_pct/100) * direction * trade['leverage_used']
                total_fees = trade['total_fees']
                net_pnl = raw_pnl - total_fees
                
                running_balance += net_pnl
                
                detailed_trades.append({
                    'timestamp': trade['timestamp'],
                    'action': 'exit',
                    'price': trade['price'],
                    'position_size': entry_trade['size'],
                    'position_value': position_value,
                    'leverage_used': trade['leverage_used'],
                    'entry_price': entry_trade['price'],
                    'price_change_pct': price_change_pct,
                    'raw_pnl': raw_pnl,
                    'exit_fee': exit_fee,
                    'total_fees': total_fees,
                    'net_pnl': net_pnl,
                    'running_balance': running_balance,
                    'trade_type': 'exit',
                    'balance_change': net_pnl,
                    'cumulative_pnl': running_balance - initial_balance,
                    'trade_duration': trade['trade_duration'],
                    'exit_reason': trade['exit_reason']
                })
                
                entry_trade = None
        
        # Convert to DataFrame and save
        detailed_df = pd.DataFrame(detailed_trades)
        
        # Calculate additional metrics
        if not detailed_df.empty:
            detailed_df['drawdown'] = (detailed_df['running_balance'].cummax() - detailed_df['running_balance']) / detailed_df['running_balance'].cummax() * 100
            detailed_df['equity_curve'] = detailed_df['running_balance'] / initial_balance
            
            # Verify calculations
            detailed_df['expected_balance'] = initial_balance + detailed_df['balance_change'].cumsum()
            detailed_df['balance_mismatch'] = abs(detailed_df['running_balance'] - detailed_df['expected_balance'])
            
            # Save main trade details
            trade_columns = [
                'timestamp', 'action', 'price', 'position_size', 'position_value',
                'leverage_used', 'entry_price', 'price_change_pct', 'raw_pnl',
                'total_fees', 'net_pnl', 'running_balance', 'drawdown',
                'equity_curve', 'trade_duration', 'exit_reason'
            ]
            detailed_df[trade_columns].to_excel(writer, sheet_name='Trade_Details', index=False)
            
            # Save verification details
            verification_columns = [
                'timestamp', 'action', 'balance_change', 'expected_balance',
                'running_balance', 'balance_mismatch'
            ]
            detailed_df[verification_columns].to_excel(writer, sheet_name='Balance_Verification', index=False)
            
            # Add summary statistics
            summary_data = {
                'Metric': [
                    'Initial Balance',
                    'Final Balance',
                    'Total Return (%)',
                    'Total PnL',
                    'Total Fees Paid',
                    'Total Trades',
                    'Maximum Drawdown (%)',
                    'Largest Single Trade Profit',
                    'Largest Single Trade Loss',
                    'Average Trade PnL',
                    'Win Rate (%)',
                    'Average Win Size',
                    'Average Loss Size',
                    'Profit Factor',
                    'Average Trade Duration (hours)'
                ],
                'Value': [
                    f"${initial_balance:,.2f}",
                    f"${final_balance:,.2f}",
                    f"{((final_balance/initial_balance)-1)*100:.2f}%",
                    f"${detailed_df['net_pnl'].sum():,.2f}",
                    f"${detailed_df['total_fees'].sum():,.2f}",
                    len(detailed_df[detailed_df['action'] == 'exit']),
                    f"{detailed_df['drawdown'].max():.2f}%",
                    f"${detailed_df[detailed_df['net_pnl'] > 0]['net_pnl'].max():,.2f}",
                    f"${detailed_df[detailed_df['net_pnl'] < 0]['net_pnl'].min():,.2f}",
                    f"${detailed_df[detailed_df['net_pnl'].notna()]['net_pnl'].mean():,.2f}",
                    f"{(detailed_df[detailed_df['net_pnl'] > 0]['net_pnl'].count() / detailed_df['net_pnl'].count())*100:.2f}%",
                    f"${detailed_df[detailed_df['net_pnl'] > 0]['net_pnl'].mean():,.2f}",
                    f"${detailed_df[detailed_df['net_pnl'] < 0]['net_pnl'].mean():,.2f}",
                    f"{abs(detailed_df[detailed_df['net_pnl'] > 0]['net_pnl'].sum() / detailed_df[detailed_df['net_pnl'] < 0]['net_pnl'].sum()):.2f}",
                    f"{detailed_df['trade_duration'].mean():.2f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Add balance mismatch warnings
            if (detailed_df['balance_mismatch'] > 0.01).any():
                warnings_df = detailed_df[detailed_df['balance_mismatch'] > 0.01][
                    ['timestamp', 'action', 'expected_balance', 'running_balance', 'balance_mismatch']
                ]
                warnings_df.to_excel(writer, sheet_name='Balance_Warnings', index=False)
    
    print(f"\nDetailed trade analysis saved to {filename}")
    
    # Return any significant mismatches for review
    significant_mismatches = detailed_df[detailed_df['balance_mismatch'] > 1.0]
    if not significant_mismatches.empty:
        print("\nWARNING: Found significant balance mismatches!")
        print(f"Number of mismatches: {len(significant_mismatches)}")
        print("\nLargest mismatches:")
        print(significant_mismatches.nlargest(5, 'balance_mismatch')[
            ['timestamp', 'action', 'expected_balance', 'running_balance', 'balance_mismatch']
        ])

def analyze_trade_patterns(trades_df):
    """Analyze trading patterns and print insights"""
    # Filter only exit trades for P&L analysis
    exit_trades = trades_df[trades_df['action'] == 'exit'].copy()
    
    # Calculate accurate win rate
    total_trades = len(exit_trades)
    winning_trades = len(exit_trades[exit_trades['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print("\n=== Trade Statistics ===")
    print(f"Total Completed Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {total_trades - winning_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${exit_trades[exit_trades['pnl'] > 0]['pnl'].mean():.2f}")
    print(f"Average Loss: ${exit_trades[exit_trades['pnl'] <= 0]['pnl'].mean():.2f}")
    print(f"Total Fees Paid: ${trades_df['fees'].sum():.2f}")
    
    # Add day of week and hour
    trades_df['day_of_week'] = pd.to_datetime(trades_df['timestamp']).dt.day_name()
    trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
    
    print("\n=== Trading Pattern Analysis ===")
    
    # 1. Day of Week Analysis
    day_analysis = trades_df[trades_df['pnl'].notna()].groupby('day_of_week').agg({
        'pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100]
    }).round(2)
    day_analysis.columns = ['Trade_Count', 'Avg_PnL', 'Total_PnL', 'Win_Rate']
    print("\nPerformance by Day of Week:")
    print(day_analysis)
    
    # 2. Hour of Day Analysis
    hour_analysis = trades_df[trades_df['pnl'].notna()].groupby('hour').agg({
        'pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100]
    }).round(2)
    hour_analysis.columns = ['Trade_Count', 'Avg_PnL', 'Total_PnL', 'Win_Rate']
    print("\nBest Trading Hours (Top 5 by PnL):")
    print(hour_analysis.nlargest(5, ('Total_PnL')))
    
    # 3. Consecutive Wins/Losses
    win_group = (trades_df['pnl'] <= 0).astype(int).cumsum()
    loss_group = (trades_df['pnl'] > 0).astype(int).cumsum()
    
    trades_df['win_streak'] = (trades_df['pnl'] > 0).astype(int).groupby(win_group).cumsum()
    trades_df['loss_streak'] = (trades_df['pnl'] <= 0).astype(int).groupby(loss_group).cumsum()
    
    print("\nStreak Analysis:")
    print(f"Longest Win Streak: {trades_df['win_streak'].max()}")
    print(f"Longest Loss Streak: {trades_df['loss_streak'].max()}")
    
    # 4. Price Level Analysis
    trades_df['price_level'] = pd.qcut(trades_df['price'], 4, labels=['Bottom', 'Lower Mid', 'Upper Mid', 'Top'])
    level_analysis = trades_df[trades_df['pnl'].notna()].groupby('price_level', observed=True).agg({
        'pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100]
    }).round(2)
    level_analysis.columns = ['Trade_Count', 'Avg_PnL', 'Total_PnL', 'Win_Rate']
    print("\nPerformance by Price Level:")
    print(level_analysis)
    
    # Save detailed analysis to Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    with pd.ExcelWriter(f'trade_pattern_analysis_{timestamp}.xlsx') as writer:
        day_analysis.to_excel(writer, sheet_name='Day_Analysis')
        hour_analysis.to_excel(writer, sheet_name='Hour_Analysis')
        level_analysis.to_excel(writer, sheet_name='Price_Level_Analysis')
    
    # Return optimization suggestions
    print("\n=== Strategy Optimization Suggestions ===")
    
    # Best days
    best_days = day_analysis.nlargest(3, 'Win_Rate').index.tolist()
    print(f"\n1. Consider trading primarily on: {', '.join(best_days)}")
    
    # Best hours
    best_hours = hour_analysis.nlargest(3, 'Win_Rate').index.tolist()
    print(f"2. Best trading hours: {', '.join(map(str, best_hours))}")
    
    return {
        'day_analysis': day_analysis,
        'hour_analysis': hour_analysis,
        'level_analysis': level_analysis
    }

def analyze_exit_patterns(trades_df):
    """Analyze exit patterns and their effectiveness"""
    exit_analysis = trades_df[trades_df['exit_reason'].notna()].groupby('exit_reason').agg({
        'pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100]
    }).round(2)
    exit_analysis.columns = ['Count', 'Avg_PnL', 'Total_PnL', 'Win_Rate']
    
    print("\n=== Exit Analysis ===")
    print(exit_analysis)
    
    # Analyze time-based exits
    time_exits = trades_df[trades_df['exit_reason'] == 'time_exit']
    if len(time_exits) > 0:
        print("\nTime-based Exit Performance:")
        print(f"Win Rate: {(time_exits['pnl'] > 0).mean()*100:.2f}%")
        print(f"Average PnL: ${time_exits['pnl'].mean():.2f}")
    
    return exit_analysis

def calculate_buy_hold_return(exchange, start_date):
    """Calculate buy & hold return using spot prices"""
    try:
        # Fetch spot price data
        spot_symbol = 'BTC/USDT'
        start_ts = exchange.parse8601(f'{start_date}T00:00:00Z')
        end_ts = exchange.milliseconds()
        
        # Get start and end prices
        start_ohlcv = exchange.fetch_ohlcv(spot_symbol, '1d', start_ts, limit=1)
        end_ohlcv = exchange.fetch_ohlcv(spot_symbol, '1d', end_ts - 86400000, limit=1)
        
        if start_ohlcv and end_ohlcv:
            start_price = start_ohlcv[0][4]  # Close price
            end_price = end_ohlcv[0][4]
            return ((end_price - start_price) / start_price) * 100
        return 0
    except Exception as e:
        print(f"Error calculating buy & hold return: {e}")
        return 0

def calculate_max_drawdown(trades_df):
    """Calculate the maximum drawdown from the trades DataFrame."""
    # Calculate the cumulative balance
    trades_df['cumulative_balance'] = trades_df['balance'].cummax()
    
    # Calculate drawdown
    trades_df['drawdown'] = trades_df['balance'] - trades_df['cumulative_balance']
    
    # Calculate maximum drawdown
    max_drawdown = trades_df['drawdown'].min()
    
    return max_drawdown

def verify_pnl_calculations(trades_df):
    """Verify P&L calculations and provide detailed diagnostics"""
    verification = {
        'pnl_accuracy': True,
        'proper_leverage': True,
        'issues': []
    }
    
    mismatches = []
    
    for i in range(len(trades_df)):
        row = trades_df.iloc[i]
        
        if row['action'] == 'exit':
            # Get the entry trade
            entry_trades = trades_df[
                (trades_df.index < i) & 
                (trades_df['action'].isin(['buy', 'sell']))
            ].iloc[-1]
            
            # Calculate expected PnL
            entry_price = entry_trades['price']
            exit_price = row['price']
            position_size = entry_trades['size']
            leverage = entry_trades['leverage_used']
            position_value = position_size * exit_price
            
            # Determine position direction
            position = 1 if entry_trades['action'] == 'buy' else -1
            
            # Calculate fees
            entry_fee = entry_trades['fees']
            exit_fee = row['position_value'] * TAKER_FEE
            total_fees = entry_fee + exit_fee
            
            # Calculate expected PnL
            price_change = (exit_price - entry_price) / entry_price
            expected_raw_pnl = position_value * price_change * position * leverage
            expected_pnl = expected_raw_pnl - total_fees
            
            # Compare with actual PnL
            pnl_diff = abs(expected_pnl - row['pnl'])
            if pnl_diff > 0.01:  # Allow for small floating point differences
                mismatches.append({
                    'timestamp': row['timestamp'],
                    'expected_pnl': expected_pnl,
                    'actual_pnl': row['pnl'],
                    'difference': pnl_diff,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'leverage': leverage,
                    'fees': total_fees,
                    'direction': 'long' if position == 1 else 'short'
                })
                verification['pnl_accuracy'] = False
                
            # Verify position size doesn't exceed leverage limit
            max_position_value = entry_trades['balance'] * leverage
            actual_position_value = position_size * entry_price
            if actual_position_value > max_position_value * 1.01:  # Allow 1% margin for rounding
                verification['proper_leverage'] = False
                verification['issues'].append(
                    f"Position value (${actual_position_value:.2f}) exceeds leverage limit (${max_position_value:.2f}) "
                    f"at {row['timestamp']}"
                )
    
    if mismatches:
        print("\n=== P&L Calculation Diagnostics ===")
        print(f"Found {len(mismatches)} mismatches in P&L calculations")
        print("\nTop 5 largest mismatches:")
        sorted_mismatches = sorted(mismatches, key=lambda x: x['difference'], reverse=True)[:5]
        for m in sorted_mismatches:
            print(f"\nTimestamp: {m['timestamp']}")
            print(f"Direction: {m['direction']}")
            print(f"Entry Price: ${m['entry_price']:.2f}")
            print(f"Exit Price: ${m['exit_price']:.2f}")
            print(f"Position Size: {m['position_size']:.8f}")
            print(f"Leverage: {m['leverage']}x")
            print(f"Total Fees: ${m['fees']:.2f}")
            print(f"Expected PnL: ${m['expected_pnl']:.2f}")
            print(f"Actual PnL: ${m['actual_pnl']:.2f}")
            print(f"Difference: ${m['difference']:.2f}")
    
    if verification['issues']:
        print("\n=== Leverage Issues ===")
        for issue in verification['issues']:
            print(issue)
    
    return verification

def verify_trade_durations(trades_df):
    """Verify trade duration calculations"""
    print("\n=== Duration Verification ===")
    
    # Create pairs of entry/exit trades
    entries = trades_df[trades_df['action'].isin(['buy', 'sell'])].reset_index()
    exits = trades_df[trades_df['action'] == 'exit'].reset_index()
    
    if len(entries) != len(exits):
        print("WARNING: Number of entries doesn't match exits!")
        return False
    
    # Calculate durations manually
    durations = []
    for i in range(len(entries)):
        duration = (exits['timestamp'].iloc[i] - entries['timestamp'].iloc[i]).total_seconds() / 3600
        durations.append(duration)
    
    # Compare with stored durations
    duration_diff = abs(pd.Series(durations) - exits['duration'])
    if (duration_diff > 0.01).any():
        print("WARNING: Found inconsistencies in duration calculations!")
        print(f"Number of inconsistencies: {(duration_diff > 0.01).sum()}")
        return False
    
    return True

def verify_trade_sequence(trades_df):
    """Verify trade sequence is valid"""
    print("\n=== Trade Sequence Verification ===")
    
    # Check for proper entry/exit sequence
    current_position = 0
    errors = []
    
    for i, row in trades_df.iterrows():
        if row['action'] in ['buy', 'sell']:
            if current_position != 0:
                errors.append(f"Entry signal at {row['timestamp']} while position already open")
            current_position = 1 if row['action'] == 'buy' else -1
        elif row['action'] == 'exit':
            if current_position == 0:
                errors.append(f"Exit signal at {row['timestamp']} without open position")
            current_position = 0
    
    if errors:
        print("Found trade sequence errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(error)
        return False
    
    print("Trade sequence verification passed")
    return True

def update_trailing_tp(current_price, entry_price, position, current_tp, tp_trail_percent=0.01):
    """Update trailing take profit level"""
    if position == 1:  # Long position
        profit_distance = current_price - entry_price
        if profit_distance > 0:  # In profit
            new_tp = current_price * (1 + tp_trail_percent)
            return max(current_tp, new_tp)
    elif position == -1:  # Short position
        profit_distance = entry_price - current_price
        if profit_distance > 0:  # In profit
            new_tp = current_price * (1 - tp_trail_percent)
            return min(current_tp, new_tp)
    return current_tp

def fetch_and_print_balance(exchange):
    """Fetch and print the current balance in BTC and USDT"""
    try:
        balance = exchange.fetch_balance()
        btc_balance = balance['total'].get('BTC', 0)
        usdt_balance = balance['total'].get('USDT', 0)
        print(f"Current BTC Balance: {btc_balance}")
        print(f"Current USDT Balance: {usdt_balance}")
    except Exception as e:
        print(f"Error fetching balance: {str(e)}")

def backtest_strategy(data, initial_balance=100, risk_per_trade=0.02, leverage=3, buy_hold_return=0):
    """Multi-timeframe strategy backtesting"""
    balance = initial_balance
    position = 0
    entry_price = 0
    entry_time = None
    trades = []
    stop_loss = 0
    take_profit = 0
    max_balance = initial_balance
    max_drawdown = 0
    peak_balance = initial_balance
    total_fees = 0
    
    # Get main timeframe data (15m)
    df_15m = data['15m'].copy()
    df_1h = data['1h'].copy()
    df_4h = data['4h'].copy()
    
    # Add indicators to all timeframes
    df_15m = add_indicators(df_15m)
    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)
    
    # Calculate previous MACD histogram values
    df_1h['macd_hist_prev'] = df_1h['macd_hist'].shift(1)
    
    # Wait for indicators to be properly initialized (avoid NaN values)
    warmup_period = 50  # Adjust this based on your longest indicator period
    df_15m = df_15m.iloc[warmup_period:]
    df_1h = df_1h.iloc[warmup_period:]
    df_4h = df_4h.iloc[warmup_period:]
    
    total_candles = len(df_15m)
    
    # Removed print statements for backtest results
    
    for i in range(len(df_15m)):
        try:
            current_time = df_15m.index[i]
            current_price = df_15m['close'].iloc[i]
            
            # Get the most recent 1h and 4h data up to current time
            h1_data = df_1h.loc[:current_time].iloc[-1] if not df_1h.loc[:current_time].empty else None
            h4_data = df_4h.loc[:current_time].iloc[-1] if not df_4h.loc[:current_time].empty else None
            
            # Skip if higher timeframe data is not available
            if h1_data is None or h4_data is None:
                continue
            
            if position == 0:
                # Long entry conditions - less restrictive
                long_condition = (
                    df_15m['rsi'].iloc[i] < 45 and  # RSI oversold (relaxed)
                    df_15m['close'].iloc[i] < df_15m['bb_lower'].iloc[i] * 1.01 and  # Price near lower BB
                    (
                        h1_data['macd_hist'] > h1_data['macd_hist_prev'] or  # MACD histogram increasing
                        df_15m['close'].iloc[i] < df_15m['bb_lower'].iloc[i] * 0.99  # Deep oversold
                    )
                )
                
                # Short entry conditions - less restrictive
                short_condition = (
                    df_15m['rsi'].iloc[i] > 55 and  # RSI overbought (relaxed)
                    df_15m['close'].iloc[i] > df_15m['bb_upper'].iloc[i] * 0.99 and  # Price near upper BB
                    (
                        h1_data['macd_hist'] < h1_data['macd_hist_prev'] or  # MACD histogram decreasing
                        df_15m['close'].iloc[i] > df_15m['bb_upper'].iloc[i] * 1.01  # Deep overbought
                    )
                )
                
                if long_condition:
                    # Position sizing
                    atr = df_15m['atr'].iloc[i]
                    stop_distance = atr * 1.5
                    position_size = calculate_position_size(balance, current_price, stop_distance, risk_per_trade, leverage)
                    
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + (stop_distance * 3.0)
                    
                    trades.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'price': current_price,
                        'size': position_size,
                        'balance': balance,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                
                elif short_condition:
                    # Position sizing
                    atr = df_15m['atr'].iloc[i]
                    stop_distance = atr * 1.5
                    position_size = calculate_position_size(balance, current_price, stop_distance, risk_per_trade, leverage)
                    
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - (stop_distance * 3.0)
                    
                    trades.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'price': current_price,
                        'size': position_size,
                        'balance': balance,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
            
            # Exit conditions
            elif position != 0:
                exit_price = None
                exit_reason = None
                
                # Check stop loss and take profit
                if position == 1:  # Long position
                    # Stop loss hit
                    if current_price <= stop_loss:
                        exit_price = current_price
                        exit_reason = 'stop_loss'
                    # Take profit hit
                    elif current_price >= take_profit:
                        exit_price = current_price
                        exit_reason = 'take_profit'
                    # RSI overbought exit
                    elif df_15m['rsi'].iloc[i] > 70:
                        exit_price = current_price
                        exit_reason = 'rsi_exit'
                
                else:  # Short position
                    # Stop loss hit
                    if current_price >= stop_loss:
                        exit_price = current_price
                        exit_reason = 'stop_loss'
                    # Take profit hit
                    elif current_price <= take_profit:
                        exit_price = current_price
                        exit_reason = 'take_profit'
                    # RSI oversold exit
                    elif df_15m['rsi'].iloc[i] < 30:
                        exit_price = current_price
                        exit_reason = 'rsi_exit'
                
                # Process exit if conditions met
                if exit_price is not None:
                    # Calculate PnL
                    price_change = (exit_price - entry_price) / entry_price
                    pnl = position * price_change * balance * leverage
                    
                    # Update balance
                    balance += pnl
                    
                    # Record exit trade
                    trades.append({
                        'timestamp': current_time,
                        'action': 'exit',
                        'price': exit_price,
                        'pnl': pnl,
                        'balance': balance,
                        'exit_reason': exit_reason,
                        'trade_duration': (current_time - entry_time).total_seconds() / 3600  # Duration in hours
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # Update drawdown
            if balance > peak_balance:
                peak_balance = balance
            current_drawdown = (peak_balance - balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, current_drawdown)
            
        except Exception as e:
            print(f"\nError at timestamp {current_time}: {str(e)}")
            continue
        
        if i % 1000 == 0:
            print(f"\rProgress: {i/total_candles*100:.1f}%", end='')
    
    # Calculate final statistics
    total_trades = len([t for t in trades if t['action'] == 'exit'])
    winning_trades = len([t for t in trades if t['action'] == 'exit' and t['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Get PnL statistics
    exit_trades = [t for t in trades if t['action'] == 'exit']
    winning_pnls = [t['pnl'] for t in exit_trades if t['pnl'] > 0]
    losing_pnls = [t['pnl'] for t in exit_trades if t['pnl'] <= 0]
    
    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
    largest_win = max(winning_pnls) if winning_pnls else 0
    largest_loss = min(losing_pnls) if losing_pnls else 0
    
    strategy_return = ((balance - initial_balance) / initial_balance) * 100
    
    # Removed print statements for backtest results
    
    return trades, balance


def backtest_strategy_2(data, initial_balance=100, risk_per_trade=0.02, leverage=3, buy_hold_return=0):
    """Multi-timeframe strategy backtesting with LSTM predictions"""
    balance = initial_balance
    position = 0
    entry_price = 0
    entry_time = None
    trades = []
    stop_loss = 0
    take_profit = 0
    max_balance = initial_balance
    max_drawdown = 0
    peak_balance = initial_balance
    
    # Get main timeframe data (15m)
    df_15m = data['15m'].copy()
    df_1h = data['1h'].copy()
    df_4h = data['4h'].copy()
    
    # Add indicators to all timeframes
    print("Calculating indicators...")
    df_15m = add_indicators(df_15m)
    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)
    
    # Load LSTM model
    print("Loading LSTM model...")
    model_path = 'models/lstm_model_with_sessions_mtf_20250418_1024.h5'
    try:
        lstm_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Trying alternative loading method...")
        lstm_model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        lstm_model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mae']
        )
    
    # Prepare data for LSTM predictions
    print("Preparing data for LSTM predictions...")
    features = [
        'close', 
        'volume', 
        'atr', 
        'rsi', 
        'macd', 
        'macd_signal', 
        'macd_hist',
        'bb_upper',
        'bb_middle', 
        'bb_lower',
        'bb_width',
        'asia_session',
        'europe_session',
        'us_session',
        'close_1h_change',
        'close_4h_change',
        'close_1d_change',
        'volume_1h_change',
        'volume_4h_change'
    ]

    # Add session data
    df_15m['asia_session'] = ((df_15m.index.hour >= 0) & (df_15m.index.hour < 8)).astype(int)
    df_15m['europe_session'] = ((df_15m.index.hour >= 8) & (df_15m.index.hour < 16)).astype(int)
    df_15m['us_session'] = ((df_15m.index.hour >= 16) & (df_15m.index.hour < 24)).astype(int)

    # Add multi-timeframe features with proper handling of infinite values
    def safe_pct_change(series, periods):
        pct = series.pct_change(periods)
        # Replace infinite values with NaN
        pct = pct.replace([np.inf, -np.inf], np.nan)
        # Forward fill NaN values
        pct = pct.ffill()
        # If any NaN values remain at the start, fill with 0
        pct = pct.fillna(0)
        # Clip extreme values to reasonable range (-100% to +100%)
        return np.clip(pct, -1, 1)
    
    print("Calculating multi-timeframe features...")
    df_15m['close_1h_change'] = safe_pct_change(df_15m['close'], 4)  # 4 15-min periods = 1 hour
    df_15m['close_4h_change'] = safe_pct_change(df_15m['close'], 16)  # 16 15-min periods = 4 hours
    df_15m['close_1d_change'] = safe_pct_change(df_15m['close'], 96)  # 96 15-min periods = 1 day
    df_15m['volume_1h_change'] = safe_pct_change(df_15m['volume'], 4)
    df_15m['volume_4h_change'] = safe_pct_change(df_15m['volume'], 16)
    
    # Handle any remaining infinite values in the features
    print("Preparing features for LSTM...")
    X = df_15m[features].values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize the data with feature-wise clipping
    print("Normalizing features...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data for LSTM input (create sequences of 60 timesteps)
    sequence_length = 60
    X_sequences = []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:(i + sequence_length)])
    X_reshaped = np.array(X_sequences)
    
    # Make predictions
    print("Making LSTM predictions...")
    predictions = lstm_model.predict(X_reshaped)

    # Create DataFrame with debugging
    predictions_df = pd.DataFrame(
        predictions,
        columns=['predicted_stop_loss', 'predicted_take_profit'],
        index=df_15m.index[sequence_length:]
    )
    
    # Forward fill any missing predictions
    predictions_df = predictions_df.reindex(df_15m.index, method='ffill')

    # Add current price and other relevant data
    predictions_df['current_price'] = df_15m['close'].iloc[sequence_length:]
    predictions_df['atr'] = df_15m['atr'].iloc[sequence_length:]
    predictions_df['rsi'] = df_15m['rsi'].iloc[sequence_length:]
    
    # Save predictions to CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    predictions_filename = f'predictions_{timestamp}.csv'
    predictions_df.to_csv(predictions_filename)
    print(f"\nPredictions saved to {predictions_filename}")
    
    # Calculate previous MACD histogram values
    df_1h['macd_hist_prev'] = df_1h['macd_hist'].shift(1)
    
    total_candles = len(df_15m)
    
    print("\n=== Backtest Results ===")
    print(f"Initial Balance: ${initial_balance:.2f}")
    
    for i in range(len(df_15m)):
        try:
            current_time = df_15m.index[i]
            current_price = df_15m['close'].iloc[i]
            
            # Get LSTM predictions for current timestep
            if current_time in predictions_df.index:
                predicted_stop_loss = predictions_df.loc[current_time, 'predicted_stop_loss']
                predicted_take_profit = predictions_df.loc[current_time, 'predicted_take_profit']
            else:
                continue  # Skip if no prediction available
            
            # Get the most recent 1h and 4h data up to current time
            h1_data = df_1h.loc[:current_time].iloc[-1] if not df_1h.loc[:current_time].empty else None
            h4_data = df_4h.loc[:current_time].iloc[-1] if not df_4h.loc[:current_time].empty else None
            
            if h1_data is None or h4_data is None:
                continue
            
            if position == 0:
                # Long entry conditions with LSTM confirmation
                long_condition = (
                    df_15m['rsi'].iloc[i] < 45 and
                    df_15m['close'].iloc[i] < df_15m['bb_lower'].iloc[i] * 1.01 and
                    h1_data['macd_hist'] > h1_data['macd_hist_prev'] and
                    current_price < predicted_stop_loss  # Use LSTM prediction
                )
                
                # Short entry conditions with LSTM confirmation
                short_condition = (
                    df_15m['rsi'].iloc[i] > 55 and
                    df_15m['close'].iloc[i] > df_15m['bb_upper'].iloc[i] * 0.99 and
                    h1_data['macd_hist'] < h1_data['macd_hist_prev'] and
                    current_price > predicted_take_profit  # Use LSTM prediction
                )
                
                if long_condition:
                    # Position sizing
                    atr = df_15m['atr'].iloc[i]
                    stop_distance = atr * 1.5
                    position_size = calculate_position_size(balance, current_price, stop_distance, risk_per_trade, leverage)
                    
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = predicted_stop_loss  # Use LSTM prediction
                    take_profit = predicted_take_profit  # Use LSTM prediction
                    
                    trades.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'price': current_price,
                        'size': position_size,
                        'balance': balance,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                
                elif short_condition:
                    # Position sizing
                    atr = df_15m['atr'].iloc[i]
                    stop_distance = atr * 1.5
                    position_size = calculate_position_size(balance, current_price, stop_distance, risk_per_trade, leverage)
                    
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = predicted_take_profit  # Use LSTM prediction (reversed for shorts)
                    take_profit = predicted_stop_loss  # Use LSTM prediction (reversed for shorts)
                    
                    trades.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'price': current_price,
                        'size': position_size,
                        'balance': balance,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
            
            # Exit conditions
            elif position != 0:
                exit_price = None
                exit_reason = None
                
                # Check stop loss and take profit
                if position == 1:  # Long position
                    # Stop loss hit
                    if current_price <= stop_loss:
                        exit_price = current_price
                        exit_reason = 'stop_loss'
                    # Take profit hit
                    elif current_price >= take_profit:
                        exit_price = current_price
                        exit_reason = 'take_profit'
                    # RSI overbought exit
                    elif df_15m['rsi'].iloc[i] > 70:
                        exit_price = current_price
                        exit_reason = 'rsi_exit'
                
                else:  # Short position
                    # Stop loss hit
                    if current_price >= stop_loss:
                        exit_price = current_price
                        exit_reason = 'stop_loss'
                    # Take profit hit
                    elif current_price <= take_profit:
                        exit_price = current_price
                        exit_reason = 'take_profit'
                    # RSI oversold exit
                    elif df_15m['rsi'].iloc[i] < 30:
                        exit_price = current_price
                        exit_reason = 'rsi_exit'
                
                # Process exit if conditions met
                if exit_price is not None:
                    # Calculate PnL
                    price_change = (exit_price - entry_price) / entry_price
                    pnl = position * price_change * balance * leverage
                    
                    # Update balance
                    balance += pnl
                    
                    # Record exit trade
                    trades.append({
                        'timestamp': current_time,
                        'action': 'exit',
                        'price': exit_price,
                        'pnl': pnl,
                        'balance': balance,
                        'exit_reason': exit_reason,
                        'trade_duration': (current_time - entry_time).total_seconds() / 3600  # Duration in hours
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # Update drawdown
            if balance > peak_balance:
                peak_balance = balance
            current_drawdown = (peak_balance - balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, current_drawdown)
            
        except Exception as e:
            print(f"\nError at timestamp {current_time}: {str(e)}")
            continue
        
        if i % 1000 == 0:
            print(f"\rProgress: {i/total_candles*100:.1f}%", end='')
    
    # Calculate final statistics
    total_trades = len([t for t in trades if t['action'] == 'exit'])
    winning_trades = len([t for t in trades if t['action'] == 'exit' and t['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Get PnL statistics
    exit_trades = [t for t in trades if t['action'] == 'exit']
    winning_pnls = [t['pnl'] for t in exit_trades if t['pnl'] > 0]
    losing_pnls = [t['pnl'] for t in exit_trades if t['pnl'] <= 0]
    
    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
    largest_win = max(winning_pnls) if winning_pnls else 0
    largest_loss = min(losing_pnls) if losing_pnls else 0
    
    strategy_return = ((balance - initial_balance) / initial_balance) * 100
    
    print("\n=== Backtest Results ===")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy vs Buy & Hold: {strategy_return - buy_hold_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Largest Win: ${largest_win:.2f}")
    print(f"Largest Loss: ${largest_loss:.2f}\n")
    
    if trades:
        # Convert trades to DataFrame with more detailed information
        trades_df = pd.DataFrame(trades)
        
        # Add timestamp for the file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'trades_with_fees_{timestamp}.csv'
        
        # Save trades with additional information
        trades_df.to_csv(filename, index=False)
        print(f"\nTrades saved to {filename}")
        
        # Save detailed trade analysis
        analysis_filename = f'trade_analysis_{timestamp}.csv'
        
        # Calculate additional metrics for analysis
        if 'pnl' in trades_df.columns:
            analysis_df = pd.DataFrame({
                'Total Trades': [len(trades_df[trades_df['action'] == 'exit'])],
                'Win Rate': [(trades_df[trades_df['pnl'] > 0]['pnl'].count() / trades_df[trades_df['pnl'].notna()].shape[0]) * 100],
                'Average Win': [trades_df[trades_df['pnl'] > 0]['pnl'].mean()],
                'Average Loss': [trades_df[trades_df['pnl'] < 0]['pnl'].mean()],
                'Largest Win': [trades_df['pnl'].max()],
                'Largest Loss': [trades_df['pnl'].min()],
                'Final Balance': [balance],
                'Total Return %': [((balance - initial_balance) / initial_balance) * 100],
                'Max Drawdown %': [max_drawdown]
            })
            
            analysis_df.to_csv(analysis_filename, index=False)
            print(f"Detailed analysis saved to {analysis_filename}")
    
    return trades, balance

def main():
    try:
        # Initialize trading variables
        position = 0  # 0: no position, 1: long, -1: short
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        print("\nLoading markets...")
        exchange.load_markets()
        print("\nUsing symbol: BTC/USDT:USDT")

        print("\nTesting authentication...")
        # Test authentication by fetching balance
        balance = exchange.fetch_balance()
        print("\nAuthentication successful!")
        print(f"USDT Balance: {balance['USDT']['free'] if 'USDT' in balance else 0}")

        # Fetch market data
        print(f"\nFetching market data for BTC/USDT:USDT...")
        ticker = exchange.fetch_ticker('BTC/USDT:USDT')
        print(f"Current price: {ticker['last']}")

        # If we got here, connection is working
        print("\nConnection to MEXC established successfully!")

        # Continue with data fetching
        start_date = '2024-01-01'
        print(f"\nFetching historical data from {start_date}...")
        data = fetch_multi_timeframe_data(exchange, since=start_date)

        if not data:
            raise Exception("Failed to fetch historical data")
        
        if data:
            trades, final_balance = backtest_strategy_2(data)
            print("final balance: ", final_balance)
        
    except ccxt.AuthenticationError as e:
        print(f"\nAuthentication error: {str(e)}")
        print("Please check your API credentials (key, secret)")
        return
    except ccxt.NetworkError as e:
        print(f"\nNetwork error: {str(e)}")
        print("Please check your internet connection")
        return
    except ccxt.ExchangeError as e:
        print(f"\nExchange error: {str(e)}")
        print("This might be due to incorrect market symbol or exchange configuration")
        return
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return
            
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
