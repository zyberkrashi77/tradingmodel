import ccxt
import json
import os
import time

def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create config.json with your settings.")
        raise
    except json.JSONDecodeError:
        print("Error: config.json is not valid JSON. Please check the file format.")
        raise

def initialize_exchange():
    """Initialize and return the MEXC exchange instance"""
    config = load_config()
    
    exchange = ccxt.mexc({
        'apiKey': config['MEXC_API_KEY'],
        'secret': config['MEXC_API_SECRET'],
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True,
            'defaultNetwork': 'MAINNET',
            'recvWindow': 60000,  # Increased recvWindow
            'createMarketBuyOrderRequiresPrice': False
        },
        'timeout': 30000,  # Increased timeout
        'headers': {
            'User-Agent': 'Mozilla/5.0'  # Added user agent
        }
    })
    
    # Sync time with exchange
    try:
        exchange.load_time_difference()
        print("Successfully synchronized time with exchange")
    except Exception as e:
        print(f"Warning: Could not sync time with exchange: {str(e)}")
    
    return exchange

def fetch_with_retry(exchange, symbol, timeframe, since=None, limit=1000, max_retries=3):
    """Fetch data with retry mechanism"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
            
            return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            continue

# Create a global exchange instance that can be imported by other files
exchange = initialize_exchange() 