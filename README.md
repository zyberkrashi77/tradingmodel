MEXC BTC Futures Trading Connection

This repository contains code for connecting to the MEXC cryptocurrency exchange API to trade BTC futures. The main issue is with authentication in the btc_futures_live.py file, causing a "Passphrase error" during API connection.

Current Issue

The script fails to authenticate with the MEXC API, returning:

mexc {"code":"152408","msg":"Passphrase error"}

Files
btc_futures_live.py: Main script for exchange connection and trading
btc_futures.py: Trading strategy implementation
requirements.txt: Python dependencies

Setup
Install dependencies:
pip install -r requirements.txt

Configure API credentials in btc_futures_live.py:
exchange_config = {
    'apiKey': 'mx0vglQq8BV2UyQOVj',
    'secret': 'f29d5254c8604bd5abf94fcdb1109bd5',
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True,
        'createMarketBuyOrderRequiresPrice': False,
        'defaultContractType': 'perpetual',
        'defaultMarginMode': 'cross',
        'password': 'YOUR_API_PASSPHRASE'
    }
}

Additional Info
API key created on MEXC with trading permissions
Account verification is pending
Passphrase verified as correct
Help Needed
Verify MEXC API configuration
Fix authentication process
Check if pending account verification affects API access
Test connection with provided credentials
Contact
Contact the repository owner for questions or assistance.
