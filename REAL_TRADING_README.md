# QUALIA Real Trading System

This README explains how to set up and run the QUALIA Trading System with real API credentials for KuCoin and Kraken exchanges.

## Overview

The QUALIA Trading System has been enhanced to support real trading on KuCoin and Kraken exchanges, incorporating:

- Holographic memory for pattern recognition
- CGR (Chaos Game Representation) analysis
- Retrocausal predictive methods
- Adaptive learning and risk management

The system is designed to run for a limited time period (default: 1 hour) with safety mechanisms to limit potential losses.

## Setup

### 1. API Credentials

Create a `.env` file in the root directory with the following API credentials:

```
# KuCoin API credentials
KUCOIN_API_KEY=your_kucoin_api_key
KUCOIN_API_SECRET=your_kucoin_api_secret
KUCOIN_API_PASSPHRASE=your_kucoin_api_passphrase

# Kraken API credentials
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_API_SECRET=your_kraken_api_secret
```

**Important**: Make sure your API keys have trading permissions. For initial testing, it's recommended to use API keys with limited permissions and small trading limits.

### 2. Install Dependencies

Ensure all dependencies are installed:

```
pip install -r requirements.txt
```

Key dependencies include:
- `numpy`
- `pandas`
- `python-dotenv`
- `requests`
- `kucoin-universal-sdk` (for KuCoin API)

## Running Real Trading Sessions

### Basic Usage

To start a real trading session with default parameters (1-hour duration, BTC/USDT and ETH/USDT pairs):

```
python run_real_trading.py
```

### Advanced Options

The script supports several command-line arguments:

```
python run_real_trading.py --duration 30 --pairs "BTC/USDT,ETH/USDT,SOL/USDT" --max-drawdown 1.5 --safe-mode
```

Parameters:
- `--duration`: Trading session duration in minutes (default: 60)
- `--pairs`: Comma-separated list of trading pairs (default: "BTC/USDT,ETH/USDT")
- `--max-drawdown`: Maximum allowed drawdown percentage (default: 2.0)
- `--safe-mode`: Enable additional safety measures for risk mitigation

### Monitoring

During the trading session, the system will log status updates to both the console and a log file. 

The log file is created in the root directory with the naming format: `trading_log_YYYYMMDD_HHMMSS.log`

## Safety Mechanisms

The system implements several safety mechanisms:

1. **Time Limit**: Trading automatically stops after the specified duration
2. **Maximum Drawdown**: Trading stops if a specified percentage loss is reached
3. **Error Handling**: Comprehensive error handling to prevent unintended behavior
4. **Simulation Fallback**: In case of API errors, the system can fall back to simulation mode

## Emergency Stop

You can stop the trading session at any time by pressing `Ctrl+C`. The system will perform an orderly shutdown, including generating a final performance report.

## Important Notes

- **Real Money Warning**: This system trades with real money. Start with small amounts until you're confident in its performance.
- **Exchange Fees**: The system accounts for exchange fees, but actual fees may vary.
- **Market Conditions**: Performance will vary based on market conditions.
- **Supervision**: Never leave the system running unsupervised.

## Components

The real trading functionality is implemented across several files:

- `quantum_trading/real_time_trader.py`: Main class for real-time trading sessions
- `quantum_trading/market_api.py`: Enhanced to support real exchange APIs
- `run_real_trading.py`: Entry point script for starting a trading session
- `quantum_trading/holographic_memory.py`: Pattern recognition for market analysis

## Troubleshooting

If you encounter issues:

1. Check your API credentials are correctly set up in the `.env` file
2. Verify your API keys have the necessary permissions
3. Ensure you have sufficient balances on the exchanges
4. Check the log files for detailed error messages

## Risk Disclaimer

Trading cryptocurrency involves significant risk and can result in the loss of your invested capital. The QUALIA Trading System is provided as-is without any guarantee of profitability. Use at your own risk.
