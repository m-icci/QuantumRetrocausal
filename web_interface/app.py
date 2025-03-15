import os
import logging
from flask import Flask, render_template, jsonify, request
from trading_system.market_api import TradingAPI
from trading_system.quantum_analysis import QuantumAnalyzer
from trading_system.cgr_analysis import CGRAnalyzer
from trading_system.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)
app.secret_key = os.environ.get("SESSION_SECRET")

# Initialize trading components
try:
    trading_api = TradingAPI()
    quantum_analyzer = QuantumAnalyzer()
    cgr_analyzer = CGRAnalyzer()
    risk_manager = RiskManager()
    logger.info("All trading components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize trading components: {e}")
    raise

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/healthcheck')
def healthcheck():
    """Basic health check endpoint"""
    return jsonify({"status": "ok"}), 200

@app.route('/api/dashboard_data')
def dashboard_data():
    """Get real-time trading dashboard data"""
    try:
        # Get real market data from Kraken
        market_data = trading_api.get_market_data()
        logger.debug(f"Market data received: {market_data}")

        # Get real portfolio data
        portfolio = trading_api.get_portfolio()
        logger.debug(f"Portfolio data received: {portfolio}")

        # Get BTC price (now using BTC-USD for Kraken)
        btc_price = market_data.get('BTC-USD', {}).get('price', 0)

        # Get active trades
        active_trades = trading_api.get_active_trades()

        # Calculate P&L (to be implemented with historical data)
        daily_pnl = 0  # Will be calculated when trading is implemented

        # Perform quantum and CGR analysis
        quantum_metrics = quantum_analyzer.analyze(market_data)
        cgr_patterns = cgr_analyzer.analyze(market_data)

        return jsonify({
            'total_value': portfolio.get('total_value', 0),
            'holdings': portfolio.get('holdings', {}),
            'daily_pnl': daily_pnl,
            'open_positions': len(active_trades),
            'quantum_coherence': quantum_metrics.get('coherence', 0),
            'current_price': btc_price,
            'cgr_points': cgr_patterns.get('points', []),
            'active_trades': active_trades,
            'exchange': 'Kraken'
        })

    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to get trading data from Kraken'
        }), 500

@app.route('/api/close_trade', methods=['POST'])
def close_trade():
    """Close an active trade"""
    try:
        trade_id = request.json.get('trade_id')
        if not trade_id:
            return jsonify({'error': 'Missing trade_id'}), 400

        result = trading_api.close_trade(trade_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)