import os
import logging
from flask import Flask, render_template, jsonify, request
from trading_system.market_api import MultiExchangeAPI
from trading_system.quantum_analysis import QuantumAnalyzer
from trading_system.cgr_analysis import CGRAnalyzer
from trading_system.risk_manager import RiskManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize trading components
try:
    market_api = MultiExchangeAPI()
    quantum_analyzer = QuantumAnalyzer()
    cgr_analyzer = CGRAnalyzer()
    risk_manager = RiskManager()
except Exception as e:
    logger.error(f"Error initializing trading components: {e}")
    # Initialize with None to allow app to start without trading functionality
    market_api = None
    quantum_analyzer = None
    cgr_analyzer = None
    risk_manager = None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/healthcheck')
def healthcheck():
    return jsonify({"status": "ok"}), 200

@app.route('/api/dashboard_data')
def dashboard_data():
    try:
        # If trading components aren't initialized, return dummy data
        if not all([market_api, quantum_analyzer, cgr_analyzer]):
            return jsonify({
                'total_value': 0,
                'daily_pnl': 0,
                'open_positions': 0,
                'quantum_coherence': 0.98,
                'current_price': 50000,
                'cgr_points': [],
                'active_trades': []
            })

        # Get market data
        market_data = market_api.get_market_data()

        # Default BTC price if no data
        btc_price = market_data.get('BTC-USDT', {}).get('price', 50000)

        # Perform quantum analysis
        quantum_metrics = quantum_analyzer.analyze(market_data)

        # Get CGR patterns
        cgr_patterns = cgr_analyzer.analyze(market_data)

        # Get active trades and portfolio data
        portfolio = market_api.get_portfolio()
        active_trades = market_api.get_active_trades()

        # Calculate P&L
        daily_pnl = sum(trade.get('pnl', 0) for trade in active_trades)

        return jsonify({
            'total_value': portfolio.get('total_value', 0),
            'daily_pnl': daily_pnl,
            'open_positions': len(active_trades),
            'quantum_coherence': quantum_metrics.get('coherence', 0.98),
            'current_price': btc_price,
            'cgr_points': cgr_patterns.get('points', []),
            'active_trades': active_trades
        })
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/close_trade', methods=['POST'])
def close_trade():
    try:
        if not market_api:
            return jsonify({'error': 'Trading system not initialized'}), 503

        trade_id = request.json.get('trade_id')
        if not trade_id:
            return jsonify({'error': 'Missing trade_id'}), 400

        result = market_api.close_trade(trade_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # ALWAYS serve on port 5000 and bind to 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=True)