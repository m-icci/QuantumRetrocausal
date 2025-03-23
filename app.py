from flask import Flask, render_template, jsonify
import numpy as np
import logging
from datetime import datetime, timedelta
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@lru_cache(maxsize=1)
def get_mock_metrics():
    """Generate mock metrics for testing with caching"""
    logger.debug("Generating mock metrics")
    return {
        'coherence': round(np.random.uniform(0.6, 0.9), 2),
        'entanglement': round(np.random.uniform(0.4, 0.8), 2),
        'market_stability': round(np.random.uniform(0.5, 0.9), 2),
        'risk_level': round(np.random.uniform(0.2, 0.6), 2),
        'market_volatility': round(np.random.uniform(0.3, 0.7), 2),
        'stability_index': round(np.random.uniform(0.4, 0.8), 2)
    }

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/api/metrics')
def get_metrics():
    logger.debug("Fetching metrics from cache")
    metrics = get_mock_metrics()
    return jsonify(metrics)

@app.route('/api/portfolio')
def get_portfolio():
    logger.debug("Fetching portfolio data")
    return jsonify({
        'total_value': 10000.00,
        'open_positions': 0,
        'active_pairs': ['BTC-USDT', 'ETH-USDT', 'XRP-USDT']
    })

@app.after_request
def add_header(response):
    response.cache_control.max_age = 300  # 5 minutes cache
    return response

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)