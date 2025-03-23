"""Test suite for Kraken API integration"""
import unittest
import os
from unittest.mock import patch, MagicMock
from quantum_trading.market_api import MarketAPI
import base64
import time

class TestMarketAPI(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Mock environment variables to ensure tests don't use real credentials
        self.env_patcher = patch.dict('os.environ', {
            'KRAKEN_API_KEY': '',
            'KRAKEN_API_SECRET': ''
        })
        self.env_patcher.start()

        self.market_api = MarketAPI()
        self.market_api.api_key = "test_key"
        self.market_api.api_secret = base64.b64encode(b"test_secret").decode('utf-8')

    def tearDown(self):
        """Clean up after each test"""
        self.env_patcher.stop()

    def test_format_symbol_for_api(self):
        """Test symbol formatting for Kraken API"""
        test_cases = [
            ('BTC/USD', 'XBTUSD'),
            ('ETH/USD', 'ETHUSD'),
            ('SOL/USD', 'SOLUSD'),
            ('BTC/USDT', 'XBTUSDT'),
            ('ETH/USDT', 'ETHUSDT')
        ]
        
        for input_symbol, expected in test_cases:
            result = self.market_api.format_symbol_for_api(input_symbol)
            self.assertEqual(result, expected, f"Failed to format {input_symbol}")

    def test_authentication_headers(self):
        """Test authentication header generation"""
        # Simular endpoint e dados para teste
        endpoint = '/private/Balance'
        data = {
            'nonce': str(int(time.time() * 1000)),
            'otp': None
        }
        
        headers = self.market_api.generate_auth_headers(endpoint, data)
        
        # Verificar componentes necessários
        self.assertIn('API-Key', headers)
        self.assertIn('API-Sign', headers)
        self.assertTrue(headers['API-Key'])
        self.assertTrue(headers['API-Sign'])

    def test_authentication_retry_mechanism(self):
        """Testar mecanismo de retry para autenticação"""
        # Simular falha de autenticação seguida de sucesso
        with patch.object(self.market_api, '_make_request', side_effect=[
            Exception("First attempt failed"),
            {"result": {"balance": {"USDT": 1000.0}}}
        ]):
            result = self.market_api.get_balance()
            
            # Verificar que o método foi chamado duas vezes
            self.assertEqual(self.market_api._make_request.call_count, 2)
            self.assertIn('balance', result)

    def test_get_balance_success(self):
        """Testar obtenção de saldo com sucesso"""
        # Simular resposta de saldo da API
        mock_balance_response = {
            'result': {
                'USDT': 1000.0,
                'BTC': 0.5
            }
        }
        
        with patch.object(self.market_api, '_make_request', return_value=mock_balance_response):
            balance = self.market_api.get_balance()
            
            # Verificar saldo retornado
            self.assertIn('USDT', balance)
            self.assertIn('BTC', balance)
            self.assertEqual(balance['USDT'], 1000.0)
            self.assertEqual(balance['BTC'], 0.5)

    def test_get_balance_error_handling(self):
        """Testar tratamento de erro na obtenção de saldo"""
        # Simular erro na API
        with patch.object(self.market_api, '_make_request', side_effect=Exception("API Error")):
            with self.assertRaises(Exception):
                self.market_api.get_balance()

    def test_get_ticker_success(self):
        """Testar obtenção de ticker com sucesso"""
        # Simular resposta de ticker
        mock_ticker_response = {
            'result': {
                'XBTUSD': {
                    'c': ['50000.00', '0.01'],  # Último preço e volume
                    'h': ['51000.00'],  # Máxima
                    'l': ['49000.00'],  # Mínima
                    'o': ['50500.00']   # Abertura
                }
            }
        }
        
        with patch.object(self.market_api, '_make_request', return_value=mock_ticker_response):
            ticker = self.market_api.get_ticker('BTC/USD')
            
            # Verificar campos importantes
            self.assertIn('last_price', ticker)
            self.assertIn('high', ticker)
            self.assertIn('low', ticker)
            self.assertIn('open', ticker)
            
            self.assertEqual(ticker['last_price'], 50000.00)

    def test_nonce_generation(self):
        """Testar geração de nonce"""
        # Gerar múltiplos nonces
        nonces = [self.market_api._generate_nonce() for _ in range(5)]
        
        # Verificar que cada nonce é único
        self.assertEqual(len(set(nonces)), 5)
        
        # Verificar que cada nonce é um inteiro
        for nonce in nonces:
            self.assertIsInstance(nonce, int)
            self.assertGreater(nonce, 0)

    def test_minimum_order_sizes(self):
        """Testar obtenção de tamanhos mínimos de ordem"""
        # Simular resposta de tamanhos mínimos
        mock_minimum_sizes = {
            'BTC/USD': 0.001,
            'ETH/USD': 0.01,
            'SOL/USD': 0.1
        }
        
        with patch.object(self.market_api, '_fetch_minimum_order_sizes', return_value=mock_minimum_sizes):
            for symbol, expected_size in mock_minimum_sizes.items():
                size = self.market_api.get_minimum_order_size(symbol)
                self.assertEqual(size, expected_size)

    def test_priority_pairs(self):
        """Testar pares de moedas prioritários"""
        priority_pairs = self.market_api.get_priority_pairs()
        
        # Verificar que a lista não está vazia
        self.assertTrue(priority_pairs)
        
        # Verificar formato dos pares
        for pair in priority_pairs:
            self.assertIn('/', pair)  # Deve ter formato de par de moedas
            base, quote = pair.split('/')
            self.assertTrue(base)
            self.assertTrue(quote)

    def test_timeframes(self):
        """Testar disponibilidade de timeframes"""
        timeframes = self.market_api.get_supported_timeframes()
        
        # Verificar timeframes padrão
        expected_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for tf in expected_timeframes:
            self.assertIn(tf, timeframes)


if __name__ == '__main__':
    unittest.main()