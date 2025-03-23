"""
Interface de integração com a KuCoin
"""

from python_kucoin.client import Trade, Market
import os
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
from core.logging.quantum_logger import quantum_logger

class KuCoinInterface:
    def __init__(self):
        """Inicializa cliente KuCoin com credenciais seguras"""
        api_key = os.environ.get('KUCOIN_API_KEY')
        api_secret = os.environ.get('KUCOIN_API_SECRET')
        api_passphrase = os.environ.get('KUCOIN_API_PASSPHRASE')

        if not all([api_key, api_secret, api_passphrase]):
            raise ValueError("Credenciais da KuCoin não encontradas")

        self.trade_client = Trade(
            key=api_key,
            secret=api_secret,
            passphrase=api_passphrase
        )

        self.market_client = Market()
        quantum_logger.info("Interface KuCoin inicializada")

    def get_market_data(self, symbol: str, timeframe: str = '1min') -> Dict[str, Any]:
        """Obtém dados do mercado"""
        try:
            klines = self.market_client.get_kline(symbol, timeframe)
            quantum_logger.debug(
                "Dados de mercado obtidos",
                {"symbol": symbol, "timeframe": timeframe}
            )
            return {
                'timestamp': datetime.now(),
                'data': klines,
                'symbol': symbol,
                'timeframe': timeframe
            }
        except Exception as e:
            quantum_logger.error(
                "Erro obtendo dados do mercado",
                {"error": str(e)}
            )
            raise

    def execute_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        quantum_validation: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Executa trade com validação quântica

        Args:
            symbol: Par de trading (ex: BTC-USDT)
            side: buy ou sell
            size: Quantidade
            price: Preço limite (opcional)
            quantum_validation: Métricas quânticas para validação
        """
        try:
            # Valida métricas quânticas
            if quantum_validation:
                coherence = quantum_validation.get('coherence', 0)
                retrocausality = quantum_validation.get('retrocausality', 0)

                if coherence < 0.7 or retrocausality < 0.5:
                    quantum_logger.warning(
                        "Validação quântica falhou",
                        {
                            "coherence": coherence,
                            "retrocausality": retrocausality
                        }
                    )
                    return {
                        'success': False,
                        'reason': 'quantum_validation_failed'
                    }

            # Executa ordem
            order_params = {
                'clientOid': f'qualia_{datetime.now().timestamp()}',
                'side': side,
                'symbol': symbol,
                'size': str(size)
            }

            if price:
                order_params['price'] = str(price)
                response = self.trade_client.create_limit_order(**order_params)
            else:
                response = self.trade_client.create_market_order(**order_params)

            quantum_logger.info(
                "Ordem executada com sucesso",
                {
                    "order_id": response.get('orderId'),
                    "params": order_params
                }
            )

            return {
                'success': True,
                'order_id': response.get('orderId'),
                'params': order_params
            }

        except Exception as e:
            quantum_logger.error(
                "Erro executando trade",
                {"error": str(e)}
            )
            raise

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Obtém status da ordem"""
        try:
            status = self.trade_client.get_order_details(order_id)
            quantum_logger.debug(
                "Status da ordem obtido",
                {"order_id": order_id, "status": status}
            )
            return status
        except Exception as e:
            quantum_logger.error(
                "Erro obtendo status da ordem",
                {"error": str(e)}
            )
            raise