import os
import sys
import time
import logging
import traceback
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any
import numpy as np
import json
import colorama
from colorama import Fore, Style

# Inicializa colorama para suporte ao terminal
colorama.init()

# Configuração de logging – com mais detalhes e níveis diferenciados
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# JSON Custom Encoder para serialização de datetime
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(CustomJSONEncoder, self).default(obj)

# Função auxiliar para converter datetime recursivamente (se necessário)
def converter_datetime_para_iso(obj):
    if isinstance(obj, dict):
        return {k: converter_datetime_para_iso(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_datetime_para_iso(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

# ========= CLASSE DE INTERAÇÃO COM A KUCOIN ==========
from kucoin_universal_sdk.api import DefaultClient
from kucoin_universal_sdk.generate.spot.market import GetTickerReqBuilder
from kucoin_universal_sdk.model import (
    ClientOptionBuilder, TransportOptionBuilder,
    GLOBAL_API_ENDPOINT, GLOBAL_FUTURES_API_ENDPOINT, GLOBAL_BROKER_API_ENDPOINT
)

class KucoinAPI:
    def __init__(self, api_key=None, api_secret=None, api_passphrase=None):
        self.client = None
        self.rest_service = None
        self.spot_market_api = None
        self.ultima_atualizacao = {}
        self.max_falhas_consecutivas = 3
        self.falhas_consecutivas = 0
        self.modo_seguro = False
        self._inicializar_kucoin_api(
            api_key or os.environ.get('API_KEY', ''),
            api_secret or os.environ.get('API_SECRET', ''),
            api_passphrase or os.environ.get('API_PASSPHRASE', '')
        )
    
    def _inicializar_kucoin_api(self, api_key, api_secret, api_passphrase):
        http_transport_option = (
            TransportOptionBuilder()
            .set_keep_alive(True)
            .set_max_pool_size(10)
            .set_max_connection_per_pool(10)
            .build()
        )
        client_option = (
            ClientOptionBuilder()
            .set_key(api_key)
            .set_secret(api_secret)
            .set_passphrase(api_passphrase)
            .set_spot_endpoint(GLOBAL_API_ENDPOINT)
            .set_futures_endpoint(GLOBAL_FUTURES_API_ENDPOINT)
            .set_broker_endpoint(GLOBAL_BROKER_API_ENDPOINT)
            .set_transport_option(http_transport_option)
            .build()
        )
        self.client = DefaultClient(client_option)
        self.rest_service = self.client.rest_service()
        self.spot_market_api = self.rest_service.get_spot_service().get_market_api()
        logger.info("KuCoin API inicializada com sucesso")
    
    def get_ticker(self, symbol):
        if self.modo_seguro:
            logger.warning(f"Modo seguro ativo. Usando preço simulado para {symbol}")
            price = self._gerar_preco_simulado(symbol)
            return {"symbol": symbol, "price": price, "simulator_mode": True}
        agora = datetime.now()
        ultima = self.ultima_atualizacao.get(symbol, datetime.min)
        if (agora - ultima).total_seconds() < 0.1:
            logger.debug(f"Taxa limite atingida para {symbol}, aguardando...")
            time.sleep(0.1)
        try:
            request = GetTickerReqBuilder().set_symbol(symbol).build()
            response = self.spot_market_api.get_ticker(request)
            self.ultima_atualizacao[symbol] = agora
            self.falhas_consecutivas = 0
            price = 0.0
            if hasattr(response, 'price'):
                price = float(response.price)
            elif hasattr(response, 'data') and hasattr(response.data, 'price'):
                price = float(response.data.price)
            elif isinstance(response, dict) and 'price' in response:
                price = float(response['price'])
            else:
                logger.warning("Estrutura de resposta inesperada para {}. Usando preço simulado.".format(symbol))
                price = self._gerar_preco_simulado(symbol)
            return {"symbol": symbol, "price": price, "timestamp": int(datetime.now().timestamp()*1000), "time": datetime.now().isoformat()}
        except Exception as e:
            self.falhas_consecutivas += 1
            logger.error(f"Erro ao obter ticker para {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            if self.falhas_consecutivas >= self.max_falhas_consecutivas:
                logger.error("Múltiplas falhas. Ativando modo seguro.")
                self.modo_seguro = True
            price = self._gerar_preco_simulado(symbol)
            return {"symbol": symbol, "price": price, "error": str(e), "simulator_mode": True}
    
    def _gerar_preco_simulado(self, symbol):
        base_prices = {'BTC-USDT': 60000.0, 'ETH-USDT': 3000.0}
        base_price = base_prices.get(symbol, 100.0)
        variacao = np.random.normal(0, 0.01)
        return max(1.0, base_price * (1 + variacao))

# ========= SIMULADOR DE TRADING QUÂNTICO ==========
class SimuladorTradingQuantico:
    def __init__(self, 
                 saldo_inicial=50.0, 
                 duracao_minutos=10, 
                 tempo_entre_trades_segundos=30,
                 pares_trading=['BTC-USDT', 'ETH-USDT'],
                 limite_forca_sinal=0.3,
                 api_key=None, 
                 api_secret=None, 
                 api_passphrase=None):
        self.saldo_inicial = saldo_inicial
        self.duracao_minutos = duracao_minutos
        self.tempo_entre_trades_segundos = tempo_entre_trades_segundos
        self.pares_trading = pares_trading
        self.limite_forca_sinal = limite_forca_sinal
        self.hora_inicio = datetime.now()
        self.hora_termino = self.hora_inicio + timedelta(minutes=duracao_minutos)
        self.maker_fee = 0.001
        self.taker_fee = 0.001
        self.total_taxas_pagas = 0.0
        self.stop_loss_percent = 0.01
        self.trailing_stop_percent = 0.005
        self.max_position_size_percent = 0.15
        self.max_daily_loss_percent = 0.05
        self.max_tempo_operacao_minutos = 5
        self.positions_info = {}
        self.ordens_pendentes = {}
        self.ordens_timeout_segundos = 60
        self.ordens_canceladas = []
        self.ordem_id_contador = 1
        self.estado_quantico = 0.0
        self.portfolio = {'USDT': saldo_inicial, 'BTC': 0.0, 'ETH': 0.0}
        self.lucro_acumulado = 0.0
        self.precos = {}
        self.precos_anteriores = {}
        self.historico_portfolio = []
        self.historico_trades = []
        self.historico_estados_quanticos = []
        self.arquivo_log_trades = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.json")
        self.arquivo_backup_estado = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estado_backup.json")
        self.kucoin_api = KucoinAPI(api_key, api_secret, api_passphrase)
        self._tentar_recuperar_estado()
        self.registrar_valor_portfolio()
        logger.info(f"Saldo inicial: {saldo_inicial:.2f} USDT")
    
    def _tentar_recuperar_estado(self):
        try:
            if os.path.exists(self.arquivo_backup_estado):
                with open(self.arquivo_backup_estado, 'r') as f:
                    estado = json.loads(f.read())
                ts_backup = datetime.fromisoformat(estado.get('timestamp', '2000-01-01T00:00:00'))
                if (datetime.now() - ts_backup).total_seconds() < 3600:
                    logger.info(f"Recuperando estado de {ts_backup.isoformat()}")
                    self.portfolio = estado.get('portfolio', self.portfolio)
                    self.lucro_acumulado = estado.get('lucro_acumulado', 0.0)
                    self.total_taxas_pagas = estado.get('total_taxas_pagas', 0.0)
                    self.historico_trades = estado.get('historico_trades', [])
                    self.historico_portfolio = estado.get('historico_portfolio', [])
                    logger.info(f"Estado recuperado. Portfolio: {self.portfolio}")
                    return True
        except Exception as e:
            logger.error(f"Erro ao recuperar estado: {str(e)}")
        return False
    
    def _salvar_estado_atual(self):
        try:
            estado = {
                'timestamp': datetime.now().isoformat(),
                'portfolio': self.portfolio,
                'lucro_acumulado': self.lucro_acumulado,
                'total_taxas_pagas': self.total_taxas_pagas,
                'historico_trades': self.historico_trades[-100:] if len(self.historico_trades) > 100 else self.historico_trades,
                'historico_portfolio': self.historico_portfolio[-100:] if len(self.historico_portfolio) > 100 else self.historico_portfolio
            }
            with open(self.arquivo_backup_estado, 'w') as f:
                json.dump(estado, f, cls=CustomJSONEncoder, indent=2)
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar estado: {str(e)}")
            return False
    
    def _registrar_trade_log(self, trade_info):
        try:
            trade_info['timestamp_registro'] = datetime.now().isoformat()
            trade_info['portfolio_antes'] = self.historico_portfolio[-2]['portfolio'] if len(self.historico_portfolio) > 1 else {}
            trade_info['portfolio_depois'] = self.historico_portfolio[-1]['portfolio'] if self.historico_portfolio else {}
            trade_info['total_taxas_pagas'] = self.total_taxas_pagas
            with open(self.arquivo_log_trades, 'a') as f:
                f.write(json.dumps(trade_info, cls=CustomJSONEncoder) + '\n')
            return True
        except Exception as e:
            logger.error(f"Erro ao registrar trade: {str(e)}")
            return False
    
    def _gerar_ordem_id(self):
        ordem_id = f"ORDER-{int(time.time())}-{self.ordem_id_contador}"
        self.ordem_id_contador += 1
        return ordem_id
    
    def _verificar_ordens_pendentes(self):
        agora = datetime.now()
        ordens_para_remover = []
        for ordem_id, ordem in list(self.ordens_pendentes.items()):
            if (agora - ordem['timestamp']).total_seconds() > self.ordens_timeout_segundos:
                logger.warning(f"Ordem {ordem_id} excedeu timeout. Tentando reexecução...")
                if ordem['tipo'] == 'compra':
                    ok, msg = self.executar_compra(
                        ordem['symbol'], ordem['valor_compra'], ordem['preco'], ordem_type=ordem.get('order_type', 'market')
                    )
                    if ok:
                        logger.info(f"Ordem {ordem_id} reexecutada com sucesso.")
                        continue
                ordem['status'] = 'cancelada'
                ordem['motivo_cancelamento'] = 'timeout'
                self.ordens_canceladas.append(ordem)
                ordens_para_remover.append(ordem_id)
        for oid in ordens_para_remover:
            del self.ordens_pendentes[oid]
    
    def get_ticker(self, symbol):
        return self.kucoin_api.get_ticker(symbol)
    
    def atualizar_precos(self):
        self._verificar_ordens_pendentes()
        self.precos_anteriores = self.precos.copy()
        precos_atualizados = {}
        for par in self.pares_trading:
            try:
                ticker = self.get_ticker(par)
                if 'price' in ticker and ticker['price'] > 0:
                    precos_atualizados[par] = ticker['price']
            except Exception as e:
                logger.error(f"Erro ao atualizar {par}: {str(e)}")
                if par in self.precos:
                    precos_atualizados[par] = self.precos[par]
        self.precos = precos_atualizados
        return self.precos
    
    def calcular_valor_portfolio(self):
        valor_total = self.portfolio['USDT']
        for par in self.pares_trading:
            moeda = par.split('-')[0]
            if moeda in self.portfolio and par in self.precos:
                valor_total += self.portfolio[moeda] * self.precos[par]
        return valor_total
    
    def registrar_valor_portfolio(self):
        valor = self.calcular_valor_portfolio()
        timestamp = datetime.now().isoformat()
        self.historico_portfolio.append({
            'timestamp': timestamp,
            'valor': valor,
            'portfolio': self.portfolio.copy(),
            'precos': self.precos.copy()
        })
        return valor

    def calcular_rsi(self, symbol, periodo=14):
        trades_par = [t for t in self.historico_trades if t.get('symbol') == symbol]
        if len(trades_par) < periodo + 1:
            return 50.0
        trades_par_sorted = sorted(trades_par, key=lambda x: x['timestamp'])
        precos = [float(t['preco']) for t in trades_par_sorted[-(periodo+1):]]
        ganhos, perdas = [], []
        for i in range(1, len(precos)):
            diff = precos[i] - precos[i-1]
            ganhos.append(diff if diff >= 0 else 0)
            perdas.append(abs(diff) if diff < 0 else 0)
        media_ganhos = np.mean(ganhos[-periodo:]) or 0.0001
        media_perdas = np.mean(perdas[-periodo:]) or 0.0001
        rs = media_ganhos / media_perdas if media_perdas != 0 else 9999
        return 100 - (100 / (1 + rs))
    
    def analisar_mercado(self):
        ruido_quantico = np.random.random() * 0.2
        tendencias = {}
        soma_tendencia = 0
        count = 0
        for par in self.pares_trading:
            if par in self.precos and par in self.precos_anteriores and self.precos_anteriores[par] > 0:
                variacao = (self.precos[par] / self.precos_anteriores[par]) - 1
                tendencias[par] = variacao
                soma_tendencia += variacao
                count += 1
        tendencia_media = soma_tendencia / count if count > 0 else 0
        self.estado_quantico = max(0, min(1, self.estado_quantico * 0.7 + tendência_media * 2 + ruido_quantico))
        logger.info(f"Estado quântico atualizado: {self.estado_quantico:.4f}")
        logger.info(f"Tendência média: {tendência_media*100:+.2f}%")
        return {'estado_quantico': self.estado_quantico, 'tendencia_media': tendência_media, 'tendencias': tendencias}
    
    def gerar_sinais_trading(self):
        analise = self.analisar_mercado()
        sinais = {}
        for par in self.pares_trading:
            if par not in self.precos:
                continue
            preco = self.precos[par]
            tendencia = analise['tendencias'].get(par, 0.0)
            estado = self.estado_quantico
            rsi = self.calcular_rsi(par, periodo=6)
            random_factor = np.random.uniform(0.8, 1.2)
            prob_compra = max(0, min(1, 0.5 * estado - tendência * 5))
            prob_venda = max(0, min(1, 0.5 * (1 - estado) + tendência * 5))
            if rsi < 40:
                prob_compra += 0.2
            elif rsi > 70:
                prob_venda += 0.2
            prob_hold = 1 - (prob_compra + prob_venda)
            if prob_hold < 0:
                total = prob_compra + prob_venda
                prob_compra /= total
                prob_venda /= total
                prob_hold = 0.0
            decisao = np.random.choice(['comprar', 'vender', 'manter'], p=[prob_compra, prob_venda, prob_hold])
            forca = 0.5
            if decisao == 'comprar':
                forca = (prob_compra * (1 + abs(tendência) * 10)) * random_factor
            elif decisao == 'vender':
                forca = (prob_venda * (1 + abs(tendência) * 10)) * random_factor
            forca = max(0, min(1, forca))
            sinais[par] = {
                'acao': decisao,
                'forca': forca,
                'preco': preco,
                'estado_quantico': estado,
                'tendencia': tendência * 100,
                'rsi': rsi,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Sinal {par}: {decisao.upper()} (força: {forca:.2f}, RSI: {rsi:.1f}, preço: {preco:.2f})")
        return sinais
    
    def executar_compra(self, symbol, valor_compra, preco, ordem_type="market"):
        crypto = symbol.split('-')[0]
        valor_total = self.calcular_valor_portfolio()
        tamanho_maximo = valor_total * self.max_position_size_percent
        if crypto in self.positions_info:
            logger.warning(f"Posição já aberta para {crypto}. Ignorando compra.")
            return False, "Já existe posição para esta moeda."
        if valor_compra > tamanho_maximo:
            logger.warning("Valor de compra excede tamanho máximo; ajustando.")
            valor_compra = tamanho_maximo
        if self.portfolio['USDT'] < valor_compra:
            logger.warning(f"Saldo insuficiente: disponível {self.portfolio['USDT']:.2f}, necessário {valor_compra:.2f}")
            return False, "Saldo insuficiente."
        ordem_id = self._gerar_ordem_id()
        quantidade = valor_compra / preco
        taxa = quantidade * self.taker_fee
        quantidade_liquida = quantidade - taxa
        taxa_usdt = taxa * preco
        if ordem_type.lower() == "limit":
            logger.info(f"Enviando ordem LIMIT de compra para {symbol} a {preco:.2f} USDT")
            # Integração real via API deve ser implementada aqui
        elif ordem_type.lower() == "market":
            logger.info(f"Enviando ordem MARKET de compra para {symbol} a preço de mercado.")
        self.portfolio['USDT'] -= valor_compra
        self.portfolio[crypto] = self.portfolio.get(crypto, 0) + quantidade_liquida
        self.total_taxas_pagas += taxa_usdt
        ordem = {
            'id': ordem_id,
            'timestamp': datetime.now(),
            'tipo': 'compra',
            'symbol': symbol,
            'valor_compra': valor_compra,
            'preco': preco,
            'quantidade': quantidade_liquida,
            'taxa': taxa_usdt,
            'status': 'concluida',
            'order_type': ordem_type
        }
        self.historico_trades.append({
            'timestamp': datetime.now().isoformat(),
            'tipo': 'compra',
            'symbol': symbol,
            'preco': preco,
            'quantidade': quantidade_liquida,
            'valor': valor_compra,
            'taxa': taxa_usdt,
            'ordem_id': ordem_id,
            'order_type': ordem_type
        })
        self.registrar_valor_portfolio()
        self._salvar_estado_atual()
        self._registrar_trade_log(ordem)
        logger.info(f"Compra executada: {quantidade_liquida:.6f} {crypto} a {preco:.2f} USDT (Valor: {valor_compra:.2f}, Taxa: {taxa_usdt:.4f})")
        return True, f"Compra realizada: {quantidade_liquida:.6f} {crypto}"
    
    def executar_venda(self, symbol, preco, ordem_type="market", venda_parcial=False, percentual_venda=1.0, motivo="sinal"):
        crypto = symbol.split('-')[0]
        if crypto not in self.portfolio or self.portfolio[crypto] <= 0:
            logger.warning(f"Sem saldo de {crypto} para venda.")
            return False, f"Sem {crypto} para vender."
        ordem_id = self._gerar_ordem_id()
        quantidade_total = self.portfolio[crypto]
        quantidade = quantidade_total * percentual_venda if venda_parcial else quantidade_total
        valor_venda = quantidade * preco
        taxa = valor_venda * self.taker_fee
        valor_liquido = valor_venda - taxa
        if ordem_type.lower() == "limit":
            logger.info(f"Enviando ordem LIMIT de venda para {symbol} a {preco:.2f} USDT")
            # Chamada real para ordem limit deve ser integrada
        elif ordem_type.lower() == "market":
            logger.info(f"Enviando ordem MARKET de venda para {symbol} a preço de mercado.")
        self.portfolio[crypto] -= quantidade
        self.portfolio['USDT'] += valor_liquido
        self.total_taxas_pagas += taxa
        ordem = {
            'id': ordem_id,
            'timestamp': datetime.now(),
            'tipo': 'venda',
            'symbol': symbol,
            'quantidade': quantidade,
            'preco': preco,
            'valor_venda': valor_venda,
            'taxa': taxa,
            'status': 'concluida',
            'order_type': ordem_type,
            'motivo': motivo
        }
        self.historico_trades.append({
            'timestamp': datetime.now().isoformat(),
            'tipo': 'venda',
            'symbol': symbol,
            'preco': preco,
            'quantidade': quantidade,
            'valor_venda': valor_venda,
            'valor_liquido': valor_liquido,
            'taxa': taxa,
            'ordem_id': ordem_id,
            'motivo': motivo,
            'order_type': ordem_type
        })
        self.registrar_valor_portfolio()
        self._salvar_estado_atual()
        self._registrar_trade_log(ordem)
        logger.info(f"Venda executada: {quantidade:.6f} {crypto} a {preco:.2f} USDT (Taxa: {taxa:.4f}). Motivo: {motivo}")
        return True, "Venda executada com sucesso."
    
    def calcular_predicao_retrocausal(self, symbol):
        sinal_forca = abs(np.sin(time.time() * np.random.random()) * np.random.uniform(0.5, 1.5))
        horizonte_max = 300  # 5 minutos
        horizonte_min = 10
        horizonte = min(horizonte_min + int(sinal_forca * (horizonte_max - horizonte_min)), horizonte_max)
        valor_aleatorio = np.random.random()
        if valor_aleatorio > 0.7:
            tipo = "compra"
        elif valor_aleatorio < 0.3:
            tipo = "venda"
        else:
            tipo = "neutro"
        ts_atual = datetime.now()
        ts_exp = ts_atual + timedelta(seconds=horizonte)
        predicao = {
            "symbol": symbol,
            "tipo_sinal": tipo,
            "forca_sinal": sinal_forca,
            "horizonte_segundos": horizonte,
            "timestamp": ts_atual.isoformat(),
            "expiracao": ts_exp.isoformat()
        }
        self.historico_estados_quanticos.append(predicao)
        return predicao
    
    def atualizar_ciclo_trading_retrocausal(self):
        for symbol in self.pares_trading:
            ticker = self.get_ticker(symbol)
            preco_atual = ticker['price']
            self.precos[symbol] = preco_atual
            predicao = self.calcular_predicao_retrocausal(symbol)
            if predicao['forca_sinal'] < self.limite_forca_sinal:
                logger.info(f"Sinal fraco para {symbol}: {predicao['forca_sinal']:.4f} < {self.limite_forca_sinal:.4f}")
                continue
            crypto = symbol.split('-')[0]
            posicao_aberta = crypto in self.positions_info
            if predicao['tipo_sinal'] == "compra" and not posicao_aberta:
                valor_portfolio = self.calcular_valor_portfolio()
                valor_compra = valor_portfolio * self.max_position_size_percent
                sucesso, msg = self.executar_compra(symbol, valor_compra, preco_atual, ordem_type="market")
                if sucesso:
                    logger.info(f"Compra retrocausal: {msg}")
            elif predicao['tipo_sinal'] == "venda" and posicao_aberta:
                sucesso, msg = self.executar_venda(symbol, preco_atual, ordem_type="market", motivo="sinal_retrocausal")
                if sucesso:
                    logger.info(f"Venda retrocausal: {msg}")
        self._salvar_estado_atual()
        valor_atual = self.registrar_valor_portfolio()
        return {
            'valor_portfolio': valor_atual,
            'posicoes_abertas': len(self.positions_info),
            'lucro_acumulado': self.lucro_acumulado,
            'taxas_pagas': self.total_taxas_pagas,
            'timestamp': datetime.now().isoformat()
        }
    
    def gerar_relatorio_final(self):
        if len(self.historico_portfolio) < 2:
            return "Dados insuficientes para relatório."
        valor_inicial = self.historico_portfolio[0]['valor']
        valor_final = self.historico_portfolio[-1]['valor']
        variacao_total = ((valor_final / valor_inicial) - 1) * 100
        total_trades = len(self.historico_trades)
        compras = sum(1 for t in self.historico_trades if t['tipo'] == 'compra')
        vendas = sum(1 for t in self.historico_trades if t['tipo'] == 'venda')
        relatorio = []
        relatorio.append("="*50)
        relatorio.append("RELATÓRIO FINAL DA SIMULAÇÃO DE TRADING QUÂNTICO")
        relatorio.append("="*50)
        relatorio.append(f"Início: {self.historico_portfolio[0]['timestamp']}")
        relatorio.append(f"Fim:    {self.historico_portfolio[-1]['timestamp']}")
        relatorio.append(f"Valor inicial: {valor_inicial:.2f} USDT")
        relatorio.append(f"Valor final:   {valor_final:.2f} USDT")
        relatorio.append(f"Variação:      {variacao_total:+.2f}%")
        relatorio.append(f"Total de trades: {total_trades} (Compras={compras}, Vendas={vendas})")
        relatorio.append(f"Taxas pagas: {self.total_taxas_pagas:.2f} USDT")
        return "\n".join(relatorio)

def main():
    try:
        load_dotenv()
        saldo_inicial = float(os.environ.get('SALDO_INICIAL', '50.0'))
        duracao_minutos = int(os.environ.get('DURACAO_MINUTOS', '10'))
        tempo_entre_trades = int(os.environ.get('TEMPO_ENTRE_TRADES', '10'))
        
        simulador = SimuladorTradingQuantico(
            saldo_inicial=saldo_inicial,
            duracao_minutos=duracao_minutos,
            tempo_entre_trades_segundos=tempo_entre_trades,
            pares_trading=['BTC-USDT', 'ETH-USDT'],
            limite_forca_sinal=0.3,
            api_key=os.environ.get('API_KEY'),
            api_secret=os.environ.get('API_SECRET'),
            api_passphrase=os.environ.get('API_PASSPHRASE')
        )
        
        logger.info("="*80)
        logger.info("TRADING QUÂNTICO REAL - EXECUÇÃO OTIMIZADA")
        logger.info(f"Saldo inicial: {saldo_inicial:.2f} USDT")
        logger.info(f"Duração: {duracao_minutos} minutos")
        logger.info(f"Intervalo entre trades: {tempo_entre_trades} segundos")
        logger.info(f"Limiar de força do sinal: {simulador.limite_forca_sinal:.2f}")
        logger.info("="*80)
        
        while datetime.now() < simulador.hora_termino:
            resultado_ciclo = simulador.atualizar_ciclo_trading_retrocausal()
            logger.info(f"Valor atual do portfólio: {resultado_ciclo['valor_portfolio']:.2f} USDT")
            logger.info(f"Posições abertas: {resultado_ciclo['posicoes_abertas']}")
            time.sleep(simulador.tempo_entre_trades_segundos)
        
        relatorio = simulador.gerar_relatorio_final()
        logger.info(relatorio)
        logger.info("Simulação concluída com sucesso!")
    except Exception as e:
        logger.error(f"Erro na simulação: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
