"""
Análise Quântica
===============

Módulo que implementa análise quântica de padrões,
memória holográfica e campos mórficos para trading.
""" 

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ..exceptions import AnalysisError

logger = logging.getLogger(__name__)

class BaseAnalyzer:
    """Classe base para analisadores."""
    
    def __init__(self):
        """Inicializa analisador base."""
        self.config = {}
    
    def configure(self, **kwargs):
        """
        Configura analisador.
        
        Args:
            **kwargs: Parâmetros de configuração.
        """
        self.config.update(kwargs)
    
    def save_configuration(self, filename: str) -> None:
        """
        Salva configuração em arquivo.
        
        Args:
            filename: Nome do arquivo.
        """
        with open(f"{filename}.json", "w") as f:
            json.dump(self.config, f)
    
    def load_configuration(self, filename: str) -> Dict[str, Any]:
        """
        Carrega configuração de arquivo.
        
        Args:
            filename: Nome do arquivo.
            
        Returns:
            Configuração carregada.
        """
        with open(f"{filename}.json", "r") as f:
            self.config = json.load(f)
        return self.config
    
    def remove_configuration(self, filename: str) -> None:
        """
        Remove arquivo de configuração.
        
        Args:
            filename: Nome do arquivo.
        """
        import os
        os.remove(f"{filename}.json")
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Retorna configuração atual.
        
        Returns:
            Configuração atual.
        """
        return self.config.copy()

class QuantumAnalyzer(BaseAnalyzer):
    """Analisador quântico."""
    
    def __init__(self):
        """Inicializa analisador quântico."""
        super().__init__()
        self.config = {
            "quantum_bits": 4,
            "entanglement_depth": 2,
            "measurement_basis": "computational"
        }
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa análise quântica.
        
        Args:
            market_data: DataFrame com dados de mercado.
            
        Returns:
            Resultados da análise.
        """
        try:
            # Prepara dados
            prices = market_data["close"].values
            volumes = market_data["volume"].values
            
            # Calcula estado quântico
            quantum_state = self._calculate_quantum_state(prices, volumes)
            
            # Calcula medida de emaranhamento
            entanglement = self._calculate_entanglement(quantum_state)
            
            # Calcula indicadores quânticos
            indicators = self._calculate_quantum_indicators(quantum_state)
            
            return {
                "quantum_state": quantum_state.tolist(),
                "entanglement_measure": float(entanglement),
                "quantum_indicators": indicators
            }
            
        except Exception as e:
            logger.error(f"Erro na análise quântica: {str(e)}")
            raise AnalysisError(f"Falha na análise quântica: {str(e)}")
    
    def _calculate_quantum_state(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Calcula estado quântico do mercado.
        
        Args:
            prices: Array de preços.
            volumes: Array de volumes.
            
        Returns:
            Estado quântico.
        """
        # Normaliza dados
        prices_norm = (prices - np.mean(prices)) / np.std(prices)
        volumes_norm = (volumes - np.mean(volumes)) / np.std(volumes)
        
        # Cria estado quântico
        n_qubits = self.config["quantum_bits"]
        state = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
        
        # Preenche estado com dados normalizados
        for i in range(min(len(prices_norm), 2**n_qubits)):
            state[i, i] = prices_norm[i] + 1j * volumes_norm[i]
        
        return state
    
    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """
        Calcula medida de emaranhamento.
        
        Args:
            state: Estado quântico.
            
        Returns:
            Medida de emaranhamento.
        """
        try:
            # Calcula matriz densidade reduzida
            n = state.shape[0] // 2
            rho = np.dot(state, state.conj().T)
            
            # Normaliza matriz densidade
            rho = rho / np.trace(rho)
            
            # Calcula matriz densidade reduzida
            rho_a = np.trace(rho.reshape(2, n, 2, n), axis1=1, axis2=3)
            
            # Calcula autovalores
            eigenvals = np.linalg.eigvalsh(rho_a)
            
            # Remove autovalores muito próximos de zero
            eigenvals = eigenvals[eigenvals > 1e-10]
            
            # Normaliza autovalores para somarem 1
            eigenvals = eigenvals / np.sum(eigenvals)
            
            # Calcula entropia de von Neumann
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            
            # Normaliza pelo máximo possível (log2(2) = 1 para 1 qubit)
            return min(max(entropy, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de emaranhamento: {str(e)}")
            return 0.0  # Retorna 0 em caso de erro
    
    def _calculate_quantum_indicators(self, state: np.ndarray) -> Dict[str, float]:
        """
        Calcula indicadores quânticos.
        
        Args:
            state: Estado quântico.
            
        Returns:
            Dicionário com indicadores.
        """
        # Calcula coerência
        coherence = np.abs(np.sum(state - np.diag(np.diag(state))))
        
        # Calcula pureza
        purity = np.abs(np.trace(np.dot(state, state.conj().T)))
        
        # Calcula fidelidade com estado base
        base_state = np.eye(state.shape[0]) / np.sqrt(state.shape[0])
        fidelity = np.abs(np.trace(np.dot(state, base_state.conj().T)))
        
        return {
            "coherence": float(coherence),
            "purity": float(purity),
            "fidelity": float(fidelity)
        }

class TechnicalAnalyzer(BaseAnalyzer):
    """Analisador técnico."""
    
    def __init__(self):
        """Inicializa analisador técnico."""
        super().__init__()
        self.indicators = []
    
    def configure_indicators(self, indicators: List[Dict[str, Any]]) -> None:
        """
        Configura indicadores técnicos.
        
        Args:
            indicators: Lista de indicadores e parâmetros.
        """
        self.indicators = indicators
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Executa análise técnica.
        
        Args:
            market_data: DataFrame com dados de mercado.
            
        Returns:
            Dicionário com indicadores calculados.
        """
        try:
            results = {}
            
            for indicator in self.indicators:
                name = indicator["name"]
                params = indicator["params"]
                
                if name == "SMA":
                    results[name] = self._calculate_sma(market_data["close"], **params)
                elif name == "RSI":
                    results[name] = self._calculate_rsi(market_data["close"], **params)
                elif name == "MACD":
                    results[name] = self._calculate_macd(market_data["close"], **params)
                else:
                    logger.warning(f"Indicador desconhecido: {name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise técnica: {str(e)}")
            raise AnalysisError(f"Falha na análise técnica: {str(e)}")
    
    def generate_signals(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Gera sinais de trading baseados nos indicadores.
        
        Args:
            indicators: Dicionário com indicadores.
            
        Returns:
            Dicionário com sinais.
        """
        signals = {
            "buy_signals": pd.Series(index=indicators["SMA"].index, data=False),
            "sell_signals": pd.Series(index=indicators["SMA"].index, data=False)
        }
        
        # Sinais baseados em cruzamento de médias
        if "SMA" in indicators:
            sma = indicators["SMA"]
            price = sma.index  # Preço de fechamento
            signals["buy_signals"] |= (price > sma) & (price.shift(1) <= sma.shift(1))
            signals["sell_signals"] |= (price < sma) & (price.shift(1) >= sma.shift(1))
        
        # Sinais baseados em RSI
        if "RSI" in indicators:
            rsi = indicators["RSI"]
            signals["buy_signals"] |= (rsi < 30) & (rsi.shift(1) >= 30)
            signals["sell_signals"] |= (rsi > 70) & (rsi.shift(1) <= 70)
        
        # Sinais baseados em MACD
        if "MACD" in indicators:
            macd = indicators["MACD"]
            signals["buy_signals"] |= (macd > 0) & (macd.shift(1) <= 0)
            signals["sell_signals"] |= (macd < 0) & (macd.shift(1) >= 0)
        
        return signals
    
    def _calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcula Média Móvel Simples."""
        return prices.rolling(window=window).mean()
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcula Índice de Força Relativa."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> pd.Series:
        """Calcula MACD."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

class StatisticalAnalyzer(BaseAnalyzer):
    """Analisador estatístico."""
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Executa análise estatística.
        
        Args:
            market_data: DataFrame com dados de mercado.
            
        Returns:
            Dicionário com estatísticas.
        """
        try:
            returns = market_data["close"].pct_change().dropna()
            
            return {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "skew": float(returns.skew()),
                "kurtosis": float(returns.kurtosis()),
                "var_95": float(np.percentile(returns, 5)),
                "var_99": float(np.percentile(returns, 1))
            }
            
        except Exception as e:
            logger.error(f"Erro na análise estatística: {str(e)}")
            raise AnalysisError(f"Falha na análise estatística: {str(e)}")
    
    def test_normality(self, series: pd.Series) -> Dict[str, float]:
        """
        Testa normalidade da série.
        
        Args:
            series: Série temporal.
            
        Returns:
            Resultados do teste.
        """
        statistic, p_value = stats.normaltest(series)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value)
        }
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, float]:
        """
        Testa estacionariedade da série.
        
        Args:
            series: Série temporal.
            
        Returns:
            Resultados do teste.
        """
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series)
        return {
            "adf_stat": float(result[0]),
            "p_value": float(result[1])
        }

class MachineLearningAnalyzer(BaseAnalyzer):
    """Analisador de machine learning."""
    
    def prepare_data(
        self,
        market_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados para ML.
        
        Args:
            market_data: DataFrame com dados de mercado.
            
        Returns:
            Features e labels.
        """
        # Calcula features
        data = market_data.copy()
        data["returns"] = data["close"].pct_change()
        data["volatility"] = data["returns"].rolling(20).std()
        data["momentum"] = data["returns"].rolling(10).mean()
        data["volume_ma"] = data["volume"].rolling(10).mean()
        
        # Remove dados faltantes
        data = data.dropna()
        
        # Cria labels (1 se retorno futuro > 0, 0 caso contrário)
        y = (data["returns"].shift(-1) > 0).astype(int)
        
        # Seleciona features
        X = data[["returns", "volatility", "momentum", "volume_ma"]]
        
        return X.values[:-1], y.values[:-1]
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> RandomForestClassifier:
        """
        Treina modelo.
        
        Args:
            X: Features.
            y: Labels.
            
        Returns:
            Modelo treinado.
        """
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    def predict(
        self,
        model: RandomForestClassifier,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Faz previsões.
        
        Args:
            model: Modelo treinado.
            X: Features.
            
        Returns:
            Previsões.
        """
        return model.predict(X)
    
    def evaluate_model(
        self,
        model: RandomForestClassifier,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Avalia modelo.
        
        Args:
            model: Modelo treinado.
            X: Features.
            y: Labels.
            
        Returns:
            Métricas de avaliação.
        """
        y_pred = model.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred))
        }
    
    def feature_importance(
        self,
        model: RandomForestClassifier
    ) -> Dict[str, float]:
        """
        Calcula importância das features.
        
        Args:
            model: Modelo treinado.
            
        Returns:
            Importância das features.
        """
        features = ["returns", "volatility", "momentum", "volume_ma"]
        importance = model.feature_importances_
        return dict(zip(features, importance))

class PortfolioAnalyzer(BaseAnalyzer):
    """Analisador de portfólio."""
    
    def analyze_portfolio(self, portfolio: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Analisa portfólio.
        
        Args:
            portfolio: Dados do portfólio.
            
        Returns:
            Análise do portfólio.
        """
        try:
            # Calcula valor total
            total_value = sum(
                p["position"] * p["current_price"]
                for p in portfolio.values()
            )
            
            # Calcula PnL total
            total_pnl = sum(p["pnl"] for p in portfolio.values())
            
            # Calcula risco do portfólio (média ponderada)
            weights = [
                p["position"] * p["current_price"] / total_value
                for p in portfolio.values()
            ]
            risks = [p["risk"] for p in portfolio.values()]
            portfolio_risk = np.average(risks, weights=weights)
            
            return {
                "total_value": float(total_value),
                "total_pnl": float(total_pnl),
                "portfolio_risk": float(portfolio_risk)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de portfólio: {str(e)}")
            raise AnalysisError(f"Falha na análise de portfólio: {str(e)}")
    
    def calculate_returns(self, portfolio: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calcula retornos do portfólio.
        
        Args:
            portfolio: Dados do portfólio.
            
        Returns:
            Retornos calculados.
        """
        total_value = sum(
            p["position"] * p["current_price"]
            for p in portfolio.values()
        )
        total_cost = sum(
            p["position"] * p["entry_price"]
            for p in portfolio.values()
        )
        
        absolute_return = total_value - total_cost
        percentage_return = (absolute_return / total_cost) * 100
        
        return {
            "absolute_return": float(absolute_return),
            "percentage_return": float(percentage_return)
        }
    
    def analyze_risk(self, portfolio: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Analisa risco do portfólio.
        
        Args:
            portfolio: Dados do portfólio.
            
        Returns:
            Métricas de risco.
        """
        # Calcula VaR
        total_value = sum(
            p["position"] * p["current_price"]
            for p in portfolio.values()
        )
        portfolio_risk = self.analyze_portfolio(portfolio)["portfolio_risk"]
        var_95 = total_value * portfolio_risk * 1.645  # 95% confiança
        
        # Calcula Sharpe Ratio (assumindo risk-free rate de 2%)
        returns = self.calculate_returns(portfolio)
        rf_rate = 0.02
        excess_return = returns["percentage_return"] - rf_rate
        sharpe = excess_return / (portfolio_risk * 100)
        
        # Calcula drawdown máximo
        max_dd = max(
            (p["entry_price"] - p["current_price"]) / p["entry_price"]
            for p in portfolio.values()
        ) * 100
        
        return {
            "var": float(var_95),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd)
        }
    
    def optimize_allocation(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        market_data: pd.DataFrame,
        risk_free_rate: float
    ) -> Dict[str, Any]:
        """
        Otimiza alocação do portfólio.
        
        Args:
            portfolio: Dados do portfólio.
            market_data: Dados de mercado.
            risk_free_rate: Taxa livre de risco.
            
        Returns:
            Alocação otimizada.
        """
        from scipy.optimize import minimize
        
        # Extrai retornos e riscos
        returns = np.array([
            (p["current_price"] - p["entry_price"]) / p["entry_price"]
            for p in portfolio.values()
        ])
        risks = np.array([p["risk"] for p in portfolio.values()])
        n_assets = len(portfolio)
        
        # Função objetivo (maximizar Sharpe Ratio)
        def objective(weights):
            portfolio_return = np.sum(returns * weights)
            portfolio_risk = np.sqrt(np.sum(risks**2 * weights**2))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            return -sharpe  # Minimiza negativo = maximiza positivo
        
        # Restrições
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Soma = 1
        ]
        bounds = [(0, 1) for _ in range(n_assets)]  # 0 <= peso <= 1
        
        # Otimiza
        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,  # Pesos iniciais iguais
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        # Calcula métricas com pesos otimizados
        weights = result.x
        portfolio_return = np.sum(returns * weights)
        portfolio_risk = np.sqrt(np.sum(risks**2 * weights**2))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
        
        return {
            "weights": weights.tolist(),
            "expected_return": float(portfolio_return),
            "expected_risk": float(portfolio_risk),
            "sharpe_ratio": float(sharpe)
        }

__all__ = [
    'QuantumAnalyzer',
    'TechnicalAnalyzer', 
    'StatisticalAnalyzer',
    'MachineLearningAnalyzer',
    'PortfolioAnalyzer'
]

from .market_analysis import MarketAnalysis

__all__ = ['MarketAnalysis']