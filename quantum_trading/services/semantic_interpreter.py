#!/usr/bin/env python3
"""
Semantic Interpreter
====================
Módulo para interpretação semântica de decisões de trading e geração
de relatórios explicativos que traduzem análises técnicas em linguagem natural.

Combina dados do Helix, LSTM, WAVE e outros componentes para criar explicações
sobre o raciocínio por trás das decisões de trading.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("semantic_interpreter")

class SemanticInterpreter:
    """
    Interpretador semântico para gerar explicações sobre decisões de trading.
    
    Analisa os dados de diversos componentes (Helix, LSTM, QCNN, etc.)
    e gera relatórios explicativos sobre o raciocínio por trás das decisões.
    """
    
    def __init__(
        self,
        reports_dir: Optional[str] = None,
        verbose: bool = True,
        threshold_map: Optional[Dict[str, Dict[str, float]]] = None,
        template_file: Optional[str] = None
    ):
        """
        Inicializa o interpretador semântico.
        
        Args:
            reports_dir: Diretório para salvar relatórios
            verbose: Se True, gera relatórios detalhados
            threshold_map: Mapeamento de thresholds para interpretação
            template_file: Arquivo de template para relatórios
        """
        self.reports_dir = Path(reports_dir) if reports_dir else Path("./reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        self.report_history = []
        
        # Carrega thresholds para interpretação ou usa padrões
        self.threshold_map = threshold_map or {
            "helix": {
                "quantum_coherence": {
                    "high": 0.7,
                    "medium": 0.4,
                    "low": 0.0
                },
                "entropy": {
                    "high": 0.7,
                    "medium": 0.4,
                    "low": 0.0
                },
                "fractal_dimension": {
                    "high": 1.8,
                    "medium": 1.5,
                    "low": 1.0
                }
            },
            "lstm": {
                "confidence": {
                    "high": 0.8,
                    "medium": 0.6,
                    "low": 0.0
                }
            },
            "wave": {
                "spread": {
                    "high": 0.02,
                    "medium": 0.01,
                    "low": 0.0
                },
                "volatility": {
                    "high": 0.05,
                    "medium": 0.02,
                    "low": 0.0
                }
            }
        }
        
        # Carrega templates para relatórios ou usa padrões
        self.templates = self._load_templates(template_file)
    
    def _load_templates(self, template_file: Optional[str]) -> Dict[str, str]:
        """
        Carrega templates para relatórios.
        
        Args:
            template_file: Arquivo de template para relatórios
            
        Returns:
            Dicionário de templates
        """
        default_templates = {
            "report_header": "Relatório de Decisão de Trading - {timestamp}",
            "decision_summary": "Decisão: {decision}. Confiança: {confidence:.2f}",
            "helix_insight": "O campo Helix apresentou {coherence_desc} coerência quântica ({coherence:.2f}), {entropy_desc} entropia ({entropy:.2f}) e dimensão fractal {fractal_desc} ({fractal:.2f}).",
            "lstm_insight": "O modelo LSTM estimou uma probabilidade de {probability:.0f}% de {direction} no preço, com confiança {confidence_desc} ({confidence:.2f}).",
            "wave_insight": "A estratégia WAVE detectou um spread {spread_desc} ({spread:.4f}) entre as exchanges, com volatilidade {volatility_desc} ({volatility:.4f}).",
            "spectra_insight": "A análise de sentimento indica um viés {sentiment_desc} ({sentiment:.2f}), com sinais {reinforcement_desc} de aprendizado por reforço.",
            "combined_rationale": "Entramos na operação porque {main_reason}, {secondary_reason}, e {tertiary_reason}.",
            "declined_rationale": "Não entramos na operação porque {main_reason}, apesar de {positive_factor}.",
            "market_conditions": "Condições de mercado: volume de {volume:.2f} {asset}, {market_desc}.",
            "report_footer": "---\nRelatório gerado por QUALIA Semantic Interpreter v1.0\n{timestamp}"
        }
        
        # Se um arquivo de template foi fornecido, carrega-o
        if template_file and Path(template_file).exists():
            try:
                with open(template_file, 'r') as f:
                    custom_templates = json.load(f)
                    default_templates.update(custom_templates)
            except Exception as e:
                logger.error(f"Erro ao carregar arquivo de template: {e}")
        
        return default_templates
    
    def interpret_helix_data(self, helix_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpreta dados do Helix Controller.
        
        Args:
            helix_data: Dados do Helix Controller
            
        Returns:
            Dicionário com interpretações semânticas
        """
        result = {}
        
        # Extrai métricas quânticas se disponíveis
        quantum_metrics = helix_data.get("quantum_metrics", {})
        coherence = quantum_metrics.get("coherence", 0.0)
        entropy = quantum_metrics.get("entropy", 0.0)
        fractal_dimension = quantum_metrics.get("fractal_dimension", 0.0)
        
        # Categoriza métricas
        thresholds = self.threshold_map["helix"]
        
        if coherence >= thresholds["quantum_coherence"]["high"]:
            coherence_desc = "alta"
        elif coherence >= thresholds["quantum_coherence"]["medium"]:
            coherence_desc = "média"
        else:
            coherence_desc = "baixa"
            
        if entropy >= thresholds["entropy"]["high"]:
            entropy_desc = "alta"
        elif entropy >= thresholds["entropy"]["medium"]:
            entropy_desc = "média"
        else:
            entropy_desc = "baixa"
            
        if fractal_dimension >= thresholds["fractal_dimension"]["high"]:
            fractal_desc = "elevada"
        elif fractal_dimension >= thresholds["fractal_dimension"]["medium"]:
            fractal_desc = "moderada"
        else:
            fractal_desc = "baixa"
        
        # Identifica padrões e os traduz semanticamente
        patterns = helix_data.get("detected_patterns", [])
        pattern_interpretations = []
        
        for pattern in patterns:
            if pattern == "entanglement_peak":
                pattern_interpretations.append("pico de entrelaçamento quântico")
            elif pattern == "fractal_convergence":
                pattern_interpretations.append("convergência fractal")
            elif pattern == "entropy_oscillation":
                pattern_interpretations.append("oscilação de entropia")
            elif pattern == "quantum_attractor":
                pattern_interpretations.append("atrator quântico")
        
        # Combina em uma interpretação unificada
        importance_factors = []
        if coherence >= thresholds["quantum_coherence"]["high"]:
            importance_factors.append(("coherence", coherence, "a coerência quântica estava alta"))
        if entropy <= thresholds["entropy"]["low"]:
            importance_factors.append(("entropy", 1.0 - entropy, "a entropia estava baixa"))
        if fractal_dimension >= thresholds["fractal_dimension"]["high"]:
            importance_factors.append(("fractal", fractal_dimension, "a dimensão fractal estava elevada"))
        if pattern_interpretations:
            pattern_str = f"o campo apresentou padrões de {' e '.join(pattern_interpretations)}"
            importance_factors.append(("patterns", 0.9, pattern_str))
        
        # Ordena fatores por importância
        importance_factors.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            "coherence": coherence,
            "coherence_desc": coherence_desc,
            "entropy": entropy,
            "entropy_desc": entropy_desc,
            "fractal_dimension": fractal_dimension,
            "fractal_desc": fractal_desc,
            "patterns": pattern_interpretations,
            "importance_factors": [factor for _, _, factor in importance_factors]
        }
        
        return result
    
    def interpret_lstm_data(self, lstm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpreta dados do modelo LSTM.
        
        Args:
            lstm_data: Dados do modelo LSTM
            
        Returns:
            Dicionário com interpretações semânticas
        """
        result = {}
        
        # Extrai métricas do LSTM
        confidence = lstm_data.get("confidence", 0.0)
        prediction = lstm_data.get("prediction", 0.0)
        
        # Categoriza métricas
        thresholds = self.threshold_map["lstm"]
        
        if confidence >= thresholds["confidence"]["high"]:
            confidence_desc = "alta"
        elif confidence >= thresholds["confidence"]["medium"]:
            confidence_desc = "média"
        else:
            confidence_desc = "baixa"
        
        # Interpreta a direção prevista
        if prediction > 0.05:  # 5% de aumento
            direction = "alta"
            probability = prediction * 100
        elif prediction < -0.05:  # 5% de queda
            direction = "queda"
            probability = abs(prediction) * 100
        else:
            direction = "estabilidade"
            probability = (1 - abs(prediction)) * 100
        
        result = {
            "confidence": confidence,
            "confidence_desc": confidence_desc,
            "direction": direction,
            "probability": probability,
            "prediction": prediction
        }
        
        return result
    
    def interpret_wave_data(self, wave_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpreta dados da estratégia WAVE.
        
        Args:
            wave_data: Dados da estratégia WAVE
            
        Returns:
            Dicionário com interpretações semânticas
        """
        result = {}
        
        # Extrai métricas do WAVE
        spread = wave_data.get("spread", 0.0)
        volatility = wave_data.get("volatility", 0.0)
        
        # Categoriza métricas
        thresholds = self.threshold_map["wave"]
        
        if spread >= thresholds["spread"]["high"]:
            spread_desc = "amplo"
        elif spread >= thresholds["spread"]["medium"]:
            spread_desc = "moderado"
        else:
            spread_desc = "baixo"
            
        if volatility >= thresholds["volatility"]["high"]:
            volatility_desc = "alta"
        elif volatility >= thresholds["volatility"]["medium"]:
            volatility_desc = "média"
        else:
            volatility_desc = "baixa"
        
        # Identifica oportunidades de arbitragem
        opportunity_types = []
        if spread >= thresholds["spread"]["high"]:
            opportunity_types.append("arbitragem direta de preço")
        if volatility >= thresholds["volatility"]["high"]:
            opportunity_types.append("arbitragem de volatilidade")
        
        result = {
            "spread": spread,
            "spread_desc": spread_desc,
            "volatility": volatility,
            "volatility_desc": volatility_desc,
            "opportunity_types": opportunity_types
        }
        
        return result
    
    def generate_explanation(
        self,
        decision_data: Dict[str, Any],
        include_market_data: bool = True
    ) -> Dict[str, Any]:
        """
        Gera uma explicação semântica para uma decisão de trading.
        
        Args:
            decision_data: Dados da decisão
            include_market_data: Se True, inclui dados de mercado na explicação
            
        Returns:
            Dicionário com explicação semântica
        """
        timestamp = datetime.now().isoformat()
        
        # Extrai dados dos diferentes componentes
        helix_insights = decision_data.get("helix_insights", {})
        lstm_insights = decision_data.get("lstm_insights", {})
        wave_insights = decision_data.get("wave_insights", {})
        spectra_insights = decision_data.get("spectra_insights", {})
        
        # Interpreta dados dos componentes
        helix_interpretation = self.interpret_helix_data(helix_insights)
        lstm_interpretation = self.interpret_lstm_data(lstm_insights)
        wave_interpretation = self.interpret_wave_data(wave_insights)
        
        # Extrai a decisão final
        should_enter = decision_data.get("should_enter", False)
        confidence = decision_data.get("confidence", 0.0)
        pair = decision_data.get("pair", "")
        
        # Prepara os fatores explicativos por importância
        reason_factors = []
        
        # Adiciona fatores do Helix
        if helix_interpretation.get("importance_factors"):
            reason_factors.extend([
                ("helix", factor) for factor in helix_interpretation.get("importance_factors", [])
            ])
        
        # Adiciona fatores do LSTM
        if lstm_interpretation.get("confidence", 0) > 0.6:
            direction = lstm_interpretation.get("direction", "")
            probability = lstm_interpretation.get("probability", 0)
            reason_factors.append(
                ("lstm", f"o LSTM estimou probabilidade de {probability:.0f}% de {direction}")
            )
        
        # Adiciona fatores do WAVE
        if wave_interpretation.get("opportunity_types"):
            opportunity = " e ".join(wave_interpretation.get("opportunity_types", []))
            reason_factors.append(
                ("wave", f"detectamos oportunidade de {opportunity}")
            )
        
        # Verifica se existem fatores suficientes
        if not reason_factors:
            reason_factors = [
                ("default", "análise técnica indicou condições favoráveis"),
                ("default", "múltiplos indicadores convergiram positivamente"),
                ("default", "padrão de mercado reconhecido")
            ]
        
        # Prepara o texto explicativo
        explanation = {}
        
        if should_enter:
            # Seleciona os 3 principais fatores para a explicação (ou menos se não houver 3)
            main_reasons = reason_factors[:min(3, len(reason_factors))]
            main_reason = main_reasons[0][1] if main_reasons else ""
            secondary_reason = main_reasons[1][1] if len(main_reasons) > 1 else ""
            tertiary_reason = main_reasons[2][1] if len(main_reasons) > 2 else ""
            
            # Formata o texto de explicação para decisão positiva
            rationale = self.templates["combined_rationale"].format(
                main_reason=main_reason,
                secondary_reason=secondary_reason,
                tertiary_reason=tertiary_reason if tertiary_reason else "outros fatores técnicos foram favoráveis"
            )
        else:
            # Para decisões negativas, identifica o principal motivo para não entrar
            # e um fator positivo que não foi suficiente
            negative_factors = []
            positive_factors = []
            
            # Analisa fatores do Helix para negativos
            if helix_interpretation.get("coherence", 0) < 0.4:
                negative_factors.append("a coerência quântica estava baixa")
            if helix_interpretation.get("entropy", 0) > 0.7:
                negative_factors.append("a entropia estava alta")
                
            # Analisa fatores do LSTM para negativos
            if lstm_interpretation.get("confidence", 0) < 0.6:
                negative_factors.append("a confiança do LSTM estava baixa")
            if lstm_interpretation.get("direction") == "queda":
                negative_factors.append("o LSTM previu queda no preço")
                
            # Analisa fatores do WAVE para negativos
            if wave_interpretation.get("spread", 0) < 0.005:
                negative_factors.append("o spread estava muito estreito")
            
            # Identifica fatores positivos que não foram suficientes
            if helix_interpretation.get("coherence", 0) > 0.6:
                positive_factors.append("a coerência quântica estava favorável")
            if lstm_interpretation.get("confidence", 0) > 0.7:
                positive_factors.append("o LSTM tinha alta confiança")
            if wave_interpretation.get("spread", 0) > 0.01:
                positive_factors.append("havia um spread significativo")
            
            # Fallbacks caso não encontre fatores específicos
            if not negative_factors:
                negative_factors = ["os indicadores combinados não atingiram o threshold mínimo"]
            if not positive_factors:
                positive_factors = ["alguns indicadores mostraram sinais promissores"]
            
            # Formata o texto de explicação para decisão negativa
            rationale = self.templates["declined_rationale"].format(
                main_reason=negative_factors[0],
                positive_factor=positive_factors[0]
            )
        
        # Formata a explicação completa
        explanation = {
            "timestamp": timestamp,
            "pair": pair,
            "decision": "ENTRAR" if should_enter else "NÃO ENTRAR",
            "confidence": confidence,
            "rationale": rationale,
            "helix_interpretation": helix_interpretation,
            "lstm_interpretation": lstm_interpretation,
            "wave_interpretation": wave_interpretation,
            "reason_factors": reason_factors
        }
        
        # Gera o texto completo do relatório
        report_text = self._format_report(explanation, decision_data, include_market_data)
        explanation["report_text"] = report_text
        
        # Salva o relatório se necessário
        self._save_report(explanation)
        
        # Adiciona ao histórico
        self.report_history.append({
            "timestamp": timestamp,
            "pair": pair,
            "decision": explanation["decision"],
            "confidence": confidence,
            "rationale": rationale
        })
        
        return explanation
    
    def _format_report(
        self,
        explanation: Dict[str, Any],
        decision_data: Dict[str, Any],
        include_market_data: bool
    ) -> str:
        """
        Formata o texto completo do relatório.
        
        Args:
            explanation: Explicação semântica
            decision_data: Dados da decisão
            include_market_data: Se True, inclui dados de mercado
            
        Returns:
            Texto do relatório
        """
        timestamp = explanation["timestamp"]
        pair = explanation["pair"]
        
        # Prepara o cabeçalho
        report_lines = [
            self.templates["report_header"].format(timestamp=timestamp),
            f"Par: {pair}",
            ""
        ]
        
        # Adiciona o resumo da decisão
        report_lines.append(
            self.templates["decision_summary"].format(
                decision=explanation["decision"],
                confidence=explanation["confidence"]
            )
        )
        report_lines.append("")
        
        # Adiciona a explicação principal
        report_lines.append("Explicação:")
        report_lines.append(explanation["rationale"])
        report_lines.append("")
        
        # Adiciona detalhes dos componentes se o modo verbose estiver ativado
        if self.verbose:
            # Detalhes do Helix
            helix = explanation["helix_interpretation"]
            if helix:
                report_lines.append("Detalhes do Helix:")
                report_lines.append(
                    self.templates["helix_insight"].format(
                        coherence=helix.get("coherence", 0.0),
                        coherence_desc=helix.get("coherence_desc", ""),
                        entropy=helix.get("entropy", 0.0),
                        entropy_desc=helix.get("entropy_desc", ""),
                        fractal=helix.get("fractal_dimension", 0.0),
                        fractal_desc=helix.get("fractal_desc", "")
                    )
                )
                
                patterns = helix.get("patterns", [])
                if patterns:
                    report_lines.append(f"Padrões detectados: {', '.join(patterns)}")
                report_lines.append("")
            
            # Detalhes do LSTM
            lstm = explanation["lstm_interpretation"]
            if lstm:
                report_lines.append("Detalhes do LSTM:")
                report_lines.append(
                    self.templates["lstm_insight"].format(
                        probability=lstm.get("probability", 0.0),
                        direction=lstm.get("direction", ""),
                        confidence=lstm.get("confidence", 0.0),
                        confidence_desc=lstm.get("confidence_desc", "")
                    )
                )
                report_lines.append("")
            
            # Detalhes do WAVE
            wave = explanation["wave_interpretation"]
            if wave:
                report_lines.append("Detalhes do WAVE:")
                report_lines.append(
                    self.templates["wave_insight"].format(
                        spread=wave.get("spread", 0.0),
                        spread_desc=wave.get("spread_desc", ""),
                        volatility=wave.get("volatility", 0.0),
                        volatility_desc=wave.get("volatility_desc", "")
                    )
                )
                
                opportunities = wave.get("opportunity_types", [])
                if opportunities:
                    report_lines.append(f"Oportunidades identificadas: {', '.join(opportunities)}")
                report_lines.append("")
        
        # Adiciona dados de mercado se solicitado
        if include_market_data:
            market_data = decision_data.get("market_data", {})
            if market_data:
                volume = market_data.get("volume_a", 0.0) + market_data.get("volume_b", 0.0)
                volume = volume / 2  # média dos volumes
                
                asset = pair.split('/')[0] if '/' in pair else "BTC"
                
                volatility = (market_data.get("volatility_a", 0.0) + market_data.get("volatility_b", 0.0)) / 2
                
                if volatility > 0.05:
                    market_desc = "mercado volátil"
                elif volatility > 0.02:
                    market_desc = "volatilidade moderada"
                else:
                    market_desc = "baixa volatilidade"
                
                report_lines.append(
                    self.templates["market_conditions"].format(
                        volume=volume,
                        asset=asset,
                        market_desc=market_desc
                    )
                )
                report_lines.append("")
        
        # Adiciona rodapé
        report_lines.append(
            self.templates["report_footer"].format(
                timestamp=timestamp
            )
        )
        
        return "\n".join(report_lines)
    
    def _save_report(self, explanation: Dict[str, Any]) -> None:
        """
        Salva o relatório em disco.
        
        Args:
            explanation: Explicação semântica
        """
        timestamp = datetime.fromisoformat(explanation["timestamp"]).strftime("%Y%m%d_%H%M%S")
        pair = explanation["pair"].replace("/", "_")
        decision = "ENTER" if explanation["decision"] == "ENTRAR" else "SKIP"
        
        # Formata o nome do arquivo
        filename = f"{timestamp}_{pair}_{decision}.txt"
        file_path = self.reports_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                f.write(explanation["report_text"])
            logger.debug(f"Relatório salvo em {file_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
    
    def get_historical_reports(
        self,
        limit: Optional[int] = None,
        pair_filter: Optional[str] = None,
        decision_filter: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém relatórios históricos, opcionalmente filtrados.
        
        Args:
            limit: Número máximo de relatórios a retornar
            pair_filter: Filtro por par
            decision_filter: Filtro por decisão (True para ENTRAR, False para NÃO ENTRAR)
            
        Returns:
            Lista de relatórios históricos
        """
        # Filtra o histórico
        filtered_history = self.report_history
        
        if pair_filter:
            filtered_history = [
                report for report in filtered_history
                if report["pair"] == pair_filter
            ]
            
        if decision_filter is not None:
            decision_str = "ENTRAR" if decision_filter else "NÃO ENTRAR"
            filtered_history = [
                report for report in filtered_history
                if report["decision"] == decision_str
            ]
        
        # Ordena por timestamp (mais recente primeiro)
        filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limita o número de resultados se necessário
        if limit:
            filtered_history = filtered_history[:limit]
            
        return filtered_history 