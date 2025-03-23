#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Streamlit para Trading Quântico com CGR
-------------------------------------------------
Interface gráfica para monitoramento e controle do sistema de trading quântico.

Autor: QUALIA (Sistema Retrocausal)
Versão: 1.0
Data: 2025-03-14
"""

import streamlit as st
import requests
import time
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt
from typing import Dict, List, Any

# Configuração da aplicação Streamlit
st.set_page_config(
    page_title="Trading Quântico CGR Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL base da API
API_BASE_URL = "http://localhost:8000"

# Funções para comunicação com a API
def iniciar_trading(apenas_monitorar=False, duracao_horas=1.0, auto_restart=True):
    try:
        response = requests.post(
            f"{API_BASE_URL}/iniciar", 
            json={
                "apenas_monitorar": apenas_monitorar,
                "duracao_horas": duracao_horas,
                "auto_restart": auto_restart
            }
        )
        return response.json()
    except Exception as e:
        st.error(f"Erro ao iniciar trading: {str(e)}")
        return {"status": "erro", "mensagem": str(e)}

def pausar_trading():
    try:
        response = requests.post(f"{API_BASE_URL}/pausar")
        return response.json()
    except Exception as e:
        st.error(f"Erro ao pausar trading: {str(e)}")
        return {"status": "erro", "mensagem": str(e)}

def retomar_trading():
    try:
        response = requests.post(f"{API_BASE_URL}/retomar")
        return response.json()
    except Exception as e:
        st.error(f"Erro ao retomar trading: {str(e)}")
        return {"status": "erro", "mensagem": str(e)}

def parar_trading():
    try:
        response = requests.post(f"{API_BASE_URL}/parar")
        return response.json()
    except Exception as e:
        st.error(f"Erro ao parar trading: {str(e)}")
        return {"status": "erro", "mensagem": str(e)}

def obter_status():
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        return response.json()
    except Exception as e:
        st.warning(f"Não foi possível conectar à API: {str(e)}")
        return {"sistema_iniciado": False, "erro_conexao": True}

# Função para formatar tempo
def formatar_tempo(segundos):
    if segundos <= 0:
        return "Finalizado"
    
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos_restantes = int(segundos % 60)
    
    if horas > 0:
        return f"{horas}h {minutos}m {segundos_restantes}s"
    elif minutos > 0:
        return f"{minutos}m {segundos_restantes}s"
    else:
        return f"{segundos_restantes}s"

# Título do dashboard
st.title("🌌 Dashboard de Trading Quântico com CGR")
st.markdown("### Sistema de Trading Automatizado com Análise Quântica e CGR")

# Layout principal com duas colunas
col1, col2 = st.columns([2, 3])

# Configurações no sidebar
with st.sidebar:
    st.header("Configurações")
    
    # Controles de inicialização
    apenas_monitorar = st.checkbox("Apenas monitorar (sem execução real de ordens)", value=True)
    duracao_horas = st.slider("Duração da sessão (horas)", min_value=0.1, max_value=24.0, value=1.0, step=0.1)
    auto_restart = st.checkbox("Auto-reiniciar após finalização", value=True)
    
    # Botões de controle
    st.header("Controles")
    controle_col1, controle_col2, controle_col3 = st.columns(3)
    
    with controle_col1:
        if st.button("▶️ Iniciar", key="btn_iniciar"):
            resultado = iniciar_trading(apenas_monitorar, duracao_horas, auto_restart)
            st.success(f"Trading iniciado: {resultado.get('status', 'ok')}")
    
    with controle_col2:
        if st.button("⏸️ Pausar", key="btn_pausar"):
            resultado = pausar_trading()
            st.info(f"Trading pausado: {resultado.get('status', 'ok')}")
    
    with controle_col3:
        if st.button("⏹️ Parar", key="btn_parar"):
            if st.session_state.get("confirmado_parar", False):
                resultado = parar_trading()
                st.warning(f"Trading finalizado: {resultado.get('status', 'ok')}")
                st.session_state["confirmado_parar"] = False
            else:
                st.session_state["confirmado_parar"] = True
                st.warning("Clique novamente para confirmar parada")
    
    if st.session_state.get("confirmado_parar", False):
        if st.button("Cancelar", key="btn_cancelar_parar"):
            st.session_state["confirmado_parar"] = False
    
    # Botão de retomar, se pausado
    status = obter_status()
    if status.get("sistema_iniciado", False) and not status.get("em_execucao", True):
        if st.button("▶️ Retomar", key="btn_retomar"):
            resultado = retomar_trading()
            st.success(f"Trading retomado: {resultado.get('status', 'ok')}")

# Principal dashboard
with col1:
    # Status cards
    st.subheader("Status do Sistema")
    status = obter_status()
    
    # Erro de conexão
    if status.get("erro_conexao", False):
        st.error("⚠️ API não disponível. Inicie o servidor com: `python trading_api.py`")
    
    # Status do sistema
    sistema_status_col1, sistema_status_col2 = st.columns(2)
    
    with sistema_status_col1:
        if status.get("sistema_iniciado", False):
            st.success("✅ Sistema iniciado")
        else:
            st.warning("⏸️ Sistema não iniciado")
    
    with sistema_status_col2:
        if status.get("em_execucao", False):
            st.success("▶️ Em execução")
        else:
            st.warning("⏸️ Pausado")
    
    # Informações sobre o portfólio
    st.subheader("Portfólio")
    portfolio_col1, portfolio_col2 = st.columns(2)
    
    valor_portfolio = status.get("valor_portfolio", 0)
    valor_inicial = status.get("valor_inicial", 0)
    variacao = status.get("variacao_sessao", 0)
    
    with portfolio_col1:
        st.metric(
            label="Valor do Portfólio", 
            value=f"{valor_portfolio:.2f} USDT",
            delta=f"{variacao:.2f}%" if variacao != 0 else None
        )
    
    with portfolio_col2:
        tempo_restante = status.get("tempo_restante_segundos", 0)
        st.metric(
            label="Tempo Restante", 
            value=formatar_tempo(tempo_restante)
        )
    
    # Status atual do sistema
    st.subheader("Status da operação")
    status_text = status.get("status", "Não iniciado")
    ultima_att = status.get("ultima_atualizacao", datetime.now().isoformat())
    try:
        ultima_att_dt = datetime.fromisoformat(ultima_att)
        ultima_att_str = ultima_att_dt.strftime("%H:%M:%S")
    except:
        ultima_att_str = ultima_att
    
    st.info(f"**Status atual:** {status_text} (Última atualização: {ultima_att_str})")
    
    # Sinais atuais de trading
    st.subheader("Últimos Sinais de Trading")
    
    ultimo_ciclo = status.get("resultado_ultimo_ciclo", {})
    if ultimo_ciclo and ultimo_ciclo.get("sinais"):
        sinais = ultimo_ciclo.get("sinais", {})
        for par, sinal in sinais.items():
            cols = st.columns([1, 1, 1, 2])
            acao = sinal.get("acao", "").upper()
            cor = "green" if acao == "COMPRAR" else "red" if acao == "VENDER" else "gray"
            
            cols[0].markdown(f"**{par}**")
            cols[1].markdown(f"<span style='color:{cor}; font-weight:bold'>{acao}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"Força: {sinal.get('forca', 0):.2f}")
            cols[3].markdown(f"CGR: {sinal.get('confianca_cgr', 0):.2f} | Tendência: {sinal.get('tendencia_cgr', 0):.2f}")
    else:
        st.warning("Nenhum sinal de trading disponível")

# Gráficos e análises detalhadas
with col2:
    st.subheader("Desempenho do Trading")
    
    # Histórico de valores
    historico = []
    
    # Tentar carregar de um arquivo se existir
    try:
        with open("estado_api_trading.json", "r") as f:
            dados = json.load(f)
            historico_trades = dados.get("historico_trades", [])
            
            if historico_trades:
                for trade in historico_trades:
                    if "timestamp" in trade and "valor_portfolio" in trade:
                        historico.append({
                            "timestamp": trade["timestamp"],
                            "valor": trade["valor_portfolio"]
                        })
    except Exception as e:
        st.warning(f"Histórico não disponível. Aguardando dados...")
    
    # Exibir o gráfico se houver dados
    if historico:
        df = pd.DataFrame(historico)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Criar gráfico
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("timestamp:T", title="Tempo"),
            y=alt.Y("valor:Q", title="Valor do Portfólio (USDT)"),
            tooltip=["timestamp:T", "valor:Q"]
        ).properties(
            width=700,
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Aguardando dados de desempenho... O gráfico será exibido após o primeiro ciclo de trading.")
    
    # Comparação de preços (KuCoin vs Kraken)
    st.subheader("Comparação de Preços")
    ultimo_ciclo = status.get("resultado_ultimo_ciclo", {})
    
    if ultimo_ciclo and ultimo_ciclo.get("sinais"):
        sinais = ultimo_ciclo.get("sinais", {})
        
        # Dados para o gráfico
        pares = []
        precos_kucoin = []
        precos_kraken = []
        
        for par, sinal in sinais.items():
            pares.append(par)
            precos_kucoin.append(sinal.get("preco_kucoin", 0))
            precos_kraken.append(sinal.get("preco_kraken", 0))
        
        if pares:
            fig = go.Figure(data=[
                go.Bar(name="KuCoin", x=pares, y=precos_kucoin, marker_color="blue"),
                go.Bar(name="Kraken", x=pares, y=precos_kraken, marker_color="orange")
            ])
            
            fig.update_layout(
                barmode="group",
                title="Comparação de Preços entre Exchanges",
                xaxis_title="Par de Trading",
                yaxis_title="Preço (USDT)",
                legend_title="Exchange",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de preço não disponíveis")
    else:
        st.info("Aguardando dados de preços... O gráfico comparativo será exibido após o primeiro ciclo de trading.")

# Tabela de histórico de operações
st.subheader("Histórico de Operações")
try:
    with open("estado_api_trading.json", "r") as f:
        dados = json.load(f)
        historico_trades = dados.get("historico_trades", [])
        
        if historico_trades:
            # Converter para DataFrame para exibição
            df_trades = pd.DataFrame(historico_trades)
            
            # Selecionar apenas algumas colunas relevantes
            colunas = ["timestamp", "valor_portfolio", "variacao_sessao"]
            colunas_disponiveis = [col for col in colunas if col in df_trades.columns]
            
            if colunas_disponiveis:
                df_exibir = df_trades[colunas_disponiveis].copy()
                
                # Formatação
                if "timestamp" in df_exibir.columns:
                    df_exibir["timestamp"] = pd.to_datetime(df_exibir["timestamp"])
                    df_exibir["timestamp"] = df_exibir["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                
                if "valor_portfolio" in df_exibir.columns:
                    df_exibir["valor_portfolio"] = df_exibir["valor_portfolio"].map("{:.2f} USDT".format)
                
                if "variacao_sessao" in df_exibir.columns:
                    df_exibir["variacao_sessao"] = df_exibir["variacao_sessao"].map("{:+.2f}%".format)
                
                # Renomear colunas para português
                df_exibir = df_exibir.rename(columns={
                    "timestamp": "Data/Hora",
                    "valor_portfolio": "Valor do Portfólio",
                    "variacao_sessao": "Variação da Sessão"
                })
                
                st.dataframe(df_exibir, use_container_width=True)
            else:
                st.warning("Histórico não contém as colunas esperadas")
        else:
            st.info("Histórico de operações vazio")
except Exception as e:
    st.warning(f"Não foi possível carregar o histórico de operações: {str(e)}")

# Nova seção: Análise de Performance Detalhada
st.header("📊 Análise Detalhada de Performance")

# Dividir em duas colunas
performance_col1, performance_col2 = st.columns(2)

# Métricas principais de desempenho
with performance_col1:
    st.subheader("Métricas de Desempenho")
    
    try:
        with open("estado_api_trading.json", "r") as f:
            dados = json.load(f)
            historico_trades = dados.get("historico_trades", [])
            
            if historico_trades and len(historico_trades) > 1:
                # Extrair valores do portfólio
                valores_portfolio = [trade.get("valor_portfolio", 0) for trade in historico_trades if "valor_portfolio" in trade]
                
                # Calcular estatísticas básicas
                if valores_portfolio:
                    valor_inicial = valores_portfolio[0]
                    valor_atual = valores_portfolio[-1]
                    ganho_absoluto = valor_atual - valor_inicial
                    ganho_percentual = ((valor_atual / valor_inicial) - 1) * 100 if valor_inicial > 0 else 0
                    
                    # Calcular drawdown máximo
                    max_valor = valor_inicial
                    max_drawdown = 0
                    
                    for valor in valores_portfolio:
                        if valor > max_valor:
                            max_valor = valor
                        
                        drawdown = (max_valor - valor) / max_valor * 100 if max_valor > 0 else 0
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                    
                    # Calcular volatilidade
                    if len(valores_portfolio) > 1:
                        retornos = [(valores_portfolio[i] / valores_portfolio[i-1]) - 1 for i in range(1, len(valores_portfolio))]
                        volatilidade = pd.Series(retornos).std() * 100 if retornos else 0
                    else:
                        volatilidade = 0
                    
                    # Métricas de sinais
                    sinais_compra = 0
                    sinais_venda = 0
                    for trade in historico_trades:
                        sinais = trade.get("sinais", {})
                        for par, sinal in sinais.items():
                            acao = sinal.get("acao", "").upper() if isinstance(sinal, dict) else ""
                            if acao == "COMPRAR":
                                sinais_compra += 1
                            elif acao == "VENDER":
                                sinais_venda += 1
                    
                    # Exibir métricas em formato tabular
                    data = {
                        "Métrica": [
                            "Ganho Total (USDT)", 
                            "Retorno (%)", 
                            "Drawdown Máximo (%)", 
                            "Volatilidade (%)",
                            "Sinais de Compra",
                            "Sinais de Venda",
                            "Total de Ciclos"
                        ],
                        "Valor": [
                            f"{ganho_absoluto:.2f} USDT",
                            f"{ganho_percentual:+.2f}%",
                            f"{max_drawdown:.2f}%",
                            f"{volatilidade:.2f}%",
                            sinais_compra,
                            sinais_venda,
                            len(historico_trades)
                        ]
                    }
                    
                    df_metricas = pd.DataFrame(data)
                    st.table(df_metricas)
                    
                    # Indicadores visuais
                    if ganho_percentual > 0:
                        st.success(f"📈 Lucrativo: +{ganho_percentual:.2f}%")
                    elif ganho_percentual < 0:
                        st.error(f"📉 Prejuízo: {ganho_percentual:.2f}%")
                    else:
                        st.info("⚖️ Neutro: 0.00%")
                else:
                    st.warning("Dados insuficientes para cálculo de métricas")
            else:
                st.info("Aguardando mais ciclos de trading para análise de performance")
    except Exception as e:
        st.error(f"Erro ao calcular métricas de desempenho: {str(e)}")

# Gráfico de distribuição de retornos
with performance_col2:
    st.subheader("Distribuição de Retornos")
    
    try:
        with open("estado_api_trading.json", "r") as f:
            dados = json.load(f)
            historico_trades = dados.get("historico_trades", [])
            
            if historico_trades and len(historico_trades) > 2:
                # Extrair valores do portfólio
                valores_portfolio = [trade.get("valor_portfolio", 0) for trade in historico_trades if "valor_portfolio" in trade]
                
                if len(valores_portfolio) > 2:
                    # Calcular retornos percentuais entre ciclos
                    retornos = [(valores_portfolio[i] / valores_portfolio[i-1]) - 1 for i in range(1, len(valores_portfolio))]
                    retornos_percent = [r * 100 for r in retornos]
                    
                    # Criar DataFrame para o histograma
                    df_retornos = pd.DataFrame({"retorno_percent": retornos_percent})
                    
                    # Criar histograma com Altair
                    # Determinar a cor com base na média dos retornos
                    cor = "green" if sum(retornos) > 0 else "red"
                    
                    chart = alt.Chart(df_retornos).mark_bar().encode(
                        alt.X("retorno_percent:Q", bin=alt.Bin(maxbins=20), title="Retorno por Ciclo (%)"),
                        alt.Y("count()", title="Frequência"),
                        color=alt.value(cor)
                    ).properties(
                        title="Distribuição dos Retornos por Ciclo",
                        width=500,
                        height=300
                    )
                    
                    # Adicionar linha vertical em zero
                    linha_zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black", strokeDash=[5, 5]).encode(
                        x="x:Q"
                    )
                    
                    # Combinar gráficos
                    st.altair_chart(chart + linha_zero, use_container_width=True)
                    
                    # Estatísticas dos retornos
                    retorno_medio = sum(retornos_percent) / len(retornos_percent)
                    retorno_max = max(retornos_percent)
                    retorno_min = min(retornos_percent)
                    
                    st.markdown(f"""
                    **Estatísticas dos Retornos:**
                    * Retorno médio por ciclo: **{retorno_medio:+.2f}%**
                    * Melhor ciclo: **{retorno_max:+.2f}%**
                    * Pior ciclo: **{retorno_min:+.2f}%**
                    * Ciclos positivos: **{sum(1 for r in retornos_percent if r > 0)}** ({sum(1 for r in retornos_percent if r > 0)/len(retornos_percent)*100:.1f}%)
                    * Ciclos negativos: **{sum(1 for r in retornos_percent if r < 0)}** ({sum(1 for r in retornos_percent if r < 0)/len(retornos_percent)*100:.1f}%)
                    """)
                else:
                    st.info("Aguardando mais ciclos de trading para gerar distribuição de retornos")
            else:
                st.info("Dados insuficientes para análise de distribuição")
    except Exception as e:
        st.error(f"Erro ao gerar distribuição de retornos: {str(e)}")

# Seção de análise por par de trading
st.subheader("Desempenho por Par de Trading")

try:
    with open("estado_api_trading.json", "r") as f:
        dados = json.load(f)
        historico_trades = dados.get("historico_trades", [])
        
        if historico_trades:
            # Reunir todos os sinais por par
            sinais_por_par = {}
            
            for trade in historico_trades:
                sinais = trade.get("sinais", {})
                for par, sinal in sinais.items():
                    if isinstance(sinal, dict):
                        if par not in sinais_por_par:
                            sinais_por_par[par] = {
                                "comprar": 0, 
                                "vender": 0, 
                                "aguardar": 0,
                                "forcas": [],
                                "confiancas": [],
                                "tendencias": []
                            }
                        
                        acao = sinal.get("acao", "").lower()
                        if acao in ["comprar", "vender", "aguardar"]:
                            sinais_por_par[par][acao] += 1
                        
                        # Acumular métricas
                        if "forca" in sinal:
                            sinais_por_par[par]["forcas"].append(sinal["forca"])
                        if "confianca_cgr" in sinal:
                            sinais_por_par[par]["confiancas"].append(sinal["confianca_cgr"])
                        if "tendencia_cgr" in sinal:
                            sinais_por_par[par]["tendencias"].append(sinal["tendencia_cgr"])
            
            if sinais_por_par:
                # Criar tabela de desempenho por par
                dados_tabela = []
                
                for par, dados_par in sinais_por_par.items():
                    total_sinais = dados_par["comprar"] + dados_par["vender"] + dados_par["aguardar"]
                    forca_media = sum(dados_par["forcas"]) / len(dados_par["forcas"]) if dados_par["forcas"] else 0
                    confianca_media = sum(dados_par["confiancas"]) / len(dados_par["confiancas"]) if dados_par["confiancas"] else 0
                    tendencia_media = sum(dados_par["tendencias"]) / len(dados_par["tendencias"]) if dados_par["tendencias"] else 0
                    
                    dados_tabela.append({
                        "Par": par,
                        "Comprar": dados_par["comprar"],
                        "Vender": dados_par["vender"],
                        "Aguardar": dados_par["aguardar"],
                        "Total": total_sinais,
                        "Força Média": f"{forca_media:.2f}",
                        "Confiança CGR": f"{confianca_media:.2f}",
                        "Tendência CGR": f"{tendencia_media:.2f}"
                    })
                
                df_pares = pd.DataFrame(dados_tabela)
                st.dataframe(df_pares, use_container_width=True)
                
                # Mostrar gráfico de distribuição de sinais
                st.subheader("Distribuição de Sinais por Par")
                
                dados_grafico = []
                for par, dados_par in sinais_por_par.items():
                    dados_grafico.extend([
                        {"Par": par, "Sinal": "Comprar", "Quantidade": dados_par["comprar"]},
                        {"Par": par, "Sinal": "Vender", "Quantidade": dados_par["vender"]},
                        {"Par": par, "Sinal": "Aguardar", "Quantidade": dados_par["aguardar"]}
                    ])
                
                df_grafico = pd.DataFrame(dados_grafico)
                
                chart = alt.Chart(df_grafico).mark_bar().encode(
                    x=alt.X("Par:N", title="Par de Trading"),
                    y=alt.Y("Quantidade:Q", title="Número de Sinais"),
                    color=alt.Color("Sinal:N", scale=alt.Scale(
                        domain=["Comprar", "Vender", "Aguardar"],
                        range=["green", "red", "gray"]
                    )),
                    tooltip=["Par", "Sinal", "Quantidade"]
                ).properties(
                    title="Distribuição de Sinais por Par de Trading",
                    width=600,
                    height=400
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Nenhum sinal de trading registrado para análise por par")
        else:
            st.info("Histórico de trading vazio")
except Exception as e:
    st.error(f"Erro ao analisar desempenho por par: {str(e)}")

# Rodapé com documentação
st.markdown("---")
st.markdown("""
**Sistema de Trading Quântico com CGR**
* Use o controle no painel lateral para iniciar, pausar ou parar o trading.
* A interface atualiza automaticamente a cada 10 segundos.
* Todos os dados são salvos localmente para análise posterior.
""")

# Atualização automática a cada 10 segundos
st.empty()
time.sleep(10)
st.experimental_rerun()
