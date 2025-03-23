"""
Interface de Trading usando Streamlit
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuração da página
st.set_page_config(
    page_title="Trading Quântico",
    page_icon="📈",
    layout="wide"
)

# Título
st.title("Sistema de Trading Quântico")

# Sidebar
st.sidebar.title("Controles")

# Estado do trading
try:
    response = requests.get("http://localhost:8000/state")
    if response.status_code == 200:
        trading_state = response.json()
        
        # Status
        status = "🟢 Ativo" if trading_state["is_running"] else "🔴 Parado"
        st.sidebar.markdown(f"**Status:** {status}")
        
        # Informações
        st.sidebar.markdown("### Informações")
        st.sidebar.markdown(f"**Par:** {trading_state['current_symbol']}")
        st.sidebar.markdown(f"**Saldo:** ${trading_state['balance']:.2f}")
        
        # Último preço
        if trading_state["last_price"]:
            st.sidebar.markdown(f"**Último Preço:** ${trading_state['last_price']:.2f}")
        
        # Posição atual
        if trading_state["current_position"]:
            st.sidebar.markdown("### Posição Atual")
            st.sidebar.json(trading_state["current_position"])
        
        # Botões de controle
        if not trading_state["is_running"]:
            if st.sidebar.button("Iniciar Trading"):
                response = requests.post("http://localhost:8000/start")
                if response.status_code == 200:
                    st.sidebar.success("Trading iniciado!")
                else:
                    st.sidebar.error("Erro ao iniciar trading")
        else:
            if st.sidebar.button("Parar Trading"):
                response = requests.post("http://localhost:8000/stop")
                if response.status_code == 200:
                    st.sidebar.success("Trading parado!")
                else:
                    st.sidebar.error("Erro ao parar trading")
    
except requests.exceptions.ConnectionError:
    st.sidebar.error("Erro ao conectar com a API")
    st.error("""
    ### Erro de Conexão
    Não foi possível conectar com a API de trading.
    
    Verifique se:
    1. A API está rodando (`python trading_api.py`)
    2. A porta 8000 está disponível
    3. Não há firewall bloqueando a conexão
    """)

# Área principal
st.markdown("## Dashboard")

# Placeholder para o gráfico
chart_placeholder = st.empty()

# Função para atualizar o gráfico
def update_chart():
    # Aqui você pode adicionar a lógica para buscar dados reais
    # Por enquanto vamos usar dados de exemplo
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-03-01', periods=100, freq='1H'),
        'price': [50000 + i * 100 for i in range(100)]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        mode='lines',
        name='BTC/USDT'
    ))
    
    fig.update_layout(
        title='Preço BTC/USDT',
        xaxis_title='Data/Hora',
        yaxis_title='Preço (USDT)',
        template='plotly_dark'
    )
    
    return fig

# Atualiza o gráfico
chart_placeholder.plotly_chart(update_chart(), use_container_width=True)

# Métricas
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Retorno Diário",
        value="2.5%",
        delta="0.5%"
    )

with col2:
    st.metric(
        label="Drawdown",
        value="-1.2%",
        delta="-0.3%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Win Rate",
        value="65%",
        delta="5%"
    )

# Logs
st.markdown("## Logs do Sistema")
log_placeholder = st.empty()
log_placeholder.code("Iniciando sistema de trading...\nConectado à exchange\nAguardando sinais...") 