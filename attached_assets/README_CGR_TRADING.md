# Sistema de Trading Quântico CGR

## Descrição
Este sistema de trading utiliza análise CGR (Chaos Game Representation) combinada com elementos quânticos e retrocausais para identificar padrões de mercado e executar operações de trading em criptomoedas nas exchanges KuCoin e Kraken.

## Características Principais
- **Trading Multi-Exchange**: Operações coordenadas entre KuCoin e Kraken
- **Análise Quântica**: Estado quântico adaptativo que responde a condições de mercado
- **Proteção contra Decoerência**: Sistema que mantém a estabilidade do algoritmo 
- **Detecção de Arbitragem**: Identifica oportunidades entre exchanges
- **Gerenciamento Avançado de Risco**: Stops dinâmicos e trailing stops
- **Relatórios Detalhados**: Métricas completas de desempenho após cada execução

## Requisitos
- Python 3.8+
- Credenciais de API para KuCoin e Kraken
- Bibliotecas: `numpy`, `pandas`, `requests`, `python-dotenv`, `colorama`

## Configuração
1. Clone o repositório
2. Configure as variáveis de ambiente com suas credenciais API:
   ```
   KUCOIN_API_KEY=sua_api_key
   KUCOIN_API_SECRET=seu_api_secret
   KUCOIN_API_PASSPHRASE=sua_api_passphrase
   KRAKEN_API_KEY=sua_api_key_kraken
   KRAKEN_API_SECRET=seu_api_secret_kraken
   ```
3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Uso
Execute o script com os parâmetros desejados:

```
python CGR_trading.py --modo [real|simulacao] --duracao 60 --pares BTC-USDT ETH-USDT
```

### Parâmetros disponíveis
- `--modo`: Escolha entre `real` (trading com dinheiro real) ou `simulacao` (sem ordens reais)
- `--saldo`: Saldo inicial para simulação (em USDT)
- `--duracao`: Duração da execução em minutos (padrão: 60 - 1 hora)
- `--intervalo`: Intervalo entre análises em segundos (padrão: 5)
- `--pares`: Lista de pares de trading (padrão: BTC-USDT ETH-USDT)
- `--limite-forca`: Limite mínimo da força do sinal para executar operações (padrão: 0.3)
- `--arquivo-config`: Caminho para um arquivo de configuração JSON

### Exemplo de arquivo de configuração JSON
```json
{
    "modo": "simulacao",
    "saldo_inicial": 1000.0,
    "duracao_minutos": 60,
    "tempo_entre_trades": 5,
    "pares_trading": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    "limite_forca_sinal": 0.3
}
```

## Modo Real vs Simulação
- **Modo Simulação**: Executa todas as análises e estratégias, mas não realiza ordens reais. Útil para testes.
- **Modo Real**: Conecta-se às exchanges e executa ordens reais com seu saldo disponível. CUIDADO: Este modo utiliza fundos reais!

## Análise do Mercado
O sistema utiliza diversas técnicas para analisar o mercado:

1. **CGR (Chaos Game Representation)**: Análise de padrões usando fractal e teoria do caos
2. **Análise Retrocausal**: Previsão usando métodos quânticos retrocausais
3. **Análise Wavelet**: Decomposição de séries temporais para identificar tendências
4. **Entropia e Dimensão Fractal**: Medidas de complexidade do mercado

## Métricas e Relatórios
Ao final da execução, o sistema gera um relatório detalhado com:
- Resumo financeiro (saldo inicial, final, lucro)
- Métricas de trading (total de operações, win rate)
- Métricas avançadas (Sharpe Ratio, Drawdown, Volatilidade)
- Composição do portfólio final
- Análise de risco
- Métricas quânticas
- Recomendações baseadas nos resultados

## Segurança e Boas Práticas
- Nunca compartilhe suas chaves de API
- Comece com valores pequenos em modo real
- Teste extensivamente em modo simulação antes de usar dinheiro real
- Monitore regularmente as operações do bot
- Utilize stop-loss para limitar perdas potenciais

## Estratégia Multi-Exchange
O sistema determina automaticamente a melhor exchange para cada operação baseado em:
- Preço mais favorável (menor para compra, maior para venda)
- Taxas de trading
- Liquidez disponível
- Histórico de execuções bem-sucedidas

## Recomendações de Uso
1. Inicie com simulações de pelo menos 24 horas para observar o comportamento em diferentes condições de mercado
2. Teste com pares de alta liquidez (BTC-USDT, ETH-USDT)
3. Comece com valores baixos em modo real
4. Monitore os relatórios para ajustar parâmetros como limite de força do sinal
5. Avalie a coerência quântica nos relatórios para verificar a estabilidade do algoritmo

## Aviso de Risco
Trading de criptomoedas envolve risco significativo e pode resultar em perda de capital. Este sistema não garante lucros e deve ser usado com cautela. O autor não se responsabiliza por perdas financeiras resultantes do uso deste software.
