# Testes de Comunicação via Partículas de Hawking

Este diretório contém os testes para o sistema de comunicação via partículas de Hawking do QUALIA.

## Estrutura

- `test_hawking_simple.py`: Implementação principal dos testes
- `test_config.json`: Configuração padrão dos testes
- `run_tests.py`: Script para executar os testes e gerar relatórios
- `analyze_results.ipynb`: Notebook Jupyter para análise dos resultados
- `jupyter_notebook_config.py`: Configuração do Jupyter Notebook
- `start_notebook.py`: Script para iniciar o Jupyter Notebook
- `test_results/`: Diretório onde os resultados são salvos

## Como Executar

1. Certifique-se de ter as dependências instaladas:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute os testes com relatório:
   ```bash
   python run_tests.py
   ```

   Ou execute apenas os testes básicos:
   ```bash
   python test_hawking_simple.py
   ```

3. Para análise detalhada dos resultados, você pode:

   a. Iniciar o Jupyter Notebook com configuração personalizada:
   ```bash
   python start_notebook.py
   ```

   b. Ou iniciar o Jupyter Notebook manualmente:
   ```bash
   jupyter notebook analyze_results.ipynb
   ```

## Configuração

Você pode modificar os parâmetros dos testes editando o arquivo `test_config.json`:

```json
{
    "num_particles": 100,  // Número de partículas a serem simuladas
    "energy": 1.0,        // Energia inicial das partículas
    "mass": 1.0,          // Massa das partículas
    "time_step": 0.01     // Passo de tempo para propagação
}
```

As configurações do Jupyter Notebook podem ser ajustadas em `jupyter_notebook_config.py`:
- Porta e endereço IP
- Limites de memória
- Intervalos de autosave
- Configurações de segurança
- Logging

## Resultados

Os resultados dos testes são salvos no diretório `test_results/` com os seguintes arquivos:

- `config_TIMESTAMP.json`: Configuração usada nos testes
- `results_TIMESTAMP.json`: Resultados dos testes, incluindo:
  - Número total de partículas
  - Energia média
  - Entropia média
  - Desvios padrão
  - Valores mínimos e máximos
  - Correlações
- `report.txt`: Relatório detalhado dos testes
- `report.png`: Visualização gráfica dos resultados

## Análise dos Resultados

O notebook `analyze_results.ipynb` fornece uma análise detalhada dos resultados, incluindo:

1. Análise Estatística:
   - Estatísticas básicas (média, desvio padrão, etc.)
   - Correlações entre variáveis
   - Distribuições

2. Visualizações:
   - Gráficos de energia vs. entropia
   - Distribuições de energia e entropia
   - Evolução temporal das métricas

3. Conclusões e Recomendações:
   - Análise da estabilidade
   - Avaliação das correlações
   - Sugestões de otimização

## Logs

Os logs são exibidos no console e incluem informações sobre:
- Início e fim dos testes
- Criação de partículas
- Propagação de partículas
- Salvamento de resultados
- Geração de relatórios
- Inicialização do Jupyter Notebook

## Contribuindo

Para adicionar novos testes:

1. Crie uma nova função de teste em `test_hawking_simple.py`
2. Adicione a chamada da função em `run_tests()`
3. Atualize a documentação conforme necessário
4. Se necessário, adicione novas visualizações em `generate_report()`
5. Atualize o notebook de análise para incluir as novas métricas
6. Ajuste as configurações do Jupyter Notebook se necessário 