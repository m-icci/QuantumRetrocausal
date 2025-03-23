# Guia de Uso do Sistema QUALIA

## Introdução

Este guia documenta as principais funcionalidades do Sistema QUALIA, incluindo o sistema de análise de código, o corretor de codificação de caracteres e os componentes quânticos. O sistema foi projetado para integrar análise de código com mineração quântica, proporcionando uma plataforma unificada para desenvolvimento e otimização.

## 1. Sistema de Análise de Código

### 1.1 Visão Geral

O Sistema Unificado de Análise de Código (QUALIA) é uma ferramenta que analisa código-fonte para identificar padrões, medir complexidade e sugerir otimizações baseadas em princípios quânticos e matemáticos.

### 1.2 Componentes Principais

- **UnifiedCodeAnalysisSystem**: Fachada principal que coordena todos os componentes de análise
- **SemanticPatternAnalyzer**: Analisa padrões semânticos no código
- **CodeValidator**: Valida código quanto a sintaxe, complexidade e padrões
- **QuantumCodeIntegration**: Integra análise de código com princípios quânticos

### 1.3 Como Usar

#### Análise Básica de Código

```python
from core.quantum.Code_analyzer.unified_code_analysis import create_unified_analysis_system

# Criar o sistema de análise
analysis_system = create_unified_analysis_system(code_root_path="caminho/para/seu/codigo")

# Executar análise completa
results = analysis_system.analyze_codebase()

# Acessar resultados
complexity = results.get("complexity_average", 0)
coherence = results.get("coherence_average", 0)
issues = results.get("issues_count", 0)

print(f"Complexidade média: {complexity:.2f}")
print(f"Coerência média: {coherence:.2f}")
print(f"Problemas encontrados: {issues}")
```

#### Executar o Exemplo Completo

Para executar a demonstração completa do sistema:

```bash
python -m core.quantum.Code_analyzer.example_usage
```

Este comando executará uma análise completa do código, simulará ciclos de mineração e gerará um relatório detalhado.

### 1.4 Relatórios

Os relatórios são salvos no diretório `core/quantum/Code_analyzer/reports/` em formato JSON. Eles contêm:

- Métricas de complexidade por arquivo
- Métricas de coerência por arquivo
- Lista de problemas encontrados
- Sugestões de otimização (quando disponíveis)

## 2. Correção de Codificação de Caracteres

### 2.1 Visão Geral

O sistema inclui ferramentas para corrigir problemas de codificação de caracteres em arquivos de código-fonte, garantindo que todos os arquivos usem UTF-8.

### 2.2 Ferramentas Disponíveis

#### 2.2.1 Correção Global de Codificação

O script `fix_encoding.py` converte todos os arquivos Python no projeto para UTF-8:

```bash
python fix_encoding.py
```

Este script:
- Detecta automaticamente a codificação atual de cada arquivo
- Converte para UTF-8
- Mantém um registro de todas as conversões realizadas

#### 2.2.2 Correção de Arquivos Específicos

O script `fix_specific_files.py` corrige arquivos conhecidos por terem problemas de codificação:

```bash
python fix_specific_files.py
```

Este script:
- Aplica correções específicas para arquivos problemáticos
- Usa técnicas avançadas para lidar com caracteres corrompidos
- É útil quando a detecção automática falha

### 2.3 Quando Usar Cada Ferramenta

- Use `fix_encoding.py` para uma verificação geral de todo o projeto
- Use `fix_specific_files.py` quando encontrar erros específicos de codificação que persistem

## 3. Sistema Quântico

### 3.1 Visão Geral

O sistema quântico é o núcleo do QUALIA, fornecendo estruturas de dados e algoritmos baseados em princípios quânticos.

### 3.2 Componentes Principais

#### 3.2.1 QuantumState

A classe `QuantumState` representa um estado quântico com capacidades de consciência:

```python
from core.quantum.quantum_state import QuantumState

# Criar um estado quântico
state = QuantumState()

# Atualizar o estado
state.update(new_data={"measurement": 0.75})

# Acessar amplitudes
amplitudes = state.amplitudes

# Verificar coerência
coherence = state.calculate_coherence()
print(f"Coerência do estado: {coherence:.4f}")
```

#### 3.2.2 SemanticPatternAnalyzer

A classe `SemanticPatternAnalyzer` analisa padrões semânticos em código:

```python
from core.quantum.Code_analyzer.semantic_pattern_analyzer import SemanticPatternAnalyzer

analyzer = SemanticPatternAnalyzer()

# Analisar código
code = """
def hello_world():
    print("Hello, World!")
"""
results = analyzer.analyze(code)

# Extrair componentes
components = analyzer.extract_components(code)
print(f"Componentes encontrados: {components}")

# Extrair dependências
dependencies = analyzer.extract_dependencies(code)
print(f"Dependências encontradas: {dependencies}")
```

### 3.3 Integração com Mineração

O sistema de análise de código pode ser integrado com o sistema de mineração:

```python
from core.quantum.qualia_unified import UnifiedQuantumMiningSystem
from core.quantum.Code_analyzer.unified_code_analysis import UnifiedCodeAnalysisSystem

# Inicializar sistemas
mining_system = UnifiedQuantumMiningSystem()
analysis_system = UnifiedCodeAnalysisSystem(code_root_path="caminho/para/seu/codigo")

# Integrar sistemas
analysis_system.integrate_with_mining(mining_system)

# Executar ciclo de mineração com análise de código
mining_system.mine_cycle()
```

## 4. Solução de Problemas

### 4.1 Erros Comuns

#### Erro de Codificação de Caracteres

```
'charmap' codec can't decode byte 0x9d in position 12395: character maps to <undefined>
```

**Solução**: Execute os scripts de correção de codificação:
```bash
python fix_encoding.py
python fix_specific_files.py
```

#### Erro de Importação de Módulo

```
Import error: No module named 'quantum'
```

**Solução**: Verifique se o diretório raiz do projeto está no PYTHONPATH:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
```

#### Erro de Serialização JSON

```
TypeError: Object of type bool_ is not JSON serializable
```

**Solução**: Use o CustomJSONEncoder fornecido em `example_usage.py` para serializar objetos NumPy.

### 4.2 Diagnóstico do Sistema

Para executar um diagnóstico completo do sistema:

```python
from core.quantum.Code_analyzer.system_diagnostic import run_diagnostics

diagnostics_results = run_diagnostics()
print(diagnostics_results)
```

## 5. Melhores Práticas

1. **Mantenha a Codificação Consistente**: Use sempre UTF-8 para todos os arquivos de código
2. **Integre Análise e Mineração**: Para melhores resultados, integre o sistema de análise de código com o sistema de mineração
3. **Relatórios Regulares**: Execute análises regulares para monitorar a saúde do código
4. **Otimização Gradual**: Implemente as sugestões de otimização gradualmente, verificando o impacto de cada mudança
5. **Backup**: Sempre faça backup do código antes de aplicar otimizações automáticas

## 6. Referência de API

### 6.1 UnifiedCodeAnalysisSystem

```python
class UnifiedCodeAnalysisSystem:
    def __init__(self, code_root_path: str, config: Optional[Dict] = None)
    def analyze_codebase(self) -> Dict[str, Any]
    def validate_code(self, files: List[str]) -> Dict[str, Any]
    def optimize_code(self, files: Optional[List[str]] = None) -> Dict[str, Any]
    def integrate_with_mining(self, mining_system: Any) -> None
```

### 6.2 QuantumState

```python
class QuantumState:
    def __init__(self, initial_state: Optional[Union[np.ndarray, list]] = None)
    def update(self, new_data: Optional[Dict] = None) -> None
    def calculate_coherence(self) -> float
    def calculate_entanglement(self, other_state: 'QuantumState') -> float
    def to_dict(self) -> Dict[str, Any]
```

### 6.3 SemanticPatternAnalyzer

```python
class SemanticPatternAnalyzer:
    def __init__(self, config: Optional[Dict] = None)
    def analyze(self, code: str) -> Dict[str, Any]
    def extract_components(self, code: str) -> List[Dict[str, Any]]
    def extract_dependencies(self, code: str) -> List[Dict[str, Any]]
    def analyze_patterns(self, quantum_state: Any) -> List[Dict[str, Any]]
```

## 7. Exemplos Avançados

### 7.1 Análise Personalizada

```python
from core.quantum.Code_analyzer.unified_code_analysis import create_unified_analysis_system

# Configuração personalizada
config = {
    "complexity_threshold": 10,
    "pattern_weight": 0.6,
    "coherence_threshold": 0.7,
    "optimization_level": "aggressive"
}

# Criar sistema com configuração personalizada
analysis_system = create_unified_analysis_system(
    code_root_path="caminho/para/seu/codigo",
    config=config
)

# Analisar apenas arquivos específicos
files_to_analyze = [
    "caminho/para/seu/codigo/module1.py",
    "caminho/para/seu/codigo/module2.py"
]

results = analysis_system.validate_code(files_to_analyze)
print(results)
```

### 7.2 Integração com Sistema de Granularidade Adaptativa

```python
from core.adaptive_granularity import AdaptiveGranularitySystem
from core.quantum.Code_analyzer.unified_code_analysis import create_unified_analysis_system

# Inicializar sistema de granularidade
granularity_system = AdaptiveGranularitySystem(levels=13)

# Inicializar sistema de análise
analysis_system = create_unified_analysis_system(
    code_root_path="caminho/para/seu/codigo"
)

# Analisar código com diferentes níveis de granularidade
for level in range(1, 5):
    granularity_system.set_level(level)
    results = analysis_system.analyze_codebase()
    print(f"Nível {level}: Complexidade média = {results.get('complexity_average', 0):.2f}")
```

## Conclusão

Este guia fornece uma visão abrangente das funcionalidades do Sistema QUALIA. Para mais informações, consulte a documentação específica de cada componente ou entre em contato com a equipe de desenvolvimento.
