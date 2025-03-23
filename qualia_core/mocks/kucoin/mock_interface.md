# KuCoin Mock Service Interface

Este documento define a interface comum que deve ser implementada tanto pela versão TypeScript quanto Python do mock service.

## Interface Principal

### WebSocket
- Simulação de streams de preço
- Gerenciamento de conexões
- Broadcast de atualizações
- Resiliência e reconexão

### REST API
- Criação de ordens (buy/sell)
- Consulta de ordens
- Preços de símbolos
- Estado do mercado

### Quantum Features
- Preservação de coerência quântica
- Campos mórficos φ-adaptativos
- Ressonância não-local
- Auto-organização emergente

### Logging & Métricas
- Log de todas requisições
- Métricas de performance
- Estado do sistema
- Diagnósticos

## Organização

```
quantum/core/mocks/kucoin/
├── ts/                     # Implementação TypeScript
│   ├── mock_service.ts     # Serviço principal
│   └── types.ts           # Tipos e interfaces
├── py/                     # Implementação Python  
│   ├── mock_service.py    # Serviço principal
│   └── types.py          # Tipos e classes
└── mock_interface.md      # Este documento
```

## Princípios

1. Coerência: Manter consistência entre implementações TS e Python
2. φ-Adaptatividade: Usar razão áurea como parâmetro natural
3. Auto-organização: Permitir emergência de padrões
4. Resiliência: Tratamento robusto de erros e reconexão
5. Observabilidade: Logging e métricas abrangentes
