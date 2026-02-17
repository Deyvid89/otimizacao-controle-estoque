# Otimização de Controle de Estoque com Previsão de Demanda

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)

**Dashboard ao vivo:**  
https://seu-app.streamlit.app  
*(substitua pelo link após o deploy)*

---

## Visão Geral

Este projeto combina **previsão de demanda com machine learning** e **políticas clássicas de Pesquisa Operacional** para otimizar o controle de estoque no varejo.

A solução integra:

- XGBoost para previsão de vendas  
- Economic Order Quantity (EOQ)  
- Ponto de reposição e safety stock  
- Simulação de custos para comparação de políticas  

A análise utiliza o dataset público *Rossmann Store Sales* (Kaggle).

---

## Objetivo

Demonstrar que políticas tradicionais de controle de estoque, quando apoiadas por previsões de demanda mais precisas, podem:

- Reduzir custos operacionais totais  
- Minimizar rupturas de estoque  
- Estabilizar níveis de inventário  

Os experimentos indicam reduções de custo de aproximadamente **23%** quando comparados a estratégias simples (ex.: pedidos mensais fixos).

---

## Contexto do Problema

A demanda no varejo normalmente apresenta:

- Sazonalidade semanal  
- Impacto de promoções  
- Tendências ao longo do tempo  

Estratégias simples de reposição frequentemente resultam em:

- Excesso de estoque (alto custo de armazenagem)  
- Falta de produtos (perda de vendas)

Este projeto aborda esses problemas por meio de:

- Engenharia de atributos para séries temporais  
- Previsão supervisionada com machine learning  
- Simulação de políticas de estoque  
- Dashboard interativo para apoio à decisão  

---

## Funcionalidades

### Previsão de Demanda

- Previsão de vendas diárias com XGBoost  
- MAE típico: ~600–800  

### Simulação de Estoque

Comparação entre:

- EOQ + ponto de reposição (política dinâmica)  
- Pedido mensal fixo (baseline)

### Análise de Custos

- Custo de armazenagem (holding)  
- Custo de pedido  
- Custo de ruptura  

### Dashboard Interativo (Streamlit)

- Cards de KPI:
  - Custo total
  - Economia gerada
- Visualizações interativas com Plotly:
  - Previsão de demanda
  - Evolução do nível de estoque
- Controles na sidebar:
  - Seleção de loja
  - Parâmetros de custo
  - Lead time
  - Horizonte de simulação  

---

## Tecnologias Utilizadas

- Python 3.12  
- XGBoost  
- Pandas  
- NumPy  
- Scikit-learn  
- Plotly  
- Streamlit  
- Matplotlib / Seaborn  

---

## Estrutura do Projeto

```
inventory-optimization/
├── data/                  # Arquivos do dataset (train.csv, store.csv, test.csv)
├── notebooks/             # Análise exploratória e prototipagem
│   ├── 01_exploracao.ipynb
│   ├── 02_forecasting.ipynb
│   └── 03_simulacao_estoque.ipynb
├── src/                   # Código modularizado
│   ├── forecasting.py
│   ├── inventory_policies.py
│   └── simulation.py
├── .streamlit/            # Configuração do Streamlit
├── app.py                 # Aplicação principal (dashboard)
├── requirements.txt
└── README.md
```

---

## Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/inventory-optimization.git
cd inventory-optimization
```

### 2. Criar um ambiente virtual (recomendado)

```bash
python -m venv .venv

# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar o dataset

Baixe o dataset **Rossmann Store Sales** no Kaggle e coloque os arquivos:

```
data/train.csv
data/store.csv
data/test.csv
```

### 5. Executar o dashboard

```bash
streamlit run app.py
```

---

## Deploy (Streamlit Community Cloud)

1. Crie uma conta em: https://share.streamlit.io  
2. Conecte seu repositório no GitHub  
3. Configure:
   - `requirements.txt`
   - `app.py` como arquivo principal  
4. O deploy será atualizado automaticamente a cada push

---

## Principais Resultados

- Redução de custo total: aproximadamente 23%  
  (Exemplo: Loja 1, horizonte de 180 dias)

- Níveis de estoque mais estáveis  
- Menor capital imobilizado  
- Redução do risco de ruptura  

O dashboard permite explorar interativamente os impactos operacionais ao ajustar parâmetros do sistema.

---

## Screenshots

Adicione aqui imagens ou GIFs demonstrando o dashboard.

---

## Autor

Deyvid Araújo  
Data Scientist | Entusiasta de Otimização e Machine Learning  

LinkedIn: (adicionar link)  
GitHub: (adicionar link)

---

## Licença

MIT License

Copyright (c) 2026 Deyvid Araújo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the conditions stated in the license.
