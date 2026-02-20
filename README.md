# Otimização de Controle de Estoque com Previsão de Demanda

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)

**Dashboard interativo ao vivo** → [https://seu-app.streamlit.app](https://seu-app.streamlit.app) *(substitua pelo seu link após deploy)*

Projeto de portfólio que combina **previsão de demanda avançada (XGBoost)** com **políticas clássicas de Pesquisa Operacional (EOQ + ponto de reposição)** para otimizar o controle de estoque no varejo, usando o dataset público Rossmann Store Sales (Kaggle).

O objetivo é demonstrar que métodos tradicionais, quando combinados com boa previsão de demanda, superam baselines simples, resultando em **redução de custos de até ~23%** e níveis de estoque mais estáveis e seguros.

## Motivação
Em ambientes de varejo com demanda variável (sazonalidade semanal, promoções), políticas simples como "pedido mensal fixo" geram estoque excessivo ou risco de ruptura. Este projeto mostra como:
- Prever demanda com precisão (XGBoost + feature engineering de séries temporais).
- Aplicar EOQ e safety stock para equilibrar custos de holding, pedidos e rupturas.
- Visualizar resultados em um dashboard interativo.

## Funcionalidades
- Previsão de vendas diárias com XGBoost (MAE típico ~600-800).
- Simulação de políticas de estoque (EOQ + reorder point vs. pedido mensal fixo).
- Comparação quantitativa de custos (holding, pedidos, rupturas).
- Dashboard Streamlit com:
  - KPI cards (custo total e economia).
  - Gráficos interativos (Plotly): previsão de demanda e níveis de estoque.
  - Ajustes em tempo real via sidebar (loja, custos, lead time, horizonte).

## Tecnologias Utilizadas
- **Python 3.11**
- **XGBoost** – Previsão de demanda
- **Pandas, NumPy** – Manipulação de dados
- **Plotly** – Visualizações interativas
- **Streamlit** – Dashboard web
- **Scikit-learn, Matplotlib/Seaborn** – Suporte a modelagem e notebooks

## Estrutura do Projeto

```
inventory-optimization/
├── data/                  # Datasets (train.csv, store.csv) - adicione do Kaggle
├── notebooks/             # Análise exploratória e prototipagem
│   ├── 01_exploracao.ipynb
│   ├── 02_forecasting.ipynb
│   └── 03_simulacao_estoque.ipynb
├── src/                   # Código modularizado
│   ├── forecasting.py
│   ├── inventory_policies.py
│   └── simulation.py
├── .streamlit/            # Configuração de tema escuro
├── app.py                 # Dashboard principal
├── requirements.txt
└── README.md
```


## Como Rodar Localmente
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/inventory-optimization.git
   cd inventory-optimization

2. Crie e ative um ambiente virtual (recomendado):
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3. Instale as dependências:
pip install -r requirements.txt

4. Baixe o dataset Rossmann do Kaggle e coloque train.csv e store.csv na pasta data/.

5. Rode o dashboard:
streamlit run app.py


## Deploy (Streamlit Community Cloud - Gratuito)

1. Crie conta em share.streamlit.io
2. Conecte seu repositório GitHub
3. Configure requirements.txt e app.py como entrada
4. Deploy automático a cada push

## Resultados Principais

* **Redução de custo total**: ~23% com EOQ vs. baseline mensal (exemplo com loja 1, 180 dias).
* Estoque mais estável, sem rupturas e com menor capital imobilizado.
* Dashboard responsivo e interativo (tema escuro forçado para melhor contraste).

## Screenshots

## Autor

Deyvid Araújo – Data Scientist | Entusiasta de Otimização e Machine Learning
LinkedIn | GitHub
Feedback e colaborações são bem-vindos!

## License
 
The MIT License (MIT)

Copyright (c) 2026 Deyvid Araújo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.