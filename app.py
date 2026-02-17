import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.forecasting import load_and_prepare_data, create_features, train_and_predict
from src.inventory_policies import eoq, reorder_point
from src.simulation import simulate_inventory

st.set_page_config(page_title="Otimização de Estoque - Rossmann", layout="wide")
st.title("🛒 Otimização de Controle de Estoque - Rossmann")
st.markdown("Previsão de demanda com XGBoost + Políticas Clássicas (EOQ) vs. Baseline Mensal")

# Sidebar
st.sidebar.header("Parâmetros da Simulação")
store_id = st.sidebar.selectbox("Loja", options=[1, 2, 3, 4, 5, 10, 20], index=0)
simulation_days = st.sidebar.slider("Dias de simulação", 60, 365, 180)
order_cost = st.sidebar.number_input("Custo fixo por pedido", 50, 500, 200)
holding_cost_day = st.sidebar.number_input("Custo holding por unidade/dia", 0.0001, 0.005, 0.0005, format="%.6f")
shortage_cost = st.sidebar.number_input("Penalidade por ruptura/unidade", 50, 500, 100)
lead_time = st.sidebar.slider("Lead time (dias)", 0, 14, 7)

# Cache da previsão
@st.cache_data
def get_forecast(store_id, simulation_days):
    model, _, _, _, _ = train_and_predict(store_id=store_id, test_days=60)
    df_full = load_and_prepare_data(store_id=store_id)
    df_feat = create_features(df_full)
    
    last_date = df_feat.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=simulation_days)
    full_df = pd.concat([df_feat, pd.DataFrame({'Sales': [0]*simulation_days}, index=future_dates)])
    full_df = create_features(full_df)
    future_features = full_df.tail(simulation_days).drop('Sales', axis=1)
    future_pred = model.predict(future_features)
    demand_series = pd.Series(future_pred.round(0).astype(int), index=future_dates)
    
    return demand_series

demand_series = get_forecast(store_id, simulation_days)

# Estatísticas e políticas
demand_mean = demand_series.mean()
demand_std = demand_series.std()

Q_eoq = eoq(demand_rate=demand_mean * 365, setup_cost=order_cost, holding_cost=holding_cost_day * 365)
s_reorder = reorder_point(demand_mean=demand_mean, demand_std=demand_std, lead_time=lead_time)
monthly_quantity = demand_mean * 30

# Simulações
costs_eoq, stock_levels_eoq = simulate_inventory(
    demand_series=demand_series, order_quantity=Q_eoq, reorder_point=s_reorder,
    lead_time=lead_time, initial_stock=Q_eoq + s_reorder,
    holding_cost_per_unit=holding_cost_day, shortage_cost_per_unit=shortage_cost, order_cost=order_cost
)

# Baseline mensal
initial_stock_monthly = monthly_quantity + demand_mean * lead_time
stock = initial_stock_monthly
total_holding = total_shortage = total_orders = 0
stock_levels_monthly = [initial_stock_monthly]
day_count = 0

for demand in demand_series:
    day_count += 1
    stock -= demand
    if stock < 0:
        total_shortage += abs(stock)
        stock = 0
    total_holding += stock
    if day_count % 30 == 0:
        stock += monthly_quantity
        total_orders += 1
    stock_levels_monthly.append(stock)

costs_monthly = {
    'total_cost': total_holding * holding_cost_day + total_shortage * shortage_cost + total_orders * order_cost,
    'avg_holding': total_holding / len(demand_series),
    'total_shortage': total_shortage,
    'total_orders': total_orders
}
stock_levels_monthly = stock_levels_monthly[1:]

# KPI cards
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Custo Total EOQ", f"R$ {costs_eoq['total_cost']:.2f}")
col_kpi2.metric("Custo Total Baseline", f"R$ {costs_monthly['total_cost']:.2f}")
col_kpi3.metric("Economia com EOQ", f"R$ {costs_monthly['total_cost'] - costs_eoq['total_cost']:.2f}")

# Tabs
tab1, tab2 = st.tabs(["Previsão e Custos", "Níveis de Estoque"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Previsão de Demanda (XGBoost)")
        fig_demand = px.line(demand_series.reset_index(), x='index', y=0, labels={'index': 'Data', 0: 'Demanda'})
        fig_demand.update_traces(line=dict(color='#2ca02c', width=3))
        fig_demand.update_layout(height=450, margin=dict(l=40, r=20, t=40, b=80))
        st.plotly_chart(fig_demand, use_container_width=True)
    
    with col2:
        st.subheader("Comparação de Custos")
        comparison = pd.DataFrame({
            'Política': ['EOQ + Reorder Point', 'Pedido Mensal Fixo'],
            'Custo Total': [costs_eoq['total_cost'], costs_monthly['total_cost']],
            'Custo Holding Médio/Dia': [costs_eoq['avg_holding'] * holding_cost_day, costs_monthly['avg_holding'] * holding_cost_day],
            'Rupturas Total': [costs_eoq['total_shortage'] * shortage_cost, costs_monthly['total_shortage'] * shortage_cost],
            'Pedidos': [costs_eoq['total_orders'], costs_monthly['total_orders']]
        })
        st.dataframe(
            comparison.style.format({
                'Custo Total': "{:.2f}",
                'Custo Holding Médio/Dia': "{:.2f}",
                'Rupturas Total': "{:.2f}",
                'Pedidos': "{:.0f}"
            })
        )  # Removido height — tabela ajusta automática ao conteúdo (sem linhas vazias)

with tab2:
    st.subheader("Níveis de Estoque ao Longo do Tempo")
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(x=demand_series.index, y=stock_levels_eoq, mode='lines', name='EOQ + Reorder Point', line=dict(color='#1f77b4', width=4)))
    fig_stock.add_trace(go.Scatter(x=demand_series.index, y=stock_levels_monthly, mode='lines', name='Pedido Mensal Fixo', line=dict(color='#ff7f0e', width=4)))
    fig_stock.add_hline(y=0, line_dash="dash", line_color="#d62728", line_width=3, annotation_text="Nível de Ruptura")
    fig_stock.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=40, b=80),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0.01)  # Legenda vertical na esquerda
    )
    st.plotly_chart(fig_stock, use_container_width=True)

st.caption("Dashboard interativo completo. Hover nos gráficos para detalhes, zoom e pan.")