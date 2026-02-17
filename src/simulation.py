import pandas as pd
import matplotlib.pyplot as plt

def simulate_inventory(demand_series, order_quantity, reorder_point, lead_time=0, initial_stock=1000,
                       holding_cost_per_unit=0.0005, shortage_cost_per_unit=100, order_cost=200):
    stock = initial_stock
    total_holding = 0
    total_shortage = 0
    total_orders = 0
    stock_levels = [initial_stock]  # Começa com estoque inicial
    
    for demand in demand_series:
        # Consumo da demanda
        stock -= demand
        
        # Shortage (estoque negativo)
        if stock < 0:
            total_shortage += abs(stock)
            stock = 0
        
        # Custo de holding (estoque positivo)
        total_holding += stock
        
        # Decisão de pedido (política (s,Q) - pedido quando <= reorder_point)
        if stock <= reorder_point:
            stock += order_quantity
            total_orders += 1
        
        # Registra o nível de estoque AO FINAL do dia (após consumo e possível pedido)
        stock_levels.append(stock)
    
    # Cálculo de custos totais
    total_cost = (total_holding * holding_cost_per_unit +
                  total_shortage * shortage_cost_per_unit +
                  total_orders * order_cost)
    
    costs = {
        'total_cost': total_cost,
        'avg_holding': total_holding / len(demand_series),
        'total_shortage': total_shortage,
        'total_orders': total_orders
    }
    
    # Retorna níveis de estoque sem o valor inicial (para alinhar com as datas)
    return costs, stock_levels[1:]