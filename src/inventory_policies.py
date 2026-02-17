import numpy as np

def eoq(demand_rate, setup_cost, holding_cost):
    """Economic Order Quantity"""
    return np.sqrt(2 * demand_rate * setup_cost / holding_cost)

def reorder_point(demand_mean, demand_std, lead_time, service_level_z=1.65):
    """Ponto de reposição com safety stock (95% service level)"""
    safety_stock = service_level_z * np.sqrt(lead_time) * demand_std
    return demand_mean * lead_time + safety_stock