import numpy as np

def eoq(demand_rate, setup_cost, holding_cost):
    """Economic Order Quantity with check for invalid values"""
    if holding_cost <= 0 or demand_rate <= 0 or setup_cost <= 0:
        return 0  # or a large default, e.g. 1e6, to avoid nan in simulation
    return np.sqrt(2 * demand_rate * setup_cost / holding_cost)

def reorder_point(demand_mean, demand_std, lead_time, service_level_z=1.65):
    """Ponto de reposição com safety stock"""
    if demand_std < 0 or lead_time < 0:
        return 0
    safety_stock = service_level_z * np.sqrt(lead_time) * demand_std
    return demand_mean * lead_time + safety_stock