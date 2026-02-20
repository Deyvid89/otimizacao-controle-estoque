import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def load_and_prepare_data(store_id=1, data_path='data/'):
    """Carrega e prepara os dados para uma loja específica"""
    train = pd.read_csv(data_path + 'train.csv', low_memory=False, parse_dates=['Date'])
    store = pd.read_csv(data_path + 'store.csv')
    
    df = train[train['Store'] == store_id].merge(store, on='Store', how='left')
    df = df[df['Open'] == 1].copy()
    df = df[['Date', 'Sales']].sort_values('Date').set_index('Date')
    return df

def create_features(df, lag_days=[1, 7, 14, 30]):
    """Cria features de séries temporais, repetindo padrão sazonal no futuro"""
    data = df.copy()
    
    # Média histórica para fallback
    historical_mean = data['Sales'][data['Sales'] > 0].mean() if (data['Sales'] > 0).any() else 0
    
    # Preencher zeros no histórico com média
    data['Sales'] = data['Sales'].replace(0, historical_mean).ffill()
    
    # Features calendário
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    
    # Lags e rolling no histórico
    for lag in lag_days:
        data[f'lag_{lag}'] = data['Sales'].shift(lag)
    
    data['rolling_mean_7'] = data['Sales'].rolling(7, min_periods=1).mean()
    data['rolling_mean_30'] = data['Sales'].rolling(30, min_periods=1).mean()
    
    # Para o futuro: repetir o padrão dos últimos 7 dias (proxy sazonal semanal)
    last_7_days = data.tail(7).copy()
    future_periods = len(data) - len(df)  # quantos dias futuros foram adicionados
    
    if future_periods > 0:
        repeated_pattern = pd.concat([last_7_days] * (future_periods // 7 + 1))
        repeated_pattern = repeated_pattern.iloc[:future_periods]
        repeated_pattern.index = data.index[-future_periods:]
        
        # Copiar features do padrão repetido para o futuro
        for col in repeated_pattern.columns:
            if col != 'Sales':
                data.loc[repeated_pattern.index, col] = repeated_pattern[col]
        
        # Preencher lags/rolling que ainda possam ter NaN com ffill
        data = data.ffill()
    
    # Fallback final
    data = data.fillna(historical_mean)
    
    return data

def train_and_predict(store_id=1, test_days=60):
    """Treina o modelo XGBoost e retorna modelo + previsão de teste"""
    df = load_and_prepare_data(store_id=store_id)
    df_feat = create_features(df)
    
    train_data = df_feat[:-test_days]
    test_data = df_feat[-test_days:]
    
    X_train = train_data.drop('Sales', axis=1)
    y_train = train_data['Sales']
    X_test = test_data.drop('Sales', axis=1)
    y_test = test_data['Sales']
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    
    return model, test_data.index, y_test, pred, mae