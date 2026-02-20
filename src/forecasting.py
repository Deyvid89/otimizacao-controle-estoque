import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def load_and_prepare_data(store_id=1, data_path='data/'):
    train = pd.read_csv(data_path + 'train.csv', low_memory=False, parse_dates=['Date'])
    store = pd.read_csv(data_path + 'store.csv')
    
    df = train[train['Store'] == store_id].merge(store, on='Store', how='left')
    df = df[df['Open'] == 1].copy()
    df = df[['Date', 'Sales']].sort_values('Date').set_index('Date')
    return df

def create_features(df, lag_days=[1, 7, 14, 30]):
    data = df.copy()
    
    # Features calendário (sempre presentes)
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    
    # Separar histórico e futuro
    historical = data[data['Sales'] > 0].copy()
    future_index = data[data['Sales'] == 0].index
    
    if not future_index.empty:
        # Repetir o padrão dos últimos 7 dias de Sales no futuro
        last_7_sales = historical['Sales'].tail(7).values
        repeated_sales = np.tile(last_7_sales, (len(future_index) // 7 + 1))[:len(future_index)]
        data.loc[future_index, 'Sales'] = repeated_sales
        
        # Calcular lags e rolling com Sales preenchido
        for lag in lag_days:
            data[f'lag_{lag}'] = data['Sales'].shift(lag)
        
        data['rolling_mean_7'] = data['Sales'].rolling(7, min_periods=1).mean()
        data['rolling_mean_30'] = data['Sales'].rolling(30, min_periods=1).mean()
        
        # Preencher qualquer NaN restante
        data = data.ffill().fillna(data['Sales'].mean())
    
    else:
        # Se não houver futuro, calcular normalmente
        for lag in lag_days:
            data[f'lag_{lag}'] = data['Sales'].shift(lag)
        
        data['rolling_mean_7'] = data['Sales'].rolling(7, min_periods=1).mean()
        data['rolling_mean_30'] = data['Sales'].rolling(30, min_periods=1).mean()
        data = data.fillna(data['Sales'].mean())
    
    return data

def train_and_predict(store_id=1, test_days=60):
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