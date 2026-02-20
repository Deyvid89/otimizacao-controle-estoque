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
    """Cria features de séries temporais, preenchendo NaN no futuro"""
    data = df.copy()
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    
    # Lags e rolling
    for lag in lag_days:
        data[f'lag_{lag}'] = data['Sales'].shift(lag)
    
    data['rolling_mean_7'] = data['Sales'].rolling(7, min_periods=1).mean()
    data['rolling_mean_30'] = data['Sales'].rolling(30, min_periods=1).mean()
    
    # Preenche NaN no futuro com o último valor conhecido (forward fill)
    data = data.ffill()
    
    # Preenche qualquer NaN restante com média global (raro)
    data = data.fillna(data['Sales'].mean())
    
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