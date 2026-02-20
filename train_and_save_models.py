import os
import joblib
from src.forecasting import load_and_prepare_data, create_features, train_and_predict

os.makedirs('models', exist_ok=True)

lojas = [1, 2, 3, 4, 5, 10, 20]   # as lojas que aparecem no selectbox

print("🔄 Treinando e salvando modelos para o deploy...\n")

for store_id in lojas:
    print(f"→ Treinando modelo para loja {store_id}...")
    model, _, _, _, mae = train_and_predict(store_id=store_id, test_days=60)
    filename = f"models/model_loja_{store_id}.joblib"
    joblib.dump(model, filename)
    print(f"  ✅ Salvo: {filename} (MAE: {mae:.2f})\n")

print("🎉 Todos os modelos foram salvos na pasta 'models/'!")
print("Agora o app vai abrir instantaneamente no Streamlit Cloud.")