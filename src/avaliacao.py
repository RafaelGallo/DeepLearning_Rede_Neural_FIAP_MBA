# src/avaliacao.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def avaliar_modelo(model, X_test, y_test, scaler_y=None, nome_modelo='', nome_ticker=''):
    """
    Avalia o modelo usando MAE, RMSE e R2. Retorna dicionário de métricas.
    
    Parâmetros:
        model: modelo treinado (Keras ou sklearn)
        X_test: dados de teste (escalados)
        y_test: alvo de teste (escalado)
        scaler_y: scaler para inversão do target (opcional)
        nome_modelo: string para identificar o modelo
        nome_ticker: string para identificar o ticker
    
    Retorna:
        dict: {'Modelo': ..., 'Ticker': ..., 'MAE': ..., 'RMSE': ..., 'R2': ...}
    """
    # Previsão
    y_pred_scaled = model.predict(X_test)
    
    # Inverter escala se necessário
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)
    else:
        y_pred = y_pred_scaled
        y_true = y_test

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    resultado = {
        'Modelo': nome_modelo,
        'Ticker': nome_ticker,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    return resultado

# Exemplo de uso:
resultado = avaliar_modelo(model, X_test, y_test, scaler_y, nome_modelo='LSTM', nome_ticker='VALE3')
print(resultado)