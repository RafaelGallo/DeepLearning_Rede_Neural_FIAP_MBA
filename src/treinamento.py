import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def treinar_modelo_lstm(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Treina um modelo LSTM simples para séries temporais.
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Regressão (preço)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return model, history

# Exemplo de uso:
if __name__ == "__main__":
    # Substitua pelos seus arrays (já processados e escalados)
    # X_train, y_train, X_val, y_val = ...

    # model, history = treinar_modelo_lstm(X_train, y_train, X_val, y_val)
    print("Função de treinamento pronta! Importe em main.py ou chame aqui com seus dados.")
