import os
import pandas as pd
from setuptools import setup

setup(
    name="quantumfinance",
    version="0.1.0",
    description="Projeto Deep Learning e ETL para previsão de ações.",
    author="Seu Nome",
    author_email="seu@email.com",
    py_modules=["main"],  # Só o main.py
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tensorflow>=2.9",
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "quantumfinance-main=main:main"
        ]
    },
)

from treinamento import treinar_modelo_lstm

# Exemplo de chamada
model, history = treinar_modelo_lstm(X_train, y_train, X_test, y_test)


# Use o mesmo caminho do seu ETL
PATH_TREINO = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\BBAS3.SA\BBAS3_SA_treino.csv'
PATH_TESTE  = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\BBAS3.SA\BBAS3_SA_teste.csv'

def main():
    for path in [PATH_TREINO, PATH_TESTE]:
        if not os.path.exists(path):
            print(f"❌ Arquivo não encontrado: {path}")
            return

    print("Carregando dados...")
    df_treino = pd.read_csv(PATH_TREINO)
    df_teste = pd.read_csv(PATH_TESTE)

    print("Dados carregados com sucesso!")
    print(df_treino.head())

if __name__ == "__main__":
    main()

