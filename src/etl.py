### ETL para preparação base dados
import os
import pandas as pd
import numpy as np

## Carregadno base dados
# # Base dados - BBAS3_SA
BBAS3_SA_train = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\BBAS3.SA\BBAS3_SA_treino.csv'
BBAS3_SA_test =  r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\BBAS3.SA\BBAS3_SA_teste.csv'

# Base dados - CSNA3_SA
CSNA3_SA_train = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\CSNA3.SA\CSNA3_SA_treino.csv'
CSNA3_SA_test = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\CSNA3.SA\CSNA3_SA_teste.csv'

# Base dados - PETR4_SA
PETR4_SA_train = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\PETR4.SA\PETR4_SA_treino.csv'
PETR4_SA_test = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\PETR4.SA\PETR4_SA_teste.csv'

# Base dados - VALE3_SA
VALE3_SA_train = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\VALE3.SA\VALE3_SA_treino.csv'
VALE3_SA_test = r'G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\data\VALE3.SA\VALE3_SA_teste.csv'

# Função auxiliar para carregar e limpar (remover Unnamed: 0)
def carrega_limpa_csv(caminho):
    return pd.read_csv(caminho, usecols=lambda c: c != 'Unnamed: 0')

## Carregando dataset
# BBAS3
df_train_bbas = carrega_limpa_csv(BBAS3_SA_train)
df_test_bbas  = carrega_limpa_csv(BBAS3_SA_test)

# CSNA3
df_train_csna = carrega_limpa_csv(CSNA3_SA_train)
df_test_csna  = carrega_limpa_csv(CSNA3_SA_test)

# PETR4
df_train_petr = carrega_limpa_csv(PETR4_SA_train)
df_test_petr  = carrega_limpa_csv(PETR4_SA_test)

# VALE3
df_train_vale = carrega_limpa_csv(VALE3_SA_train)
df_test_vale  = carrega_limpa_csv(VALE3_SA_test)

# Carregar as bases
df_bbas = pd.read_csv(BBAS3_SA_train)
df_csna = pd.read_csv(CSNA3_SA_train)
df_petr = pd.read_csv(PETR4_SA_train)
df_vale = pd.read_csv(VALE3_SA_train)

# Renomear colunas para evitar conflito, exceto Unnamed: 0
df_bbas = df_bbas.add_prefix('BBAS3_')
df_csna = df_csna.add_prefix('CSNA3_')
df_petr = df_petr.add_prefix('PETR4_')
df_vale = df_vale.add_prefix('VALE3_')

# Carregar bases
dfs = []

#
for ticker, path in [('BBAS3', BBAS3_SA_train),
                     ('CSNA3', CSNA3_SA_train),
                     ('PETR4', PETR4_SA_train),
                     ('VALE3', VALE3_SA_train)]:

    df = pd.read_csv(path)
    df['Ticker'] = ticker
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    dfs.append(df)

# Concatenar tudo (long format)
data = pd.concat(dfs, ignore_index=True)

# (Opcional) Organizar colunas
cols = ['Ticker', 'Date'] + [col for col in data.columns if col not in ['Ticker','Date']]
data = data[cols]

# Visualizando dataset
print(data.head())

# Salvando dataset
data.to_csv(r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Deep learning\notebooks\Projeto_final_2\Rede_Neural_FIAP_MBA_QuantumFinance\output\dataset.csv", index=False)
print("✔️ Dataset salvo com sucesso!")