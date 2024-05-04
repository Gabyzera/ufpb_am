import numpy as np
import pandas as pd

def calcular_intensidade_e_simetria(row):
    pixelx = row[1:].values.reshape(28, 28)

    intensidade = pixelx.sum() / 255
    simetria_vertical = np.sum(np.abs(pixelx[:, :14] - pixelx[:, ::-1][:, :14])) / 255
    simetria_horizontal = np.sum(np.abs(pixelx[:14, :] - pixelx[::-1, :][:14, :])) / 255
    simetria_completa = simetria_vertical + simetria_horizontal

    return pd.Series({'intensidade': intensidade, 'simetria': simetria_completa})

def dados_reduzidos():
    TEST_DATA_PATH = ('dados/test.csv')
    TRAIN_DATA_PATH = ('dados/train.csv')
    data_test_df = pd.read_csv(TEST_DATA_PATH, sep=';')
    data_train_df = pd.read_csv(TRAIN_DATA_PATH, sep=';')

    data_train_df[['intensidade', 'simetria']] = data_train_df.apply(calcular_intensidade_e_simetria, axis=1)
    data_test_df[['intensidade', 'simetria']] = data_test_df.apply(calcular_intensidade_e_simetria, axis=1) 

    colunas_relevantes = ['label', 'intensidade', 'simetria']
    TEST_REDU_DATA_PATH = ('dados/test_redu.csv')
    TRAIN_REDU_DATA_PATH = ('dados/train_redu.csv')

    data_test_df[colunas_relevantes].to_csv('dados/test_redu.csv', index=False)
    data_train_df[colunas_relevantes].to_csv('dados/train_redu.csv', index=False)

    data_test_redu_df = pd.read_csv(TEST_REDU_DATA_PATH)
    data_train_redu_df = pd.read_csv(TRAIN_REDU_DATA_PATH)
    
    return {
        "data_test_redu_df": data_test_redu_df,
        "data_train_redu_df": data_train_redu_df
    }