import pandas as pd

def reduzir_dados():    
    ED_BRASILEIRA_2022_PATH = ('tratamento_dados/dados/ed_brasileira_2022.csv')
    ed_brasileira_2022_df = pd.read_csv(ED_BRASILEIRA_2022_PATH)
    
    colunas_relevantes = ['taxa_de_abandono_ensino_medio',
                          'agua_filtrada',
                          'energia', 
                          'saneamento_basico',
                          'coleta_de_lixo',
                          'internet',
                          'alimentacao']

    ed_brasileira_2022_df[colunas_relevantes].to_csv('tratamento_dados/dados/ed_brasileira_2022_redu.csv', index=False)
    
    ED_BRASILEIRA_2022_REDU_PATH = ('tratamento_dados/dados/ed_brasileira_2022_redu.csv')
    ed_brasileira_2022_redu_df = pd.read_csv(ED_BRASILEIRA_2022_REDU_PATH)

    return ed_brasileira_2022_redu_df