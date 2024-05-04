import pandas as pd

def reduzir_dados():    
    REND_ESCOLAS_2020_PATH = ('tratamento_dados/dados/rend_escolas_manipulados_2020.csv')
    rend_escolas_2020_df = pd.read_csv(REND_ESCOLAS_2020_PATH)
    REND_ESCOLAS_2022_PATH = ('tratamento_dados/dados/rend_escolas_manipulados_2022.csv')
    rend_escolas_2022_df = pd.read_csv(REND_ESCOLAS_2022_PATH)
    
    colunas_relevantes_2020 = ['taxa_de_abandono_ensino_medio',
                          'zona_de_localizacao', 
                          'dependencia_administrativa']
    
    colunas_relevantes_2022 = ['taxa_de_abandono_ensino_medio',
                               'zona_de_localizacao',
                               'dependencia_administrativa']

    rend_escolas_2020_df[colunas_relevantes_2020].to_csv('tratamento_dados/dados/rend_escolas_redu_2020.csv', index=False)
    rend_escolas_2022_df[colunas_relevantes_2022].to_csv('tratamento_dados/dados/rend_escolas_redu_2022.csv', index=False)
    
    REND_ESCOLAS_REDU_2020_PATH = ('tratamento_dados/dados/rend_escolas_redu_2020.csv')
    rend_escolas_redu_2020_df = pd.read_csv(REND_ESCOLAS_REDU_2020_PATH)
    REND_ESCOLAS_REDU_2022_PATH = ('tratamento_dados/dados/rend_escolas_redu_2022.csv')
    rend_escolas_redu_2022_df = pd.read_csv(REND_ESCOLAS_REDU_2022_PATH)

    return {'rend_escolas_redu_2020_df': rend_escolas_redu_2020_df,
            'rend_escolas_redu_2022_df': rend_escolas_redu_2022_df}