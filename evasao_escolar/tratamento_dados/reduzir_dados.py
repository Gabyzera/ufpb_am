import pandas as pd

def reduzir_dados(ed_brasileira_df_path, edu_brasileira_redu_df_path):    
    ed_brasileira_2022_df = pd.read_csv(ed_brasileira_df_path)
    
    colunas_relevantes = ['taxa_de_abandono',
                          'taxa_de_reprovacao',
                          'regiao',
                          'agua_filtrada',
                          'energia', 
                          'saneamento_basico',
                          'coleta_de_lixo',
                          'tratamento_de_lixo',
                          'computador',
                          'educacao_especial',
                          'laboratorio_ciencias',
                          'laboratorio_informatica',
                          'internet',
                          'tipo_de_dependencia',
                          'biblioteca_sala_leitura',
                          'tipo_localizacao',
                          'localizacao_diferenciada',
                          'alimentacao',
                          'nenhuma_acessibilidade']

    ed_brasileira_2022_df[colunas_relevantes].to_csv(edu_brasileira_redu_df_path, index=False)
    ed_brasileira_2022_redu_df = pd.read_csv(edu_brasileira_redu_df_path)

    return ed_brasileira_2022_redu_df