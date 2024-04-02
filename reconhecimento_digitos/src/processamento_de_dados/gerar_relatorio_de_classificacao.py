from sklearn.metrics import classification_report

def gerar_relatorio_classificacao(y_teste, y_previsao, classes = ['0', '1', '4', '5']):
    relatorio_classificacao = classification_report(y_teste, y_previsao, target_names=classes)
    
    print(relatorio_classificacao)