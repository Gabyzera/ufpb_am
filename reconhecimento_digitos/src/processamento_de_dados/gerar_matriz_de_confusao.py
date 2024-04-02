import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def gerar_matriz_de_confusao(y_teste, y_previsao, classes = ['0', '1', '4', '5'], modelo = ''):
    matriz_confusao = confusion_matrix(y_teste, y_previsao)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predições')
    plt.ylabel('Valores Verdadeiros')
    plt.title(f'Matriz de Confusão [{modelo}]')
    plt.show()