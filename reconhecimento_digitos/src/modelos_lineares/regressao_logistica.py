import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegressaoLogistica():
    def __init__(self, taxa_aprendizado = 0.001, n_iteracoes = 1000, lambda_decaimento_peso = 0.01):
        self.vies = None
        self.pesos = None
        self.n_iteracoes = n_iteracoes
        self.taxa_aprendizado = taxa_aprendizado
        # Weight decay 
        self.lambda_decaimento_peso = lambda_decaimento_peso

    def ajuste(self, X, y):
        n_amostras, n_caracteristicas = X.shape
        
        self.vies = 0.0
        self.pesos = np.zeros(n_caracteristicas)
        
        for _ in range(self.n_iteracoes):
            previsao_linear = np.dot(X, self.pesos) + self.vies
            y_previsao = sigmoid(previsao_linear)
            
            derivada_vies = (1/n_amostras) * np.sum(y_previsao - y)
            # Derivada de peso levando em consideração o weight decay
            derivada_peso = (1/n_amostras) * np.dot(X.T, (y_previsao - y)) + (
                                self.lambda_decaimento_peso / n_amostras
                            ) * self.pesos 
            
            self.vies -= self.taxa_aprendizado * derivada_vies
            self.pesos -= self.taxa_aprendizado * derivada_peso
    
    def prever(self, X):
        previsao_linear = np.dot(X, self.pesos) + self.vies
        y_probabilidade = sigmoid(previsao_linear)
        
        previsao_classes = np.where(y_probabilidade <= 0.5, -1, 1)
        
        return previsao_classes
    
    def plotar(self, X, y, resolucao = 0.5):
        X1 = X[y == 1]
        X2 = X[y == -1]
        
        xmin = np.min(X[:, 0]) - resolucao
        xmax = np.max(X[:, 0]) + resolucao

        x = np.linspace(xmin, xmax, 100)
        pesos_intensidade, pesos_simetria = self.pesos
        y_linha = (-pesos_intensidade * x - self.vies) / pesos_simetria

        plt.ylabel("Simetria")
        plt.xlabel("Intensidade")
        plt.title("Regressão Logística")
        plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='1', edgecolors='white')
        plt.scatter(X2[:, 0], X2[:, 1], c='red', label='-1', edgecolors='white')
        plt.plot(x, y_linha, label="Pesos", c='black', linewidth=2)
        plt.legend()
        plt.xlim(xmin, xmax)
        plt.ylim(np.min(X[:, 0]) - resolucao, np.max(X[:, 0]) + resolucao)       
        plt.show()