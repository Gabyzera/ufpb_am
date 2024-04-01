import numpy as np
import matplotlib.pyplot as plt

class RegressaoLinear():
    def __init__(self, taxa_aprendizado = 0.001, n_iteracoes = 1000):
        self.vies = None
        self.pesos = None
        self.n_iteracoes = n_iteracoes
        self.taxa_aprendizado = taxa_aprendizado

    def ajuste(self, X, y):
        n_amostras, n_caracteristicas = X.shape
        
        self.vies = 0.0
        self.pesos = np.zeros(n_caracteristicas)
        
        for _ in range(self.n_iteracoes):
            y_previsao = np.dot(X, self.pesos) + self.vies
            
            derivada_vies = (2/n_amostras) * np.sum(y_previsao - y)
            derivada_peso = (2/n_amostras) * np.dot(X.T, (y_previsao - y))
            
            self.vies = self.vies - self.taxa_aprendizado * derivada_vies
            self.pesos = self.pesos - self.taxa_aprendizado * derivada_peso
        

    def prever(self, X):
        y_previsao = np.dot(X, self.pesos) + self.vies
        
        return y_previsao
    
    
    def erro_medio_quadratico(y, previsoes):
        return np.mean(pow(y - previsoes), 2)
    
    def plotagem_regressao_linear(self, X, y):
        X1 = X[y == 1]
        X2 = X[y == -1]
        
        xmin = np.min(X[:, 0]) - 0.5
        xmax = np.max(X[:, 0]) + 0.5

        x = np.linspace(xmin, xmax, 100)
        pesos_intensidade, pesos_simetria = self.pesos
        y_linha = (-pesos_intensidade*x - self.vies) / pesos_simetria

        plt.ylabel("Simetria")
        plt.xlabel("Intensidade")
        plt.title("Regressão Linear")
        plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='1', edgecolors='white')
        plt.scatter(X2[:, 0], X2[:, 1], c='red', label='-1', edgecolors='white')
        plt.plot(x, y_linha, label="Pesos", c='black', linewidth=2)
        plt.legend()
        plt.xlim(xmin, xmax)
        plt.ylim(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5)       
        plt.show()