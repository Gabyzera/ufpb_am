import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, taxa_aprendizado=0.01, n_iteracoes=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.n_iteracoes = n_iteracoes

    def ajuste(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.total_erros = []  
        
        for _ in range(self.n_iteracoes):
            erros_cada_amosta = 0
            
            for xi, target in zip(X, y):
                update = self.taxa_aprendizado * (target - self.prever(xi))

                self.weight[1:] += update * xi
                self.weight[0] += update

                erros_cada_amosta += int(update != 0)

            self.total_erros.append(erros_cada_amosta)

    
    def prever(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    
    def net_input(self, X):
        # w * x + b 
        return np.dot(X, self.weight[1:]) + self.weight[0]
    
    def plotar(self, X, y, resolution = 0.02):
        marker = ('o')
    
        x1_min, x1_max = X[:, 0].min() - 1 , X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
        Z = self.prever(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
    
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=('blue' if cl == 1 else 'red'), marker=marker, 
                    label=f'{cl}', edgecolors='white')

        weights = self.weight
        a = -weights[1] / weights[2] # inclinação da reta
        b = -weights[0] / weights[2] # interceptro da linha
        decision_boundary = np.array([x1_min, x1_max])

        plt.plot(decision_boundary, a * decision_boundary + b, "k-", linewidth=2)
    
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.xlabel('Intensidade')
        plt.ylabel('Simetria')
        plt.title('Perceptron')
        plt.legend(loc='upper left')
        plt.show()