import numpy as np
from tqdm import tqdm

class Perceptron():
    def __init__(self, taxa_aprendizado=0.01, n_iteracoes=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.n_iteracoes = n_iteracoes
        
    def fit(self, X_train, y_train):
        self.weight = np.zeros(1 + X_train.shape[1])
        self.errors_ = []  
        
        for _ in tqdm(range(self.n_iteracoes)):
            errors = 0
            for xi, target in zip(X_train, y_train):
                update = self.taxa_aprendizado * (target - self.predict(xi))
                self.weight[1:] += update * xi
                self.weight[0] += update
                errors += int(update != 0)
                
            self.errors_.append(errors)
    
    def predict(self, X_train):
        return np.where(self.net_input(X_train) >= 0, 1, -1)
    
    def net_input(self, X_train):
        return np.dot(X_train, self.weight[1:]) + self.weight[0]