import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron():
    # Função construtora que insere o modo como o modelo aprende, possibilita a alteração na taxa 
    # de aprendizado (controla o quão rápido ou lento há a alteração nos pesos) e o número de iterações. 
    def __init__(self, taxa_aprendizado=0.01, n_iteracoes=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.n_iteracoes = n_iteracoes
    # Inicializa com os pesos = 0
    def ajuste(self, X_train, y_train):
        self.weight = np.zeros(1 + X_train.shape[1])
        self.total_erros = []  
        # Loop que percorre o número de iterações e define que a cada iteração o erros da amostra é zerado
        # para calcular independentemente o erro de cada amostra.
        for _ in tqdm(range(self.n_iteracoes)):
            erros_cada_amosta = 0
            # Loop que passa pelos pontos xi e target dentro de X_train e y_train respectivamente para criar
            # um novo iterável de tuplas com os valores de xi e target.
            for xi, target in zip(X_train, y_train):
                # O update é a diferença entre o target e a previsão do ponto xi, multiplicada pela taxa de 
                # aprendizado. Isso determina quão grande é o passo que precisamos dar na direção do gradiente 
                # para ajustar os pesos.
                update = self.taxa_aprendizado * (target - self.prever(xi))
                # Atualização dos pesos das características por meio da multiplicação do valor do update 
                # pelo xi. 
                self.weight[1:] += update * xi
                # E o ajuste dos pesos dos vieses [0].
                self.weight[0] += update
                # Conta quantas vezes o modelo fez uma previsão incorreta. 
                erros_cada_amosta += int(update != 0)
            # Armazenamento de erros após cada iteração, permitindo análise posterior do desempenho do modelo.
            self.total_erros.append(erros_cada_amosta)
    # Prevê as entradas de X_train, quando a entrada líquida é maior do que 0, retorna 1, caso contrário, 
    # retorna -1.
    def prever(self, X_train):
        return np.where(self.net_input(X_train) >= 0, 1, -1)
    # Calcula a entrada líquida (características de entrada + vieses) de dados no modelo.
    def net_input(self, X_train):
        return np.dot(X_train, self.weight[1:]) + self.weight[0]

    def plotagem_traco_decisao(self, X, y, classifier, resolution=0.02):
        # Define os marcadores e cores para o gráfico
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        # Cria um mapa de cores a partir da lista de cores
        cmap = ListedColormap(colors[:len(np.unique(y))])
    
        # Determina os valores mínimos e máximos para as duas características
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        # Cria uma grade de pontos com a resolução especificada
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
        # Usa o classificador para fazer previsões para cada ponto na grade
        Z = classifier.prever(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
    
        # Preenche o gráfico com as regiões de decisão coloridas
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    
        # Plota os pontos de dados usando os marcadores e cores definidos
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], 
                    label=cl, edgecolor='black')
    
        # Extrai os pesos do classificador e calcula a linha de decisão
        weights = classifier.weight
        a = -weights[1] / weights[2]  # Coeficiente angular da linha de decisão
        b = -weights[0] / weights[2]  # Interceptação no eixo y da linha de decisão
        decision_boundary = np.array([x1_min, x1_max])
    
        # Plota a linha de decisão
        plt.plot(decision_boundary, a * decision_boundary + b, "k--", linewidth=2)
    
        # Define os limites do gráfico e os rótulos dos eixos, adiciona um título e a legenda
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.xlabel('Intensidade')
        plt.ylabel('Simetria')
        plt.title('Perceptron - Regiões de Decisão')
        plt.legend(loc='upper left')
    
        # Exibe o gráfico
        plt.show()