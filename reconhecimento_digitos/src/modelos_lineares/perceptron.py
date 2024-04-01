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
    def ajuste(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.total_erros = []  
        # Loop que percorre o número de iterações e define que a cada iteração o erros da amostra é zerado
        # para calcular independentemente o erro de cada amostra.
        for _ in tqdm(range(self.n_iteracoes)):
            erros_cada_amosta = 0
            # Loop que passa pelos pontos xi e target dentro de X e y respectivamente para criar
            # um novo iterável de tuplas com os valores de xi e target.
            for xi, target in zip(X, y):
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
    # Prevê as entradas de X, quando a entrada líquida é maior do que 0, retorna 1, caso contrário, 
    # retorna -1.
    def prever(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    # Calcula a entrada líquida (características de entrada + vieses) de dados no modelo.
    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]
    # Plota o traço de decisão usando X, y, o classificador que já foi treinado anteriormente e a resolução
    # para definir o quão fina é a malha que será usada para visualizar a região. 
    def plotagem_perceptron(self, X, y, classifier, resolution = 0.02):
        # Define o marcador (símbolo gráfico que representa os pontos de dados no gráfico).
        marker = ('o')
    
        # Determina os valores mínimos e máximos para as duas características com a subtração e adição para 
        # criação de espaços extras para melhor visualização. 
        x1_min, x1_max = X[:, 0].min() - 1 , X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        # Cria uma grade de pontos com a resolução especificada, em que usa o np.meshgrid para receber dois 
        # arrays 1D e retorna 2 arrays 2D. 
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
        # Usa o classificador para fazer previsões para cada ponto na grade e remodelar.
        Z = classifier.prever(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
    
        # Plota os pontos de dados usando os marcadores e cores definidos
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=('blue' if cl == 1 else 'red'), marker=marker, 
                    label=f'{cl}', edgecolors='white')
    
        # Extrai os pesos do classificador e calcula a linha de decisão
        weights = classifier.weight
        a = -weights[1] / weights[2]  # Coeficiente angular da linha de decisão
        b = -weights[0] / weights[2]  # Interceptação no eixo y da linha de decisão
        decision_boundary = np.array([x1_min, x1_max])
    
        # Plota a linha de decisão
        plt.plot(decision_boundary, a * decision_boundary + b, "k-", linewidth=2)
    
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.xlabel('Intensidade')
        plt.ylabel('Simetria')
        plt.title('Perceptron')
        plt.legend(loc='upper left')
        plt.show()