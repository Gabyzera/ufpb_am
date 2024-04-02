import numpy as np

def acuracia(y_previsao, y_teste):
    return np.mean(y_previsao == y_teste)