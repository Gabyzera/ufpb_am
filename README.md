# 🤖 Projetos de Aprendizagem de Máquina
Este repositório contém três projetos de aprendizagem de máquina: reconhecimento de dígitos, evasão escolar e perceptron learning algorithm.

## 🏗️ Projetos
### 🔢 1. Reconhecimento de Dígitos
O reconhecimento de dígitos é um problema clássico de classificação na área de visão computacional. O problema consiste em receber uma imagem de um número escrito à mão, codificada em tons de cinza, e classificar o dígito decimal (0-9) ali contido.

**Especificações**:

- Redução da dimensão das amostras:
    - Intensidade da imagem
    - Simetria da imagem
- Classificação dos dígitos 1 x 5
- Classificador de dígitos completo
- Comparação entre os classificadores
- Implementações avançadas

### 🏫 2. Evasão Escolar
O projeto visa implementar redes neurais e árvores de decisões para prever o abandono escolar em municípios brasileiros. 

**Especificações**:

#### Dataset:
Dados retirados do site do INEP, contendo informações em formatos “xlsx” e “csv”.

#### Tratamento de dados básico:
Limpeza e organização dos dados, mesclagem de dataframes, identificação e categorização de outliers e preparação dos dados para análise.

#### Implementação do modelo baseado em rede neural:
Definição da arquitetura da rede neural, escolha de parâmetros, análise de overfitting, escolha de batch size, número de épocas e visualização de resultados com gráficos.

#### Construção do modelo de Árvore de Decisão:
Divisão dos dados entre treinamento e teste, uso de 'DecisionTreeClassifier', análise de overfitting, poda de complexidade mínima de custo, e avaliação de desempenho com métricas.

#### Resultados:
Comparação entre a acurácia, precisão, recall e F1 Score dos modelos de rede neural e árvore de decisão, demonstrando as vantagens e desvantagens de cada abordagem.


## 📖 Referências
- INEP - Indicadores Educacionais
- INEP - Microdados Censo Escolar
- Teorema da Aproximação Universal - Wikipedia
- Deep Learning Book - Early Stopping
- LinkedIn - Diferença entre Adam e SGD
