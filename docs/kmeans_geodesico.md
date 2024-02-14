# Algoritmo topológico de K-Means

## Conceito
Método baseado um algoritmo de grafos para _clusterização_ onde ele usa um grafo
k-NN para aproximar o _manifold_ de dados no espaço de entrada.

## Hipótese de _manifold_
Conjuntos com alta dimensionalidade quem ocorrem no mundo real podem ser expressos
em estruturas geométricas não linear de menor dimensionalidade.

## Como construir um grafo de um conjunto de dados multivariado (n dimensões)

### 1. Grafo k-NN
Para cada ponto dos dados $\vec{x}_i, \space i = 1, 2, ..., n$ deve-se calcular a
distancia euclidiana para todos os outros porntos $\vec{x}_j, \space j \neq i,
\space j = 1, 2,..., n$ então se seleciona $k$ amostras com menor distancia e
cria uma aresta entre elas.

### 2. Grafo ϵ-vizinhaçã
É definido um raio $\epsilon$ e para cada amostra $\vec{x}_i, \space i = 1, 2,
..., n$ e toda outra amostra $\vec{x}_j, \space j \neq i, \space j = 1, 2, ...,
n$, calcula a distância euclidiana entre eles. Para todo ponto dentro da esfera
de raio $\epsilon$ e centro em $\vec{x}_i$ uma aresta deve ser criada.

O quão bem o tamanho dos menores caminhos no grafo k-NN pode aproximar as
verdadeiras distâncias geodésicas subjacentes no manifold dos dados.
-
O Teorema da Convergência Assintótica explica que, sob certas condições de
regularidade, o comprimentos dos caminhos mais curtos em grafos k-NN
$d_G(\vec{x}_i, \vec{x}_j)$ converge para a distância geodesica $d_M(\vec{x}_i,
\vec{x}_j)$

**Teorema da Convergência Assintótica**, dado $\lambda_1, \lambda_2, \mu \gt 0$,
porém tão pequeno quanto desejada, então, para uma desidade de pontos
suficientemente grande, a seguente desigualdade é válida:
$$
1 - \lambda_1 \leq \frac{d_G(\vec{x}_i, \vec{x}_j)}{d_M(\vec{x}_i, \vec{x}_j)}
\leq 1 + \lambda_2 
$$
com probabilidade $1 - \mu$, onde $d_G(\vec{x}_i, \vec{x}_j)$ é a distancia
recuperada (comprimento do caminho mais curto) e $d_M(\vec{x}_i, \vec{x}_j)$ é
a verdadeira distância no manifold.
