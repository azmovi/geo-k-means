### K-Means
##### Conceito:
- Sub dividir nosso grupo.
- Comummente utilizado em aprendizado não supervisionado.
- Uso de distancias euclidianas entre centroides e pontos do gráfico.

##### Abordagens:
###### _Hard Clustering_:
- Simples e tradicional.
- Cada ponto pertence a um único cluster.
###### _Soft Clustering_:
- Abordagem mais flexível
- Cada ponto pode pertencer a outros clusteres.
- Sendo atribuído um grau de pertencimento.

##### Número de _Clusters_:
- Um problema difícil é determinar o numero de clusteres que serão criados para um 
determinado _data set_, denotado por $K$

- Para encontrar o melhor valor de $K$, podemos usar a distancia média dos pontos
dentro de um cluster até seu centroide, 
- Essa técnica é denominada de Elbow method (método do cotovelo)
$$ WCSS = \sum_{i=1}^{N_C} \sum_{x \in C_{i}}(\text{distance}(x, C{i}))^2 $$

- Onde $C_i$ é um grupo e $N_C$ é o número de grupos.
- Objetivo: Encontrar um WCSS baixo com número de clusters pequeno.
> WCSS = within cluster sum of squares.

##### K Médias:
- **Objetivo**: minimizar a distância euclidiana quadrada média.
$$ RSS = \sum_{j=1}^{k} \sum_{i=1}^{n} |x_{i}^{j} - c_{j}|^2 $$

RSS = residual square sum

k = número de clusteres

n = número de pontos 

c = centroide do cluster j

x = dado do banco de dados

###### Método de desenvolvimento:
- Encontrar os centroides de forma aleatória. 
- O algoritmo move os centroides com objetivo de minimizar o valor de RSS
- Pode apresentar vários critério de parada:
    - Fixar um número máximo de iterações.
    - Não ocorre a mudança do local dos centroides.
    - RSS muito baixo, garantindo a convergência.

###### Referências:
[_Flat Clustering_ - Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze
](https://nlp.stanford.edu/IR-book/pdf/16flat.pdf)

[K-Means from scratch - Turner Luke
](https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670)

