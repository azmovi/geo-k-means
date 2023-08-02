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

$$ WSS = \sum_{i=1}^{N_{C}}\sum_{\bold{x} \in C_{i}} d(\bold{x}, \bar{x}_{C_{i}}) $$

- Onde $C_i$ é um grupo e $N_C$ é o número de grupos.


###### Referências:
[_Flat Clustering_ - Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze
](https://nlp.stanford.edu/IR-book/pdf/16flat.pdf)

[K-Means from scratch - Turner Luke
](https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670)



