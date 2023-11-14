import numpy as np
import kmeans as kmedias

from sklearn.cluster import KMeans

tamanho_do_ponto = 3
quantidade_de_pontos = 10
numero_clusters = 2

data = np.random.uniform(-10, 10, size=(quantidade_de_pontos, tamanho_do_ponto))

clusters, centroides, rotulo = kmedias.fit(data, 2)
print(type(clusters))
