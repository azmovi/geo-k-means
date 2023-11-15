import numpy as np
import kmeans as kmedias

from sklearn.cluster import KMeans

tamanho_do_ponto = 2
quantidade_de_pontos = 15
numero_clusters = 3

data = np.random.uniform(-10, 10, size=(quantidade_de_pontos, tamanho_do_ponto))

atributos = kmedias.fit(data, 3)

print(atributos)

