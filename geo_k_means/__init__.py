import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# from geo_k_means.kmeans import KMeans  # usando o taskipy

import kmeans


def duas_dimensoes():
    data, groupos = make_blobs(
        n_samples=300, centers=3, cluster_std=1, random_state=42
    )

    atributos = kmeans.fit(data, 3)

    cores = ['r', 'g', 'b']
    centroides = atributos['centroides']
    clusters = atributos['clusters']
    rotulo = atributos['rotulo']

    for i, vetores in enumerate(clusters):
        cor = cores[i]
        for vetor in vetores:
            plt.scatter(vetor[0], vetor[1], color=cor, label=f'Cluster {i+1}')

    for i, vetor in enumerate(centroides):
        plt.scatter(vetor[0], vetor[1], color='black', marker='x', s=100, label=f'Centróides {i+1}')

    plt.xlabel('Dimensão X')
    plt.ylabel('Dimensão Y')
    plt.title('KMeans Clustering')
    plt.show()


duas_dimensoes()
