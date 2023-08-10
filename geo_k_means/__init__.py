import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from geo_k_means.kmeans import KMeans


def duas_dimensoes():
    data, groupos = make_blobs(
        n_samples=300, centers=3, cluster_std=1, random_state=42
    )

    test = KMeans(numero_clusters=3)

    test.fit(data)

    centroides = test.centroides
    clusters = test.clusters

    cores = ['r', 'g', 'b']

    for i, (centroide, pontos) in enumerate(clusters.items()):
        cor = cores[i]
        pontos = list(zip(*pontos))
        plt.scatter(*pontos, color=cor, label=f'Cluster {i+1}')

    centroides = list(zip(*centroides))
    plt.scatter(
        *centroides, color='black', marker='x', s=100, label='Centróides'
    )

    plt.xlabel('Dimensão X')
    plt.ylabel('Dimensão Y')
    plt.title('KMeans Clustering')
    plt.legend()
    plt.show()


duas_dimensoes()
