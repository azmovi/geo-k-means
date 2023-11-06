import numpy as np

from kmeans import KMeans as KMedias  # Execução Padrão
# from geo_k_means.kmeans import KMeans as KMedias  # Pytest

from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# Importações de métricas no uso no algoritmo
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score


# def metricas():


def desempenho_iris() -> None:
    df = fetch_openml(name='iris', version=1, parser='auto')
    soma_rand_score = 0
    soma_adjusted_rand_score = 0
    for _ in range(30):
        kmedias = KMedias(3)
        kmedias.fit(df.data.values)
        kmeans = KMeans(n_clusters=3, n_init='auto', init='random')
        kmeans.fit(df.data.values)
        soma_rand_score += rand_score(kmeans.labels_, kmedias.labels)
        soma_adjusted_rand_score +=  adjusted_rand_score(kmeans.labels_, kmedias.labels)

    print(f'Media da rand score = {soma_rand_score/30}')
    print(f'Media da rand score = {soma_adjusted_rand_score/30}')

def main():
    desempenho_iris()


if __name__ == '__main__':
    main()


adjusted_rand_score
adjusted_rand_score
adjusted_rand_score
