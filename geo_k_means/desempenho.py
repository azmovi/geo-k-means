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
    kmedias = KMedias(3)
    kmedias.fit(df.data.values)

    #print(df.data.values)
    #for centroide in kmedias.clusters.keys():
        
        #print("---------", centroide, "--------")
        #for valores in kmedias.clusters[centroide]:
        #    print(valores)

    kmeans = KMeans(n_clusters=3, n_init='auto')
    kmeans.fit(df.data.values)

    #print(rand_score(kmeans.labels_, kmedias.labels))
    #print(adjusted_rand_score(kmeans.labels_, kmedias.labels))
    print(kmeans.labels_, kmedias.labels)

def main():
    desempenho_iris()


if __name__ == '__main__':
    main()
