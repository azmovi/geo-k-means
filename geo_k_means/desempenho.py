# from kmeans import KMeans as KMedias  # Execução Padrão
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# Importações de métricas no uso no algoritmo
from sklearn.metrics import adjusted_rand_score, rand_score

from geo_k_means.kmeans import KMeans as KMedias  # Pytest


def convert_label_to_int(target: pd.Series) -> list[int]:
    """
    Converte uma série de rótulos em uma lista de inteiros, onde cada rótulo
    possui um número correspondente de acordo com a quantidade de classes.

    Parameters:
        target: Uma serie do pandas que retrata as classificações reais
        presentes na base de dados.

    Returns:
        Uma lista de inteiros com o tamanho do target que contem números entre
        zero a número classificações - 1

    Examples:
        >>> labels = pd.Series(['A', 'B', 'A', 'C', 'B'])
        >>> convert_labels_to_int(labels)
        [0, 1, 0, 2, 1]

    """
    label = {}
    lista_de_rotulos = [0] * len(target)
    for index, rotulo in enumerate(target.unique()):
        label[rotulo] = index

    for index, tipo in enumerate(target):
        lista_de_rotulos[index] = label[tipo]

    return lista_de_rotulos


def rand_index(
    data: pd.DataFrame, target: pd.Series, n_class: int, n_iter: int = 30
) -> list[float]:

    soma_rand_score = 0
    soma_adjusted_rand_score = 0
    for _ in range(n_iter):
        # Algoritmo feito por mim
        kmedias = KMedias(n_class)
        kmedias.fit(data_frame)

        # Algoritmo feito pelo sklearn
        kmeans = KMeans(n_clusters=n_class, n_init='auto', init='random')
        kmeans.fit(data_frame)

        soma_rand_score += rand_score(kmeans.labels_, kmedias.labels)
        soma_adjusted_rand_score += adjusted_rand_score(
            kmeans.labels_, kmedias.labels
        )
    return [soma_rand_score / n_iter, soma_adjusted_rand_score / n_iter]


def desempenho_iris() -> None:
    df = fetch_openml(name='iris', version=1, parser='auto')
    convert_label_to_int(df['target'])

    return


def main():
    desempenho_iris()


if __name__ == '__main__':
    main()
