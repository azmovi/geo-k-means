from time import perf_counter

import numpy as np
import pandas as pd

# Importações de métricas no uso no algoritmo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

from geo_k_means.kmeans import KMeans as KMedias  # Pytest

# from kmeans import KMeans as KMedias  # Execução Padrão


DICT_DE_METRICAS = {
    'completeness_score': metrics.completeness_score,
    'fowlkes_mallows_score': metrics.fowlkes_mallows_score,
    'homogeneity_score': metrics.homogeneity_score,
    'v_measure_score': metrics.v_measure_score,
    'rand_score': metrics.rand_score,
    'adjusted_rand_score': metrics.adjusted_rand_score,
    'mutual_info_score': metrics.mutual_info_score,
    'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score,
    'normalized_mutual_info_score': metrics.normalized_mutual_info_score,
}


def sklearn_parametrizer(
    data: pd.DataFrame,
    labels: list[int],
    n_class: int,
    dict_de_metricas: dict[str, callable],
    n_iter: int = 30,
) -> dict[str, list[float]]:
    """
    Esta função calcula o desempenho de diferentes métricas de avaliação
    disponíveis na biblioteca sklearn, bem como o tempo de execução para cada
    métrica usando o classificador da biblioteca sklearn.

    Parametres:
        data: Um conjunto de dados para o treinamento.
        labels: Rótulos corretos do conjunto de dados.
        n_class: A quantidade de classes do conjunto de dados.
        dict_de_metricas: Dicionario contendo as funções das métricas presentes
        na biblioteca do sklearn.
        n_iter: Número de iterações para o treinamento do modelo. Padrão é 30.

    Returns:
        Um dicionário contendo o desempenho de cada métrica e o tempo de
        execução correspondente.

    Examples:
        >>> df = fetch_openml(name='iris', version=1, parser='auto')
        >>> labels = convert_label_to_int(df['target'])
        >>> sklearn_parametrizer(df['data'], labels, 3, DICT_DE_METRICAS)
        {'completeness_score': [...], ...}


    """

    dict_dos_resultados = {}

    for key in dict_de_metricas.keys():
        soma_da_metrica = 0
        start = perf_counter()
        for _ in range(n_iter):
            kmeans = KMeans(n_clusters=n_class, n_init='auto', init='random')
            kmeans.fit(data)
            soma_da_metrica += dict_de_metricas[key](labels, kmeans.labels_)

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [
            round(soma_da_metrica / n_iter, 3),
            round(duration, 3),
        ]

    return dict_dos_resultados


def kmedias_parametrizer():
    ...


def convert_label_to_int(target: pd.Series) -> list[int]:
    """
    Converte uma série de rótulos em uma lista de inteiros, onde cada
    rótulo possui um número correspondente de acordo com a quantidade de
    classes.

    Parameters:
        target: Uma serie do pandas que retrata as classificações reais
        presentes na base de dados.

    Returns:
        Uma lista de inteiros com o tamanho do target que contem números
        entre zero a número classificações - 1

    Examples:
        >>> labels = pd.Series(['A', 'B', 'A', 'C', 'B'])
        >>> convert_label_to_int(labels)
        [0, 1, 0, 2, 1]

    """
    label = {}
    lista_de_rotulos = [0] * len(target)
    for index, rotulo in enumerate(target.unique()):
        label[rotulo] = index

    for index, tipo in enumerate(target):
        lista_de_rotulos[index] = label[tipo]

    return lista_de_rotulos


def desempenho_iris() -> None:
    df = fetch_openml(name='iris', version=1, parser='auto')
    labels = convert_label_to_int(df['target'])

    resultados_sklearn = sklearn_parametrizer(
        df['data'], labels, 3, DICT_DE_METRICAS
        )

    return resultados_sklearn


def main():
    desempenho_iris()


if __name__ == '__main__':
    main()
