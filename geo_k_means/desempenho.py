from time import perf_counter

import numpy as np
import pandas as pd

# Importações de métricas no uso no algoritmo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# from geo_k_means.kmeans import KMeans as KMedias  # Pytest

from kmeans import KMeans as KMedias  # Execução Padrão


METRICAS = {
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

OPENML_DATASETS = [
    'iris',
    'amazon-commer-reviews',
    'semion',
    'mfeat-pixel',
    'micro-mass',
    'monks-problems-1',
    'breast-tissue',
    'fri_c2_100_10',
    'datatrive',
    'fri_c3_250_25',
    'GCM',
    'collins',
    'pyrim',
    'balance-scale',
    'tr45.wc',
    'cloud',
    'servo',
    'AP_Breast_Lung',
    'leukemia',
    'AP_Colon_Prostate',
    'AP_Colon_Kidney',
]


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
        >>> n_class = len(df['target'].unique())
        >>> lista_de_classes = df['target'].unique().categories
        >>> labels = convert_label_to_int(
        ... df['target'], n_class, lista_de_classes
        ... )
        >>> sklearn_parametrizer(df['data'], labels, 3, METRICAS)
        {'completeness_score': [...], ...}
    """

    dict_dos_resultados = {}

    for key in dict_de_metricas.keys():
        soma_da_metrica = 0
        start = perf_counter()
        for _ in range(n_iter):
            kmeans = KMeans(n_clusters=n_class, n_init='auto', init='random')
            kmeans.fit(data.values)
            soma_da_metrica += dict_de_metricas[key](labels, kmeans.labels_)

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [
            round(soma_da_metrica / n_iter, 3),
            round(duration, 3),
        ]

    return dict_dos_resultados


def kmedias_parametrizer(
    data: pd.DataFrame,
    labels: list[int],
    n_class: int,
    dict_de_metricas: dict[str, callable],
    n_iter: int = 30,
) -> dict[str, list[float]]:
    """
    Esta função calcula o desempenho de diferentes métricas de avaliação
    disponíveis na biblioteca sklearn, bem como o tempo de execução para cada
    métrica usando o classificador feito por mim.

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
        >>> n_class = len(df['target'].unique())
        >>> lista_de_classes = df['target'].unique().categories
        >>> labels = convert_label_to_int(
        ... df['target'], n_class, lista_de_classes
        ... )
        >>> kmedias_parametrizer(df['data'], labels, 3, METRICAS)
        {'completeness_score': [...], ...}
    """

    dict_dos_resultados = {}

    for key in dict_de_metricas.keys():
        soma_da_metrica = 0
        start = perf_counter()
        for _ in range(n_iter):
            kmedias = KMedias(n_class)
            kmedias.fit(data.values)
            soma_da_metrica += dict_de_metricas[key](labels, kmedias.labels)

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [
            round(soma_da_metrica / n_iter, 3),
            round(duration, 3),
        ]

    return dict_dos_resultados


def convert_label_to_int(
    target: pd.Series, n_class: int, lista_de_classes: pd.Index
) -> list[int]:
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

        >>> df = fetch_openml(name='iris', version=1, parser='auto')
        >>> n_class = len(df['target'].unique())
        >>> lista_de_classes = df['target'].unique().categories
        >>> convert_label_to_int(df['target'], n_class, lista_de_classes)
        [0, 0, 0, 0, ..., 2]

    """
    label = {}
    lista_de_rotulos = [0] * len(target)
    for index, rotulo in enumerate(lista_de_classes):
        label[rotulo] = index

    for index, tipo in enumerate(target):
        lista_de_rotulos[index] = label[tipo]

    return lista_de_rotulos


def cria_data_frame(dict_de_metricas: dict[str, callable]) -> pd.DataFrame:
    """
    Cria um DataFrame utilizando a biblioteca pandas, com colunas para métricas pré-definidas
    e seus respectivos tempos de execução.

    Parameters:
        dict_de_metricas: Um dicionário onde as chaves são os nomes das
        métricas e os valores são suas respectivas funções.

    Returns:
        Um DataFrame com colunas para as métricas e os tempos de execução para
        cada métrica.

    Examples:
        >>> metricas = {'completeness_score': metrics.completeness_score}
        >>> cria_data_frame(metricas)
        Empty DataFrame
        Columns: [completeness_score, Tempo 1]
        Index: []
    """

    colunas = {}
    colunas = pd.DataFrame(colunas)

    for index, key in enumerate(dict_de_metricas.keys(), start=1):
        colunas[key] = None
        colunas[f'Tempo {index}'] = None

    return colunas


def formata_resultados(
    sklearn_resultados: dict[str, list[float]],
    kmedias_resultados: dict[str, list[float]],
    data_frame: pd.DataFrame,
) -> None:

    for index, chave in enumerate(
        zip(sklearn_resultados, kmedias_resultados), start=1
    ):
        desempenho1 = sklearn_resultados[chave][0]
        desempenho2 = kmedias_resultados[chave][0]

        tempo1 = sklearn_resultados[chave][1]
        tempo2 = kmedias_resultados[chave][1]

        razao_acertos = desempenho1 / desempenho2
        razao_tempo = tempo1 / tempo2

        linha_dataframe[chave1] = razao_acertos
        linha_dataframe[f'Tempo {index}']
        df = df.append(nova_linha, ignore_index=True)

def executa_datasets() -> None:
    df = fetch_openml(name='iris', version=1, parser='auto')
    n_class = len(df['target'].unique())
    lista_de_classes = df['target'].unique().categories

    labels = convert_label_to_int(df['target'], n_class, lista_de_classes)

    sklearn_resultados = sklearn_parametrizer(
        df['data'], labels, n_class, METRICAS, 1
    )

    kmedias_resultados = kmedias_parametrizer(
        df['data'], labels, n_class, METRICAS, 1
    )

    df = cria_data_frame(METRICAS)
    print(df)
    #formata_resultados(df, sklearn_resultados, kmedias_resultados)


def main():
    executa_datasets()


if __name__ == '__main__':
    main()
