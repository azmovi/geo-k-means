from time import perf_counter

import pandas as pd
import numpy as np

# Importações de métricas no uso no algoritmo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError

# import geo_k_means.kmeans # pytest

import kmeans


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
    'micro-mass',
    'monks-problems-1',
    'breast-tissue',
    'GCM',
    'collins',
    'balance-scale',
    'servo',
    'AP_Breast_Lung',
    'AP_Colon_Kidney',
    'AP_Breast_Kidney',
    'AP_Breast_Ovary',
    'AP_Breast_Colon',
    'AP_Colon_Prostate',
    'AP_Endometrium_Breast',
    'AP_Colon_Omentum',
    'AP_Breast_Omentum',
    'AP_Prostate_Kidney',
    'AP_Omentum_Kidney',
    'AP_Breast_Prostate',
    'AP_Uterus_Kidney',
    'AP_Prostate_Uterus',
    'AP_Omentum_Lung',
    'AP_Lung_Uterus',
    'AP_Ovary_Kidney',
    'AP_Lung_Kidney',
    'AP_Colon_Lung',
    'AP_Endometrium_Colon',
    'AP_Ovary_Lung',
    'AP_Endometrium_Kidney',
    'AP_Colon_Uterus',
    'leukemia',
    'hepatitisC',
    'Ovarian',
    'SRBCT',
    'Colon',
    'climate-model-simulation-crashes',
    'anneal',
    'pasture',
    'glass',
    'lowbwt',
    'kc1-binary',
    'dermatology',
    'backache',
    'lsvt',
    'thoracic-surgery',
    'planning-relax',
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
        >>> labels = make_label(
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
            round(duration, 3)
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
    """

    dict_dos_resultados = {}
    for key in dict_de_metricas.keys():
        soma_da_metrica = 0
        start = perf_counter()
        for _ in range(n_iter):
            atributos = kmeans.fit(data.values, n_class)
            soma_da_metrica += dict_de_metricas[key](
                labels, atributos['rotulo']
            )

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [
            round(soma_da_metrica / n_iter, 3),
            round(duration, 3),
        ]

    return dict_dos_resultados


def make_label(
    target: pd.Series, n_class: int, lista_de_classes: pd.Categorical
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
        >>> categorias = df['target'].unique()
        >>> n_class = len(categorias)
        >>> make_label(df['target'], n_class, categorias)
        array([0, 0, 0, 0, ..., 2])

    """
    rotulo = {rotulo: index for index, rotulo in enumerate(lista_de_classes)}

    return np.array([rotulo[tipo] for tipo in target])


def cria_dataframe(dict_de_metricas: dict[str, callable]) -> pd.DataFrame:
    """
    Cria um DataFrame utilizando a biblioteca pandas, com colunas para métricas
    pré-definidas e seus respectivos tempos de execução.

    Parameters:
        dict_de_metricas: Um dicionário onde as chaves são os nomes das
        métricas e os valores são suas respectivas funções.

    Returns:
        Um DataFrame com colunas para as métricas e os tempos de execução para
        cada métrica.

    Examples:
        >>> metricas = {'completeness_score': metrics.completeness_score}
        >>> cria_dataframe(metricas)
        Empty DataFrame
        Columns: [Nome, completeness_score, Tempo 1]
        Index: []
    """
    colunas = pd.DataFrame({}, columns=['Nome'])

    for index, key in enumerate(dict_de_metricas.keys(), start=1):
        colunas[key] = None
        colunas[f'Tempo {index}'] = None

    return colunas


def formata_resultado_de_uma_metrica(
    dataframe: pd.DataFrame,
    nome_da_metrica: str,
    dict_de_metricas: dict[str, callable],
    sklearn_resultados_de_uma_metrica: dict[str, list[float]],
    kmedias_resultados_de_uma_metrica: dict[str, list[float]],
) -> pd.DataFrame:

    linha_dataframe = {'Nome': nome_da_metrica}

    for index, chave in enumerate(kmedias_resultados_de_uma_metrica, start=1):

        desempenho_km = kmedias_resultados_de_uma_metrica[chave][0]
        desempenho_sk = sklearn_resultados_de_uma_metrica[chave][0]

        tempo_km = kmedias_resultados_de_uma_metrica[chave][1]
        tempo_sk = sklearn_resultados_de_uma_metrica[chave][1]

        razao_acertos = round(desempenho_km / desempenho_sk, 3)
        razao_tempo = round(tempo_km / tempo_sk, 3)

        linha_dataframe[chave] = razao_acertos
        linha_dataframe[f'Tempo {index}'] = razao_tempo

    temporario = pd.DataFrame(linha_dataframe, index=[0], dtype='object')

    dataframe = pd.concat([dataframe, temporario], ignore_index=True)

    return dataframe


def preprocess(dataset: np.ndarray):
    return


def executa_datasets(lista_de_datasets: list[str]) -> pd.DataFrame:

    dataframe = cria_dataframe(METRICAS)

    for nome_dataset in lista_de_datasets:

        version = 1
        df = None

        while version <= 5:
            try:
                df = fetch_openml(
                    name=nome_dataset, version=version, parser='auto'
                )
                break

            except OpenMLError:
                version += 1

        if version > 5:
            print('Não achei a base de dados')

        if df is not None:
            categorias = df['target'].unique()
            n_class = len(categorias)
            n_iter = 1

            labels = make_label(df['target'], n_class, categorias)

            sklearn_resultados_de_uma_metrica = sklearn_parametrizer(
                df['data'], labels, n_class, METRICAS, n_iter
            )

            kmedias_resultados_de_uma_metrica = kmedias_parametrizer(
                df['data'], labels, n_class, METRICAS, n_iter
            )

            dataframe = formata_resultado_de_uma_metrica(
                dataframe,
                nome_dataset,
                METRICAS,
                sklearn_resultados_de_uma_metrica,
                kmedias_resultados_de_uma_metrica,
            )

    return dataframe


def main():
    #df = executa_datasets(OPENML_DATASETS)
    #df.to_csv(df, index=False)

    df = fetch_openml(name='anneal', version=1 , parser='auto')
    kmeans = KMeans(n_clusters=3, n_init='auto', init='random')
    print(df['data'].values)
    kmeans.fit(df['data'].values)
    return


if __name__ == '__main__':
    main()
