from time import perf_counter
import asyncio
import pandas as pd
import numpy as np

# Importações de métricas no uso no algoritmo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError
from sklearn.utils._bunch import Bunch

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
    'pasture',
    'glass',
    'kc1-binary',
    'dermatology',
    'backache',
    'lsvt',
    'thoracic-surgery',
    'planning-relax',
]


def sklearn_parametrizer(
    data: np.ndarray,
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
            kmeans.fit(data)
            soma_da_metrica += dict_de_metricas[key](labels, kmeans.labels_)

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [soma_da_metrica, duration]

    return dict_dos_resultados


def kmedias_parametrizer(
    data: np.ndarray,
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
            atributos = kmeans.fit(data, n_class)
            soma_da_metrica += dict_de_metricas[key](
                labels, atributos['rotulo']
            )

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [soma_da_metrica, duration]

    return dict_dos_resultados


def _preprocess_data(dataframe: pd.DataFrame):
    dataframe = dataframe.bfill()
    data = dataframe.values
    if data.dtype != np.float64:
        conversor = {}
        valor = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                try:
                    data[i][j] = float(data[i][j])
                except ValueError:
                    if data[i][j] not in conversor:
                        conversor[data[i][j]] = valor
                        valor += 1
                    data[i][j] = conversor[data[i][j]]
    return data


def _preprocess_target(target: pd.Series) -> np.ndarray:
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
    categorias = target.unique()
    rotulo = {rotulo: index for index, rotulo in enumerate(categorias)}

    return np.array([rotulo[tipo] for tipo in target])


def preprocess(dataframe: Bunch) -> tuple[np.ndarray]:

    data = _preprocess_data(dataframe['data'])
    target = _preprocess_target(dataframe['target'])
    return data, target


async def executa_datasets(nome: str) -> pd.DataFrame:

    version = 1
    while version <= 5:
        try:
            df = fetch_openml(name=nome, version=version, parser='auto')
            break
        except OpenMLError:
            version += 1

    if df is not None:
        categorias = df['target'].unique()
        n_class = len(categorias)
        n_iter = 30
        data, labels = preprocess(df)

        sklearn_metricas_de_um_dataset = sklearn_parametrizer(
            data, labels, n_class, METRICAS, n_iter
        )

        kmedias_metricas_de_um_dataset = kmedias_parametrizer(
            data, labels, n_class, METRICAS, n_iter
        )

    dicio_base = {'nome': nome}
    for index, key in enumerate(
            kmedias_metricas_de_um_dataset.keys(), start=1
    ):

        # Razão de acertos
        dicio_base[key] = round(
            kmedias_metricas_de_um_dataset[key][0]
            / sklearn_metricas_de_um_dataset[key][0],
            3,
        )
        # Razão de tempo
        dicio_base[f'time {index}'] = round(
            kmedias_metricas_de_um_dataset[key][1]
            / sklearn_metricas_de_um_dataset[key][1],
            3,
        )

    return dicio_base


async def main():
    tasks = []
    for nome in OPENML_DATASETS:
        task = asyncio.create_task(executa_datasets(nome))
        tasks.append(task)

    # df = fetch_openml(name='dermatology', version=1, parser='auto')
    return await asyncio.gather(*tasks)


if __name__ == '__main__':
    lista_de_dict = asyncio.run(main())
    df = pd.DataFrame(lista_de_dict)
    df.to_csv('arquivo.csv', index=False)
