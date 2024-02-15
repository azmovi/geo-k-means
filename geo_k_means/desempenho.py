import asyncio
from time import perf_counter

# import kmeans
import numpy as np
import pandas as pd

# Importações de métricas no uso no algoritmo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError
from sklearn.utils._bunch import Bunch

from geo_k_means.kmedias import fit_kmedias  # pytest

# from kmedias import fit_kmedias


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
    dict_dos_resultados = {metrica: [] for metrica in METRICAS.keys()}
    start = perf_counter()
    for _ in range(n_iter):
        kmeans = KMeans(n_clusters=n_class, n_init='auto', init='random')
        kmeans.fit(data)
        for metrica in dict_de_metricas.keys():
            dict_dos_resultados[metrica].append(
                dict_de_metricas[metrica](labels, kmeans.labels_)
            )

    for key, value in dict_dos_resultados.items():
        dict_dos_resultados[key] = round(np.mean(value), 3)

    end = perf_counter()
    dict_dos_resultados['tempo'] = end - start
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
    dict_dos_resultados = {key: [] for key in METRICAS.keys()}
    start = perf_counter()
    for _ in range(n_iter):
        atributos = fit_kmedias(data, n_class)
        for metrica in dict_de_metricas.keys():
            dict_dos_resultados[metrica].append(
                dict_de_metricas[metrica](labels, atributos['rotulo'])
            )

    for key, value in dict_dos_resultados.items():
        dict_dos_resultados[key] = round(np.mean(value), 3)

    end = perf_counter()
    dict_dos_resultados['tempo'] = end - start

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
        >>> _preprocess_target(df['target'])
        array([0, 0, 0, 0, ..., 2])

    """
    categorias = target.unique()
    rotulo = {rotulo: index for index, rotulo in enumerate(categorias)}

    return np.array([rotulo[tipo] for tipo in target])


def preprocess(dataframe: Bunch) -> tuple[np.ndarray]:

    data = _preprocess_data(dataframe['data'])
    target = _preprocess_target(dataframe['target'])
    return data, target


async def desempenho_abordagem(
    nome: str, n_iter: int, tipo: bool
) -> dict[str, str | float]:

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
        dicio_de_metricas = {'nome': nome}
        data, labels = preprocess(df)

        if tipo:
            dicio_de_metricas.update(
                sklearn_parametrizer(data, labels, n_class, METRICAS, n_iter)
            )
        else:
            dicio_de_metricas.update(
                kmedias_parametrizer(data, labels, n_class, METRICAS, n_iter)
            )

    return dicio_de_metricas


def exec_razao(
    list_dict_sklearn: list[dict[str, str | float]],
    list_dict_kmedias: list[dict[str, str | float]],
) -> dict[str, str | float]:

    list_dict_razao = list()

    for index, dict_parameters in enumerate(list_dict_sklearn):
        dict_base = dict()
        for key in dict_parameters.keys():
            # Razão de acertos
            if key == 'nome':
                dict_base[key] = dict_parameters[key]
            else:
                dict_base[key] = round(
                    (
                        list_dict_kmedias[index][key]
                        / list_dict_sklearn[index][key],
                        3,
                    )
                )
        list_dict_razao.append(dict_base)

    return list_dict_razao


async def exec_abordagem(tipo: bool):
    tasks = []
    n_iter = 30
    for nome in OPENML_DATASETS:
        task = asyncio.create_task(desempenho_abordagem(nome, n_iter, tipo))
        tasks.append(task)

    return await asyncio.gather(*tasks)


async def cria_csv_kmeans(lib: bool) -> dict[str, str | float]:
    string = '_desempenho.csv'
    if lib:
        lista_de_dict = await exec_abordagem(lib)
        string = 'sklearn' + string
    else:
        lista_de_dict = await exec_abordagem(lib)
        string = 'kmedias' + string

    if lista_de_dict:
        df = pd.DataFrame(lista_de_dict)
        df.to_csv(string, index=False)
        print('Arquivo csv criado')
    return lista_de_dict


async def main() -> None:
    resultados = []
    lib = True
    for i in range(2):
        resultados.append(await cria_csv_kmeans(lib))
        lib = False
    lista_de_dict = exec_razao(resultados[0], resultados[1])
    df = pd.DataFrame(lista_de_dict)
    df.to_csv('razao_desempenho.csv', index=False)

    return


if __name__ == '__main__':
    start = perf_counter()
    asyncio.run(main())
    end = perf_counter()
    print(f'Time total {end - start}')
