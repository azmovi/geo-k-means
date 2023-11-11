from time import perf_counter

import pandas as pd
from kmeans import KMeans as KMedias  # Execução Padrão

# Importações de métricas no uso no algoritmo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError

# from geo_k_means.kmeans import KMeans as KMedias  # Pytest


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
    'amazon-commerce-reviews',
    'semeion',
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
            kmeans.fit(data.map(int).values)
            soma_da_metrica += dict_de_metricas[key](labels, kmeans.labels_)

        end = perf_counter()
        duration = end - start

        dict_dos_resultados[key] = [soma_da_metrica / n_iter, duration, 3]

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
            kmedias.fit(data.map(int).values)
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
        >>> cria_data_frame(metricas)
        Empty DataFrame
        Columns: [completeness_score, Tempo 1]
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


def executa_datasets(lista_de_datasets: list[str]) -> pd.DataFrame:

    dataframe = cria_data_frame(METRICAS)

    for nome_dataset in lista_de_datasets:

        version = 1
        df = None

        while version <= 5:
            try:
                df = fetch_openml(
                        name=nome_dataset,
                        version=version,
                        parser='auto'
                )
                break

            except OpenMLError:
                version += 1

        if version > 5:
            print("Não achei a base de dados")

        if df is not None:
            n_class = len(df['target'].unique())
            lista_de_classes = df['target'].unique().categories
            n_iter = 1

            labels = convert_label_to_int(
                df['target'], n_class, lista_de_classes
            )

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
    data_sets = ['iris', 'balance-scale']
    df = executa_datasets(data_sets)

    return df.to_csv('docs/test.csv', index=False)


if __name__ == '__main__':
    main()
