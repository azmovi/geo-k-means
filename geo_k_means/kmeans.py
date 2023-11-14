import numpy as np
from numba import jit


def tipo_um_dict(centroides: np.ndarray):
    """
    Cria uma array do numpy baseado em um centroide interligado com os
    pontos que fazem parte desse cluster

    Parameters:
        centoides: um array com os centroides que foram escolhidos
        previamente

    Examples:
        >>> array = np.array([[2, 5, 9], [10, 10, 6]])
        >>> tipo_um_dict(array)
        array([([ 2.,  5.,  9.], ...)

    """

    tipo = [('centroide', float, centroides.shape[1]), ('pontos', list)]

    clusters = np.array(
        [(centroide, []) for centroide in centroides], dtype=tipo
    )

    return clusters


def update_clusters(
    data_x: np.ndarray, centroides: np.ndarray
) -> np.ndarray:
    """
    Relaciona os clusters aos seus melhores pontos, ou seja, aqueles que
    apresentam a menor distância a partir da base de dados.

    Parameters:
        data_x: Uma lista de pontos, onde cada ponto é uma lista de
        coordenadas.

        centroides: Uma lista de centroides, onde cada centróide é uma
        lista de coordenadas.

    Returns:
        Um dicionário onde as chaves são os centroides e os valores são as
        listas dos melhores pontos associados a cada centróide.

    Examples:
        >>> test1 = KMeans(2)
        >>> data1_x = np.array(
        ... [[2, 1], [6, 4], [3, 5], [8, 7], [9, 8], [10, 7]]
        ... )
        >>> centroides1 = np.array([[3, 4], [8, 8]])
        >>> test1.update_clusters(data1_x, centroides1)
        array([([3., 4.],...)
    """
    clusters = tipo_um_dict(centroides)

    for index, ponto in enumerate(data_x):
        distancias = np.linalg.norm(centroides - ponto, axis=1)
        indice_menor_distancia = np.argmin(distancias)
        clusters[indice_menor_distancia]['pontos'].append(ponto)

    return clusters


def update_centroides(clusters: np.ndarray, centroides: np.ndarray) -> None:
    """
    Atualiza os valores dos centróides com base na média das coordenadas
    dos pontos em cada cluster.

    Parameters:
        clusters: Um dicionário onde as chaves são as coordenadas dos
        centróides e os valores são listas de pontos atribuídos a cada
        centróide.

    Returns:
        Uma lista de novos centróides, onde cada centróide é uma lista de
        coordenadas recalculadas.

    Examples:
        >>> kmeans = KMeans(2)
        >>> clusters = {
        ... (3, 4): [[3, 3], [6, 4], [3, 5]],
        ... (8, 8): [[8, 7], [9, 8],[10, 9]]
        ... }
        >>> kmeans.update_centroides(clusters)
        [[4.0, 4.0], [9.0, 8.0]]
        >>> clusters = {
        ... (2, 3, 1): [[0, 0, 0], [-1, -1, -1], [1, 2, 3], [3, 2, 1]],
        ... (8, 8, 8): [[5, 6, 7], [9, 8, 10]]
        ... }
        >>> kmeans.update_centroides(clusters)
        [[0.75, 0.75, 0.75], [7.0, 7.0, 8.5]]
    """

    for index, lista_de_pontos in enumerate(clusters['pontos']):
        media = np.mean(lista_de_pontos, axis=0)
        centroides[index] = media

    return centroides


def rotula_os_dados(quantidade_de_linhas: int, clusters: np.ndarray):
    rotulos = np.zeros(quantidade_de_linhas)

    for valor, (_, valores) in enumerate(clusters):
        indices = np.arange(len(valores))
        rotulos[indices] = valor

    return rotulos


def fit(
        data: np.ndarray[float],
        n_class: int,
        n_iter: int = 300
) -> np.ndarray:
    """
    Executa o algoritmo KMeans para clusterização dos dados.

    Parameters:
        data: Um dataset que será treinado.

    Returns:
        bool: True se a convergência foi alcançada, False caso contrário.

    Examples:
    """

    if len(data) != 0:
        centroides = data[
                np.random.choice(data.shape[0], n_class, replace=False)
        ]

        for _ in range(n_iter):
            centroides_antigo = centroides
            clusters = update_clusters(data, centroides)
            centroides = update_centroides(clusters, centroides)

            if np.array_equal(centroides, centroides_antigo):
                rotulo = rotula_os_dados(data.shape[0], clusters)
                return clusters, centroides, rotulo

        rotulo = rotula_os_dados(data.shape[0], clusters)
        return clusters, centroides, rotulo
