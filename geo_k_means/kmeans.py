import numpy as np
from numba import jit


def distancia_euclidiana(ponto1: np.ndarray, ponto2: np.ndarray) -> float:
    """
    Calcula a distância entre dois pontos no espaço euclidiano n-dimensional.

    Parameters:
        ponto1: lista de coordenadas do primeiro ponto
        ponto2: lista de coordenadas do segundo ponto

    Returns:
        Um float que represeta a distância entre os dois pontos.

    Raises:
        ValueError: Se os pontos tiverem dimensões diferentes.

    Examples:
        >>> euclidean_distance([2, 1], [6, 4])
        5.0

        >>> euclidean_distance([-2, 0, 1], [0, 2, 2])
        3.0

        >>> euclidean_distance([2, 1], [6, 4, 3])
        Traceback (most recent call last):
        ...
        ValueError: Os pontos devem ter a mesma dimensão.
    """
    if len(ponto1) != len(ponto2):
        raise ValueError('Os pontos devem ter a mesma dimensão.')

    distance = np.linalg.norm(ponto1 - ponto2)

    return distance


#@jit(nopython=True)
def update_clusters(
        dataset: np.ndarray,
        lista_de_centroides: np.ndarray,
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
    clusters = [[]] * len(lista_de_centroides)

    for index, ponto in enumerate(dataset):
        distancias = [
            distancia_euclidiana(ponto, centroide) for centroide in lista_de_centroides
        ]
        indice = np.argmin(distancias)
        print(indice)
        print(clusters[indice])
        clusters[indice].append(ponto)

    return np.array(clusters)


def update_centroides(
    clusters: np.ndarray,
    lista_de_centroides: np.ndarray
) -> np.ndarray:
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

    for index, lista_de_pontos in enumerate(clusters):
        media = np.mean(lista_de_pontos, axis=0)
        lista_de_centroides[index] = media

    return lista_de_centroides


def rotula_os_dados(
        data: np.ndarray,
        clusters: np.ndarray
) -> np.ndarray:
    """

    """
    labels = np.zeros()
    for i, cluster in enumerate(clusters):
        labels[[data.tolist().index(point.tolist()) for point in cluster]] = i

    return labels 


def fit(
        data: np.ndarray[float],
        n_class: int,
        n_iter: int = 100 
) -> dict[str, list[float]]:
    """
    Executa o algoritmo KMeans para clusterização dos dados.

    Parameters:
        data: Um dataset que será treinado.

    Returns:
        bool: True se a convergência foi alcançada, False caso contrário.

    Examples:
    """
    centroides = data[
            np.random.choice(data.shape[0], n_class, replace=False)
    ]
    centroides_antigo= np.zeros_like(centroides)


    for _ in range(n_iter):
        clusters = update_clusters(data, centroides)
        centroides = update_centroides(clusters, centroides)

        if np.array_equal(centroides, centroides_antigo):
            break

        centroides_antigo = centroides.copy()


    rotulo = np.zeros(data.shape[0])
    for i, cluster in enumerate(clusters):
        rotulo[[data.tolist().index(point.tolist()) for point in cluster]] = i


    atributos = {
            'clusters': clusters,
            'centroides': centroides,
            'rotulo': rotulo
    }
    return atributos
