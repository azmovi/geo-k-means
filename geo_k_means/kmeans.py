import numpy as np


def update_clusters(
    dataset: np.ndarray,
    centroides: np.ndarray,
) -> list(list[float]):
    """
    Cria uma relação entre um centroide e um vetor do dataset, baseado na norma
    entre os dois vetores.

    Parameters:
        dataset: Uma lista de centroides, onde cada centroide é um vetor.

        centroides: Uma lista de centroides, onde cada centroide é um vetor.

    Returns:
        Uma lista de listas onde a quantidade de listas sera igual a quantidade
        de centroides, e dentro das listas terão os vetores que estão mais
        próximos de um centroide.

    Examples:
        >>> data = np.array([4, 1, 2, 10, 4, 5, 2, 1])
        >>> centroides = np.array([3, 5])
        >>> update_clusters(data, centroides)
        [[4, 1, 2, 4, 2, 1], [10, 5]]
    """
    clusters = [[] for _ in range(len(centroides))]
    for ponto in dataset:
        distancias = np.empty(len(centroides), dtype=np.float64)
        for i in range(len(centroides)):
            distancias[i] = np.linalg.norm(ponto - centroides[i])

        indice = np.argmin(distancias)
        clusters[indice].append(ponto)
    return clusters


def update_centroides(
    clusters: list[list[float]], centroides: np.ndarray
) -> np.ndarray:
    """
    Atualiza os valores dos centroides com base na média das coordenadas
    dos vetores presentes em cada clusters.

    Parameters:
        clusters: uma lista de vetores da base de dados.
        Centroide: uma lista com os centroides.

    Returns:
        Uma lista de novos centróides, onde cada centróide é uma lista de
        coordenadas recalculadas.

    Examples:

        >>> clusters = [[4, 1, 2, 4, 2, 1], [10, 5]]
        >>> centroides = np.array([3, 5])
        >>> update_centroides(clusters, centroides)
        array([2, 7])
    """

    for index, lista_de_pontos in enumerate(clusters):
        media = np.mean(lista_de_pontos, axis=0)
        centroides[index] = media

    return centroides


def fit(
    data: np.ndarray[float], n_class: int, n_iter: int = 100
) -> dict[str, list[float]]:
    """
    Executa o treinamento de classificação atualizando os clusters e centroides
    com o passar das iterações e no final rotula os dados.

    Parameters:
        data: Um dataset que será treinado.
        n_class: O numero de classes que terá o conjunto de dados.
        n_iter: A quantidade máxima de iterações que o algoritmo pode fazer.

    Returns:
        Retorna um dicionario que contem as chaves: clusters, centroides,
        rotulo.

    Examples:
        >>> data = np.array([4, 1, 2, 10, 4, 5, 2, 1])
        >>> fit(data, 2)
        {'clusters': ...}

    """
    centroides = data[np.random.choice(data.shape[0], n_class, replace=False)]
    centroides_antigo = np.zeros_like(centroides)

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
        'rotulo': rotulo,
    }
    return atributos
