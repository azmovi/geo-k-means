import random
from math import sqrt

import numpy as np


def euclidean_distance(ponto1: list[float], ponto2: list[float]) -> float:
    """
    Calcula a distância entre dois pontos no espaço euclidiano n-dimensional.

    Parameters:
        ponto1: lista de coordenadas do primeiro ponto
        ponto2: lista de coordenadas do segundo ponto
    Returns:
        Um float que represeta a distância entre os dois pontos.

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

    distance = sum((x - y) ** 2 for x, y in zip(ponto1, ponto2)) ** 0.5

    return distance


class KMeans:
    def __init__(self, numero_clusters: int, maximo_iteracoes: int = 100):
        self.numero_clusters = numero_clusters
        self.maximo_iteracoes = maximo_iteracoes

    def update_clusters(
        self, data_x: list[list[float]], centroides: list[list[float]]
    ) -> dict[tuple(list[float]), list[float]]:
        """
        Relaciona os clusters aos seus melhores pontos, ou seja, aqueles que apresentam a menor distância a partir da base de dados.

        Parameters:
            data_x: Uma lista de pontos, onde cada ponto é uma lista de coordenadas.
            centroides: Uma lista de centroides, onde cada centróide é uma lista de coordenadas.

        Returns:
            Um dicionário onde as chaves são os centroides e os valores são as listas dos melhores pontos associados a cada centróide.

        Examples:
            >>> test1 = KMeans(2)
            >>> data1_x = [[2, 1], [6, 4], [3, 5], [8, 7], [9, 8], [10, 7]]
            >>> centroides1 = [[3, 4], [8, 8]]
            >>> test1.update_clusters(data1_x, centroides1)
            {(3, 4): [[2, 1], [6, 4], [3, 5]], (8, 8): [[8, 7], [9, 8], [10, 7]]}
            >>> test2 = KMeans(3)
            >>> data2_x = [[0, 1, 0], [2, 1, 0], [-1, -1, -1], [1, 1, 1], [6, 7, 8], [6, 6, 6], [9, 10, 11]]
            >>> centroides2 = [[0, 0, 0], [8, 8, 8]]
            >>> test2.update_clusters(data2_x, centroides2)
            {(0, 0, 0): [[0, 1, 0], [2, 1, 0], [-1, -1, -1], [1, 1, 1]], (8, 8, 8): [[6, 7, 8], [6, 6, 6], [9, 10, 11]]}
        """
        clusters = {}
        for centroide in centroides:
            clusters[tuple(centroide)] = []

        for x in data_x:
            menor_distancia = float('inf')
            melhor_centroide = None

            for centroide in centroides:
                distancia = euclidean_distance(x, centroide)
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    melhor_centroide = centroide

            clusters[tuple(melhor_centroide)].append(x)

        return clusters

    def update_centroides(
        self, clusters: dict[tuple(list[float]), list[float]]
    ) -> list[float]:
        """
        Atualiza os valores dos centróides com base na média das coordenadas dos pontos em cada cluster.

        Parameters:
            clusters: Um dicionário onde as chaves são as coordenadas dos centróides e os valores são listas de pontos atribuídos a cada centróide.

        Returns:
            Uma lista de novos centróides, onde cada centróide é uma lista de coordenadas recalculadas.

        Examples:
            >>> kmeans = KMeans(2)
            >>> clusters = {(3, 4): [[3, 3], [6, 4], [3, 5]], (8, 8): [[8, 7], [9, 8], [10, 9]]}
            >>> kmeans.update_centroides(clusters)
            [[4.0, 4.0], [9.0, 8.0]]
            >>> clusters = {(2, 3, 1): [[0, 0, 0], [-1, -1, -1], [1, 2, 3], [3, 2, 1]], (8, 8, 8): [[5, 6, 7], [9, 8, 10]]}
            >>> kmeans.update_centroides(clusters)
            [[0.75, 0.75, 0.75], [7.0, 7.0, 8.5]]
        """
        novos_centroides = []

        for cluster, pontos in clusters.items():
            novo_centroide = []
            num_pontos = len(pontos)

            for dimen in range(len(pontos[0])):
                soma = 0
                for ponto in pontos:
                    soma += ponto[dimen]

                media = soma / num_pontos
                novo_centroide.append(media)
            novos_centroides.append(novo_centroide)

        return novos_centroides

    def fit(self, data: list[list[float]]) -> bool:
        """
        Executa o algoritmo KMeans para clusterização dos dados.

        Parameters:
            data (list[list[float]]): Uma lista contendo os pontos de dados a serem clusterizados.

        Returns:
            bool: True se a convergência foi alcançada, False caso contrário.

        Examples:
            >>> test = KMeans(2)
            >>> data = [[2, 1], [6, 4], [3, 5], [8, 7], [9, 8], [10, 7], [1, 2], [4, 3], [5, 5]]
            >>> test.fit(data)
            True
        """

        centroides = random.sample(list(data), self.numero_clusters)

        for i in range(self.maximo_iteracoes):
            clusters = self.update_clusters(data, centroides)

            novos_centroides = self.update_centroides(clusters)

            if np.array_equal(novos_centroides, centroides):
                self.centroides = centroides
                self.clusters = clusters
                return True

            centroides = novos_centroides
