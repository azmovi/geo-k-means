from math import sqrt
from random import sample

import numpy as np


def euclidean_distance(ponto1: list[int], ponto2: list[int]) -> float:
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
    def __init__(self, numero_clusters: int, maximo_interacoes: int = 100):
        self.numero_clusters = numero_clusters
        self.maximo_interacões = maximo_interacoes

    # def fit(self, data_x:list[int]):
    # centroides = sample(data_x, self.numero_clusters)

    def update_clusters(
        self, data_x: list[list[float]], centroides: list[list[float]]
    ) -> dict[tuple(list[float]), list[float]]:
        """
        Relaciona os clusters aos seus melhores pontos, ou seja, aqueles que apresentam a menor distância a partir da base de dados.

        Parameters:
            data_x (list[list[float]]): Uma lista de pontos, onde cada ponto é uma lista de coordenadas.
            centroides (list[list[float]]): Uma lista de centroides, onde cada centróide é uma lista de coordenadas.

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
