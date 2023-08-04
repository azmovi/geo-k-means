from math import sqrt

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
