from math import sqrt

import numpy as np


def euclidean_distance(
    ponto1: list[int, int], ponto2: list[int, int]
) -> float:
    """
    Calcula a distância Euclidiana entre dois pontos no plano cartesiano.

    Parameters:
        ponto1: primeiro ponto do plano cartesiano
        ponto2: segundo ponto do plano cartesiano
    Returns:
        Um float que represeta a distância entre os dois pontos.

    Examples:
        >>> euclidean_distance([2, 1], [6, 4])
        5.0

        >>> euclidean_distance([0, 0], [20, 15])
        25.0

    """

    return sqrt((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2)
