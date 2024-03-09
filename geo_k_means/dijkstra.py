"""
Modulo capaz de realizar distancias minimas usando o algortimo de dijkstra
"""
import heapq


class Grafo:
    """
    Classe para criar um grafo
    """

    def __init__(self, size: int) -> None:
        """
        Inicialização de um objeto grafo baseado no número de vértices

        Parameters:
            size: inteiro que representa o tamanho da matriz (n x n)

        Examples:
            >>> grafo = Grafo(10)
        """
        self.size = size
        self.matrix = [[0 for _ in range(size)] for _ in range(size)]

    def adiciona_aresta(self, linha, coluna, distancia) -> bool:
        """
        Adiciona uma aresta no grafo não direcionado dado dois vértices e a
        distancia entre eles

        Parameters:
            linha: o valor do primeiro vértice
            coluna: o valor do segundo vértice
            distancia: a distancia entre os dois vértices

        Returns:
            Retorno de um booleano caso a adição ocorra de forma correta ou
            não.

        Examples:
            >>> grafo = Grafo(10)
            >>> grafo.adiciona_aresta(2, 6, 5)
            True
            >>> grafo.adiciona_aresta(3, 3, 10)
            False
            >>> grafo.adiciona_aresta(3, 5, -1)
            False
        """
        if linha == coluna:
            return False

        if distancia < 0:
            return False

        self.matrix[coluna][linha] = distancia
        self.matrix[linha][coluna] = distancia
        return True


def dijsktra(grafo: Grafo, inicio: int) -> dict[int, int]:
    """
    Encontra os caminhos mínimos possíveis para os vértices com distancia
    positiva de um dado grafo.

    Parameters:
        grafo: Uma instancia da classe Grafo
        inicio: O vértice que será a referencia principal de distancia

    Returns:
        Um dicionario com a distancia minima de um determinado vértice

    Examples:
        >>> grafo = Grafo(5)
        >>> grafo.adiciona_aresta(0, 4, 20)
        True
        >>> grafo.adiciona_aresta(0, 1, 10)
        True
        >>> grafo.adiciona_aresta(1, 4, 50)
        True
        >>> grafo.adiciona_aresta(3, 4, 70)
        True
        >>> grafo.adiciona_aresta(1, 3, 40)
        True
        >>> grafo.adiciona_aresta(2, 3, 60)
        True
        >>> grafo.adiciona_aresta(1, 2, 30)
        True
        >>> dijsktra(grafo, 0)
        {0: 0, 1: 10, 2: 40, 3: 50, 4: 20}
    """
    distancias = {}
    visitados = set()
    for vertice in range(grafo.size):
        distancias[vertice] = float('inf')
    distancias[inicio] = 0
    fila_de_prioridade = []
    heapq.heappush(fila_de_prioridade, (0, inicio))

    while len(fila_de_prioridade) != 0:
        distancia_atual, vertice_atual = heapq.heappop(fila_de_prioridade)

        for vizinho in range(grafo.size):
            if (
                grafo.matrix[vertice_atual][vizinho] != 0
                and vizinho not in visitados
            ):
                distancia_possivel = grafo.matrix[vertice_atual][vizinho]

                if distancia_atual + distancia_possivel < distancias[vizinho]:
                    distancias[vizinho] = distancia_atual + distancia_possivel
                    heapq.heappush(
                        fila_de_prioridade, (distancias[vizinho], vizinho)
                    )

    return distancias
