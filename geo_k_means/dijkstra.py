import heapq


class Grafo:
    def __init__(self, size: int) -> None:
        """
        Inicialização de um objeto grafo baseado no número de vértices

        Parameters:
            size: inteiro que representa o tamanho da matriz (n x n)
        Examples:
            >>> grafo = Grafo(10)
        """
        self.size = size
        self.matrix = [[0 for linha in range(size)] for coluna in range(size)]
        return

    def adiciona_aresta(self, linha, coluna, distancia):
        """
        Adiciona uma aresta no grafo não direcionado dado dois vértices e a distancia entre eles

        Parameters:
            linha: o valor do primeiro vértice
            coluna: o valor do segundo vértice
            distancia: a distancia entre os dois vértices
        Returns:
            Retorno de um booleano caso a adição ocorra de forma correta ou não.
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
