"""
Modulo capaz de realizar o algortimo de Kmeans
"""
import numpy as np


class Kmedias:
    """Classe que treina um modelo baseado no algortimo kmeans"""

    def __init__(self, data, n_clusters: int, n_iter: int = 300) -> None:
        """
        A função construtora do nosso algoritmo de clusterização com os
        atributos data, n_clusters,  n_iter, n_amostra, rotulo
        Parameters:
            data: o conjunto de dados para ser treinado
            n_clusters: o numero de centroides que o nosso algortimo precisa
            n_iter: o numero máximo de iterações possíveis para encontrar a
            convergência
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = Kmedias(data, 2)
        """
        self.data = data
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_amostra = self.data.shape[0]
        self.rotulo = np.zeros(self.n_amostra)
        self.idx_centroides = np.zeros(self.n_clusters)

    def update_clusters(self, centroides: np.ndarray) -> list(list[float]):
        """
        Atualiza os valores dos clusters baseado nas distancias euclidianas e
        rotula os respectivos dados baseado no cluster que ele é mais próximo
        Parameters:
            centroides: O vetor de centroides
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = Kmedias(data, 2, 3)
            >>> obj.idx_centroides = [1, 4]
            >>> obj.update_clusters(np.array([[2, 10], [0, 0]]))
            [[array([ 2, 10]), ...]]
        """
        clusters = [[] for _ in range(len(centroides))]

        for idx, ponto in enumerate(self.data):
            distancias = np.empty(len(centroides), dtype=np.float64)
            for i, centro in enumerate(centroides):
                distancias[i] = np.linalg.norm(ponto - centro)

            indice = np.argmin(distancias)
            clusters[indice].append(ponto)
            self.rotulo[idx] = indice

        return clusters

    def update_centroides(
        self, clusters: list[list[float]], centroides: np.ndarray
    ) -> np.ndarray:
        """
        Atualiza o valor dos centroides baseado na media das coordenadas
        presentes em cada vetor do cluster respectivo.
        Parameters:
            clusters: A lista de vetores em seu respectivo cluster
        Returns:
            Um vetor com as coordenadas dos vetores dos novos centroides
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = Kmedias(data, 2)
            >>> centroides = [
            ... np.array([[4, 1], [2, 10], [4, 5], [2, 1]]), np.array([[0, 0]])
            ... ]
            >>> obj.update_centroides(centroides, data)
            array([[3, 4],
                   [0, 0],
                   [4, 5],
                   [2, 1],
                   [0, 0]])
        """
        for index, lista_de_pontos in enumerate(clusters):
            media = np.mean(lista_de_pontos, axis=0)
            centroides[index] = media

        return centroides

    def fit(self) -> None:
        """
        Executa o treinamento de classificação atualizando os clusters e
        centroides com o passar das iterações e no final rotula os dados.
        Parameters:
            data: Um dataset que será treinado.
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = Kmedias(data, 2)
            >>> obj.fit()
        """
        self.idx_centroides = np.random.choice(
            self.n_amostra, self.n_clusters, replace=False
        )
        centroides = self.data[self.idx_centroides]
        centroides_antigo = np.zeros_like(centroides)

        for _ in range(self.n_iter):
            clusters = self.update_clusters(centroides)
            centroides = self.update_centroides(clusters, centroides)

            if np.array_equal(centroides, centroides_antigo):
                break

            centroides_antigo = centroides.copy()
