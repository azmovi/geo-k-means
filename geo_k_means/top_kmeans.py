"""
Modulo que implementa o algoritmo kmeans topologic
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph


class TopKmeans:
    """Classe que implementa o algoritmo de kmedias topologico"""

    def __init__(
        self,
        data: np.ndarray,
        n_clusters: int,
        n_vizinhos: int,
        n_iter: int = 300,
    ) -> None:
        """
        A função construtora do nosso algoritmo de clusterização com os
        atributos data, n_clusters, n_vizinho, n_iter, n_amostra, rotulo
        Parameters:
            data: o conjunto de dados para ser treinado
            n_clusters: o numero de centroides que o nosso algortimo precisa
            n_vizinhos: o numero de vizinhos para determinar em que cluster o
            ponto se adequa
            n_iter: o numero máximo de iterações possíveis para encontrar a
            convergência
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = TopKmeans(data, 2, 3)
        """
        self.data = data
        self.n_clusters = n_clusters
        self.n_vizinhos = n_vizinhos
        self.n_iter = n_iter
        self.n_amostra = self.data.shape[0]
        self.rotulo = np.zeros(self.n_amostra)
        self.idx_centroides = np.zeros(self.n_clusters)

    def update_clusters(self) -> list[np.ndarray]:
        """
        Atualiza os valores dos clusters baseado nas distancias geodésicas e
        rotula os respectivos dados baseado no cluster que ele é mais próximo
        Parameters:
            centroides: O vetor de centroides
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = TopKmeans(data, 2, 3)
            >>> obj.idx_centroides = [1, 4]
            >>> obj.update_clusters()
            [[array([4, 1]), ...]]
        """
        clusters = [[] for _ in range(self.n_clusters)]
        grafo = kneighbors_graph(self.data, self.n_vizinhos)
        distancias = np.zeros((self.n_clusters, self.n_amostra))
        for i in range(self.n_clusters):
            distancias[i] = shortest_path(
                csgraph=grafo, directed=False, indices=self.idx_centroides[i]
            )

        for i in range(self.n_amostra):
            idx_vizinho_mais_proximo = distancias[:, i].argmin()
            clusters[idx_vizinho_mais_proximo].append(self.data[i])
            self.rotulo[i] = idx_vizinho_mais_proximo
        return clusters

    def update_centroides(self, clusters: list[np.ndarray]) -> np.ndarray:
        """
        Atualiza o valor dos centroides baseado na media das coordenadas
        presentes em cada vetor do cluster respectivo.
        Parameters:
            clusters: A lista de vetores em seu respectivo cluster
        Returns:
            Um vetor com as coordenadas dos vetores dos novos centroides
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = TopKmeans(data, 2, 3)
            >>> centroides = [
            ... np.array([[4, 1], [2, 10], [4, 5], [2, 1]]), np.array([[0, 0]])
            ... ]
            >>> obj.update_centroides(centroides)
            [array([3.  , 4.25]), array([0., 0.])]
        """
        centroides = [[] for _ in range(self.n_clusters)]
        for idx, lista_de_pontos in enumerate(clusters):
            media = np.mean(lista_de_pontos, axis=0)
            centroides[idx] = media

        return centroides

    def fit(self) -> None:
        """
        Executa o treinamento de classificação atualizando os clusters e
        centroides com o passar das iterações e no final rotula os dados.
        Parameters:
            data: Um dataset que será treinado.
        Examples:
            >>> data = np.array([[4, 1], [2, 10], [4, 5], [2, 1], [0, 0]])
            >>> obj = TopKmeans(data, 2, 3)
            >>> obj.fit()
        """

        self.idx_centroides = np.random.choice(
            self.n_amostra, self.n_clusters, replace=False
        )
        centroides = self.data[self.idx_centroides]
        centroides_antigo = np.zeros_like(centroides)

        for _ in range(self.n_iter):
            clusters = self.update_clusters()
            centroides = self.update_centroides(clusters)
            self.data[self.idx_centroides] = centroides

            if np.array_equal(centroides, centroides_antigo):
                break

            centroides_antigo = centroides.copy()
