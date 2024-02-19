import numpy as np
from preprocessamento import preprocess
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import fetch_openml

from scipy.sparse.csgraph import shortest_path


class Top_kmeans:
    def __init__(
        self,
        data: np.ndarray[float],
        n_clusters: int,
        n_vizinhos: int,
        n_iter: int = 100,
    ) -> None:

        self.data = data
        self.n_clusters = n_clusters
        self.n_vizinhos = n_vizinhos
        self.n_iter = n_iter
        self.n_amostra = self.data.shape[0]
        self.rotulo = np.zeros(self.n_amostra)
        return

    def update_clusters(
            self, centroides: np.ndarray[float]
    ) -> list[np.ndarray[float]]:
        """ """
        clusters = [[] for _ in range(self.n_clusters)]
        grafo = kneighbors_graph(self.data, self.n_vizinhos)
        distancias = np.zeros((self.n_clusters, self.n_amostra))
        for i in range(self.n_clusters):
            distancias[i] = shortest_path(
                csgraph=grafo, directed=False, indices=self.idx_centroides[i]
            )

        for i in range(self.n_amostra):
            idx_vizinho_mais_proximo = distancias[:, i].argmin()
            clusters[idx_vizinho_mais_proximo].append(
                data[i]
            )
            self.rotulo[i] = idx_vizinho_mais_proximo
        return clusters

    def update_centroides(
        self, clusters: list[np.ndarray[float]]
    ) -> np.ndarray[float]:
        """

        """
        centroides = [[] for _ in range(self.n_clusters)]
        for idx, lista_de_pontos in enumerate(clusters):
            media = np.mean(lista_de_pontos, axis=0)
            centroides[idx] = media

        return centroides

    def fit(self) -> dict[str, list[float]]:
        """
        Executa o treinamento de classificação atualizando os clusters e
        centroides com o passar das iterações e no final rotula os dados.
        Parameters:
            data: Um dataset que será treinado.
        """

        idx_centroides = np.random.choice(
            self.n_amostra, self.n_clusters, replace=False
        )
        centroides = data[idx_centroides]
        centroides_antigo = np.zeros_like(centroides)

        for i in range(5):
            clusters = self.update_clusters(centroides)
            centroides = self.update_centroides(clusters)

            if np.array_equal(centroides, centroides_antigo):
                print(i)
                break

            centroides_antigo = centroides.copy()

        return


from sklearn import metrics

df = fetch_openml(name='iris', version=1, parser='auto')
data, target = preprocess(df)
obj = Top_kmeans(data, 3, 11)
obj.fit()
print(metrics.rand_score(target, obj.rotulo))
