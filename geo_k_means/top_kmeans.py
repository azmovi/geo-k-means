import numpy as np
from preprocessamento import preprocess
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import fetch_openml

from scipy.sparse.csgraph import shortest_path
# single_source_dijkstra_path_length -> dict do tamanho dos menores caminhos com todas as aretas


def top_fit(
    data: np.ndarray[float], n_class: int, n_neighbors: int, n_iter: int = 100
) -> dict[str, list[float]]:
    """
    Executa o treinamento de classificação atualizando os clusters e centroides
    com o passar das iterações e no final rotula os dados.
    Parameters:
        data: Um dataset que será treinado.
    """
    idx_centroides = np.random.choice(data.shape[0], n_class, replace=False)
    centroides = data[idx_centroides]
    distancias = np.zeros((n_class, data.shape[0]))

    for _ in range(1):
        grafo = kneighbors_graph(data, n_neighbors)
        # Update cluters?
        for i in range(n_class):
            distancias[i] = shortest_path(
                    csgraph=grafo, directed=False, indices=idx_centroides[i]
            )
    return distancias


df = fetch_openml(name='iris', version=1, parser='auto')
data, target = preprocess(df)
print(top_fit(data, 3, 11))
