import numpy as np
from sklearn.neighbors import kneighbors_graph
from networkx import single_source_dijkstra_path_length

# single_source_dijkstra_path_length -> dict do tamanho dos menores caminhos com todas as aretas


def top_fit(
        data: np.ndarray[float],
        n_class: int,
        n_neighbors: int,
        n_iter: int = 100
) -> dict[str, list[float]]:
    """
    Executa o treinamento de classificação atualizando os clusters e centroides
    com o passar das iterações e no final rotula os dados.
    Parameters:
        data: Um dataset que será treinado.
    """
    centroides = data[np.random.choice(data.shape[0], n_class, replace=False)]
    distancias = np.zeros((n_class, data.shape[0]))
    novos_centroides = centroides
    for _ in range(n_iter):
        grafo = kneighbors_graph(data, n_neighbors)
        # Update cluters?
        clusters = [[] for _ in range(n_class)]
        for i in range(n_class):
            distancias[i] = single_source_dijkstra_path_length(
                    grafo, novos_centroides
            )
