from sklearn.datasets import fetch_openml

# from geo_k_means.kmeans import KMeans  # Pytest

from kmeans import KMeans  # Execução Padrão


def desempenho_iris() -> None:
    df = fetch_openml(name='iris', version=1, parser='auto')
    kmeans = KMeans(3)
    print(kmeans.fit(df.data.values))


def main():
    desempenho_iris()


if __name__ == "__main__":
    main()
