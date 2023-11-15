

import kmeans as kmedias
import time

from sklearn.datasets import fetch_openml
# from sklearn.cluster import KMeans


df = fetch_openml(name='amazon-commerce-reviews', version=1, parser='auto')
data = df['data'].values

n_class = len(df['target'].unique())

inicio = time.perf_counter()
atributos = kmedias.fit(data, n_class)

fim = time.perf_counter()

print(fim - inicio)
