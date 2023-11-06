from sklearn.datasets import fetch_openml
from openml import tasks, runs
from kmeans import KMeans

df = fetch_openml(name='iris', version=1, parser='auto')

task = tasks.get_task(4537)
print(task)
kmeans = KMeans(3)
print(kmeans.fit(df.data.values))
#print(cluster.clusters)
#print(kmeans.centroides)
