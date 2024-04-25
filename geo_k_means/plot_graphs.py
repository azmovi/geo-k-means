import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

kmeans_df = pd.read_csv('../docs/desempenhos/sklearn_desempenho.csv')
top_kmenas_df = pd.read_csv('../docs/desempenhos/topkmeans_desempenho.csv')

linhas_usadas = [
    'AP Breast Colon',
    'AP Colon Prostate',
    'AP  Lung  Kidney',
    'tr31.wc',
    'tr45.wc',
    'leukemia',
    'SRBCT',
    'GCM',
    'lsvt',
]

kmeans_df = df[df['nome'].isin(linhas_usadas)]
