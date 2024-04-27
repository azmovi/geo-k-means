import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

kmeans_df = pd.read_csv('/home/azevedo/Code/geo-k-means/docs/desempenhos/sklearn_desempenho.csv')
top_df = pd.read_csv('/home/azevedo/Code/geo-k-means/docs/desempenhos/topkmeans_desempenho.csv')

DATASETS = [
    'AP_Breast_Colon',
    'AP_Colon_Prostate',
    'AP_Lung_Kidney',
    'tr31.wc',
    'tr45.wc',
    'leukemia',
    'SRBCT',
    'GCM',
    'lsvt',
]

def create_df_tratado(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    df = df.drop(columns=['calinski harabasz', 'tempo', 'davies bouldin'])
    df = df[df['nome'].isin(datasets)].reset_index(drop=True)
    return df


def plot_metrics_overlap(
    kmeans_df: pd.DataFrame,
    top_df: pd.DataFrame
) -> None: 
    kmeans_df = create_df_tratado(kmeans_df, DATASETS)
    top_df = create_df_tratado(top_df, DATASETS)
    colunas = kmeans_df.columns

    for top_linha, kmeans_linha in zip(top_df.values, kmeans_df.values):
        fig, ax = plt.subplots(figsize=(14, 10))  # Ajuste o tamanho da figura aqui
        ax.bar(0, 0, color='blue', label='topológico')
        ax.bar(0, 0, color='orange', label='clássico')
        nome_dataset = top_linha[0]
        nome_metricas = colunas[1:]

        for metrica, top_metrica, kmeans_metrica in zip(nome_metricas, top_linha[1:], kmeans_linha[1:]):
            if top_metrica < 0 and kmeans_metrica < 0:
                if top_metrica > kmeans_metrica:
                    ax.bar(metrica, kmeans_metrica, color='orange')
                    ax.bar(metrica, top_metrica, color='blue')
                else:
                    ax.bar(metrica, top_metrica, color='blue')
                    ax.bar(metrica, kmeans_metrica, color='orange')
            
            else:
                if top_metrica > kmeans_metrica:
                    ax.bar(metrica, top_metrica, color='blue')
                    ax.bar(metrica, kmeans_metrica, color='orange')
                else:
                    ax.bar(metrica, kmeans_metrica, color='orange')
                    ax.bar(metrica, top_metrica, color='blue')



        ax.set_title(f'{nome_dataset} Dataset')
        ax.set_xlabel('Metricas')
        ax.set_ylabel('Score')
        plt.xticks(rotation=30)
        ax.legend()
        #plt.show()
        plt.savefig(f'{nome_dataset}_plot.png', format='png')

    return


def create_df_time(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    df = df[df['nome'].isin(datasets)].reset_index(drop=True)
    return df[['nome', 'tempo']]



def plot_time_datasets(
    kmeans_df: pd.DataFrame,
    top_df: pd.DataFrame
) -> None:
    kmeans_df = create_df_time(kmeans_df, DATASETS)
    top_df = create_df_time(top_df, DATASETS)

    fig, ax = plt.subplots(figsize=(14, 10))  # Ajuste o tamanho da figura aqui
    ax.bar(0, 0, color='blue', label='topológico')
    ax.bar(0, 0, color='orange', label='clássico')

    for kmeans_linha, top_linha in zip(kmeans_df.values, top_df.values):
        ax.bar(top_linha[0], np.log(top_linha[1]), color='blue')
        ax.bar(kmeans_linha[0], np.log(kmeans_linha[1]), color='orange')

    titulo = kmeans_df.values[0][0]

    ax.set_title(f'Relação de tempo entre os algortimos')
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Tempo em escala logarítmica')
    plt.xticks(rotation=30)
    ax.legend()
    plt.savefig('tempos_plot.png', format='png')
    #plt.show()
            

#plot_metrics_overlap(kmeans_df, top_df)
plot_time_datasets(kmeans_df, top_df)
