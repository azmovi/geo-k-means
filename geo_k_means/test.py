from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError
import numpy as np

OPENML_DATASETS = [
    'iris',
    'micro-mass',
    'monks-problems-1',
    'breast-tissue',
    'GCM',
    'balance-scale',
    'servo',
    'AP_Breast_Lung',
    'AP_Colon_Kidney',
    'AP_Breast_Kidney',
    'AP_Breast_Ovary',
    'AP_Breast_Colon',
    'AP_Colon_Prostate',
    'AP_Endometrium_Breast',
    'AP_Colon_Omentum',
    'AP_Breast_Omentum',
    'AP_Prostate_Kidney',
    'AP_Omentum_Kidney',
    'AP_Breast_Prostate',
    'AP_Uterus_Kidney',
    'AP_Prostate_Uterus',
    'AP_Omentum_Lung',
    'AP_Lung_Uterus',
    'AP_Ovary_Kidney',
    'AP_Lung_Kidney',
    'AP_Colon_Lung',
    'AP_Endometrium_Colon',
    'AP_Ovary_Lung',
    'AP_Endometrium_Kidney',
    'AP_Colon_Uterus',
    'leukemia',
    'hepatitisC',
    'Ovarian',
    'SRBCT',
    'Colon',
    'climate-model-simulation-crashes',
    'pasture',
    'glass',
    'kc1-binary',
    'dermatology',
    'backache',
    'lsvt',
    'thoracic-surgery',
    'planning-relax',
    'tr31.wc',
    'tr45.wc',
]

def fetch_dataset_features(name: str, numero: int):
    for version in range(1, 6):
        try:
            dataset = fetch_openml(name=name, version=version, parser='auto')
            break
        except OpenMLError:
            continue
    else:
        print(f"[WARN] Dataset '{name}' não encontrado em até 5 versões.")
        return None

    if not dataset:
        return

    if hasattr(dataset, 'data'):
        num_amostras = dataset.data.shape[0]
        num_features = dataset.data.shape[1]
        num_classes = len(np.unique(dataset['target']))


        return {
            'Numero': numero,
            'Conjunto de Dados': name, 
            'amostras': num_amostras,
            'Atributos': num_features,
            'Classes': num_classes
        }

def main():
    for idx, dataset in enumerate(OPENML_DATASETS, start=1):
        info = fetch_dataset_features(dataset, idx)
        if info:
            print(info)
        else:
            print(f"[INFO] Nenhuma característica encontrada para '{dataset}'.")


import pandas as pd

def only_datasets():
    dataset_names = [
        'micro-mass',
        'monks-problems-1',
        'breast-tissue',
        'GCM',
        'balance-scale',
        'servo',
        'AP_Prostate_Uterus',
        'AP_Colon_Kidney',
        'AP_Breast_Kidney',
        'AP_Breast_Ovary',
        'AP_Breast_Colon',
        'leukemia',
        'hepatitisC',
        'Ovarian',
        'SRBCT',
        'Colon',
        'climate-model-simulation-crashes',
        'pasture',
        'glass',
        'kc1-binary',
        'dermatology',
        'backache',
        'lsvt',
        'thoracic-surgery',
        'planning-relax',
        'tr31.wc',
        'tr45.wc'
    ]
    filepath = 'docs/desempenhos/sklearn_desempenho.csv'
    #filepath = 'docs/desempenhos/topkmeans_desempenho.csv'
    colunas = ['nome', 'completeness', 'fowlkes mallows', 'homogeneity', 'v measure',                
       'adjusted rand', 'normalized mutual info', 'silhouette', 'calinski harabasz',                          
       'davies bouldin', 'tempo']
    df = pd.read_csv(filepath, usecols=colunas)
    new_csv = df[df['nome'].isin(dataset_names)]
    new_csv = new_csv.reset_index(drop=True)
    new_csv.index = new_csv.index + 1
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', '{:,.3f}'.format)
    pd.set_option('display.width', None)
    print(new_csv.describe())
    #new_csv.to_csv('test.csv')


only_datasets()
