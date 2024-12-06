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
    for version in range(1, 6):  # Tenta até 5 versões
        try:
            dataset = fetch_openml(name=name, version=version, parser='auto')
            break
        except OpenMLError:
            continue
    else:
        print(f"[WARN] Dataset '{name}' não encontrado em até 5 versões.")
        return None

    # Obtém informações do dataset
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
#print(len(OPENML_DATASETS))
main()
