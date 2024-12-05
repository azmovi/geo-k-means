from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError

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
from openml import fetch_openml
from openml.exceptions import OpenMLError

def caracteristicas(nome: str):
    version = 1
    dataset = None
    while version <= 5:
        try:
            # Obtém o conjunto de dados a partir do OpenML
            dataset = fetch_openml(name=nome, version=version, parser='auto')
            break
        except OpenMLError:
            version += 1
    
    if not dataset:
        return

    nome_dataset = dataset.name
    num_amostras = dataset.shape[0]
    num_features = dataset.shape[1]
    
    if 'class' in dataset.features:
        num_classes = len(dataset.features['class'].values)
    else:
        num_classes = 'Não disponível'
    
    return {
        'nome': nome_dataset,
        'amostras': num_amostras,
        'features': num_features,
        'num_classes': num_classes
    }

for dataset in OPENML_DATASETS:
    print(caracteristicas(dataset))

