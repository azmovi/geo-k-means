import numpy as np
import pandas as pd
from sklearn.utils._bunch import Bunch


def _preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    faz o preprocessamento do conjunto de dados colocando valores para float e
    caso tenha strings presentes no conjunto de dados converte para valores
    numéricos

    Parameters:
        dataframe: O conjunto de dados para ser pre processado

    Returns:
        O conjunto de dados com os valores possíveis de ser trabalhados

    Example:
        >>> from sklearn.datasets import fetch_openml
        >>> df = fetch_openml(name='servo', version=1, parser='auto')
        >>> _preprocess_data(df['data'])
        array([...], dtype=object)

    """
    dataframe = dataframe.bfill()
    data = dataframe.values
    if data.dtype != np.float64:
        conversor = {}
        valor = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                try:
                    data[i][j] = float(data[i][j])
                except ValueError:
                    if data[i][j] not in conversor:
                        conversor[data[i][j]] = valor
                        valor += 1
                    data[i][j] = conversor[data[i][j]]
    return data


def _preprocess_target(target: pd.Series) -> np.ndarray:
    """
    Converte uma série de rótulos em uma lista de inteiros, onde cada
    rótulo possui um número correspondente de acordo com a quantidade de
    classes.

    Parameters:
        target: Uma serie do pandas que retrata as classificações reais
        presentes na base de dados.

    Returns:
        Uma lista de inteiros com o tamanho do target que contem números
        entre zero a número classificações - 1

    Examples:

        >>> from sklearn.datasets import fetch_openml
        >>> df = fetch_openml(name='iris', version=1, parser='auto')
        >>> _preprocess_target(df['target'])
        array([...])

    """
    categorias = target.unique()
    rotulo = {rotulo: index for index, rotulo in enumerate(categorias)}

    return np.array([rotulo[tipo] for tipo in target])


def preprocess(dataframe: Bunch) -> tuple[np.ndarray]:
    """
    Faz o pre processamento dos dados e da sua classificação de um conjunto de
    dados

    Paramenters:
        dataframe: o cojunto de dados presente na biblioteca sklearn

    Returns:
        Uma tupla com os valores dos dados e da classificação processados da
        forma correta

    Examples:
        >>> from sklearn.datasets import fetch_openml
        >>> df = fetch_openml(name='iris', version=1, parser='auto')
        >>> preprocess(df)
        (array([...]))
    """
    data = _preprocess_data(dataframe['data'])
    target = _preprocess_target(dataframe['target'])
    return data, target
