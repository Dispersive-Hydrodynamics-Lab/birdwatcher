# stdlib

# third party
import numpy as np
import scipy.misc as sc_ms

from typing import Union, Any, List, Dict


def get_frequencies(x: List[Any], normed: bool=True) -> Dict[Any, Union[float, int]]:
    counts = {}
    divisor = len(x) if normed else 1
    for item in x:
        try:
            counts[item] += 1 / divisor
        except KeyError:
            counts[item] = 1 / divisor
    return counts


def safelog(x: Union[np.ndarray, Union[int, float]], zeros: bool=False) -> Union[np.ndarray, Union[int, float]]:
    if isinstance(x, int) or isinstance(x, float):
        if x < 0:
            return None
        else:
            return np.log(x)
    else:
        if zeros:
            return np.nan_to_num(np.log(x))
        else:
            return np.where(np.logical_not(np.isnan(np.log(x))))


def hoaglin_tukey(f_k, k, F):
    """
    \phi(f_k) = log(k! f_k / F)

    f_k - observed frequency for category k
    F - total number of observations
    """
    return safelog(sc_ms.factorial(k) * f_k / F)


def mass_hoaglin_tukey(freqs, data):
    F = len(data)
    result = [hoaglin_tukey(freqs[x], x, F) for x in data]
    return np.array(result)
