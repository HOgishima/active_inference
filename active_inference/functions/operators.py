import numpy as np
import xarray as xr
from numpy.typing import NDArray

# array operation
#----------------------------------------
def softmax(x: xr.DataArray) -> xr.DataArray:
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
    
def ln(x: xr.DataArray) -> xr.DataArray:
    new_x: xr.DataArray = x.copy()
    new_x = np.log(x + np.exp(-16))
    return new_x

def entropy(x: xr.DataArray) -> xr.DataArray:
    log_x = ln(x)
    product = dot(log_x, x)
    entropy = -product.diagonal()
    
    # 元の座標系を保持したDataArrayとして返す
    return entropy.copy()

# TODO: need to fix. It's only working for 2D array.
def dot(x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
    new_y: xr.DataArray = y.copy()
    new_y = np.dot(x.T, y)
    return new_y

#----------------------------------------
def sampler(x: xr.DataArray) -> xr.DataArray:
    '''Sample from a categorical distribution and return a one-hot vector.'''
    new_x: xr.DataArray = x.copy()
    index: int = np.random.choice(range(len(x)), p=x)
    new_x.values = np.eye(x.size)[index]
    return new_x

def to_one_hot_vector(x: xr.DataArray) -> xr.DataArray:
    '''Convert a probability vector to a one-hot vector.'''
    new_x: xr.DataArray = x.copy()
    new_x.values = np.zeros_like(x.values)
    index: int = np.argmax(x.values)
    new_x.values[index] = 1
    return new_x