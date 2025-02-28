from typing import Callable
import numpy as np
import xarray as xr
from scipy.stats import dirichlet

# Pure Functions for Parameter Updates
#----------------------------------------
def update_a(block_dataset: xr.Dataset, time: int, batch_size: int = 1, learning_rate: float = 1.0) -> xr.DataArray:
    end_idx: int = time + 1
    start_idx: int = max(0, end_idx - batch_size)

    # filter
    a: xr.DataArray = block_dataset.A.isel(trial = time)
    Ss: xr.DataArray = block_dataset.S.isel(trial = slice(start_idx, end_idx))
    Os: xr.DataArray = block_dataset.O.isel(trial = slice(start_idx, end_idx))

    # compute
    new_a: xr.DataArray = a.copy()
    gradient: np.ndarray = np.zeros_like(new_a.values)
    for i in range(Ss.shape[1] - 1):
        gradient += np.kron(Os.values[i], Ss.values[i]).reshape(a.shape[0], a.shape[1])
    new_a.values = a.values + learning_rate * gradient
    
    return new_a

def update_a_rev(block_dataset: xr.Dataset, time: int, batch_size: int = 1, learning_rate: float = 1.0) -> xr.DataArray:
    start_idx: int = max(0, end_idx - batch_size)
    end_idx: int = time + 1
    slice_start: int = max(0, end_idx + slice_start)

    # filter
    a: xr.DataArray = block_dataset.A.isel(trial=time)
    
    # スライスの範囲を指定して Ss と Os を取得
    Ss: xr.DataArray = block_dataset.S.isel(trial=slice(max(0, end_idx + slice_start), end_idx))
    Os: xr.DataArray = block_dataset.O.isel(trial=slice(max(0, end_idx + slice_start), end_idx))

    # compute
    new_a: xr.DataArray = a.copy()
    gradient: np.ndarray = np.zeros_like(new_a.values)
    for i in range(Ss.shape[1] - 1):
        gradient += np.kron(Os.values[i], Ss.values[i]).reshape(a.shape[0], a.shape[1])
    new_a.values = a.values + learning_rate * gradient
    
    return new_a

def update_c(block_dataset: xr.Dataset, n: int, memory: int) -> xr.DataArray:
    start_idx: int = max(0, n - memory)
    current_C: xr.DataArray = block_dataset.C.isel(trial = n)
    Os: xr.DataArray = block_dataset.O.isel(trial = slice(start_idx, n+1))
    
    sum_Os = Os.sum(dim = 'trial')
    param = current_C + sum_Os + np.exp(-16)
    new_C = dirichlet.rvs(param, size=1)[0]  # 1次元配列にアクセス

    return xr.DataArray(new_C, coords={'observation': ['shock', 'no_shock']})

# Learning Function
#----------------------------------------
def learning(ds: xr.Dataset, time_index: int, params: list[str], memory: int = 1, learning_rate: float = 1.0 # TODO: to list
    ) -> xr.Dataset:

    updated_params_dict = {}
    
    # 各パラメータの更新
    if "A" in params:
        updated_params_dict["A"] = update_a(ds, time_index, memory)
    
    if "C" in params:
        updated_params_dict["C"] = update_c(ds, time_index, memory)
    
    updated_params: xr.Dataset = xr.Dataset(data_vars=updated_params_dict)

    return updated_params

# todo: fix
def learning_with_updates(ds: xr.Dataset, time_index: int, params: list[str], memory: int = 1, learning_rate: float = 1.0) -> xr.Dataset:
    ds_updated: xr.Dataset = ds.copy()
    updated_params: xr.Dataset = learning(ds_updated, time_index, params, memory, learning_rate)
    
    for key, value in updated_params.items():
        ds_updated[key].loc[dict(trial=time_index)] = value
    return ds_updated