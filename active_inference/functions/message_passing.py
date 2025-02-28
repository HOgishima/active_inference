import numpy as np
import xarray as xr
from typing import Callable
from active_inference.functions.operators import ln, dot

# Pure Functions for Message Passing
#----------------------------------------
def message_forward(block_dataset: xr.Dataset, time: int) -> xr.DataArray:
    if time == 0:
        return block_dataset.D.isel(trial = time)
    else:
        # filter
        prev_S: xr.DataArray = block_dataset.S.isel(trial = time - 1)
        prev_u: xr.DataArray = block_dataset.u.isel(trial = time - 1)
        index_u: int = prev_u.argmax().item()
        prev_B: xr.DataArray = block_dataset.B.isel(trial = time - 1).isel(action = index_u)
        # compute
        new_message: xr.DataArray = block_dataset.S.isel(trial = time).copy()
        new_message.values = np.dot(ln(prev_B), prev_S)
        return new_message

def message_upward(block_dataset: xr.Dataset, time: int) -> xr.DataArray:
    # filter
    O: xr.DataArray = block_dataset.O.isel(trial = time)
    A: xr.DataArray = block_dataset.A.isel(trial = time)
    # compute
    new_message: xr.DataArray = block_dataset.S.isel(trial = time).copy()
    new_message.values = dot(ln(A), O)
    return new_message

def message_upward_f(block_dataset: xr.Dataset, time: int) -> xr.DataArray:
    # filter
    A: xr.DataArray = block_dataset.A.isel(trial = time)
    S: xr.DataArray = block_dataset.S.isel(trial = time)
    C: xr.DataArray = block_dataset.C.isel(trial = time)
    # compute
    AS: xr.DataArray = np.dot(A, S)
    H: xr.DataArray = -np.diag(np.dot(ln(A), A))
    new_message: xr.DataArray = S.copy()
    #new_message.values = H + dot(A, ln(AS)) + ln(C)
    new_message.values = H - dot(A, (ln(AS) - ln(C)))
    return new_message

def message_backward(blanket: xr.Dataset, time: int) -> xr.DataArray:
    try:
        # filter
        next_S: xr.DataArray = blanket.S.isel(trial = time + 1)
    except:
        next_S: xr.DataArray = blanket.S.isel(trial = time).copy()
    B: xr.DataArray = blanket.B.isel(trial = time)
    # compute
    new_message: xr.DataArray = blanket.S.isel(trial = time).copy()
    #new_message.values = dot(B, next_S)
    prev_u: xr.DataArray = blanket.u.isel(trial = time - 1)
    index_u: int = prev_u.argmax().item()
    new_message.values = dot(B.sel(action = B.coords['action'][index_u]).values, next_S.values)  #TODO: need fix
    return new_message

# Pure Functions for sum product algorithm
#----------------------------------------
def marginal_message(block_dataset: xr.Dataset, time: int, *args: Callable) -> xr.DataArray:
    # compute each message
    messages: list[xr.DataArray] = [func(block_dataset, time) for func in args]
    # compute marginal message
    marginal_message: xr.DataArray = block_dataset.S.isel(trial = time).copy()
    marginal_message.values = np.sum([m.values for m in messages], axis=0)
    return marginal_message