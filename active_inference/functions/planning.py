import numpy as np
import xarray as xr
from active_inference.functions.operators import entropy, dot, ln, softmax
from active_inference.functions.learning import learning

def efe_of_next_action(block_dataset: xr.Dataset, time: int, preference: str) -> xr.DataArray:
    G: xr.DataArray = block_dataset.G.isel(trial = time).copy(deep=True)

    #A: xr.DataArray = block_dataset.A.isel(trial = time+1)
    B: xr.DataArray = block_dataset.B.isel(trial = time)
    #C: xr.DataArray = block_dataset.C.isel(trial = time+1)
    S: xr.DataArray = block_dataset.S.isel(trial = time)

    if time == block_dataset.trial.values[-1]:
        #A, C = learning(block_dataset, time, params=["A", "C"]) # TODO: it is not working
        updated_params = learning(block_dataset, time, params=["A", "C"])
        A: xr.DataArray = block_dataset.A.isel(trial = time)
        C: xr.DataArray = updated_params.C
    else:
        updated_params = learning(block_dataset, time, params=["A", "C"], memory=10)
        A: xr.DataArray = block_dataset.A.isel(trial = time+1)
        #C: xr.DataArray = block_dataset.C.isel(trial = time+1)
        C: xr.DataArray = updated_params.C

    for action in B.coords['action']:
        BS: np.ndarray = np.dot(B.sel(action = action).values, S.values) # next state: S_u(n+1)
        AS: np.ndarray = np.dot(A.values, BS) # next observation: O_u(n+1)
        H: np.ndarray = entropy(A.values) # entoropy

        if preference == 'S' or preference == 'AS':
            G.loc[dict(action = action)] = dot(BS, ln(BS)+C+H)
        if preference == 'O':
            G.loc[dict(action = action)] = dot(AS, ln(AS)+C.values) + dot(BS,H)
            #G.loc[dict(action = action)] = np.dot(AS.T, ln(AS)+C.values) + np.dot(BS.T,H)

    return G