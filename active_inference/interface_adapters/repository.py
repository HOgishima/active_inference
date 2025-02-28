import xarray as xr
import numpy as np
from typing import Dict

def create_trial_dataset(data_vars: Dict[str, xr.DataArray]) -> xr.Dataset:
    # TODO: もし変数間でcoordsが一致しない場合はエラーを吐き出すようにする
    return xr.Dataset(data_vars)

def create_block_dataset(
    trial_dataset: xr.Dataset,
    trial_number: int
) -> xr.Dataset:
    return trial_dataset.expand_dims(trial=trial_number).assign_coords(trial=np.arange(trial_number))

def create_dataset(
    data_vars: Dict[str, xr.DataArray],
    trial_number: int,
    block_number: int
) -> xr.Dataset:

    # Create time index array
    time_id = np.arange(trial_number * block_number).reshape(
        block_number,
        trial_number
    )
    
    # Create trial dataset
    trial_dataset = create_trial_dataset(data_vars)

    # Create block dataset
    block_dataset = create_block_dataset(trial_dataset, trial_number)

    # Add time dimensions and coordinates
    dataset = (block_dataset
        .expand_dims(block=block_number)      # add block dimension
        .assign_coords(block=np.arange(block_number))  # set block index
        .assign_coords(time=(['block', 'trial'], time_id))  # set time index
    )
    
    return dataset