import numpy as np
import xarray as xr

def block_mean_and_stderr(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    return ds.mean(dim='block'), ds.std(dim='block') / np.sqrt(ds.sizes['block'])

def block_mean(ds: xr.Dataset) -> xr.Dataset:
    return ds.mean(dim='block')

def block_stderr(ds: xr.Dataset) -> xr.Dataset:
    return ds.std(dim='block') / np.sqrt(ds.sizes['block'])

def trial_mean(ds: xr.Dataset) -> xr.Dataset:
    return ds.mean(dim='trial')

def trial_stderr(ds: xr.Dataset) -> xr.Dataset:
    return ds.std(dim='trial') / np.sqrt(ds.sizes['trial'])