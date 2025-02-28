import numpy as np
import xarray as xr
from simulation.configs import TimeSteps

# Initialize observations
def create_observations(
    LSS: float = 0.9,
    baseline_steps: int = TimeSteps.baseline,
    cs_steps: int = TimeSteps.cs,
    post_cs_steps: int = TimeSteps.post_cs
) -> np.ndarray:
    observations_b = np.tile(np.array([0, 1]), (baseline_steps, 1))
    observations_s = np.tile(np.array([LSS, 1-LSS]), (cs_steps, 1))
    observations_n = np.tile(np.array([0, 1]), (post_cs_steps, 1))
    return np.concatenate((observations_b, observations_s, observations_n), axis=0)

def generative_process(
    block_dataset: xr.Dataset,
    time_index: int
) -> xr.DataArray:
    return block_dataset.O.isel(trial=time_index)