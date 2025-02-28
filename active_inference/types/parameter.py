from typing import TypeAlias, Dict, Optional, List
from dataclasses import dataclass
import xarray as xr
import numpy as np
from active_inference.types.matrix import Matrix

# Parameters Types (値オブジェクトを集めたもの)
#----------------------------------------
LikelihoodArray: TypeAlias = xr.DataArray
TransitionArray: TypeAlias = xr.DataArray


#----------------------------------------
# Likelihood
#----------------------------------------
@dataclass(frozen=True)
class Likelihood():
    matrix: Matrix

    def __post_init__(self):
        #TODO: fix it if it's not necessary
        if self.matrix.dims != ["observation", "state"]:
            raise ValueError("Likelihood matrix must have dims=['observation', 'state']")

        #TODO: fix it if it's not necessary
        if set(self.matrix.coords.keys()) != {"observation", "state"}:
            raise ValueError("Likelihood matrix must have coords={'observation': list[str], 'state': list[str]}")

    @staticmethod
    def create(data: np.ndarray, coords: Dict[str, list[str]], dims: Optional[List[str]] = None) -> "Likelihood":
        """Smart constructor: Create an instance while validating data."""
        if dims is None:
            dims = list(coords.keys())
        return Likelihood(Matrix.create(data, dims, coords))

    def to_xarray(self) -> xr.DataArray:
        """Convert to xarray.DataArray."""
        return self.matrix.to_xarray()

@dataclass(frozen=True)
class Transition:
    matrix: Matrix

    def __post_init__(self):
        #TODO: fix it if it's not necessary
        if self.matrix.dims != ["next_state", "state"]:
            raise ValueError("Transition matrix must have dims=['next_state', 'state']")
        
        if set(self.matrix.coords.keys()) != {"next_state", "state"}:
            raise ValueError("Transition matrix must have coords={'next_state': list[str], 'state': list[str]}")

    @staticmethod
    def create(data: np.ndarray, coords: Dict[str, list[str]], dims: Optional[List[str]] = None) -> "Transition":
        """Smart constructor: Create an instance while validating data."""
        if dims is None:
            dims = list(coords.keys())
        return Transition(Matrix.create(data, dims, coords))

    def to_xarray(self) -> xr.DataArray:
        """Convert to xarray.DataArray."""
        return self.matrix.to_xarray()
