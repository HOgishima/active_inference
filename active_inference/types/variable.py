from typing import Dict, TypeAlias, Optional, List
from dataclasses import dataclass
import xarray as xr
import numpy as np
from active_inference.types.matrix import Vector


# Variables Types
#----------------------------------------
StateArray: TypeAlias = xr.DataArray
ActionArray: TypeAlias = xr.DataArray
ObservationArray: TypeAlias = xr.DataArray

#----------------------------------------
@dataclass(frozen=True)
class State:
    vector: Vector

    def __post_init__(self):
        # `State` に特化した制約
        if self.vector.dims != ["state"]:
            raise ValueError("State vector must have dims=['state']")
        
        if set(self.vector.coords.keys()) != {"state"}:
            raise ValueError("State vector must have coords={'state': list[str]}")

    @staticmethod
    def create(data: np.ndarray, coords: Dict[str, list[str]], dims: Optional[List[str]] = None) -> "State":
        """ スマートコンストラクタ: Vector を事前に作らなくても State を作成 """
        if dims is None:
            dims = list(coords.keys())
        return State(Vector.create(data, dims, coords))

    def to_xarray(self) -> StateArray:
        """ xarray.DataArray に変換 """
        return self.vector.to_xarray()

@dataclass(frozen=True)
class Action:
    vector: Vector

    def __post_init__(self):
        if self.vector.dims != ["action"]:
            raise ValueError("Action vector must have dims=['action']")
        
        if set(self.vector.coords.keys()) != {"action"}:
            raise ValueError("Action vector must have coords={'action': list[str]}")
        
    @staticmethod
    def create(data: np.ndarray, coords: Dict[str, list[str]], dims: Optional[List[str]] = None) -> "Action":
        if dims is None:
            dims = list(coords.keys())
        return Action(Vector.create(data, dims, coords))

    def to_xarray(self) -> ActionArray:
        return self.vector.to_xarray()
    
@dataclass(frozen=True)
class Observation:
    vector: Vector

    def __post_init__(self):
        if self.vector.dims != ["observation"]:
            raise ValueError("Observation vector must have dims=['observation']")
        
        if set(self.vector.coords.keys()) != {"observation"}:
            raise ValueError("Observation vector must have coords={'observation': list[str]}")

    @staticmethod
    def create(data: np.ndarray, coords: Dict[str, list[str]], dims: Optional[List[str]] = None) -> "Observation":
        if dims is None:
            dims = list(coords.keys())
        return Observation(Vector.create(data, dims, coords))

    def to_xarray(self) -> ObservationArray:
        return self.vector.to_xarray()