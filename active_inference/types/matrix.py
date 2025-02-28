from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import xarray as xr

# Matrix Types (複数の値オブジェクトを集めたもの)
#----------------------------------------
@dataclass(frozen=True)
class Vector:
    data: np.ndarray
    dims: List[str]
    coords: Dict[str, list[str]]
    
    def __post_init__(self):
        # データの妥当性チェック
        # - データが1次元のnumpy配列でない場合はエラーを返す
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 1:
            raise ValueError("data must be a one-dimensional numpy array")

        # dimsの妥当性チェック
        # - dimsがリストでないか、要素数が1でない場合はエラーを返す
        if not isinstance(self.dims, list) or len(self.dims) != 1:
            raise ValueError("dims must be a list with exactly one element")

        # coordsの妥当性チェック
        # - coordsが辞書でないか、keysがdimsと一致しない場合はエラーを返す
        if not isinstance(self.coords, dict) or set(self.coords.keys()) != set(self.dims):
            raise ValueError("coords keys must match dims")

        # coordsの要素の妥当性チェック
        # - coordsの要素がリストでないか、要素の長さがdataと一致しない場合はエラーを返す
        for key in self.dims:
            if not isinstance(self.coords[key], list) or len(self.coords[key]) != len(self.data):
                raise ValueError(f"coords[{key}] must have the same length as data")

    @staticmethod
    def create(data: np.ndarray, dims: List[str], coords: Dict[str, list[str]]) -> "Vector":
        """ スマートコンストラクタ: データの妥当性を検証しながらインスタンスを作成 """
        return Vector(data=data, dims=dims, coords=coords)

    def to_xarray(self) -> xr.DataArray:
        """ xarray.DataArray に変換 """
        return xr.DataArray(self.data, dims=self.dims, coords=self.coords)


@dataclass(frozen=True)
class Matrix:
    data: np.ndarray
    dims: List[str]
    coords: Dict[str, list[str]]

    def __post_init__(self):
        # データの妥当性チェック
        # - データが2次元のnumpy配列でない場合はエラーを返す
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 2:
            raise ValueError("data must be a two-dimensional numpy array")

        # dimsの妥当性チェック
        # - dimsがリストでないか、要素数が2でない場合はエラーを返す
        if not isinstance(self.dims, list) or len(self.dims) != 2:
            raise ValueError("dims must be a list with exactly two elements")

        # coordsの妥当性チェック
        # - coordsが辞書でないか、keysがdimsと一致しない場合はエラーを返す
        if not isinstance(self.coords, dict) or set(self.coords.keys()) != set(self.dims):
            raise ValueError("coords keys must match dims")

        # coordsの要素の妥当性チェック
        # - coordsの要素がリストでないか、要素の長さがdataの各次元と一致しない場合はエラーを返す
        for key in self.dims:
            if not isinstance(self.coords[key], list) or len(self.coords[key]) != self.data.shape[self.dims.index(key)]:
                raise ValueError(f"coords[{key}] must have the same length as data in dimension {key}")

    @staticmethod
    def create(data: np.ndarray, dims: List[str], coords: Dict[str, list[str]]) -> "Matrix":
        """ スマートコンストラクタ: データの妥当性を検証しながらインスタンスを作成 """
        return Matrix(data=data, dims=dims, coords=coords)

    def to_xarray(self) -> xr.DataArray:
        """ xarray.DataArray に変換 """
        return xr.DataArray(self.data, dims=self.dims, coords=self.coords)