from core.type.matrix import Vector
from typing import Final
import pandas as pd

class Message(Vector):
    """メッセージを表す値オブジェクト"""
    index_names: Final[list[str]] = ["fear", "extinction"]
    
    def to_pandas(self) -> pd.Series:
        return pd.Series(self, index = self.index_names)