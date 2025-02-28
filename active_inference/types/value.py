from typing import TypeAlias

# Value Objects (単純な値オブジェクト)
#----------------------------------------
Binary: TypeAlias = int  # 0か1の値を表す値オブジェクト 
Natural: TypeAlias = int  # 自然数を表す値オブジェクト
Positive: TypeAlias = float  # 正の値を表す値オブジェクト
Probability: TypeAlias = float  # 0.0から1.0の間の値を表す値オブジェクト
