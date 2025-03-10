from dataclasses import dataclass
import numpy as np

# Simple Type
#----------------------------------------
sec = float


# Data Class
#----------------------------------------
@dataclass(frozen=True)
class TimeLength:
    baseline: sec = 40
    cs: sec = 20
    post_cs: sec = 20
    trial: sec = baseline + cs + post_cs
    window: sec = 0.5

@dataclass(frozen=True)
class TimeSteps:
    baseline: int = int(TimeLength.baseline / TimeLength.window)
    cs: int = int(TimeLength.cs / TimeLength.window)
    post_cs: int = int(TimeLength.post_cs / TimeLength.window)
    trials: int = int(TimeLength.trial / TimeLength.window)
    blocks: int = 20
    total: int = blocks * trials

@dataclass(frozen=True)
class GradientDescentParams:
    iterations: int = 1
    learning_rate: float = 0.05

@dataclass(frozen=True)
class Optgenetics:
    method: str = 'none' # 'none', 'activate', 'inhibit'
    value: float = 0.35
    start_time: int = TimeSteps.baseline + TimeSteps.cs
    end_time: int = TimeSteps.trials

@dataclass(frozen=True)
class LearningParams:
    memory: int = 10

@dataclass(frozen=True)
class ExperimentDesign:
    block_continuity: bool = True
