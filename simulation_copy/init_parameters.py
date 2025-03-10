from typing import List, Dict
from dataclasses import dataclass, field
import numpy as np


# Names of variables
#================================================================
state_coord: dict[str, list[str]] = {'state': ['positive', 'neutral', 'negative']}
next_state_coord: dict[str, list[str]] = {'next_state': ['positive', 'neutral', 'negative']}
observation_coord: dict[str, list[str]] = {'observation': ['shock', 'no_shock']}
action_coord: dict[str, list[str]] = {'action': ['freezing', 'non_freezing']}

# Parameters
#================================================================
# Likelihood (A)
#-------------------------------------------------
@dataclass(frozen=True)
class InitA:
    data: np.ndarray = field(default_factory=lambda: np.array([
        # fear, extinction (S_t)
        [0.9, 0.1],  # shock    (O_t)
        [0.1, 0.9]   # no shock (O_t)
    ]))
    dims: List[str] = field(default_factory=lambda: ['observation', 'state']) # it's not necessary, but it's good to have
    coords: Dict[str, list[str]] = field(default_factory=lambda: observation_coord | state_coord)
    learning: bool = False
    learning_rate: float = 1.0
    learning_batch: int = 1

# Transition (B)
#-------------------------------------------------
# - B_freezing
@dataclass(frozen=True)
class InitB_f:
    data: np.ndarray = field(default_factory=lambda: np.array([
        # fear, extinction (S_t)
        [0.9, 0.9],             # fear       (S_t+1)
        [0.1, 0.1]              # extinction (S_t+1)
    ]))
    dims: List[str] = field(default_factory = lambda: ['next_state', 'state'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: next_state_coord | state_coord)
    learning: bool = False
    learning_rate: float = 1.0
    learning_batch: int = 1


# - B_non_freezing 
@dataclass(frozen=True)
class InitB_nf:
    data: np.ndarray = field(default_factory = lambda: np.array([
        # fear, extinction (S_t)
        [0.1, 0.1],             # fear       (S_t+1)
        [0.9, 0.9]              # extinction (S_t+1)
    ]))
    dims: List[str] = field(default_factory = lambda: ['next_state', 'state'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: next_state_coord | state_coord)
    learning: bool = False
    learning_rate: float = 1.0
    learning_batch: int = 1

# Prior for Observations (C)
#-------------------------------------------------
@dataclass(frozen=True)
class InitC:
    data: np.ndarray = field(default_factory = lambda: np.array([0.5, 0.5]))  # Sensitivity for outcome [shock, no shock]
    dims: List[str] = field(default_factory = lambda: ['observation'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: observation_coord)
    learning: bool = True
    learning_rate: float = 1.0
    learning_batch: int = 5
    

# Initial state probability (D)
#-------------------------------------------------
@dataclass(frozen=True)
class InitD:
    data: np.ndarray = field(default_factory = lambda: np.array([0,1])) 
    dims: List[str] = field(default_factory = lambda: ['state'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: state_coord)
    learning: bool = False
    learning_rate: float = 1.0
    learning_batch: int = 1


# Energy
#================================================================
# Variational Free Energy (F)
#-------------------------------------------------

# Expected Free Energy (G)
#-------------------------------------------------
@dataclass(frozen=True)
class InitG:
    data: np.ndarray = field(default_factory = lambda: np.array([0.5, 0.5]))
    dims: List[str] = field(default_factory = lambda: ['action'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: action_coord)


# Variables
#================================================================
# State (S)
#-------------------------------------------------
@dataclass(frozen=True)
class InitS:
    data: np.ndarray = field(default_factory = lambda: np.array([0.5, 0.5]))
    dims: List[str] = field(default_factory = lambda: ['state'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: state_coord)


# Observation (O)
#-------------------------------------------------
@dataclass(frozen=True)
class InitO:
    data: np.ndarray = field(default_factory = lambda: np.array([0.5, 0.5]))
    dims: List[str] = field(default_factory = lambda: ['observation'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: observation_coord)

# Action (U)
#-------------------------------------------------
@dataclass(frozen=True)
class InitU:
    data: np.ndarray = field(default_factory = lambda: np.array([0.5, 0.5]))
    dims: List[str] = field(default_factory = lambda: ['action'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: action_coord)

@dataclass(frozen=True)
class Initu:
    data: np.ndarray = field(default_factory = lambda: np.array([1, 0]))
    dims: List[str] = field(default_factory = lambda: ['action'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: action_coord)

# state_prediction error
#-------------------------------------------------
@dataclass(frozen=True)
class Inite:
    data: np.ndarray = field(default_factory = lambda: np.array([0.5, 0.5]))
    dims: List[str] = field(default_factory = lambda: ['state'])
    coords: Dict[str, list[str]] = field(default_factory = lambda: state_coord)
