import xarray as xr
import numpy as np
from active_inference.functions.operators import softmax, ln
from simulation.configs import Optgenetics

# State Prediction Error
#----------------------------------------
def optogenetic_activation(
    ds: xr.Dataset,
    time_index: int,
    start_time: int,
    end_time: int,
    optgenetics_method: str,
    optgenetics_value: float,
) -> float:
    
    start_idx = start_time
    end_idx = end_time

    if start_idx <= time_index <= end_idx:
        if optgenetics_method == 'activate':
            return optgenetics_value
        elif optgenetics_method == 'inhibit':
            return -optgenetics_value
    
    return 0.0

def optogenetic_activate(
    pe: xr.DataArray,
    start_time: int,
    end_time: int,
    time_index: int,
    optgenetics_method: str,
    optgenetics_value: float,
) -> xr.DataArray:
        
    start_idx = start_time
    end_idx = end_time

    activated_pe = pe.copy(deep=True)

    if start_idx <= time_index <= end_idx:
        if optgenetics_method == 'activate':
            activated_pe.values = pe.values + np.exp(pe.values * optgenetics_value)
        elif optgenetics_method == 'inhibit':
            activated_pe.values = pe.values - np.exp(pe.values) * optgenetics_value
        
    return activated_pe

# State Prediction Error
#----------------------------------------
def state_prediction_error(marginal_messages: xr.DataArray, old_message: xr.DataArray) -> xr.DataArray:
    updated_message = marginal_messages.copy()
    updated_message.values = marginal_messages.values - old_message.values
    return updated_message

# Pure Functions for State Updates
#----------------------------------------
def gradient_descent(marginal_messages: xr.DataArray, old_message: xr.DataArray, learning_rate: float, iteration: int = 1) -> xr.DataArray:
    updated_message: xr.DataArray = marginal_messages.copy()
    
    for _ in range(iteration):
        pe: xr.DataArray = state_prediction_error(updated_message, old_message)
        updated_message.values = old_message.values + learning_rate * pe.values
    return updated_message

def gradient_descent_with_optgenetics(
    marginal_messages: xr.DataArray,
    old_message: xr.DataArray,
    activation_value: float,
    learning_rate: float,
    iteration: int = 1
) -> tuple[xr.DataArray, xr.DataArray]:
    updated_message: xr.DataArray = marginal_messages.copy()
    
    for _ in range(iteration):
        pe: xr.DataArray = state_prediction_error(updated_message, old_message)
        #activated_pe: xr.DataArray = pe + pe * float(activation_value)
        #activated_pe: xr.DataArray = pe + pe * ln(xr.DataArray(activation_value))
        #activated_pe: xr.DataArray = pe +pe * activation_value
        #activated_pe: xr.DataArray = pe + ln(pe)
        #activated_pe: xr.DataArray = pe + float(activation_value)
        #activated_pe: xr.DataArray = pe + optogenetic_activate(pe, Optgenetics.start_time, Optgenetics.end_time, pe.trial, Optgenetics.method, Optgenetics.value)
        activated_pe: xr.DataArray = optogenetic_activate(pe, Optgenetics.start_time, Optgenetics.end_time, pe.trial, Optgenetics.method, Optgenetics.value)
        print('activaton = ', optogenetic_activate(pe, Optgenetics.start_time, Optgenetics.end_time, pe.trial, Optgenetics.method, Optgenetics.value))
        updated_message.values = old_message.values + learning_rate * activated_pe
    
    return updated_message, activated_pe

def update_state(updated_message: xr.DataArray) -> xr.DataArray:
    return softmax(updated_message)