from typing import Callable
import xarray as xr
import numpy as np
from active_inference.functions.message_passing import marginal_message, message_forward, message_upward_f, message_upward, message_backward
from active_inference.functions.operators import ln, softmax, sampler
from active_inference.functions.perception import gradient_descent, update_state, optogenetic_activation, gradient_descent_with_optgenetics
from active_inference.functions.planning import efe_of_next_action
from simulation.configs import GradientDescentParams, Optgenetics

# perceptual inference
#----------------------------------------
def perceptual_inference(ds: xr.Dataset, time_index: int) -> xr.DataArray:
    # Calculate marginal messages through message passing
    marginal_messages = marginal_message(ds, time_index, message_forward, message_upward_f)
    # Get current state belief
    old_message = ln(ds.S.isel(trial=time_index-1))
    # Update belief through gradient descent
    updated_message = gradient_descent(
        marginal_messages, 
        old_message, 
        GradientDescentParams.learning_rate
    )
    # Convert to state distribution
    new_state = update_state(updated_message)
    return new_state

# with optgenetics
def perceptual_inference_with_optgenetics(ds: xr.Dataset, time_index: int, optgenetics: Optgenetics = Optgenetics()) -> tuple[xr.DataArray, xr.DataArray]:
    # Calculate marginal messages through message passing
    marginal_messages: xr.DataArray = marginal_message(ds, time_index, message_forward, message_upward_f)
    #marginal_messages: xr.DataArray = marginal_message(ds, time_index, message_forward, message_upward, message_backward)
    #marginal_messages: xr.DataArray = marginal_message(ds, time_index, message_forward, message_upward)
    #marginal_messages: xr.DataArray = marginal_message(ds, time_index, message_forward, message_upward_f, message_backward)
    # Get current state belief
    old_message: xr.DataArray = ln(ds.S.isel(trial=time_index-1))
    # Update belief through gradient descent
    activation_value: float = optogenetic_activation(ds, time_index, optgenetics.start_time, optgenetics.end_time, optgenetics.method, optgenetics.value)

    updated_message, prediction_error = gradient_descent_with_optgenetics(
        marginal_messages, 
        old_message, 
        activation_value,
        GradientDescentParams.learning_rate
    )
    new_state: xr.DataArray = update_state(updated_message)
    return new_state, prediction_error

def perceptual_inference_with_updates(ds: xr.Dataset, time_index: int, optgenetics: Optgenetics = Optgenetics(), method: Callable = perceptual_inference) -> xr.Dataset:
    ds_updated = ds.copy()
    new_state, prediction_error = method(ds_updated, time_index, optgenetics)
    ds_updated.S.loc[dict(trial=time_index)] = new_state
    ds_updated.e.loc[dict(trial=time_index)] = prediction_error
    return ds_updated

# active inference
#----------------------------------------
def active_inference(ds: xr.Dataset, time_index: int) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    # planning
    efe: xr.DataArray = efe_of_next_action(ds, time_index, 'O')
    new_policy: xr.DataArray = softmax(efe)
    
    # with scaling
    #habituation: np.ndarray = np.array([[0.8, 0.4], [0.2, 0.6]])
    #new_policy: xr.DataArray = ds.U.isel(trial=time_index).copy()
    ##vector = habituation.T @ softmax(efe).values
    #vector = habituation @ softmax(efe).values
    #norm = np.sum(np.abs(vector))
    #normalized_values = vector / norm
    #new_policy.values = normalized_values
    
    # with habituation
    #habituation: np.ndarray = np.array([0.8, 0.2])
    #new_policy = softmax(efe + habituation)

    # action
    new_action: xr.DataArray = ds.u.isel(trial=time_index).copy()
    #new_action.values = sampler(new_policy) # probability sampling
    new_action.values = np.eye(2)[np.argmax(efe.values)] # deterministic sampling
    
    return new_policy, new_action, efe

def active_inference_with_updates(ds: xr.Dataset, time_index: int) -> xr.Dataset:
    ds_updated = ds.copy()
    new_policy, new_action, efe = active_inference(ds_updated, time_index)
    ds_updated.U.loc[dict(trial=time_index)] = new_policy
    ds_updated.u.loc[dict(trial=time_index)] = new_action
    ds_updated.G.loc[dict(trial=time_index)] = efe
    return ds_updated