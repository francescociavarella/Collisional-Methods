import numpy as np

from numba import jit, njit, prange
from numba import complex128 as ncomplex

##############################################################################
##############################################################################
# FIDELITY

@njit
def Fidelity_WF(wf1,wf2):
    """
    Fidelity between two wavefunctions
    """
    return np.abs(np.vdot(wf1, wf2))**2

@njit
def Fidelity_DM(rho1, rho2):
    """
    Fidelity between two density matrices
    """
    return np.real(np.trace(np.sqrt(np.sqrt(rho1) @ rho2 @ np.sqrt(rho1)))**2)

@njit
def Fidelity_DM_pure(rho, psi):
    """
    Fidelity between a density matrix and a pure state
    """
    return np.real(np.vdot(np.conjugate(psi), rho @ psi))

@njit
def Fidelity_intime(fidelity_func, state1_t, state2_t):
    """
    Fidelity between two time-dependent states, computed at each time step.
    
    Parameters:
        fidelity_func: function
            The function to compute the fidelity between two states (e.g., Fidelity_WF or Fidelity_DM).
        state1_t: array-like
            The first time-dependent state (e.g., wavefunction or density matrix) with shape (dim, dim, time_steps) for density matrices or (dim, time_steps) for wavefunctions.
        state2_t: array-like
            The second time-dependent state with the same shape as state1_t.
    Returns:
        fidelity_time: array-like
            The fidelity between the two states at each time step, with shape (time_steps,).
    """
    time_steps = state1_t.shape[-1]
    fidelity_time = np.zeros(time_steps)
    for t in range(time_steps):
        fidelity_time[t] = fidelity_func(state1_t[:,:, t], state2_t[:,:, t]) if state1_t.ndim == 3 else fidelity_func(state1_t[:, t], state2_t[:, t])
    return fidelity_time


##############################################################################
##############################################################################
# TRACE DISTANCE and BLP MEASURE

@njit
def trace_distance(rho1,rho2):
    '''
    Compute the trace distance between two density matrices rho1 and rho2, under the trace norm.
    '''
    mat_meas = rho1 - rho2
    trace_norm = np.sqrt( mat_meas.conj().T @ mat_meas )
    return np.real(np.trace(trace_norm) * 0.5)

@njit
def trace_distance_intime(rho1_t,rho2_t):
    '''
    Compute the trace distance between two time-dependent density matrices rho1_t and rho2_t.
    '''
    trace_distance_time = np.zeros(rho1_t.shape[-1])#, dtype=ncomplex)
    for t in range(rho1_t.shape[-1]):
        trace_distance_time[t] = trace_distance(rho1_t[:,:, t], rho2_t[:,:, t])
    return trace_distance_time



def finite_diff_gradient(x, y):
    """
    compute the finite difference gradient from two arrays x and y
    
    Parameters:
        x: array-like
            The independent variable values.
        y: array-like
            The dependent variable values.
    Returns:
        gradient: array-like
            The finite difference gradient of y with respect to x.
    """
    gradient = np.zeros_like(y)
    gradient[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    gradient[0] = (y[1] - y[0]) / (x[1] - x[0])  # Forward difference for the first point
    gradient[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])  # Backward difference for the last point
    return gradient

def get_positive_part(y):
    """
    Get the positive part of y with respect to x.
    
    Parameters:
        x: array-like
            The independent variable values.
        y: array-like
            The dependent variable values.
    Returns:
        positive_y: array-like
            The positive part of y with respect to x.
    """
    positive_y = np.zeros_like(y)
    for i in range(len(y)):
        if y[i] > 0:
            positive_y[i] = y[i]
    return positive_y

def integrate_on_positive_part(x, y):
    """
    Integrate the positive part of y with respect to x.
    
    Parameters:
        x: array-like
            The independent variable values.
        y: array-like
            The dependent variable values.
    Returns:
        integral: float
            The integral of the positive part of y with respect to x.
    """
    positive_y = np.zeros(y.shape)
    for i in range(len(y)):
        if y[i] > 0:
            positive_y[i] = y[i]
    #integral = np.trapz(positive_y, x)
    integral_t = np.cumsum(positive_y * np.diff(x, prepend=x[0]))  # Cumulative sum for integration
    integral = integral_t[-1]  # Return the last value as the integral
    return integral, integral_t
    


@njit
def BLP_meas(rho1_t, rho2_t, time):
    '''
    Compute the BLP measure between two density matrices rho1 and rho2.
    '''
    trace_distance_time = trace_distance_intime(rho1_t, rho2_t)
    rate_of_change = finite_diff_gradient(time, trace_distance_time)
    positive_domain = get_positive_part(rate_of_change)
    blp_meas, _ = integrate_on_positive_part(time, positive_domain)
    return blp_meas