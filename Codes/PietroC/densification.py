import numpy as np

from numba import jit, njit, prange
from numba import complex128 as ncomplex

#from gutils import utils
#from sse import averages as sav

#from qme.pauli import sigma_0, sigma_x, sigma_y, sigma_z

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def bloch_coords(rho):
    """Compute Bloch vector from a 2x2 density matrix."""
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    return np.array([x, y, z])

## NJIT FUNCTIONS 
@njit
def NJIT_bloch_coords(rho):
    """Compute Bloch vector from a 2x2 density matrix."""
    x = np.real(np.trace(rho @ np.array([[0, 1], [1, 0]], dtype=ncomplex)))
    y = np.real(np.trace(rho @ np.array([[0, -1j], [1j, 0]], dtype=ncomplex)))
    z = np.real(np.trace(rho @ np.array([[1, 0], [0, -1]], dtype=ncomplex)))
    return x, y, z

@njit
def NJIT_vectors_inCartesian_coords(many_rho_trjs, t_idx):
    """
    Compute the Bloch vectors in Cartesian coordinates for a set of density matrices.
    Returns an array of shape (N_traj, 3) to match the standard data matrix format.
    """
    n_traj = many_rho_trjs.shape[0]
    vects = np.empty((n_traj, 3))  # Shape is now (N_traj, 3)
    
    for i in range(n_traj):
        rho = many_rho_trjs[i, t_idx]
        bloch_vec = NJIT_bloch_coords(rho)  # Assuming this returns [x, y, z]
        vects[i, 0] = bloch_vec[0]
        vects[i, 1] = bloch_vec[1]
        vects[i, 2] = bloch_vec[2]
        
    return vects

@njit
def NJIT_angle_between_vectors(u, v):
    """
    Compute the angle between two 3D vectors.

    Parameters
    ----------
    u : np.ndarray
        A 1D array of shape (3,) representing the first vector.
    v : np.ndarray
        A 1D array of shape (3,) representing the second vector.

    Returns
    -------
    float
        The angle between the two vectors in radians.
    """

    dot_product = 0.0
    norm_u = 0.0
    norm_v = 0.0
    for i in range(3):
        dot_product += u[i] * v[i]
        norm_u += u[i] * u[i]
        norm_v += v[i] * v[i]
    norm_u = np.sqrt(norm_u)
    norm_v = np.sqrt(norm_v)
    cos_theta = dot_product / (norm_u * norm_v)
    # clamp to [-1, 1] to avoid NaNs from acos
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    return np.arccos(cos_theta)


@njit(parallel=True)
def NJIT_mean_angle_parallel(many_rho_trjs, t_idx, norm=np.pi):
    """
    Compute the mean angle using multi-core parallelization.
    This distributes the O(N^2) workload across all available CPU cores.
    """
    n = many_rho_trjs.shape[0]
    vects = NJIT_vectors_inCartesian_coords(many_rho_trjs, t_idx)
    
    av_angle = 0.0
    c = 0
    
    # Replace 'range' with 'prange' in the outer loop to parallelize
    for i in prange(n):
        local_angle = 0.0
        local_c = 0
        
        # Inner loop remains standard 'range'
        for j in range(i + 1, n):
            vec1 = vects[:, i]
            vec2 = vects[:, j]
            local_angle += NJIT_angle_between_vectors(vec1, vec2)
            local_c += 1
            
        # Numba automatically handles this reduction safely across threads
        av_angle += local_angle
        c += local_c
        
    if c > 0:
        av_angle /= (c * norm)
    else:
        av_angle = 0.0
        
    return av_angle

@njit
def NJIT_syncr_measure_time(many_rho_trjs: np.ndarray,
                            norm: float = np.pi/2,
                            minusone: bool = True) -> np.ndarray:
    '''
    Compute the synchronization measure over time for a set (family) of stochastic density matrices.
    The synchronization measure is defined as the average angle between the Bloch vectors
    of the density matrices at each time step. The synchronization measure takes values in [0, 1],
    where 1 indicates perfect synchronization and 0 indicates no synchronization.

    Parameters
    ----------
    many_rho_trjs : np.ndarray
        A 3D array of shape (n_traj, n_time, 2, 2) containing the density matrices.
    norm : float
        The normalization factor for the angles.
    minusone : bool
        If True, subtract the angles from 1 to get the synchronization measure.

    Returns
    -------
    np.ndarray
        A 1D array of shape (n_time,) containing the synchronization measure at each time step.
    '''

    ntime = many_rho_trjs.shape[1]
    syncr_meas = np.zeros(ntime)
    
    
    
    if minusone:
        try:
            syncr_meas[0] = 1. - NJIT_mean_angle_parallel(many_rho_trjs=many_rho_trjs, t_idx=0, norm=norm)
        except:
            syncr_meas[0] = 1.
        for i in range(1, ntime):
            syncr_meas[i] = 1 - NJIT_mean_angle_parallel(many_rho_trjs=many_rho_trjs, t_idx=i, norm=norm)
    else:
        try:
            syncr_meas[0] = NJIT_mean_angle_parallel(many_rho_trjs=many_rho_trjs, t_idx=0, norm=norm)
        except:
            syncr_meas[0] = 0.
        for i in range(1, ntime):
            syncr_meas[i] = NJIT_mean_angle_parallel(many_rho_trjs=many_rho_trjs, t_idx=i, norm=norm)
    return syncr_meas
