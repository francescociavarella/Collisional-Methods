#!/usr/bin/env python
# coding: utf-8

# ## Libraries & Function

# In[35]:


import numpy as np
from scipy.linalg import expm
from qutip import *
import numba
from numba import njit, prange
import os
import time
import sys

sz = np.array(([[1,0], [0,-1]]), dtype=complex); sx = np.array(([[0,1],[1,0]]), dtype=complex); sy = np.array(([[0,-1j],[1j,0]]), dtype=complex) ; sm = np.array(([[0.0, 1.0],[0.0,0.0]]), dtype=complex) ; sp = np.array(([[0.0,0.0],[1.0,0.0]]), dtype=complex)

# ============================
# Hamiltonians and U operator
# ============================


def interaction_Hamiltonian(c_CM, P12, P21, sp, sm):
    """
    Builds the Interaction Hamiltonian for the 3-level System and Ancilla collision.
    Implements the model: c * (|1><2| @ sigma_plus_a + |2><1| @ sigma_minus_a)
        
    Parameters: 
    - c_CM : float, Interaction strength for the collision
    - P12_sys : numpy array, System relaxes from |2> to |1>
    - P21_sys : numpy array, System excites from |1> to |2>
    - sp : numpy array, Ancilla raising operator
    - sm : numpy array, Ancilla lowering operator
        
    Returns: 
    - H_int : numpy array (6x6), Interaction Hamiltonian
    """
    # Tensor product for relaxation: System relaxes (P12), Ancilla excites (sp)
    term_relaxation = np.kron(P12, sp)
    
    # Tensor product for excitation: System excites (P21), Ancilla relaxes (sm)
    term_excitation = np.kron(P21, sm)
    
    # Total Collisional Hamiltonian
    H_int = c_CM * (term_relaxation + term_excitation)
    
    return H_int


def complete_Hamiltonian(H_Sys, c_CM, P12, P21, sp, sm):
    """
    Generates the Hamiltonians for the 3-level system collision model using pure NumPy:
                - H_system : system Hamiltonian
                - H_collision : interaction Hamiltonian with 1 ancilla
                - H_tot : complete Hamiltonian (system + collision)

    Parameters: 
    - H_Sys: numpy array (3x3), System Hamiltonian
    - c_CM : float, Interaction Force
    - P12_sys, P21_sys, sp, sm : numpy arrays, operators for the interaction
                
    Returns: 
    - H_system, H_collision, H_tot (all as numpy arrays)
    """
    # 1. System Hamiltonian 
    H_system = H_Sys
    
    # 2. Collision Hamiltonian
    H_collision = interaction_Hamiltonian(c_CM, P12, P21, sp, sm) 
    
    # 3. Total Hamiltonian
    # Expand H_sys in the total space: System (3x3) tensor Identity_ancilla (2x2)
    Id_ancilla = np.eye(2, dtype=complex)
    H_system_expanded = np.kron(H_system, Id_ancilla)  
        
    H_tot = H_system_expanded + H_collision
        
    return H_system, H_collision, H_tot


def evolution_operator(H, dt, method='expm', hermitian=True):
    """
    Build up of the evolution operator U = exp(-i H dt) using Expm or analytic diagonalization.
   
    Parameters: - H : Qobj or nparray, System Hamiltonian
                - dt : float, Timestep
    
    Method : - "expm"-> build up of the Matrix Exponential with expm
             - "diagonalization"->  build up of the propagater U as V @(exp(-i W dt))@ V_dag with W eigenvalues and V eigenvector of the Hamiltonian 

    Returns : Evolution Operator U, 
    """
    H = H.full() if hasattr(H, "full") else np.array(H)
    
    # -----------
    # Expm method
    # -----------
    
    if method == 'expm':
        U = expm(-1j * H * dt)
        return U
        
    # ---------------
    # Diagonalization
    # ---------------
    
    elif method == 'diagonalization':
        if hermitian:
            w, V = np.linalg.eigh(H)
            V_inv = V.conj().T
        else:
            w, V = np.linalg.eig(H) 
            V_inv = np.linalg.inv(V)
                
        U_diag = np.diag(np.exp(-1j * w * dt))
        U = V @ U_diag @ V_inv
        return U, U_diag, w, V

    else:
        raise ValueError("method must be 'expm' or 'diagonalization'")

# ===================
# Lindblad functions
# ===================


def Liouvillian(H, gamma_k, L_k):
    """
    Build the Liouvillian superoperator using row-major convention (NumPy).
    
    Parameters: - H : nparray, Hamiltonian matrix
                - gamma_k : list, Decay rates
                - L_k : list, Jump Operators
    
    Returns: - super_L : nparray, Liouvillian superoperator
    """    
    I = np.eye(H.shape[0], dtype=complex)
    
    # Unitary evolution: -i * [H, rho]
    super_L = -1.j * (np.kron(H, I) - np.kron(I, H.T))
    
    # Dissipator terms
    for k in range(len(gamma_k)):
        L = L_k[k]
        L_dag = np.conj(L).T
        L_dag_L = L_dag @ L
        
        super_L += gamma_k[k] * (np.kron(L, np.conj(L)) - 0.5 * np.kron(L_dag_L, I) - 0.5 * np.kron(I, L_dag_L.T))
    
    return super_L


@njit(cache=True)
def _evolve_expm_core(super_U, rho_vec_initial, n_times):
    """
    Core evolution loop with expm method (Numba JIT)
    """
    rho_size = rho_vec_initial.shape[0]
    rho_vec_list = np.zeros((rho_size, n_times), dtype=np.complex128)
    rho_vec_list[:, 0] = rho_vec_initial
    
    for i in range(1, n_times):
        rho_vec_list[:, i] = super_U @ rho_vec_list[:, i - 1]
    
    return rho_vec_list


@njit(cache=True)
def _evolve_diagonal_core(V, V_inv, U_diag, rho_vec_initial, n_times):
    """
    Core evolution loop with diagonal method (Numba JIT)
    """
    n_states = len(U_diag)
    
    # Initial coefficients in eigenbasis
    coeff = V_inv @ rho_vec_initial
    coeff_list = np.zeros((n_states, n_times), dtype=np.complex128)
    coeff_list[:, 0] = coeff
    
    # Evolution of coefficients
    for i in range(1, n_times):
        coeff_list[:, i] = U_diag * coeff_list[:, i - 1]
    
    # Transform back to original basis
    rho_vec_list = V @ coeff_list
    
    return rho_vec_list


def Lindblad_evo(rho, H, gamma_k, L_k, times, method="expm", vectorized=True):
    """
    Evolution of the density matrix with the Lindblad Eq. (Optimized with Numba)
    
    Method: - "expm" -> propagator = expm(super_L * dt)
            - "diagonal" -> diagonalization of the super-operator
        
    Vectorized: True/False to choose the output format
    
    Parameters: - H : nparray, System Hamiltonian
                - rho : Qobj or nparray, Initial Density Matrix
                - gamma_k : list, List of Decay Rates
                - L_k : list, List of Jump Operators
                - times : array, Time array
        
    Returns : - if vectorized=True → array (N^2, Nt)
              - if vectorized=False → array (Nt, N_site, N_site)
              - if method="diagonal" also returns V, W
    """
    # Convert to NumPy
    L_k = [L.full() if hasattr(L, "full") else np.array(L, dtype=complex) for L in L_k]
    H = H.full() if hasattr(H, "full") else np.array(H, dtype=complex)
    rho = rho.full() if hasattr(rho, "full") else np.array(rho, dtype=complex)
    
    rho_shape = H.shape[0]
    dt = times[1] - times[0]
    n_times = len(times)
    
    # Build Liouvillian
    super_L = Liouvillian(H, gamma_k, L_k)
    
    # Vectorize initial state
    rho_vec = rho.reshape(rho_shape * rho_shape)
    
    # -------------
    # Expm method
    # -------------
    if method == "expm":
        # Compute propagator 
        super_U = expm(super_L * dt)
        
        # evolution loop
        rho_vec_list = _evolve_expm_core(super_U, rho_vec, n_times)
        
        # Output
        if vectorized:
            return rho_vec_list
        else:
            return rho_vec_list.T.reshape(n_times, rho_shape, rho_shape)
    
    # ------------------
    # Diagonal method
    # ------------------
    elif method == "diagonal":
        # Diagonalize Liouvillian 
        W, V = np.linalg.eig(super_L)
        V_inv = np.linalg.inv(V)
        
        # Diagonal evolution operator
        U_diag = np.exp(W * dt)
        
        # evolution loop
        rho_vec_list = _evolve_diagonal_core(V, V_inv, U_diag, rho_vec, n_times)
        
        # Output
        if vectorized:
            return rho_vec_list, V, W
        else:
            return rho_vec_list.T.reshape(n_times, rho_shape, rho_shape), V, W
    
    else:
        raise ValueError("method must be 'expm' or 'diagonal'")

# ================
# Isolated system
# ================

@njit(cache=True)
def _compute_trajectory_isolated_core_density(psi_initial, U_site, n_times):
    """
    Core evolution loop optimized with Numba.
    Computes and stores the full density matrix rho(t) = |psi><psi| at each time step.
    """
    N_dim = len(psi_initial)
    
    # Pre-allocate array for the density matrix trajectory: shape (N, N, n_times)
    rho_traj = np.zeros((N_dim, N_dim, n_times), dtype=np.complex128)
    
    # Calculate initial density matrix (explicit loop is ultra-fast and safe in Numba)
    for i in range(N_dim):
        for j in range(N_dim):
            rho_traj[i, j, 0] = psi_initial[i] * np.conj(psi_initial[j])
            
    # Time Evolution loop
    psi = psi_initial.copy()
    for step in range(1, n_times):
        # Evolve the state vector
        psi = U_site @ psi
        
        # Calculate and store the density matrix for the current time step
        for i in range(N_dim):
            for j in range(N_dim):
                rho_traj[i, j, step] = psi[i] * np.conj(psi[j])

    return rho_traj


def compute_trajectory_wf_isolated(times, psi_sys_initial, U_site):
    """
    Optimized isolated system evolution.
    Calculates the full density matrix over a given time array.
    """
    # Convert QuTiP objects to NumPy arrays if necessary
    U_site_np = U_site.full() if hasattr(U_site, 'full') else np.array(U_site, dtype=complex)
    psi_initial_np = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else np.array(psi_sys_initial, dtype=complex)
    
    # Flatten the state vector if it's explicitly written as a column matrix
    if psi_initial_np.ndim > 1:
        psi_initial_np = psi_initial_np.flatten()

    # Number of time steps
    n_times = len(times)
        
    # Call the JIT-compiled core function
    rho_traj_isolated = _compute_trajectory_isolated_core_density(
        psi_initial_np, U_site_np, n_times
    )

    return rho_traj_isolated

# =============================
# Collisional Method functions
# =============================

# ==========================================================
# Evolution with U_{complete} and then trace on the ancilla
# ==========================================================

@njit(cache=True)
def _compute_trace_ancilla_core_density(rho_sys, rho_anc, U_step, U_step_dag, n_times, dim_sys, dim_anc):
    """
    Core computation optimized with Numba for an N-level system + single ancilla.
    Evolves the total state and traces out the ancilla at each step, 
    returning the full system density matrix over time.
    """
    # Pre-allocate array for the density matrix trajectory: shape (N, N, n_times)
    rho_trace = np.zeros((dim_sys, dim_sys, n_times), dtype=np.complex128)
    
    # Store initial state density matrix
    for i in range(dim_sys):
        for j in range(dim_sys):
            rho_trace[i, j, 0] = rho_sys[i, j]
    
    # Time Evolution loop
    for t in range(1, n_times):
        # 1: Expansion (System tensor Ancilla)
        rho_tot = np.kron(rho_sys, rho_anc)
        
        # 2: Evolution (single time step)
        rho_tot = U_step @ rho_tot @ U_step_dag
        
        # 3: Partial Trace over the Ancilla
        rho_tot_reshaped = rho_tot.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
        
        # Manual trace (summing over the ancilla indices)
        rho_sys = np.zeros((dim_sys, dim_sys), dtype=np.complex128)
        for i in range(dim_sys):
            for j in range(dim_sys):
                for k in range(dim_anc):
                    rho_sys[i, j] += rho_tot_reshaped[i, k, j, k]
        
        # 4: Store density matrix for the current step
        for i in range(dim_sys):
            for j in range(dim_sys):
                rho_trace[i, j, t] = rho_sys[i, j]
    
    return rho_trace


def compute_trace_ancilla_density(rho_sys_initial, rho_anc_single, U_diag, V, times):
    """
    Evolution with complete collisional Hamiltonian and trace on the Ancilla 
    degrees of freedom. Corresponds to the deterministic dynamics of the 
    open quantum system.
    """
    # Convert system and ancilla states to numpy arrays if they are QuTiP objects
    rho_anc = rho_anc_single.full() if hasattr(rho_anc_single, 'full') else np.array(rho_anc_single, dtype=complex)
    rho_sys = rho_sys_initial.full() if hasattr(rho_sys_initial, 'full') else np.array(rho_sys_initial, dtype=complex)

    # Time parameters
    n_times = len(times)

    # Dimensions
    dim_sys = rho_sys.shape[0]
    dim_anc = rho_anc.shape[0]
    
    # Evolution operator for a single time step
    V_np = V.full() if hasattr(V, 'full') else np.array(V, dtype=complex)
    U_diag_np = U_diag.full() if hasattr(U_diag, 'full') else np.array(U_diag, dtype=complex)
    
    # Reconstruct U_step = V * U_diag * V_dagger
    U_step = V_np @ U_diag_np @ V_np.conj().T
    U_step_dag = U_step.conj().T
    
    # Call JIT-compiled core function
    rho_traj_complete = _compute_trace_ancilla_core_density(
        rho_sys, rho_anc, U_step, U_step_dag, n_times, dim_sys, dim_anc
    )
    
    return rho_traj_complete

# =============================================
# Stochastic Trajectories with Kraus Operators
# =============================================

@njit(cache=True)
def sigma_xyz_expectation_value(psi):
    """
    Calculates the expectation values of the Pauli operators <sigma_x>, 
    <sigma_y>, and <sigma_z> for the subspace spanned by states |1> and |2> 
    in a 3-level system.

    Parameters: 
    - psi : numpy array, wave function at time t (shape: 3,)

    Returns: 
    - S_x : float, expectation value of <sigma_x>
    - S_y : float, expectation value of <sigma_y>
    - S_z : float, expectation value of <sigma_z>
    """
    
    # Pauli X embedded in the |1>, |2> subspace
    sigma_x = np.array([[0.0, 0.0, 0.0], 
                        [0.0, 0.0, 1.0], 
                        [0.0, 1.0, 0.0]], dtype=np.complex128) 
    
    # Pauli Y embedded in the |1>, |2> subspace
    sigma_y = np.array([[0.0, 0.0, 0.0], 
                        [0.0, 0.0, -1.0j], 
                        [0.0, 1.0j, 0.0]], dtype=np.complex128) 
    
    # Pauli Z embedded in the |1>, |2> subspace
    sigma_z = np.array([[0.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0], 
                        [0.0, 0.0, -1.0]], dtype=np.complex128) 

    # Compute expectation values 
    S_x = np.real(np.vdot(psi, sigma_x @ psi))
    S_y = np.real(np.vdot(psi, sigma_y @ psi))
    S_z = np.real(np.vdot(psi, sigma_z @ psi))

    return S_x, S_y, S_z


def generate_kraus_operators(c_CM, dt, phi_rad):
    """
    Generates the generalized Kraus operators M0 and M1 for the 3-level system
    using exclusively the intermediate angle phi_rad.
    
    Parameters: 
    - c_CM: float, Interaction coefficient
    - dt: float, Time step
    - phi_rad: float, Angle for the measurement basis (in radians)
                
    Returns: 
    - M0, M1: numpy arrays, The generalized Kraus operators
    """
    c_dt = c_CM * dt
    cos_val = np.cos(c_dt)
    sin_val = np.sin(c_dt)
    
    # Base Quantum Jump operators (equations 290 and 291)
    K0_QJ = np.array([[1.0, 0.0, 0.0], 
                      [0.0, 1.0, 0.0], 
                      [0.0, 0.0, cos_val]], dtype=np.complex128)
                       
    K1_QJ = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, -1j * sin_val], 
                      [0.0, 0.0, 0.0]], dtype=np.complex128)
    
    # Trigonometric coefficients for the intermediate basis transformation
    sin_phi2 = np.sin(phi_rad / 2.0)
    cos_phi2 = np.cos(phi_rad / 2.0)
    
    # Generalized intermediate operators (equations 299 and 300)
    M0 = sin_phi2 * K0_QJ + cos_phi2 * K1_QJ
    M1 = cos_phi2 * K0_QJ - sin_phi2 * K1_QJ
        
    return M0, M1


import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True, fastmath=True)
def compute_trajectory_wf_core_density(psi_initial, U_site, M0, M1, N_traj, n_times, seeds):
    """
    Core trajectory evolution optimized with Numba for an N-level system.
    Probabilities are dynamically computed at each time step using Kraus operators.
    Computes and stores the full density matrix rho(t) for each trajectory, 
    and records every quantum jump occurrence.
    """
    N_dim = len(psi_initial)
    
    # Pre-allocate array for all trajectories: shape (N_dim, N_dim, n_times, N_traj)
    rho_traj = np.zeros((N_dim, N_dim, n_times, N_traj), dtype=np.complex128)
    
    # Pre-allocate array to record jumps safely across threads: shape (n_times, N_traj)
    jump_records = np.zeros((n_times, N_traj), dtype=np.int32)
    
    # Loop over independent trajectories in parallel
    for traj in prange(N_traj):
        np.random.seed(seeds[traj])
        psi = psi_initial.copy()
        
        # Initialization at t=0
        for i in range(N_dim):
            for j in range(N_dim):
                rho_traj[i, j, 0, traj] = psi[i] * np.conj(psi[j])

        # Time evolution loop
        for step in range(1, n_times):
            # 1. Deterministic evolution given by the isolated System Hamiltonian
            psi = U_site @ psi

            # 2. Apply Kraus operator M1 to test the jump probability
            v1 = M1 @ psi
            
            # The probability P1 is exactly the squared norm of the resulting vector
            P1 = np.real(np.vdot(v1, v1))
            
            # 3. Stochastic jump Monte Carlo selection
            r = np.random.rand()
            if r < P1:
                psi = v1 # Quantum Jump occurs (M1 was already applied)
                # Safely record the jump for this specific trajectory and time step
                jump_records[step, traj] = 1 
            else:
                psi = M0 @ psi # No jump occurs (Null measurement)

            # 4. State Normalization
            norm_psi = np.linalg.norm(psi)
            for i in range(N_dim):
                psi[i] = psi[i] / norm_psi

            # 5. Store the full density matrix for the current step
            for i in range(N_dim):
                for j in range(N_dim):
                    rho_traj[i, j, step, traj] = psi[i] * np.conj(psi[j])

    return rho_traj, jump_records


def compute_trajectory_wf(c_CM, dt, N_traj, times, 
                          psi_sys_initial, U_site, 
                          phi, batch_size=1000):
    """
    Wrapper function to handle batching and random seeds before calling the JIT core.
    Returns the collection of all single trajectories and an array containing 
    the total count of jumps evaluated at each time step.
    """
    # Convert objects to numpy arrays if necessary
    U_site_np = U_site.full() if hasattr(U_site, 'full') else np.array(U_site, dtype=complex)
    psi_sys_initial_np = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else np.array(psi_sys_initial, dtype=complex)
    
    if psi_sys_initial_np.ndim > 1:
        psi_sys_initial_np = psi_sys_initial_np.flatten()
        
    n_times = len(times)
    N_dim = len(psi_sys_initial_np)
    
    # Generate the specific Kraus Operators according to the selected mode
    M0, M1 = generate_kraus_operators(c_CM, dt, phi)

    # Pre-generate seeds for reproducible parallel execution
    rng_seeds = np.random.RandomState(42)
    all_seeds = rng_seeds.randint(0, 2**30, size=N_traj)

    # Pre-allocate the complete array for all trajectories
    rho_tot_all = np.zeros((N_dim, N_dim, n_times, N_traj), dtype=np.complex128)
    
    # Pre-allocate array to sum up all jumps across all batches
    total_jump_counts = np.zeros(n_times, dtype=np.int64)

    N_done = 0
    n_batches = int(np.ceil(N_traj / batch_size))

    # Batch execution to manage memory footprint
    for b in range(n_batches):
        N_batch = min(batch_size, N_traj - N_done)
        seeds_b = all_seeds[N_done : N_done + N_batch]

        # Call the Numba JIT compiled core
        rho_batch, jumps_batch = compute_trajectory_wf_core_density(
            psi_sys_initial_np, U_site_np, M0, M1,
            N_batch, n_times, seeds_b)

        # Store batch results
        rho_tot_all[:, :, :, N_done : N_done + N_batch] = rho_batch
        
        # Accumulate the jump counts for this batch by summing across the trajectory axis
        total_jump_counts += np.sum(jumps_batch, axis=1)

        N_done += N_batch
        del rho_batch, jumps_batch

    return rho_tot_all, total_jump_counts

# ======================================
# Main Loop for varying dt and N_{traj}
# ======================================

# ===================
# System's Parameters
# ===================
np.random.seed(1) # always use the same seed 
N_site = 3  # Number of sites
#V_array = [1.0]    NO potential
E0 = 0.0  # Energy of the ground state |0>
E1 = 1.5  # Energy of the first excited state |1>
E2 = 2.0  # Energy of the second excited state |2>    

H_Sys = np.diag([E0, E1, E2])  # System Hamiltonian

# =========================
# Time Evolution Parameters
# =========================
dt_list = [0.01]     # change : time step
tf = 50.0    # Final Time
steps_list = [ int(tf / dt_list[i]) for i in range (len(dt_list)) ]
times_list = [ np.linspace(0, tf, int(steps_list[i])) for i in range(len(dt_list))]

N_traj = 10000  # change numberof trajectories

# ===================
# Dephasing Parameter 
# ===================
gamma_r = 0.1   # Gamma rate for the decay
# Lindblad Rates list
gamma_k = [gamma_r ]

# Scaling for the collsional algorithm c = sqrt(gamma / dt)
c_CM_list = np.array([np.sqrt(gamma_r / dt_list[j]) for j in range(len(dt_list))] )  


# ========================================
# Initial wave function and density matrix
# ========================================

# ======
# System
# ======
pop_0 = np.sqrt(1 - 10**(-3)) # Population in |0> is close to 1, but not exactly 1 to avoid numerical issues
pop_1 = 0.0
pop_2 = np.sqrt(10**(-3))

psi_sys_initial = np.array([pop_0, pop_1, pop_2], dtype=complex) # System is initialized in |0> mainly and|2> perturbatively
rho_sys_initial = np.outer(psi_sys_initial, psi_sys_initial.conj()) # Density matrix of the system at t=0     

# =======
# Ancilla
# =======
# Ancilla is strictly initialized in |0> 
psi_anc_single = np.array([1.0, 0.0], dtype=complex)  # ancilla initialized in |0> always
rho_anc_single = np.outer(psi_anc_single, psi_anc_single.conj())

# =========
# Projectors
# =========
P00 = np.array([[1, 0, 0],
                 [0, 0, 0], 
                 [0, 0, 0]], dtype=complex) # Projector on |0><0|

P11 = np.array([[0, 0, 0], 
                [0, 1, 0], 
                [0, 0, 0]], dtype=complex) # Projector on |1><1|

P22 = np.array([[0, 0, 0], 
                [0, 0, 0], 
                [0, 0, 1]], dtype=complex) # Projector on |2><2|

P01 = np.array([[0, 1, 0], 
                [0, 0, 0], 
                [0, 0, 0]], dtype=complex) # Projector on |0><1|

P10 = np.array([[0, 0, 0], 
                [1, 0, 0], 
                [0, 0, 0]], dtype=complex) # Projector on |1><0|

P12 = np.array([[0, 0, 0], 
                [0, 0, 1], 
                [0, 0, 0]], dtype=complex) # Projector on |1><2|

P21 = np.array([[0, 0, 0], 
                [0, 0, 0], 
                [0, 1, 0]], dtype=complex) # Projector on |2><1|

P02 = np.array([[0, 0, 1], 
                [0, 0, 0], 
                [0, 0, 0]], dtype=complex) # Projector on |0><2|

P20 = np.array([[0, 0, 0], 
                [0, 0, 0], 
                [1, 0, 0]], dtype=complex) # Projector on |2><0|


projectors = np.array([P00, P11, P22], dtype=complex) 
projectors_cohe = np.array([P01, P10,P12, P21, P02, P20], dtype=complex) 

# ======================
# Lindblad Jump Operator
# ======================
L_r = P12 # Jump operator for relaxation |1><2| 
L_k = [L_r]

# ============
# Calculation
# ============

# Lettura degli argomenti passati dal file Bash:
if len(sys.argv) > 1:
    phi_deg = float(sys.argv[1]) 
    bash_mode = sys.argv[2] if len(sys.argv) > 2 else "unknown" 
else:
    phi_deg = 90.0 
    bash_mode = "local_test"

phi_rad = np.radians(phi_deg)

# ======================
# Output directory setup
# ======================
results_dir = "../../Results/Data/Complete_rho/"
os.makedirs(results_dir, exist_ok=True)
BATCH_SIZE = 1000

def _make_fname_npz(results_dir, phi, dt, N_traj):
    dt_str = f"{dt:.6f}".replace(".", "p")
    # Replace dot with 'p' to prevent issues in file names on the cluster
    phi_str = f"{phi:.4f}".replace(".", "p")
    return os.path.join(results_dir, f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}.npz")

print(f"Starting computation for phi = {phi_rad:.4f}")

for dt_idx, dt in enumerate(dt_list):
    times = times_list[dt_idx]
    steps = steps_list[dt_idx]
    c_CM  = c_CM_list[dt_idx]

    H_site, H_coll, H_tot = complete_Hamiltonian(H_Sys, c_CM, P12, P21, sp, sm)
    U_tot, U_diag, w, V = evolution_operator(H_tot, dt, method='diagonalization', hermitian=True)
    U_site, U_diag_site, w_site, V_site = evolution_operator(H_site, dt, method='diagonalization', hermitian=True)

    rho_list_lindblad, V_lindblad, W_lindblad = Lindblad_evo(
        rho_sys_initial, H_site, gamma_k, L_k, times, method="diagonal", vectorized=False
    )
    rho_traj_isolated = compute_trajectory_wf_isolated(times, psi_sys_initial, U_site)
    rho_trace = compute_trace_ancilla_density(rho_sys_initial, rho_anc_single, U_diag, V, times)

    # Compute trajectories using the dynamically passed phi value
    rho_tot_all, total_jumps = compute_trajectory_wf(
        c_CM, dt, N_traj, times,
        psi_sys_initial, U_site,
        phi=phi_rad, batch_size=BATCH_SIZE
    )
    
    fname_npz = _make_fname_npz(results_dir, phi_rad, dt, N_traj)

    np.savez_compressed(
        fname_npz,
        rho_tot_all=rho_tot_all,
        total_jumps=total_jumps,
        rho_trace=rho_trace,
        rho_list_lindblad=rho_list_lindblad,
        V_lindblad=V_lindblad,
        W_lindblad=W_lindblad,
        rho_traj_isolated=rho_traj_isolated,
        phi=phi_rad, dt=dt, N_traj=N_traj,
        times=times, steps=steps, c_CM=c_CM
    )

    print(f"Saved -> {os.path.basename(fname_npz)}")
    del rho_tot_all, total_jumps, rho_list_lindblad, rho_traj_isolated, rho_trace

print("\n" + "=" * 40)
print("COMPUTATION COMPLETED!")
print("Results saved for:")
# Updated print statements to reflect the new angle-based logic
print(f"  - Angle (phi): {phi_deg} degrees ({phi_rad:.4f} rad)")
print(f"  - {len(dt_list)} dt values: {dt_list}")
print(f"  - Fixed N_traj: {N_traj}")
print("=" * 40)




