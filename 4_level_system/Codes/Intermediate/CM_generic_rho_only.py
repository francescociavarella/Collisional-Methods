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


def interaction_Hamiltonian(c_CM, L, L_dag, sp, sm):
    """
    Builds the Interaction Hamiltonian for the 3-level System and Ancilla collision.
    Implements the model: c * (|1><2| @ sigma_plus_a + |2><1| @ sigma_minus_a)
        
    Parameters: 
    - c_CM : float, Interaction strength for the collision
    - L : numpy array, System jump operator
    - L_dag : numpy array, System jump dagger operator
    - sp : numpy array, Ancilla raising operator
    - sm : numpy array, Ancilla lowering operator
        
    Returns: 
    - H_int : numpy array (6x6), Interaction Hamiltonian
    """
    # Tensor product for relaxation: System relaxes (L), Ancilla excites (sp)
    term_relaxation = np.kron(L, sp)
    
    # Tensor product for excitation: System excites (L_dag), Ancilla relaxes (sm)
    term_excitation = np.kron(L_dag, sm)
    
    # Total Collisional Hamiltonian
    H_int = c_CM * (term_relaxation + term_excitation)
    
    return H_int


def complete_Hamiltonian(H_Sys, c_IC, c_ISC, P12, P21, P32, P23, sp, sm):
    """
    Generates the Hamiltonians for the 4-level system with TWO collisional channels (IC and ISC).
    The total environmental space is composed of two ancillae (2x2 @ 2x2 = 4x4).
    
    Parameters:
    - H_Sys: numpy array (4x4), System Hamiltonian
    - c_IC, c_ISC: floats, Interaction Forces for the two channels
    - Pij: numpy arrays (4x4), System transition projectors
    - sp, sm: numpy arrays (2x2), Ancilla ladder operators
    """
    Id_anc_single = np.eye(2, dtype=complex)
    Id_anc_double = np.eye(4, dtype=complex) # Total environment space
    
    # 1. Expand System Hamiltonian
    H_system_expanded = np.kron(H_Sys, Id_anc_double)
    
    # 2. Internal Conversion (IC) - Acts on Ancilla A (first subspace)
    sp_A = np.kron(sp, Id_anc_single)
    sm_A = np.kron(sm, Id_anc_single)
    H_IC = c_IC * (np.kron(P12, sp_A) + np.kron(P21, sm_A))
    
    # 3. Intersystem Crossing (ISC) - Acts on Ancilla B (second subspace)
    sp_B = np.kron(Id_anc_single, sp)
    sm_B = np.kron(Id_anc_single, sm)
    H_ISC = c_ISC * (np.kron(P32, sp_B) + np.kron(P23, sm_B))
    
    H_collision = H_IC + H_ISC
    H_tot = H_system_expanded + H_collision
    
    return H_Sys, H_collision, H_tot


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
    Core computation optimized with Numba.
    Evolves the total state (System 4x4 @ Environment 4x4 = Total 16x16)
    and traces out the 4-dimensional environment at each step.
    """
    # Pre-allocate array for the density matrix trajectory: shape (4, 4, n_times)
    rho_trace = np.zeros((dim_sys, dim_sys, n_times), dtype=np.complex128)
    
    # Store initial state density matrix
    for i in range(dim_sys):
        for j in range(dim_sys):
            rho_trace[i, j, 0] = rho_sys[i, j]
    
    # Time Evolution loop
    for t in range(1, n_times):
        # 1: Expansion (System tensor double Ancilla) -> 16x16 matrix
        rho_tot = np.kron(rho_sys, rho_anc)
        
        # 2: Evolution (single time step with U_tot)
        rho_tot = U_step @ rho_tot @ U_step_dag
        
        # 3: Partial Trace over the 4D environmental space
        rho_tot_reshaped = rho_tot.reshape((dim_sys, dim_anc, dim_sys, dim_anc))
        
        rho_sys = np.zeros((dim_sys, dim_sys), dtype=np.complex128)
        
        # Manual trace: summing over the environment index k (0 to 3)
        for i in range(dim_sys):
            for j in range(dim_sys):
                for k in range(dim_anc):
                    rho_sys[i, j] += rho_tot_reshaped[i, k, j, k]
        
        # 4: Store density matrix for the current step
        for i in range(dim_sys):
            for j in range(dim_sys):
                rho_trace[i, j, t] = rho_sys[i, j]
    
    return rho_trace


def compute_trace_ancilla_density(rho_sys_initial, rho_anc_double, U_diag, V, times):
    """
    Wrapper for the deterministic dynamics via partial trace.
    Takes the double ancilla state as the environmental input.
    """
    # Ensure inputs are standard numpy arrays
    rho_anc = np.array(rho_anc_double, dtype=complex)
    rho_sys = np.array(rho_sys_initial, dtype=complex)
    n_times = len(times)
    
    # Extract dimensions dynamically: dim_sys = 4, dim_anc = 4
    dim_sys = rho_sys.shape[0]
    dim_anc = rho_anc.shape[0] 
    
    # Reconstruct U_step = V * U_diag * V_dagger
    V_np = np.array(V, dtype=complex)
    U_diag_np = np.array(U_diag, dtype=complex)
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

# @njit(cache=True)
# def sigma_xyz_expectation_value(psi):
#     """
#     Calculates the expectation values of the Pauli operators <sigma_x>, 
#     <sigma_y>, and <sigma_z> for the subspace spanned by states |1> and |2> 
#     in a 3-level system.

#     Parameters: 
#     - psi : numpy array, wave function at time t (shape: 3,)

#     Returns: 
#     - S_x : float, expectation value of <sigma_x>
#     - S_y : float, expectation value of <sigma_y>
#     - S_z : float, expectation value of <sigma_z>
#     """
    
#     # Pauli X embedded in the |1>, |2> subspace
#     sigma_x = np.array([[0.0, 0.0, 0.0], 
#                         [0.0, 0.0, 1.0], 
#                         [0.0, 1.0, 0.0]], dtype=np.complex128) 
    
#     # Pauli Y embedded in the |1>, |2> subspace
#     sigma_y = np.array([[0.0, 0.0, 0.0], 
#                         [0.0, 0.0, -1.0j], 
#                         [0.0, 1.0j, 0.0]], dtype=np.complex128) 
    
#     # Pauli Z embedded in the |1>, |2> subspace
#     sigma_z = np.array([[0.0, 0.0, 0.0], 
#                         [0.0, 1.0, 0.0], 
#                         [0.0, 0.0, -1.0]], dtype=np.complex128) 

#     # Compute expectation values 
#     S_x = np.real(np.vdot(psi, sigma_x @ psi))
#     S_y = np.real(np.vdot(psi, sigma_y @ psi))
#     S_z = np.real(np.vdot(psi, sigma_z @ psi))

#     return S_x, S_y, S_z

def generate_kraus_operators_separated(c_IC, c_ISC, dt, phi_IC_rad, phi_ISC_rad):
    """
    Genera gli operatori di Kraus separati per i due canali.
    Non esegue il prodotto matriciale combinato.
    """
    # --- Internal Conversion (IC) ---
    c_dt_IC = c_IC * dt
    K0_IC = np.array([[1.0, 0, 0, 0], 
                      [0, 1.0, 0, 0], 
                      [0, 0, np.cos(c_dt_IC), 0],
                      [0, 0, 0, 1.0]], dtype=np.complex128)
    K1_IC = np.zeros((4,4), dtype=np.complex128)
    K1_IC[1, 2] = -1j * np.sin(c_dt_IC)
    
    cos_IC, sin_IC = np.cos(phi_IC_rad / 2.0), np.sin(phi_IC_rad / 2.0)
    M0_IC = cos_IC * K0_IC + sin_IC * K1_IC
    M1_IC = -sin_IC * K0_IC + cos_IC * K1_IC

    # --- Intersystem Crossing (ISC) ---
    c_dt_ISC = c_ISC * dt
    K0_ISC = np.array([[1.0, 0, 0, 0], 
                       [0, 1.0, 0, 0], 
                       [0, 0, np.cos(c_dt_ISC), 0],
                       [0, 0, 0, 1.0]], dtype=np.complex128)
    K1_ISC = np.zeros((4,4), dtype=np.complex128)
    K1_ISC[3, 2] = -1j * np.sin(c_dt_ISC)
    
    cos_ISC, sin_ISC = np.cos(phi_ISC_rad / 2.0), np.sin(phi_ISC_rad / 2.0)
    M0_ISC = cos_ISC * K0_ISC + sin_ISC * K1_ISC
    M1_ISC = -sin_ISC * K0_ISC + cos_ISC * K1_ISC

    return M0_IC, M1_IC, M0_ISC, M1_ISC

@njit(parallel=True, cache=True, fastmath=True)
def compute_trajectory_wf_core_separated(psi_initial, U_site, M0_IC, M1_IC, M0_ISC, M1_ISC, N_traj, n_times, seeds):
    N_dim = len(psi_initial)
    rho_traj = np.zeros((N_dim, N_dim, n_times, N_traj), dtype=np.complex128)
    
    jumps_IC = np.zeros((n_times, N_traj), dtype=np.int32)
    jumps_ISC = np.zeros((n_times, N_traj), dtype=np.int32)
    
    for traj in prange(N_traj):
        np.random.seed(seeds[traj])
        psi = psi_initial.copy()
        
        # Salvataggio t=0
        for i in range(N_dim):
            for j in range(N_dim):
                rho_traj[i, j, 0, traj] = psi[i] * np.conj(psi[j])

        # Time evolution loop
        for step in range(1, n_times):
            # 1. Evoluzione Deterministica
            psi = U_site @ psi

            # ==========================================
            # 2. CANALE 1: Internal Conversion (IC)
            # ==========================================
            v1_IC = M1_IC @ psi
            P1_IC = np.real(np.vdot(v1_IC, v1_IC))
            
            r_IC = np.random.rand()
            if r_IC < P1_IC:
                # Salto avvenuto!
                psi = v1_IC / np.sqrt(P1_IC) # Normalizzazione esatta
                jumps_IC[step, traj] = 1 
            else:
                # Nessun salto, la funzione d'onda si aggiorna (State Diffusion / No-jump dynamics)
                psi = M0_IC @ psi
                psi = psi / np.linalg.norm(psi) # Normalizzazione fondamentale!

            # ==========================================
            # 3. CANALE 2: Intersystem Crossing (ISC)
            # ==========================================
            v1_ISC = M1_ISC @ psi
            P1_ISC = np.real(np.vdot(v1_ISC, v1_ISC))
            
            r_ISC = np.random.rand()
            if r_ISC < P1_ISC:
                # Salto avvenuto!
                psi = v1_ISC / np.sqrt(P1_ISC)
                jumps_ISC[step, traj] = 1
            else:
                # Nessun salto
                psi = M0_ISC @ psi
                psi = psi / np.linalg.norm(psi)

            # 4. Salvataggio della matrice densità
            for i in range(N_dim):
                for j in range(N_dim):
                    rho_traj[i, j, step, traj] = psi[i] * np.conj(psi[j])

    return rho_traj, jumps_IC, jumps_ISC


def compute_trajectory_wf(c_IC, c_ISC, dt, N_traj, times, 
                               psi_sys_initial, U_site, 
                               phi_IC_rad, phi_ISC_rad, batch_size=1000):
    """
    Wrapper function to handle batching and random seeds before calling the JIT core.
    Adapted for the 4-level system with two independent sequential channels (IC and ISC).
    """
    # Convert objects to numpy arrays if they are QuTiP Qobjs
    U_site_np = U_site.full() if hasattr(U_site, 'full') else np.array(U_site, dtype=complex)
    psi_sys_initial_np = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else np.array(psi_sys_initial, dtype=complex)
    
    if psi_sys_initial_np.ndim > 1:
        psi_sys_initial_np = psi_sys_initial_np.flatten()
        
    n_times = len(times)
    N_dim = len(psi_sys_initial_np) # Automatically resolves to 4
    
    # Generate the separated Kraus Operators for IC and ISC
    M0_IC, M1_IC, M0_ISC, M1_ISC = generate_kraus_operators_separated(
        c_IC, c_ISC, dt, phi_IC_rad, phi_ISC_rad
    )

    # Pre-generate seeds for reproducible parallel execution
    rng_seeds = np.random.RandomState(42)
    all_seeds = rng_seeds.randint(0, 2**30, size=N_traj)

    # Pre-allocate the complete array for all trajectories
    rho_tot_all = np.zeros((N_dim, N_dim, n_times, N_traj), dtype=np.complex128)
    
    # Pre-allocate arrays to sum up jumps across all batches for both channels
    total_jumps_IC = np.zeros(n_times, dtype=np.int64)
    total_jumps_ISC = np.zeros(n_times, dtype=np.int64)

    N_done = 0
    n_batches = int(np.ceil(N_traj / batch_size))

    # Batch execution to manage memory footprint
    for b in range(n_batches):
        N_batch = min(batch_size, N_traj - N_done)
        seeds_b = all_seeds[N_done : N_done + N_batch]

        # Call the new Numba JIT compiled core with separated sequential operators
        rho_batch, j_IC_batch, j_ISC_batch = compute_trajectory_wf_core_separated(
            psi_sys_initial_np, U_site_np, 
            M0_IC, M1_IC, M0_ISC, M1_ISC,
            N_batch, n_times, seeds_b
        )

        # Store batch results
        rho_tot_all[:, :, :, N_done : N_done + N_batch] = rho_batch
        
        # Accumulate the jump counts for this batch by summing across the trajectory axis
        total_jumps_IC += np.sum(j_IC_batch, axis=1)
        total_jumps_ISC += np.sum(j_ISC_batch, axis=1)

        N_done += N_batch
        
        # Free memory for the next batch
        del rho_batch, j_IC_batch, j_ISC_batch

    return rho_tot_all, total_jumps_IC, total_jumps_ISC

# ======================================
# Main Loop for varying dt and N_{traj}
# ======================================

# ===================
# System's Parameters
# ===================
np.random.seed(1) # Always use the same seed 
N_site = 4        # Now 4 levels (0, 1, 2, 3)
E0 = 0.0          # Ground state
E1 = 1.5          # First excited (Internal Conversion target)
E2 = 2.0          # Second excited (Initial excitation)
E3 = 1.8          # Third state (Intersystem Crossing target / triplet)

H_Sys = np.diag([E0, E1, E2, E3])  # System Hamiltonian (4x4)

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
gamma_IC = 0.1   # Gamma rate for the Internal Convertion
gamma_ISC = 0.08   # Gamma rate for the Intersystem Crossing

# Lindblad Rates list
gamma_k = [gamma_IC, gamma_ISC]

# ========================================
# Initial wave function and density matrix
# ========================================

# ======
# System
# ======
pop_0 = np.sqrt(1 - 10**(-3)) # Population in |0> is close to 1, but not exactly 1 to avoid numerical issues
pop_1 = 0.0
pop_2 = np.sqrt(10**(-3))
pop_3 = 0.0

psi_sys_initial = np.array([pop_0, pop_1, pop_2, pop_3], dtype=complex) # System is initialized in |0> mainly and|2> perturbatively
rho_sys_initial = np.outer(psi_sys_initial, psi_sys_initial.conj()) # Density matrix of the system at t=0     

# =======
# Ancilla
# =======
# Single ancilla state |0>
psi_anc_single = np.array([1.0, 0.0], dtype=complex)

# Double ancilla state |0_a> @ |0_b> (Tensor product)
psi_anc_double = np.kron(psi_anc_single, psi_anc_single) 

# Density matrix for the combined environment (4x4 matrix)
rho_anc_double = np.outer(psi_anc_double, psi_anc_double.conj())

# =========
# Projectors
# =========
# Only define the ones needed for the Hamiltonians and Jump Operators
P12 = np.zeros((4,4), dtype=complex); P12[1,2] = 1.0
P21 = np.zeros((4,4), dtype=complex); P21[2,1] = 1.0
P32 = np.zeros((4,4), dtype=complex); P32[3,2] = 1.0
P23 = np.zeros((4,4), dtype=complex); P23[2,3] = 1.0

# ======================
# Lindblad Jump Operators
# ======================
L_k = [P12, P32]

# ============
# Calculation
# ============

# Bash argument reading for the intermediate angle
if len(sys.argv) > 1:
    phi_deg = float(sys.argv[1]) 
    bash_mode = sys.argv[2] if len(sys.argv) > 2 else "unknown" 
else:
    phi_deg = 90.0 # Default to SD
    bash_mode = "local_test"

# Assuming the same measurement angle for both channels. 
# You can split this into phi_IC and phi_ISC if needed.
phi_rad = np.radians(phi_deg)

# ======================
# Output directory setup
# ======================
results_dir = "../../Results/Data/Complete_rho/"
os.makedirs(results_dir, exist_ok=True)
BATCH_SIZE = 1000

def _make_fname_npz(results_dir, phi_deg, dt, N_traj):
    dt_str = f"{dt:.6f}".replace(".", "p")
    phi_str = f"{phi_deg:.4f}".replace(".", "p") 
    return os.path.join(results_dir, f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}.npz")

print(f"Starting 4-level computation for phi = {phi_deg:.4f}")

for dt_idx, dt in enumerate(dt_list):
    times = times_list[dt_idx]
    steps = steps_list[dt_idx]
    
    # Calculate interaction strengths dynamically for the current dt
    c_IC  = np.sqrt(gamma_IC / dt)
    c_ISC = np.sqrt(gamma_ISC / dt)

    # Build up the deterministic components
    H_site, H_coll, H_tot = complete_Hamiltonian(H_Sys, c_IC, c_ISC, P12, P21, P32, P23, sp, sm)
    U_tot, U_diag, w, V = evolution_operator(H_tot, dt, method='diagonalization', hermitian=True)
    U_site, U_diag_site, w_site, V_site = evolution_operator(H_site, dt, method='diagonalization', hermitian=True)

    # 1. Lindblad Reference
    rho_list_lindblad, V_lindblad, W_lindblad = Lindblad_evo(
        rho_sys_initial, H_site, gamma_k, L_k, times, method="diagonal", vectorized=False
    )
    
    # 2. Isolated System
    rho_traj_isolated = compute_trajectory_wf_isolated(times, psi_sys_initial, U_site)
    
    # 3. Deterministic Dynamics (Trace over both ancillae)
    rho_trace = compute_trace_ancilla_density(rho_sys_initial, rho_anc_double, U_diag, V, times)

    # 4. Stochastic Trajectories (Monte Carlo)
    rho_tot_all, jumps_IC, jumps_ISC = compute_trajectory_wf(
        c_IC, c_ISC, dt, N_traj, times,
        psi_sys_initial, U_site,
        phi_IC_rad=phi_rad, phi_ISC_rad=phi_rad, batch_size=BATCH_SIZE
    )
    
    # Save everything correctly
    fname_npz = _make_fname_npz(results_dir, phi_deg, dt, N_traj)

    np.savez_compressed(
        fname_npz,
        rho_tot_all=rho_tot_all,
        jumps_IC=jumps_IC,           # Saved separately to track IC events
        jumps_ISC=jumps_ISC,         # Saved separately to track ISC events
        rho_trace=rho_trace,
        rho_list_lindblad=rho_list_lindblad,
        V_lindblad=V_lindblad,
        W_lindblad=W_lindblad,
        rho_traj_isolated=rho_traj_isolated,
        phi=phi_rad, dt=dt, N_traj=N_traj,
        times=times, steps=steps, 
        c_IC=c_IC, c_ISC=c_ISC       # Saved both coupling constants
    )

    print(f"Saved -> {os.path.basename(fname_npz)}")
    del rho_tot_all, jumps_IC, jumps_ISC, rho_list_lindblad, rho_traj_isolated, rho_trace

print("\n" + "=" * 40)
print("COMPUTATION COMPLETED (4-Level System)!")
print("Results saved for:")
print(f"  - Angle (phi): {phi_deg} degrees ({phi_rad:.4f} rad)")
print(f"  - {len(dt_list)} dt values: {dt_list}")
print(f"  - Fixed N_traj: {N_traj}")
print("=" * 40)




