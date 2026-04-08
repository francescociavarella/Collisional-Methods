#!/usr/bin/env python
# coding: utf-8

# ## Libraries & Function

# In[ ]:


import numpy as np
from scipy.linalg import expm
from qutip import *
import numba
from numba import njit, prange
import os
import time


# In[2]:


sz = np.array(([[1,0], [0,-1]]), dtype=complex); sx = np.array(([[0,1],[1,0]]), dtype=complex); sy = np.array(([[0,-1j],[1j,0]]), dtype=complex) ; sp = np.array(([[0,1],[0,0]]), dtype=complex) ; sm = np.array(([[0,0],[1,0]]), dtype=complex)


# In[3]:


#funzione per plottare in LaTex delle matrici
def array_to_latex(array, real = False, array_name = None):
    array = array.real if real else array
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    if array_name != None:
        display(Math(array_name+r' = \begin{bmatrix}'+matrix+r'\end{bmatrix}'))
    else:
        display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))


# ### Hamiltonians and U operator

# In[4]:


def system_Hamiltonian(N_site, E, V_array, mode="complete"):
    """
    Build up of the System's Hamiltonian for the complete basis (ground & excited states) or only excited states.

    Method: - "complete"-> complete basis (ground & excited states)
            - "exc"-> excited basis (only excited states)

    Parameters: - E: Float, System's Site Energies (randomly generated)
                - V_array: Float, Hopping Potential
                - N_site : Int, Number of Sites

    Returns : System's Hamiltonian as Numpy array
    """
    # -----------------------------------------------------
    # Build symmetric matrix from upper triangular elements
    # -----------------------------------------------------
    V_matrix = np.zeros((N_site, N_site))
    idx = 0  # runs over V_array
    for i in range(N_site):
        for j in range(i+1, N_site):
            V_matrix[i, j] = V_array[idx]
            V_matrix[j, i] = V_array[idx]  # Symmetric
            idx += 1

    # -------------------------
    # Only Excited States Basis
    # -------------------------
    if mode == "exc":   
        H_sys = np.zeros((N_site, N_site), dtype=complex)
        for i in range(N_site):
            H_sys[i, i] = E[i]

        for i in range(N_site):
            for j in range(N_site):
                if i != j:
                    H_sys[i, j] = V_matrix[i, j]
        return H_sys

    # --------------
    # Complete Basis 
    # --------------    
    elif mode == "complete":   
        H_sys = np.zeros((2**N_site, 2**N_site), dtype='complex')

        for i in range(N_site):
            H_i = (E[i]/2) * (tensor(identity(2**i), identity(2)-sigmaz(), identity(2**(N_site-i-1))))
            H_sys += H_i.full()

            for j in range(i+1, N_site):
               H_ij = V_matrix[i, j]/2 * (tensor(identity(2**i), sigmax(), identity(2**(j-i-1)), sigmax(), identity(2**(N_site-j-1))) + tensor(identity(2**i), sigmay(), identity(2**(j-i-1)), sigmay(), identity(2**(N_site-j-1))))
               H_sys += H_ij.full()

        return H_sys

    else:
        raise ValueError("mode : 'complete' or 'exc'")


# In[5]:


def interaction_Hamiltonian(N_site, c_CM, g_x, g_z):   
    """
    Build up of the Hamiltonian of Interaction for the Collision System - Ancilla in both Quantum Jump and Diffusive Limit

    Parameters: - N_site : int, Number of Sites
                - c_CM : list, Interaction Forces for the System - Ancilla intercation/collsion
                - g_x : float, parametr for the sigma x interaction
                - g_z : float, parametr for the sigma z interaction

    Returns : Hamiltonian of Interaction as Qutip object
    """
    dim_tot = 2**(2 * N_site)
    H_int = np.zeros((dim_tot, dim_tot), dtype=complex)   #inizialization

    for j in range(N_site):
        # Create fresh lists for Z and X terms
        op_list_z = [qeye(2) for _ in range(2 * N_site)]
        op_list_x = [qeye(2) for _ in range(2 * N_site)]  

       # Z_sys tensor Z_anc
        op_list_z[j] = sigmaz()             # System j
        op_list_z[N_site + j] = sigmaz()    # Ancilla j

        # Z_sys tensor X_anc
        op_list_x[j] = sigmaz()             # System j
        op_list_x[N_site + j] = sigmax()    # Ancilla j 

        # Tensor product between the element of the list
        term_z = tensor(op_list_z)
        term_x = tensor(op_list_x)

        H_term = c_CM[j] * (g_z * term_z + g_x * term_x)  

        H_int += H_term.full()

    return H_int


# In[6]:


def hamiltonian_N_ancillas(N_site, E, V_array, c_CM, g_x, g_z):
    """
    Generation of 3 Hamiltonians for the collision model with N ancillas:
                - H_system : system Hamiltonian
                - H_collision : interaction Hamiltonian with N ancillas
                - H_tot : complete Hamiltonian (system + collision)

    Parameters: - E: Float, System's Site Energies (randomly generated)
                - V_array: Float, Hopping Potential
                - N_site : int, Number of Sites
                - c_CM : list, Interaction Forces for the System - Ancilla intercation/collsion

    Returns : H_system, H_collision, H_tot
    """

    H_collision = interaction_Hamiltonian(N_site, c_CM, g_x, g_z) 

    H_system = system_Hamiltonian(N_site, E, V_array, mode="complete")
    H_system = H_system.full() if hasattr(H_system, "full") else H_system

    dim_anc = 2**N_site
    Id_ancillas = np.eye(dim_anc, dtype=complex)
    H_system_expanded = np.kron(H_system, Id_ancillas)  #expand H_sys in the total space

    H_tot = H_system_expanded + H_collision

    return H_system, H_collision, H_tot


# In[7]:


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


# ### Lindblad functions

# In[8]:


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


# In[9]:


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


# ### Isolated system

# In[10]:


@njit(cache=True)
def _compute_trajectory_isolated_core_general(psi_initial, U_site, projectors, n_times):
    """
    Core evolution loop - general for any number of sites (Populations only)
    """
    N_site = projectors.shape[0]
    pop_traj = np.zeros((N_site, n_times), dtype=np.float64)

    # Initial populations
    for site in range(N_site):
        P_psi = projectors[site] @ psi_initial
        pop_traj[site, 0] = np.real(np.vdot(psi_initial, P_psi))

    # Evolution
    psi = psi_initial.copy()
    for step in range(1, n_times):
        psi = U_site @ psi

        for site in range(N_site):
            P_psi = projectors[site] @ psi
            pop_traj[site, step] = np.real(np.vdot(psi, P_psi))

    return pop_traj

def compute_trajectory_wf_isolated(N_site, times, projectors, psi_sys_initial, U_site):
    """
    Optimized isolated system evolution with Numba (works for any N_site).
    """
    # Convert to NumPy
    U_site_np = U_site.full() if hasattr(U_site, 'full') else np.array(U_site, dtype=complex)
    psi_initial_np = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else np.array(psi_sys_initial, dtype=complex)

    # Flatten if needed
    if psi_initial_np.ndim > 1:
        psi_initial_np = psi_initial_np.flatten()

    # Times 
    n_times = len(times)

    # JIT-compiled evolution
    pop_traj_isolated = _compute_trajectory_isolated_core_general(psi_initial_np, U_site_np, projectors, n_times)

    return pop_traj_isolated


# ### Collisional Method functions

# #### Evolution with $ U_{complete} $ and then trace on the ancilla

# In[11]:


@njit(cache=True)
def _compute_trace_ancilla_core_general(rho_sys, rho_anc, U_step, U_step_dag, projectors, n_times, dim_sys, dim_anc, N_site):
    """
    Core computation optimized with Numba - generalized for N sites
    """
    pops_complete = np.zeros((N_site, n_times), dtype=np.float64)

    # Initial state - all sites
    for site in range(N_site):
        pops_complete[site, 0] = np.real(np.trace(projectors[site] @ rho_sys))

    # Time Evolution
    for t in range(1, n_times):
        # 1: Expansion
        rho_tot = np.kron(rho_sys, rho_anc)

        # 2: Evolution
        rho_tot = U_step @ rho_tot @ U_step_dag

        # 3: Partial Trace
        rho_tot_reshaped = rho_tot.reshape(dim_sys, dim_anc, dim_sys, dim_anc)

        # Manual trace
        rho_sys = np.zeros((dim_sys, dim_sys), dtype=np.complex128)
        for i in range(dim_sys):
            for j in range(dim_sys):
                for k in range(dim_anc):
                    rho_sys[i, j] += rho_tot_reshaped[i, k, j, k]

        # 4: Store populations - all sites
        for site in range(N_site):
            pops_complete[site, t] = np.real(np.trace(projectors[site] @ rho_sys))

    return pops_complete


def compute_trace_ancilla(rho_sys_initial, rho_anc_single, U_diag, V, times, projectors, N_site):
    """
    Evolution with complete collisional hamiltonian and then trace on the Ancilla degrees of freedom; corresponds
    to an average over infinite number of trajectories.
    """

    rho_anc = (tensor([rho_anc_single for _ in range(N_site)])).full() #for N ancilla

    # Convert to numpy
    rho_sys = rho_sys_initial.full() if hasattr(rho_sys_initial, 'full') else rho_sys_initial.copy()

    #Times 
    n_times = len(times)

    # Dimensions
    dim_sys = rho_sys.shape[0]
    dim_anc = rho_anc.shape[0]

    # Evolution operator
    V_np = V.full() if hasattr(V, 'full') else V
    U_diag_np = U_diag.full() if hasattr(U_diag, 'full') else U_diag
    U_step = V_np @ U_diag_np @ V_np.conj().T
    U_step_dag = U_step.conj().T

    # Call JIT-compiled function
    pops_complete = _compute_trace_ancilla_core_general(rho_sys, rho_anc, U_step, U_step_dag, 
                                                  projectors, n_times, dim_sys, dim_anc, N_site )

    return pops_complete


# #### Single trajectory Quantum Jump

# In[12]:


@njit
def sigma_xyz_expectation_value(psi, Sx_1, Sx_2, Sy_1, Sy_2, Sz_1, Sz_2):
    """
    Function to calculate expectation value of <sigmaz> for both the site at every time step

    Parameters: -psi : nparray, wf at time t of the complete systems (wf site1 otimes wf site2)
                - Sx_1 : nparray, operator Sx on site 1, sx otimes identity
                - Sx_2 : nparray, operator Sx on site 2, identity otimes sx 
                - Sy_1 : nparray, operator Sy on site 1, sy otimes identity
                - Sy_2 : nparray, operator Sy on site 2, identity otimes sy
                - Sz_1 : nparray, operator Sz on site 1, sz otimes identity
                - Sz_2 : nparray, operator Sz on site 2, identity otimes sz 

    returns : - S_x_site_1, float expectation value of <sigmax> on site 1 at time t
              - S_x_site_2, float expectation value of <sigmax> on site 2 at time t
              - S_y_site_1, float expectation value of <sigmay> on site 1 at time t
              - S_y_site_2, float expectation value of <sigmay> on site 2 at time t
              - S_z_site_1, float expectation value of <sigmaz> on site 1 at time t
              - S_z_site_2, float expectation value of <sigmaz> on site 2 at time t
    """


    S_x_site_1 = np.real(np.vdot(psi, Sx_1 @ psi))
    S_x_site_2 = np.real(np.vdot(psi, Sx_2 @ psi))

    S_y_site_1 = np.real(np.vdot(psi, Sy_1 @ psi))
    S_y_site_2 = np.real(np.vdot(psi, Sy_2 @ psi))

    S_z_site_1 = np.real(np.vdot(psi, Sz_1 @ psi))
    S_z_site_2 = np.real(np.vdot(psi, Sz_2 @ psi))

    return S_x_site_1, S_x_site_2, S_y_site_1, S_y_site_2, S_z_site_1, S_z_site_2



# In[13]:


@njit
def compute_Bloch_Sphere(psi):
    """
    Function to extract the expectation value of the Bloch's Sphere components <sigmax>, <sigmay>, <sigmaz> associated to the 2x2 space of only excited states,
    with base |10> (exc. on site 1, -z) & |01> (exc. on site 2, +z) 

    Parameters: -psi : nparray, wf at time t of the complete systems (wf site1 otimes wf site2)

    Returns: - r_x_step, float expectation value of x component <sigmax>
             - r_y_step, float expectation value of y component <sigmay>
             - r_z_step, float expectation value of z component <sigmaz>
    """

    # wf element
    c_01 = psi[1] ; c_01_conj = np.conj(c_01) # site 2 
    c_10 = psi[2] ; c_10_conj = np.conj(c_10) # site 1

    # Blochs components
    r_x_step = 2 * np.real(c_10 * c_01_conj)

    r_y_step = -2 * np.imag(c_10 * c_01_conj)

    r_z_step = np.abs(c_01)**2 - np.abs(c_10)**2 

    return r_x_step, r_y_step, r_z_step



# In[14]:


def M_operators_list(dt, c_CM, g_z, g_x, g_0, g_1, N_site):
    """
    Calculate the two generic M0 and M1 operatos for the WF evolution

    Parameters: - dt : float, Time Step
                - c_CM : array, Collisional model Coefficients
                - N_site : int, Number of Sites
                - times : array, Time array
                - projectors : list/array, Projection Operators [P_10, P_01, ...]

    Returns: - M0_list: nparray, list of operators M0 for every site
             - M1_list: nparray, list of operators M1 for every site
    """
    # parenthesis    
    par_0 = g_0 * g_z + g_1 * g_x
    par_1 = g_0 * g_x - g_1 * g_z

    M0_list = []
    M1_list = []

    for site_idx in range(N_site):

        cos_cdt = np.cos(c_CM[site_idx] * dt)
        sin_cdt = np.sin(c_CM[site_idx] * dt)

        # M0
        m0_site = (g_0 * cos_cdt * qeye(2) - 1j * par_0 * sin_cdt * sigmaz())

        # M1
        m1_site = (g_1 * cos_cdt * qeye(2) - 1j * par_1 * sin_cdt * sigmaz())

        # Build up of single site operators
        op_0_list = [qeye(2) for _ in range(N_site) ]
        op_1_list = [qeye(2) for _ in range(N_site) ]

        op_0_list[site_idx] = m0_site 
        op_1_list[site_idx] = m1_site 

        M0_list.append((tensor(op_0_list)).full())
        M1_list.append((tensor(op_1_list)).full())

    return np.array(M0_list, dtype=np.complex128), np.array(M1_list, dtype=np.complex128)



# In[15]:


@njit(parallel=True, cache=True, fastmath=True)
def compute_trajectory_wf_core(psi_sys_initial, U_site, M0_list, M1_list, projectors, projectors_cohe,
                               N_traj, N_site, n_times, Pr_0_site, Pr_1_site, seeds):
    """
    Core computation optimized with Numba (JIT compiled with parallelization).
    Alloca (n_times, N_traj) — chiamare con N_traj = BATCH_SIZE.
    """
    pop_traj = np.zeros((N_site, n_times, N_traj), dtype=np.float64)
    coh_traj = np.zeros((N_site, n_times, N_traj), dtype=np.complex128)

    # Inizializzazione al tempo t=0 per tutte le traiettorie
    for site in range(N_site):
        pop_0 = np.real(np.vdot(psi_sys_initial, projectors[site] @ psi_sys_initial))
        coh_0 = np.vdot(psi_sys_initial, projectors_cohe[site] @ psi_sys_initial)
        pop_traj[site, 0, :] = pop_0
        coh_traj[site, 0, :] = coh_0

    for traj in prange(N_traj):
        np.random.seed(seeds[traj])
        psi = psi_sys_initial.copy()

        for step in range(1, n_times):
            psi = U_site @ psi

            for j in range(N_site):
                M0 = M0_list[j]
                M1 = M1_list[j]
                Pr_0 = Pr_0_site[j]

                r = np.random.rand()
                if r < Pr_0:
                    psi = M0 @ psi
                else:
                    psi = M1 @ psi

            psi = psi / np.linalg.norm(psi)

            # Estrazione dei dati per la singola traiettoria al tempo corrente
            for site in range(N_site):
                pop_traj[site, step, traj] = np.real(np.vdot(psi, projectors[site] @ psi))

                # Calcolo della coerenza complessa usando lo stato 'psi' evoluto
                coh_traj[site, step, traj] = np.vdot(psi, projectors_cohe[site] @ psi)

    return pop_traj, coh_traj

def compute_trajectory_wf(dt, c_CM, g_z, g_x, g_0, g_1, N_traj, N_site, times,
                           projectors, projectors_cohe, psi_sys_initial, U_site, Pr_0_site, Pr_1_site,
                           batch_size=1000):

    U_site = U_site.full() if hasattr(U_site, 'full') else np.array(U_site, dtype=complex)
    psi_sys_initial = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else np.array(psi_sys_initial, dtype=complex)
    if psi_sys_initial.ndim > 1:
        psi_sys_initial = psi_sys_initial.flatten()

    # Conversione operatori in numpy array per Numba
    projectors = np.array([P.full() if hasattr(P, 'full') else np.array(P, dtype=complex) for P in projectors])
    projectors_cohe = np.array([P.full() if hasattr(P, 'full') else np.array(P, dtype=complex) for P in projectors_cohe])

    n_times  = len(times)
    M0_list, M1_list = M_operators_list(dt, c_CM, g_z, g_x, g_0, g_1, N_site)

    rng_seeds = np.random.RandomState(42)
    all_seeds = rng_seeds.randint(0, 2**30, size=N_traj)

    # 1. PRE-ALLOCAZIONE (Adesso le coerenze sono native complesse per salvare reale+immaginario)
    pop_00 = np.zeros((n_times, N_traj), dtype=np.float64)
    pop_11 = np.zeros((n_times, N_traj), dtype=np.float64)
    coh_01_10 = np.zeros((n_times, N_traj), dtype=np.complex128)
    coh_10_01 = np.zeros((n_times, N_traj), dtype=np.complex128)

    N_done = 0
    n_batches = int(np.ceil(N_traj / batch_size))

    for b in range(n_batches):
        N_batch = min(batch_size, N_traj - N_done)
        seeds_b = all_seeds[N_done : N_done + N_batch]

        # Chiamata al core (che già calcola le coerenze complesse intere)
        pop_b, coh_b = compute_trajectory_wf_core(
            psi_sys_initial, U_site, M0_list, M1_list, projectors, projectors_cohe,
            N_batch, N_site, n_times, Pr_0_site, Pr_1_site, seeds_b)

        # 2. INSERIMENTO NEGLI ARRAY TRAMITE SLICING
        pop_00[:, N_done : N_done + N_batch] = pop_b[0, :, :]
        pop_11[:, N_done : N_done + N_batch] = pop_b[1, :, :]

        # Salvataggio delle due coerenze complete (indice 0 per 01_10, indice 1 per 10_01)
        coh_01_10[:, N_done : N_done + N_batch] = coh_b[0, :, :]
        coh_10_01[:, N_done : N_done + N_batch] = coh_b[1, :, :]

        N_done += N_batch

        # Pulizia memoria del batch
        del pop_b, coh_b

    return pop_00, pop_11, coh_01_10, coh_10_01


# ## Main Loop for varying $ dt $ and $ N_{traj} $

# In[16]:


# ===================
# System's Parameters
# ===================
np.random.seed(1) # always use the same seed 
N_site = 2    # Number of sites
V_array = [1.0]    # Hopping Potential : V12, V13, ... V1N_site, V23, ..., V2N_site, V34...V3N_site
E = 1.5 + np.random.randn(N_site)*0.1     #random inizialization of the system energies

# =========================
# Time Evolution Parameters
# =========================
dt_list = [0.01]     # change : time step
tf = 100.0    # Final Time
steps_list = [ int(tf / dt_list[i]) for i in range (len(dt_list)) ]
times_list = [ np.linspace(0, tf, int(steps_list[i])) for i in range(len(dt_list))]

N_traj_list = [20000]

# ===================
# Dephasing Parameter (come in MATLAB)
# ===================
# Lindblad
g_deph = 0.1   # Gamma rate
#Lindblad Rates
gamma_k = [g_deph, g_deph]

# Scaling for the collsional algorithm c = sqrt(gamma / 4dt)
c_CM_list = np.array([[np.sqrt(g_deph / (4 * dt_list[j])) for j in range(len(dt_list))] for _ in range(N_site)])  # same Coupling for the 2 sites


# In[ ]:


# ========================================
# Initial wave function and density matrix
# ========================================

# ======
# System
# ======
psi_sys_initial = tensor(basis(2, 0), basis(2, 1)) # I set the population only in site 2   

rho_sys_initial = (ket2dm(psi_sys_initial)).full()       

# =======
# Ancilla
# =======
# -----------------
# Angles definition
# -----------------

MODE = "close_to_90_deg"  # change : "normal" or "close_to_90_deg"

if MODE == "normal":
    theta_list = np.radians([90, 60, 45, 30, 0])
elif MODE == "close_to_90_deg": 
    theta_list = np.radians([0, 90, 89.9, 89.7, 89.5, 89, 88.5, 88, 87, 86])


# In[18]:


# ----------
# Projectors
# ----------
P0 = (np.eye(2, dtype=complex) + sz) / 2 # projector on |0>
P1 = (np.eye(2, dtype=complex) - sz) / 2 # projector su |1>

P_00 = np.kron(P0, P0) # |00><00|
P_01 = np.kron(P0, P1) # |01><01|
P_10 = np.kron(P1, P0) # |10><10|
P_11 = np.kron(P1, P1) # |11><11|
P_01_10 = np.kron(sp, sm) # |01><10|
P_10_01 = np.kron (sm, sp) # |10><01|

projectors = np.array([P_10, P_01], dtype=complex) # population for only excited states
projectors_cohe = np.array([P_01_10, P_10_01], dtype=complex) # coherences for only excited states

# ----------------------
# Lindblad Jump Operator
# ----------------------
L_1 = P_10  # projector on |10><10|
L_2 = P_01  # projector on |01><01|
L_k = [L_1, L_2]


# ### Calculation

# In[ ]:


# ======================
# Output directory setup
# ======================

if MODE == "normal":
    results_dir = "../../Results/Data/Complete_rho/normal/"
elif MODE == "close_to_90_deg":
    results_dir = "../../Results/Data/Complete_rho/close_90_deg/"

os.makedirs(results_dir, exist_ok=True)

# Traiettorie per batch nel wrapper
BATCH_SIZE = 1000

def _make_fname_npz(results_dir, theta, dt, N_traj):
    t_str  = f"{theta:.6f}".replace(".", "p")
    dt_str = f"{dt:.6f}".replace(".", "p")
    return os.path.join(results_dir, f"result_theta{t_str}_dt{dt_str}_Ntraj{N_traj}.npz")

def _already_done_npz(results_dir, theta, dt, N_traj):
    return os.path.isfile(_make_fname_npz(results_dir, theta, dt, N_traj))

# =====================
# Main computation loop
# =====================
print("Starting computation for different theta, dt and N_traj")

for theta_idx, theta in enumerate(theta_list):

    print("=" * 50)
    print(f"THETA = {theta:.4f} rad = {np.degrees(theta):.2f}° ({theta_idx+1}/{len(theta_list)})")
    print("=" * 50)

    phi = theta - np.pi/2
    g_z = np.cos(theta)
    g_x = np.sin(theta)
    g_0 = np.cos(phi / 2)
    g_1 = np.sin(phi / 2)

    print(f"phi = {phi:.4f} rad, g_z = {g_z:.4f}, g_x = {g_x:.4f}, g_0 = {g_0:.4f}, g_1 = {g_1:.4f}")

    psi_anc_single = (g_0 * basis(2,0) + g_1 * basis(2,1))
    rho_anc_single = ket2dm(psi_anc_single)

    Pr_0_list = np.zeros((N_site, len(dt_list)))
    Pr_1_list = np.zeros((N_site, len(dt_list)))

    for j in range(len(dt_list)):
        for i in range(N_site):
            Pr_0_list[i,j] = (g_0**2) * (np.cos(c_CM_list[i,j]*dt_list[j]))**2 + ((g_0 * g_z + g_1 * g_x)**2) * (np.sin(c_CM_list[i,j]*dt_list[j]))**2
            Pr_1_list[i,j] = (g_1**2) * (np.cos(c_CM_list[i,j]*dt_list[j]))**2 + ((g_0 * g_x - g_1 * g_z)**2) * (np.sin(c_CM_list[i,j]*dt_list[j]))**2

    for dt_idx, dt in enumerate(dt_list):

        print("=" * 40)
        print(f"dt = {dt:.4f} ({dt_idx+1}/{len(dt_list)})")

        times     = times_list[dt_idx]
        steps     = steps_list[dt_idx]
        c_CM      = c_CM_list[:, dt_idx]
        Pr_0_site = Pr_0_list[:, dt_idx]
        Pr_1_site = Pr_1_list[:, dt_idx]

        print("Recalculating Hamiltonians")
        H_site, H_coll, H_tot = hamiltonian_N_ancillas(N_site, E, V_array, c_CM, g_x, g_z)
        U_tot, U_diag, w, V = evolution_operator(H_tot, dt, method='diagonalization', hermitian=True)
        U_diag_dag = U_diag.conj().T; V_dag = V.conj().T
        H_system = system_Hamiltonian(N_site, E, V_array, mode="complete")
        U_site, U_diag_site, w_site, V_site = evolution_operator(H_system, dt, method='diagonalization', hermitian=True)

        print("Computing Lindblad")
        start_time = time.time()
        rho_list_lindblad, V_lindblad, W_lindblad = Lindblad_evo(rho_sys_initial, H_system, gamma_k, L_k, times, method="diagonal", vectorized=False)
        print(f"Completed in {time.time() - start_time:.2f}s")

        print("Computing Trajectory Isolated")
        start_time = time.time()
        pop_traj_isolated = compute_trajectory_wf_isolated(N_site, times, projectors, psi_sys_initial, U_site)
        print(f"Completed in {time.time() - start_time:.2f}s")

        print("Computing Trace Ancilla")
        start_time = time.time()
        pops_trace = compute_trace_ancilla(rho_sys_initial, rho_anc_single, U_diag, V, times, projectors, N_site)
        print(f"Completed in {time.time() - start_time:.2f}s")

        for N_traj in N_traj_list:

            print("-" * 40)
            print(f"N_traj = {N_traj}")

            if _already_done_npz(results_dir, theta, dt, N_traj):
                print("  Already done, skipping.")
                continue

            print("Computing Trajectory WF (All Distributions)")
            start_time = time.time()

            pop_00, pop_11, coh_01_10, coh_10_01 = compute_trajectory_wf(
                dt, c_CM, g_z, g_x, g_0, g_1, N_traj, N_site, times,
                projectors, projectors_cohe, psi_sys_initial, U_site, Pr_0_site, Pr_1_site,
                batch_size=BATCH_SIZE)

            print(f"Completed in {time.time() - start_time:.2f}s")

            fname_npz = _make_fname_npz(results_dir, theta, dt, N_traj)

            np.savez_compressed(
                fname_npz,
                # 1. Dati Raw Traiettorie (Distribuzioni)
                pop_00=pop_00,
                pop_11=pop_11,
                coh_01_10=coh_01_10, # Ora è un array complex128 completo
                coh_10_01=coh_10_01, # Ora è un array complex128 completo

                # 2. Baseline Analitiche e Traccia
                pops_trace=pops_trace,
                rho_list_lindblad=rho_list_lindblad,
                V_lindblad=V_lindblad,
                W_lindblad=W_lindblad,

                # 3. Dati Sistema Isolato
                pop_traj_isolated=pop_traj_isolated,

                # 4. Parametri
                theta=theta, phi=phi, dt=dt, N_traj=N_traj,
                times=times, steps=steps, c_CM=c_CM,
                g_z=g_z, g_x=g_x, g_0=g_0, g_1=g_1)

            size_mb = os.path.getsize(fname_npz) / (1024**2)
            print(f"  Saved → {os.path.basename(fname_npz)}  ({size_mb:.2f} MB)")

            # Pulizia immediata della RAM 
            del pop_00, pop_11, coh_01_10, coh_10_01

        del rho_list_lindblad, pop_traj_isolated

print("\n" + "=" * 40)
print("COMPUTATION COMPLETED!")
print(f"Results saved for:")
print(f"  - {len(theta_list)} theta values: {[f'{t:.4f}' for t in theta_list]}")
print(f"  - {len(dt_list)} dt values: {dt_list}")
print(f"  - {len(N_traj_list)} N_traj values: {N_traj_list}")
print("=" * 40)


# In[ ]:




