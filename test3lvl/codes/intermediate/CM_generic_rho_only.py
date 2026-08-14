#!/usr/bin/env python
# coding: utf-8

# ## Libraries & Function

import numpy as np
from scipy.linalg import expm
from qutip import *
import numba
from numba import njit, prange
import os
import time
import sys
import h5py

sz = np.array(([[1,0], [0,-1]]), dtype=complex); sx = np.array(([[0,1],[1,0]]), dtype=complex); sy = np.array(([[0,-1j],[1j,0]]), dtype=complex) ; sm = np.array(([[0.0, 1.0],[0.0,0.0]]), dtype=complex) ; sp = np.array(([[0.0,0.0],[1.0,0.0]]), dtype=complex)

# ============================
# Numba thread control
# ============================
# IMPORTANTE: imposta esplicitamente il numero di thread Numba in base ai core
# reali della macchina. Macchina target: 36 core, 251 GB RAM (nproc / free -h).
# Il clamp con numba.config.NUMBA_NUM_THREADS evita errori se lo script gira
# per sbaglio su una macchina/container con meno core disponibili.
N_THREADS = min(
    int(os.environ.get("NUMBA_NUM_THREADS_OVERRIDE", 36)),
    numba.config.NUMBA_NUM_THREADS
)
numba.set_num_threads(N_THREADS)

# Evita oversubscription dei BLAS interni di NumPy dentro il loop parallelo di Numba:
# ogni thread Numba fa gia' il suo lavoro, se BLAS prova a sua volta a parallelizzare
# le moltiplicazioni matrice-vettore 3x3 dentro prange si crea contesa tra thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

    K0_QJ = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, cos_val]], dtype=np.complex128)

    K1_QJ = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, -1j * sin_val],
                      [0.0, 0.0, 0.0]], dtype=np.complex128)

    # Trigonometric coefficients for the intermediate basis transformation
    cos_phi2 = np.cos(phi_rad / 2.0)
    sin_phi2 = np.sin(phi_rad / 2.0)

    M0 = cos_phi2 * K0_QJ + sin_phi2 * K1_QJ
    M1 = -sin_phi2 * K0_QJ + cos_phi2 * K1_QJ

    return M0, M1


@njit(parallel=True, cache=True, fastmath=True)
def compute_trajectory_wf_core_density_strided(psi_initial, U_site, M0, M1,
                                                N_traj, n_times, seeds,
                                                save_stride, n_saved):
    """
    Core trajectory evolution optimized with Numba for an N-level system.
    Probabilities are dynamically computed at each time step using Kraus operators.

    Rispetto alla versione originale, qui la densita' viene scritta in output
    solo ogni `save_stride` passi (n_saved = numero di punti temporali salvati),
    mentre l'evoluzione stocastica avviene comunque ad ogni singolo step per
    non perdere accuratezza numerica. Questo riduce l'array di output di un
    fattore save_stride mantenendo la fisica intatta.

    jump_records invece resta ad ogni step (int32, costo trascurabile) cosi'
    da non perdere informazione sui salti quantistici tra un salvataggio e l'altro.
    """
    N_dim = len(psi_initial)

    # Pre-allocate array for all trajectories: shape (N_dim, N_dim, n_saved, N_traj)
    rho_traj = np.zeros((N_dim, N_dim, n_saved, N_traj), dtype=np.complex128)

    # Pre-allocate array to record jumps at EVERY step (non sottocampionato)
    jump_records = np.zeros((n_times, N_traj), dtype=np.int32)

    # Loop over independent trajectories in parallel
    for traj in prange(N_traj):
        np.random.seed(seeds[traj])
        psi = psi_initial.copy()

        # Initialization at t=0 (sempre salvato, e' il primo punto salvato)
        for i in range(N_dim):
            for j in range(N_dim):
                rho_traj[i, j, 0, traj] = psi[i] * np.conj(psi[j])

        save_idx = 0

        # Time evolution loop (ad ogni singolo step di dt)
        for step in range(1, n_times):
            # 1. Deterministic evolution given by the isolated System Hamiltonian
            psi = U_site @ psi

            # 2. Apply Kraus operator M1 to test the jump probability
            v1 = M1 @ psi
            P1 = np.real(np.vdot(v1, v1))

            # 3. Stochastic jump Monte Carlo selection
            r = np.random.rand()
            if r < P1:
                psi = v1
                jump_records[step, traj] = 1
            else:
                psi = M0 @ psi

            # 4. State Normalization
            norm_psi = np.linalg.norm(psi)
            for i in range(N_dim):
                psi[i] = psi[i] / norm_psi

            # 5. Store the density matrix SOLO ogni save_stride step
            if step % save_stride == 0:
                save_idx += 1
                for i in range(N_dim):
                    for j in range(N_dim):
                        rho_traj[i, j, save_idx, traj] = psi[i] * np.conj(psi[j])

    return rho_traj, jump_records


def compute_trajectory_wf_streaming(c_CM, dt, N_traj, times,
                                     psi_sys_initial, U_site,
                                     phi, h5_file, batch_size=1000,
                                     save_stride=10, save_dtype=np.complex64):
    """
    Wrapper con scrittura incrementale su HDF5.

    Invece di accumulare tutti i batch in RAM (rho_tot_all completo) prima di
    salvare, ogni batch viene scritto direttamente nel dataset HDF5 gia' creato
    (resizable lungo l'asse delle traiettorie). La RAM di picco resta quindi
    legata SOLO alla dimensione di un batch, non al totale di N_traj.

    Parameters aggiuntivi rispetto alla versione originale:
    - h5_file : h5py.File gia' aperto in modalita' scrittura, con i dataset
                'rho_traj' e 'jump_records' gia' creati (vedi main).
    - save_stride : int, salva la densita' ogni save_stride step temporali.
    - save_dtype : dtype con cui castare la densita' salvata su disco
                   (complex64 dimezza lo spazio rispetto a complex128;
                   l'evoluzione interna resta sempre in complex128).
    """
    U_site_np = U_site.full() if hasattr(U_site, 'full') else np.array(U_site, dtype=complex)
    psi_sys_initial_np = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else np.array(psi_sys_initial, dtype=complex)

    if psi_sys_initial_np.ndim > 1:
        psi_sys_initial_np = psi_sys_initial_np.flatten()

    n_times = len(times)
    N_dim = len(psi_sys_initial_np)

    # Numero di punti temporali salvati: t=0 piu' un punto ogni save_stride step
    n_saved = 1 + (n_times - 1) // save_stride

    # Generate the specific Kraus Operators according to the selected mode
    M0, M1 = generate_kraus_operators(c_CM, dt, phi)

    # Pre-generate seeds for reproducible parallel execution
    rng_seeds = np.random.RandomState(42)
    all_seeds = rng_seeds.randint(0, 2**30, size=N_traj)

    dset_rho = h5_file["rho_traj"]      # shape (N_dim, N_dim, n_saved, N_traj)
    dset_jumps = h5_file["jump_records"]  # shape (n_times, N_traj)

    total_jump_counts = np.zeros(n_times, dtype=np.int64)

    N_done = 0
    n_batches = int(np.ceil(N_traj / batch_size))

    for b in range(n_batches):
        t_batch_start = time.time()
        N_batch = min(batch_size, N_traj - N_done)
        seeds_b = all_seeds[N_done: N_done + N_batch]

        rho_batch, jumps_batch = compute_trajectory_wf_core_density_strided(
            psi_sys_initial_np, U_site_np, M0, M1,
            N_batch, n_times, seeds_b, save_stride, n_saved)

        # Scrittura diretta su HDF5: cast a save_dtype solo qui, in scrittura,
        # cosi' la precisione dell'evoluzione stocastica non e' mai intaccata.
        dset_rho[:, :, :, N_done: N_done + N_batch] = rho_batch.astype(save_dtype)
        dset_jumps[:, N_done: N_done + N_batch] = jumps_batch

        total_jump_counts += np.sum(jumps_batch, axis=1)

        N_done += N_batch
        del rho_batch, jumps_batch

        elapsed = time.time() - t_batch_start
        print(f"    Batch {b+1}/{n_batches} ({N_batch} traiettorie) "
              f"completato in {elapsed:.1f} s "
              f"[{N_done}/{N_traj} traiettorie totali]", flush=True)

    return total_jump_counts, n_saved


# ======================================
# Main Loop for varying dt and N_{traj}
# ======================================

if __name__ == "__main__":

    # ===================
    # System's Parameters
    # ===================
    np.random.seed(1)  # always use the same seed
    N_site = 3  # Number of sites
    E0 = 0.0  # Energy of the ground state |0>
    E1 = 5.14  # Energy of the first excited state |1>
    E2 = 5.49  # Energy of the second excited state |2>

    H_Sys = np.diag([E0, E1, E2])  # System Hamiltonian

    # =========================
    # Time Evolution Parameters
    # =========================
    dt_list = [0.01]     # change : time step
    tf = 100.0    # Final Time
    steps_list = [int(tf / dt_list[i]) for i in range(len(dt_list))]
    times_list = [np.linspace(0, tf, int(steps_list[i])) for i in range(len(dt_list))]

    N_traj = 10000  # change number of trajectories

    # ==========================
    # Salvataggio sottocampionato
    # ==========================
    SAVE_STRIDE = 10          # salva la densita' ogni 10 step (ogni 0.1 in tempo, con dt=0.01)
    SAVE_DTYPE = np.complex64  # complex64 dimezza lo spazio su disco rispetto a complex128

    # ===================
    # Dephasing Parameter
    # ===================
    gamma_r = 0.1   # Gamma rate for the decay
    gamma_k = [gamma_r]

    # Scaling for the collisional algorithm c = sqrt(gamma / dt)
    c_CM_list = np.array([np.sqrt(gamma_r / dt_list[j]) for j in range(len(dt_list))])

    # ========================================
    # Initial wave function and density matrix
    # ========================================

    # ======
    # System
    # ======
    pop_0 = np.sqrt(1 - 10**(-3))
    pop_1 = 0.0
    pop_2 = np.sqrt(10**(-3))

    psi_sys_initial = np.array([pop_0, pop_1, pop_2], dtype=complex)
    rho_sys_initial = np.outer(psi_sys_initial, psi_sys_initial.conj())

    # =======
    # Ancilla
    # =======
    psi_anc_single = np.array([1.0, 0.0], dtype=complex)
    rho_anc_single = np.outer(psi_anc_single, psi_anc_single.conj())

    # =========
    # Projectors
    # =========
    P00 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
    P11 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
    P22 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex)
    P01 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
    P10 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    P12 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
    P21 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=complex)
    P02 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
    P20 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=complex)

    projectors = np.array([P00, P11, P22], dtype=complex)
    projectors_cohe = np.array([P01, P10, P12, P21, P02, P20], dtype=complex)

    # ======================
    # Lindblad Jump Operator
    # ======================
    L_r = P12
    L_k = [L_r]

    # ============
    # Calculation
    # ============

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
    # Con 251 GB di RAM disponibili, un batch di 10000 traiettorie (l'intero
    # N_traj in un colpo solo) occupa in RAM circa:
    #   3*3*n_saved*N_batch*16 byte (complex128, array interno pre-cast)
    #   = 3*3*1000*10000*16 byte ~= 1.44 GB
    # ben al di sotto della RAM disponibile, quindi eseguiamo tutto in un
    # unico batch per evitare l'overhead di rilanciare piu' volte il loop
    # parallelo Numba. Se in futuro N_traj crescesse molto (es. 10^6),
    # riabbassare BATCH_SIZE per restare entro un budget di RAM ragionevole.
    BATCH_SIZE = min(N_traj, 10000)

    def _make_fname_h5(results_dir, phi_deg, dt, N_traj, save_stride):
        dt_str = f"{dt:.6f}".replace(".", "p")
        phi_str = f"{phi_deg:.4f}".replace(".", "p")
        return os.path.join(
            results_dir,
            f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}_stride{save_stride}.h5"
        )

    print(f"Starting computation for phi = {phi_deg:.4f}")
    print(f"Numba threads: {numba.get_num_threads()} (impostati a N_THREADS={N_THREADS})")

    for dt_idx, dt in enumerate(dt_list):
        times = times_list[dt_idx]
        steps = steps_list[dt_idx]
        c_CM = c_CM_list[dt_idx]
        n_times = len(times)
        n_saved = 1 + (n_times - 1) // SAVE_STRIDE

        H_site, H_coll, H_tot = complete_Hamiltonian(H_Sys, c_CM, P12, P21, sp, sm)
        U_tot, U_diag, w, V = evolution_operator(H_tot, dt, method='diagonalization', hermitian=True)
        U_site, U_diag_site, w_site, V_site = evolution_operator(H_site, dt, method='diagonalization', hermitian=True)

        # Indici temporali che verranno effettivamente salvati su disco: usati
        # per allineare TUTTI i dataset (Lindblad, trace, isolated, traiettorie
        # stocastiche) sullo stesso identico asse temporale sottocampionato.
        # Cosi' nel plotting non servono piu' due assi diversi (times vs times_saved).
        save_indices = np.arange(0, n_times, SAVE_STRIDE)
        times_saved = times[save_indices]

        print("Calcolo Lindblad, isolato e traccia ancilla (economici, a piena risoluzione)...")
        # Questi tre calcoli restano a piena risoluzione temporale (n_times):
        # sono deterministici e a basso costo (nessun Monte Carlo, singola
        # matrice 3x3 evoluta linearmente), quindi non c'e' motivo di
        # sottocampionare durante il calcolo. Il sottocampionamento avviene
        # solo qui sotto, al momento di selezionare cosa scrivere su HDF5,
        # cosi' l'accuratezza numerica intermedia non e' mai intaccata.
        rho_list_lindblad_full, V_lindblad, W_lindblad = Lindblad_evo(
            rho_sys_initial, H_site, gamma_k, L_k, times, method="diagonal", vectorized=False
        )
        rho_traj_isolated_full = compute_trajectory_wf_isolated(times, psi_sys_initial, U_site)
        rho_trace_full = compute_trace_ancilla_density(rho_sys_initial, rho_anc_single, U_diag, V, times)

        # Sottocampionamento allo stesso identico stride/indici usati per le
        # traiettorie stocastiche, cosi' ogni dataset nel file HDF5 ha
        # esattamente n_saved punti lungo l'asse tempo.
        rho_list_lindblad = rho_list_lindblad_full[save_indices]       # (n_saved, 3, 3)
        rho_traj_isolated = rho_traj_isolated_full[:, :, save_indices]  # (3, 3, n_saved)
        rho_trace = rho_trace_full[:, :, save_indices]                  # (3, 3, n_saved)

        fname_h5 = _make_fname_h5(results_dir, phi_deg, dt, N_traj, SAVE_STRIDE)

        print(f"Apertura file HDF5 di output: {fname_h5}")
        with h5py.File(fname_h5, "w") as f:

            # Dataset principale: (N_dim, N_dim, n_saved, N_traj), chunk per traiettoria
            # cosi' la scrittura di ogni batch e' un blocco contiguo efficiente.
            dset_rho = f.create_dataset(
                "rho_traj",
                shape=(N_site, N_site, n_saved, N_traj),
                dtype=SAVE_DTYPE,
                chunks=(N_site, N_site, n_saved, min(BATCH_SIZE, N_traj)),
                compression="lzf",   # veloce, nessuna dipendenza extra rispetto a h5py
                shuffle=True,        # migliora il rapporto di compressione senza costo significativo
            )

            dset_jumps = f.create_dataset(
                "jump_records",
                shape=(n_times, N_traj),
                dtype=np.int32,
                chunks=(n_times, min(BATCH_SIZE, N_traj)),
                compression="lzf",
                shuffle=True,
            )

            # Metadati e risultati "economici" salvati come attributi/dataset dedicati.
            # Tutti questi dataset (rho_trace, rho_list_lindblad, rho_traj_isolated)
            # sono ora allineati sullo STESSO asse temporale sottocampionato
            # 'times' di questo file (n_saved punti) usato per rho_traj, quindi
            # nel plotting basta un solo array 'times' per tutte le serie.
            f.create_dataset("rho_trace", data=rho_trace)
            f.create_dataset("rho_list_lindblad", data=rho_list_lindblad)
            f.create_dataset("V_lindblad", data=V_lindblad)
            f.create_dataset("W_lindblad", data=W_lindblad)
            f.create_dataset("rho_traj_isolated", data=rho_traj_isolated)

            # Versioni a piena risoluzione temporale (n_times punti, asse
            # 'times_full'). Costano pochissimo extra: rho_trace_full e
            # rho_traj_isolated_full sono al piu' pochi MB anche per tf/dt
            # grandi (es. 3*3*10000*16 byte ~= 1.4 MB), e rho_list_lindblad_full
            # e' della stessa dimensione. Utili per stimare l'errore introdotto
            # dal sottocampionamento o per confronti che richiedano la massima
            # granularita' sulla dinamica deterministica.
            f.create_dataset("rho_trace_full", data=rho_trace_full)
            f.create_dataset("rho_list_lindblad_full", data=rho_list_lindblad_full)
            f.create_dataset("rho_traj_isolated_full", data=rho_traj_isolated_full)

            # 'times' salvato e' gia' l'asse sottocampionato (n_saved punti):
            # e' l'unico array temporale necessario per interpretare qualunque
            # dataset "corto" in questo file (rho_traj, rho_trace,
            # rho_list_lindblad, rho_traj_isolated, total_jumps).
            f.create_dataset("times", data=times_saved)
            f.create_dataset("saved_time_indices", data=save_indices)

            # 'times_full' e' l'asse temporale a piena risoluzione (n_times
            # punti), da usare con i dataset '_full' e con total_jumps_full /
            # jump_records.
            f.create_dataset("times_full", data=times)

            f.attrs["phi_rad"] = phi_rad
            f.attrs["phi_deg"] = phi_deg
            f.attrs["dt"] = dt
            f.attrs["N_traj"] = N_traj
            f.attrs["steps"] = steps
            f.attrs["c_CM"] = c_CM
            f.attrs["save_stride"] = SAVE_STRIDE
            f.attrs["save_dtype"] = str(SAVE_DTYPE)

            print(f"Avvio calcolo traiettorie stocastiche (N_traj={N_traj}, batch_size={BATCH_SIZE})...")
            t_start = time.time()

            total_jumps, n_saved_check = compute_trajectory_wf_streaming(
                c_CM, dt, N_traj, times,
                psi_sys_initial, U_site,
                phi=phi_rad, h5_file=f, batch_size=BATCH_SIZE,
                save_stride=SAVE_STRIDE, save_dtype=SAVE_DTYPE
            )

            # total_jumps resta a piena risoluzione temporale (n_times): e' un
            # array 1D economico (int64) e utile con la massima granularita'
            # per individuare esattamente a quale step avvengono i salti.
            f.create_dataset("total_jumps_full", data=total_jumps)
            # Versione sottocampionata (somma dei jump in ogni intervallo tra
            # due punti salvati), allineata con 'times' per un plot rapido
            # senza dover caricare total_jumps_full + times_full.
            total_jumps_saved = np.add.reduceat(total_jumps, save_indices)
            f.create_dataset("total_jumps", data=total_jumps_saved)

            t_elapsed = time.time() - t_start
            print(f"Calcolo traiettorie completato in {t_elapsed:.1f} s "
                  f"({t_elapsed/60:.1f} min)")

        print(f"Saved -> {os.path.basename(fname_h5)}")
        del rho_list_lindblad, rho_traj_isolated, rho_trace

    print("\n" + "=" * 40)
    print("COMPUTATION COMPLETED!")
    print("Results saved for:")
    print(f"  - Angle (phi): {phi_deg} degrees ({phi_rad:.4f} rad)")
    print(f"  - {len(dt_list)} dt values: {dt_list}")
    print(f"  - Fixed N_traj: {N_traj}")
    print(f"  - Save stride: {SAVE_STRIDE} (ogni {SAVE_STRIDE * dt_list[0]:.3f} in tempo)")
    print(f"  - Save dtype: {SAVE_DTYPE}")
    print("=" * 40)