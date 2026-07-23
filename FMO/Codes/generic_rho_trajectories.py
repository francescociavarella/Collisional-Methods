import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from qutip import *
import numba
from numba import njit, prange
import os
import time
import sys

sz = np.array(([[1.0,0.0], [0.0,-1.0]]), dtype=complex); sx = np.array(([[0.0,1.0],[1.0,0.0]]), dtype=complex); sy = np.array(([[0.0,-1j],[1j,0.0]]), dtype=complex) ; sm = np.array(([[0.0, 1.0],[0.0,0.0]]), dtype=complex) ; sp = np.array(([[0.0,0.0],[1.0,0.0]]), dtype=complex)

# ===================================================
# Environment SPectral Function : Drude-Lorentz Bath
# ===================================================

def check_KMS(C_func, omega_test, beta, rtol=1e-8):
    """
    Sanity check of the KMS relation C(-omega) = C(omega) * exp(-beta*omega).
    Returns the relative error (should be ~0).
    """
    lhs = C_func(-omega_test)
    rhs = C_func(omega_test) * np.exp(-beta * omega_test)
    return np.abs(lhs - rhs) / (np.abs(rhs) + 1e-30)


def bose_factor_stable(x):
    """
    Numerically stable computation of 1/(1 - exp(-x)) for any real x
    (avoids overflow for large |x|).
    """
    scalar_input = np.isscalar(x) or np.asarray(x).ndim == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)

    pos = x > 0
    out[pos] = 1.0 / (1.0 - np.exp(-x[pos]))

    neg = ~pos
    ex = np.exp(x[neg])          # x[neg] <= 0, so ex in (0,1], safe
    out[neg] = ex / (ex - 1.0)

    return out[0] if scalar_input else out


def C_drude_lorentz(omega, lam, Omega, beta, omega_tol=1e-10):
    """
    Drude-Lorentz spectral function C(omega) = 4*lam*omega*Omega/(omega^2+Omega^2) * 1/(1-exp(-omega*beta)).
    Handles the omega -> 0 limit analytically, and uses a numerically stable
    Bose factor to avoid overflow for large |omega|.
    """
    scalar_input = np.isscalar(omega) or np.asarray(omega).ndim == 0
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.empty_like(omega)

    small = np.abs(omega) < omega_tol
    out[small] = 4.0 * lam / (Omega * beta)

    w = omega[~small]
    if w.size > 0:
        lorentz = 4.0 * lam * w * Omega / (w**2 + Omega**2)
        out[~small] = lorentz * bose_factor_stable(w * beta)

    return out[0] if scalar_input else out


def lamb_shift_Lambda(omega, C_func, bound=500.0, limit=500):
    """
    Computes Lambda(omega) = (1/2pi) * P.V. Integral[ C(omega')/(omega-omega'), d omega' ]
    via scipy.integrate.quad with a Cauchy principal-value weight.
    """
    integrand = lambda wp: float(np.real(C_func(wp)))

    pv_result, _ = quad(integrand, -bound, bound, weight='cauchy', wvar=omega,
                         limit=limit)   # 'points' removed, incompatible with weight='cauchy'

    return -pv_result / (2.0 * np.pi)
    

# =======================================
# Exciton Basis and Site Projectors
# =======================================

# w_{alpha,beta} = sum_i |<alpha|S_i|beta>|^2  (Eq. V.44 / w_{alpha beta})
def compute_w_alphabeta(S_exc):
    """
    Computes the geometric coupling factor w_{alpha,beta} = sum_i |<alpha|S_i|beta>|^2.
    Returns array of shape (N_alpha, N_beta).
    """
    N = S_exc.shape[1]
    w = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            w[a, b] = np.sum(np.abs(S_exc[:, a, b])**2)
    return w

# =======================================
# Full Secular Redfield master equation
# =======================================

# ---------------------------------------------------
# Jump operators for the full secular Redfield (exciton basis)
# ---------------------------------------------------
def build_redfield_jump_operators(eigenergies, s_weights, w_ab, C_func):
    """
    Builds the list of Lindblad jump operators (with rates already folded in
    as gamma_k, kept separate for use with the Liouvillian function) for the
    full secular Redfield equation, Eq. V.45:
      - N Pure Dephasing operators: L_i = S_i(0)  (diagonal, exciton basis)
        with rate gamma_i = C(0)
      - N(N-1) Eigenstate Transition operators: L_{alpha,beta} = |alpha><beta|
        with rate gamma_{alpha,beta} = C(eigenergies_beta - eigenergies_alpha) * w_{alpha,beta}

    Returns: L_list (list of NxN complex arrays), gamma_list (list of floats)
    """
    N = len(eigenergies)
    L_list = []
    gamma_list = []

    # --- Pure Dephasing (omega = 0) ---
    C0 = np.real(C_func(0.0))
    for i in range(N):
        L_i = np.diag(s_weights[i]).astype(complex)
        L_list.append(L_i)
        gamma_list.append(C0)

    # --- Eigenstate Transition (omega != 0) ---
    for alpha in range(N):
        for beta in range(N):
            if alpha == beta:
                continue
            L_ab = np.zeros((N, N), dtype=complex)
            L_ab[alpha, beta] = 1.0
            omega = eigenergies[beta] - eigenergies[alpha]
            gamma_ab = np.real(C_func(omega)) * w_ab[alpha, beta]
            L_list.append(L_ab)
            gamma_list.append(gamma_ab)

    return L_list, gamma_list


# ---------------------------------------------------
# Liouvillian superoperator (row-major convention, as in your reference code)
# ---------------------------------------------------
def Liouvillian(H, gamma_k, L_k):
    """
    Build the Liouvillian superoperator using row-major convention (NumPy).
    (Same construction as in the reference collision-model code.)
    """
    I = np.eye(H.shape[0], dtype=complex)
    super_L = -1.j * (np.kron(H, I) - np.kron(I, H.T))

    for k in range(len(gamma_k)):
        L = L_k[k]
        L_dag = np.conj(L).T
        L_dag_L = L_dag @ L
        super_L += gamma_k[k] * (np.kron(L, np.conj(L))
                                 - 0.5 * np.kron(L_dag_L, I)
                                 - 0.5 * np.kron(I, L_dag_L.T))
    return super_L


# ---------------------------------------------------
# Evolution wrapper (expm method)
# ---------------------------------------------------
def Redfield_evo(rho0_exc, H_tot_exc, gamma_list, L_list, times):
    """
    Propagates the full secular Redfield equation in the exciton basis.

    Parameters:
    - rho0_exc    : (N,N) initial density matrix, exciton basis
    - H_tot_exc   : (N,N) total system Hamiltonian (exciton energies + Lamb shift), diagonal
    - gamma_list, L_list : dissipator rates/operators (exciton basis)
    - times       : array, time grid (fs)

    Returns: rho_traj (n_times, N, N), density matrix at each time step, exciton basis
    """
    N = H_tot_exc.shape[0]
    dt = times[1] - times[0]
    n_times = len(times)

    super_L = Liouvillian(H_tot_exc, gamma_list, L_list)
    super_U = expm(super_L * dt)

    rho_vec = rho0_exc.reshape(N * N).astype(complex)
    rho_vec_list = np.zeros((N * N, n_times), dtype=complex)
    rho_vec_list[:, 0] = rho_vec

    for t in range(1, n_times):
        rho_vec_list[:, t] = super_U @ rho_vec_list[:, t - 1]

    return rho_vec_list.T.reshape(n_times, N, N)

# ======================================================================
# Deterministic trace-out-ancilla evolution (infinite-trajectory limit)
# ======================================================================

# ---------------------------------------------------
# Ancilla Pauli operators
# ---------------------------------------------------

I_anc = np.eye(2, dtype=complex)

def build_channel_unitaries(eigenenergies, s_weights, w_ab, C_func, dt):
    """
    Builds the list of collisional unitary propagators U_channel = exp(-i H_channel dt)
    for the full secular Redfield model (exciton basis), following:
      - Pure Dephasing:      H_i^PD    = sqrt(C(0)/dt) * (S_i(0) x sigma_x)
      - Eigenstate Transition: H_ab^Trans = sqrt(C(eps_b-eps_a) w_ab/dt) *
                                            (|a><b| x sigma_+ + |b><a| x sigma_-)

    Each channel acts on the (N_sys x 2)-dimensional system+ancilla space.

    Returns: list of dicts, each with keys 'U', 'type', 'label'
    """
    N = len(eigenenergies)
    C0 = np.real(C_func(0.0))
    channels = []

    # --- Pure Dephasing channels ---
    for i in range(N):
        L_i = np.diag(s_weights[i]).astype(complex)
        c_i = np.sqrt(C0 / dt)
        H_i = c_i * np.kron(L_i, sx)
        U_i = expm(-1j * H_i * dt)
        channels.append({'U': U_i, 'type': 'PD', 'label': i})

    # --- Eigenstate Transition channels ---
    for alpha in range(N):
        for beta in range(N):
            if alpha == beta:
                continue
            omega = eigenenergies[beta] - eigenenergies[alpha]
            gamma_ab = np.real(C_func(omega)) * w_ab[alpha, beta]
            c_ab = np.sqrt(gamma_ab / dt)

            L_ab = np.zeros((N, N), dtype=complex)
            L_ab[alpha, beta] = 1.0
            L_ab_dag = L_ab.conj().T

            H_ab = c_ab * (np.kron(L_ab, sp) + np.kron(L_ab_dag, sm))
            U_ab = expm(-1j * H_ab * dt)
            channels.append({'U': U_ab, 'type': 'Trans', 'label': (alpha, beta)})

    return channels


def build_free_evolution_unitary(eigenenergies, dt):
    """
    Builds U_H(dt) = exp(-i H_exc dt), the free (unitary) evolution operator
    of the isolated system, in the exciton basis (trivially diagonal).

    Returns: (N,N) unitary propagator
    """
    return np.diag(np.exp(-1j * eigenenergies * dt)).astype(complex)


def apply_channel_and_trace(rho_sys, U_channel, rho_anc, dim_sys, dim_anc=2):
    """
    Applies a single collision (expand -> evolve -> partial trace over ancilla).
    """
    rho_tot = np.kron(rho_sys, rho_anc)
    rho_tot = U_channel @ rho_tot @ U_channel.conj().T

    rho_tot_reshaped = rho_tot.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
    rho_sys_new = np.einsum('ikjk->ij', rho_tot_reshaped)

    return rho_sys_new


def collisional_trace_evo(rho0_exc, channels, U_H, times):
    """
    Deterministic collisional evolution, including the free system Hamiltonian
    propagation at each step: U(dt) = U_exc-ph(dt) * U_H(dt), cf. Eq. V.11.

    At each time step:
      1. apply the free unitary evolution U_H (excitonic Hamiltonian)
      2. sequentially apply all collisional channels (each with a freshly
         reset ancilla in |0_a>), tracing out the ancilla after each collision.

    This corresponds to the exact infinite-trajectory-average limit of the
    collisional algorithm.

    Parameters:
    - rho0_exc : (N,N) initial density matrix, exciton basis
    - channels : list of channel dicts from build_channel_unitaries (built for THIS dt)
    - U_H      : (N,N) free evolution unitary for THIS dt, from build_free_evolution_unitary
    - times    : array, time grid (fs), must be consistent with dt used in channels/U_H

    Returns: rho_traj (n_times, N, N), density matrix at each time step, exciton basis
    """
    N = rho0_exc.shape[0]
    n_times = len(times)

    rho_anc = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # |0_a><0_a|

    rho_traj = np.zeros((n_times, N, N), dtype=complex)
    rho_traj[0] = rho0_exc

    rho_sys = rho0_exc.copy()
    for t in range(1, n_times):
        # 1. Free (unitary) evolution under H_exc
        rho_sys = U_H @ rho_sys @ U_H.conj().T

        # 2. Collisional (dissipative) channels, one fresh ancilla each
        for ch in channels:
            rho_sys = apply_channel_and_trace(rho_sys, ch['U'], rho_anc, N)

        rho_traj[t] = rho_sys

    return rho_traj

# ================
# Kraus operators 
# ================

def build_kraus_operators(eigenenergies, s_weights, w_ab, C_func, dt):
    """
    Closed-form Kraus operators (K0, K1) for each of the N + N(N-1) collisional
    channels of the full secular Redfield model (exciton basis).
    """
    N = len(eigenenergies)
    I_sys = np.eye(N, dtype=complex)
    C0 = np.real(C_func(0.0))
    g_i_global = np.sqrt(C0 * dt)

    kraus_list = []

    # --- Pure Dephasing ---
    for i in range(N):
        s_alpha = s_weights[i]
        K0_i = np.diag(np.cos(g_i_global * s_alpha)).astype(complex)
        K1_i = -1j * np.diag(np.sin(g_i_global * s_alpha)).astype(complex)
        kraus_list.append({'K0': K0_i, 'K1': K1_i, 'type': 'PD', 'label': i})

    # --- Eigenstate Transition ---
    for alpha in range(N):
        for beta in range(N):
            if alpha == beta:
                continue
            omega = eigenenergies[beta] - eigenenergies[alpha]
            gamma_ab = np.real(C_func(omega)) * w_ab[alpha, beta]
            g_ab = np.sqrt(gamma_ab * dt)

            P_beta = np.zeros((N, N), dtype=complex)
            P_beta[beta, beta] = 1.0

            K0_ab = I_sys - (1.0 - np.cos(g_ab)) * P_beta

            K1_ab = np.zeros((N, N), dtype=complex)
            K1_ab[alpha, beta] = -1j * np.sin(g_ab)

            kraus_list.append({'K0': K0_ab, 'K1': K1_ab, 'type': 'Trans', 'label': (alpha, beta)})

    return kraus_list


def extract_kraus_from_unitary(channels, dim_sys):
    """
    Cross-check: extracts K0 = <0_a|U|0_a>, K1 = <1_a|U|0_a> directly from the
    collisional unitaries built by build_channel_unitaries.
    """
    kraus_list = []
    for ch in channels:
        U = ch['U']
        U_reshaped = U.reshape(dim_sys, 2, dim_sys, 2)
        K0 = U_reshaped[:, 0, :, 0]
        K1 = U_reshaped[:, 1, :, 0]
        kraus_list.append({'K0': K0, 'K1': K1, 'type': ch['type'], 'label': ch['label']})
    return kraus_list


def check_kraus_completeness(kraus_list, N):
    I_sys = np.eye(N, dtype=complex)
    max_dev = 0.0
    for ch in kraus_list:
        completeness = ch['K0'].conj().T @ ch['K0'] + ch['K1'].conj().T @ ch['K1']
        max_dev = max(max_dev, np.max(np.abs(completeness - I_sys)))
    return max_dev


def rotate_kraus_operators(kraus_list, theta):
    """
    M0(theta) = cos(theta/2) K0 + sin(theta/2) K1
    M1(theta) = sin(theta/2) K0 - cos(theta/2) K1
    theta = 0 -> standard quantum jump; theta = pi/2 -> diffusive-type unravelling.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    rotated_list = []
    for ch in kraus_list:
        M0 = c * ch['K0'] + s * ch['K1']
        M1 = s * ch['K0'] - c * ch['K1']
        rotated_list.append({'M0': M0, 'M1': M1, 'type': ch['type'], 'label': ch['label']})
    return rotated_list


# =================================
# Monte Carlo trajectory algorithm
# (pure-Python reference version)
# =================================

def sample_trajectory(psi0_exc, U_H, kraus_list, n_steps, theta=0.0, rng=None):
    """
    Single pure-state trajectory: at each step, free evolution U_H then
    sequential stochastic application of all channels in kraus_list.
    """
    if rng is None:
        rng = np.random.default_rng()

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    N = psi0_exc.shape[0]
    psi_traj = np.zeros((n_steps + 1, N), dtype=complex)
    psi_traj[0] = psi0_exc

    psi = psi0_exc.copy()

    for t in range(1, n_steps + 1):
        psi = U_H @ psi

        for ch in kraus_list:
            K0, K1 = ch['K0'], ch['K1']
            M0 = c * K0 + s * K1
            M1 = s * K0 - c * K1

            psi_M0 = M0 @ psi
            p0 = np.real(np.vdot(psi_M0, psi_M0))
            p0 = min(max(p0, 0.0), 1.0)

            if rng.random() < p0:
                psi = psi_M0 / np.sqrt(p0)
            else:
                psi_M1 = M1 @ psi
                p1 = np.real(np.vdot(psi_M1, psi_M1))
                psi = psi_M1 / np.sqrt(p1)

        psi_traj[t] = psi

    return psi_traj


def average_trajectories(psi0_exc, U_H, kraus_list, n_steps, n_traj, theta=0.0, seed=None):
    """
    Ensemble average of n_traj independent Monte Carlo trajectories.
    """
    N = psi0_exc.shape[0]
    rho_sum = np.zeros((n_steps + 1, N, N), dtype=complex)

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_traj)

    for k in range(n_traj):
        rng = np.random.default_rng(child_seeds[k])
        psi_traj = sample_trajectory(psi0_exc, U_H, kraus_list, n_steps, theta=theta, rng=rng)
        rho_sum += np.einsum('ti,tj->tij', psi_traj, psi_traj.conj())

    return rho_sum / n_traj


# ==========================================
# Monte Carlo trajectory algorithm (Numba)
# Stores psi (not full rho) for every single
# trajectory and time step. Per-trajectory
# seeds depend ONLY on (master_seed, n_traj),
# NOT on theta -> reproducible across angles.
# ==========================================

def kraus_list_to_arrays(kraus_list, N):
    """
    Converts the list-of-dicts Kraus representation into stacked arrays,
    required as input for the Numba JIT core.

    Returns: K0_arr, K1_arr, each of shape (n_channels, N, N), complex128
    """
    n_channels = len(kraus_list)
    K0_arr = np.zeros((n_channels, N, N), dtype=np.complex128)
    K1_arr = np.zeros((n_channels, N, N), dtype=np.complex128)
    for k, ch in enumerate(kraus_list):
        K0_arr[k] = ch['K0']
        K1_arr[k] = ch['K1']
    return K0_arr, K1_arr


def rotate_kraus_arrays(K0_arr, K1_arr, theta):
    """
    Applies the generalized measurement-basis rotation to the stacked
    Kraus arrays: M0(theta) = cos(theta/2) K0 + sin(theta/2) K1, etc.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    M0_arr = (c * K0_arr + s * K1_arr).astype(np.complex128)
    M1_arr = (s * K0_arr - c * K1_arr).astype(np.complex128)
    return M0_arr, M1_arr


@njit(parallel=True, cache=True, fastmath=True)
def _mc_trajectories_core(psi0, U_H, M0_arr, M1_arr, n_traj, n_times, seeds):
    """
    Computes n_traj independent quantum-jump trajectories in parallel.
    Each trajectory: free evolution U_H, then sequential stochastic
    application of all channels (M0_arr[k], M1_arr[k]).

    Returns: psi_traj, shape (N, n_times, n_traj), complex128
    """
    N = psi0.shape[0]
    n_channels = M0_arr.shape[0]
    psi_traj = np.zeros((N, n_times, n_traj), dtype=np.complex128)

    for traj in prange(n_traj):
        np.random.seed(seeds[traj])
        psi = psi0.copy()

        for i in range(N):
            psi_traj[i, 0, traj] = psi[i]

        for t in range(1, n_times):
            # 1. Free evolution
            psi = np.dot(U_H, psi)

            # 2. Sequential collisional channels
            for k in range(n_channels):
                psi_M0 = np.dot(M0_arr[k], psi)
                p0 = np.real(np.vdot(psi_M0, psi_M0))
                if p0 > 1.0:
                    p0 = 1.0
                elif p0 < 0.0:
                    p0 = 0.0

                r = np.random.rand()
                if r < p0:
                    psi = psi_M0 / np.sqrt(p0)
                else:
                    psi_M1 = np.dot(M1_arr[k], psi)
                    p1 = np.real(np.vdot(psi_M1, psi_M1))
                    psi = psi_M1 / np.sqrt(p1)

            for i in range(N):
                psi_traj[i, t, traj] = psi[i]

    return psi_traj


def compute_trajectories_numba(psi0_exc, U_H, kraus_list, theta, n_traj, n_times, master_seed=42):
    """
    Computes n_traj Monte Carlo trajectories via the Numba JIT core.

    Per-trajectory seeds are generated from a FIXED master_seed and depend
    ONLY on (master_seed, n_traj) -- NOT on theta. This guarantees that
    trajectory #k uses the identical random seed across different theta
    values, exactly as in the reference implementation.

    Returns: psi_traj (N, n_times, n_traj), seeds (n_traj,)
    """
    N = psi0_exc.shape[0]
    K0_arr, K1_arr = kraus_list_to_arrays(kraus_list, N)
    M0_arr, M1_arr = rotate_kraus_arrays(K0_arr, K1_arr, theta)

    rng_master = np.random.RandomState(master_seed)
    seeds = rng_master.randint(0, 2**30, size=n_traj)

    psi0 = psi0_exc.astype(np.complex128)
    U_H_np = U_H.astype(np.complex128)

    psi_traj = _mc_trajectories_core(psi0, U_H_np, M0_arr, M1_arr, n_traj, n_times, seeds)
    return psi_traj, seeds


def psi_traj_to_rho_avg(psi_traj):
    """
    Ensemble-averaged density matrix at each time step, from stored psi.
    Returns: rho_avg (n_times, N, N)
    """
    return np.einsum('itk,jtk->tij', psi_traj, np.conj(psi_traj)) / psi_traj.shape[2]


def site_population_stats(psi_traj, eigenvectors):
    """
    Computes mean and standard error of the SITE-basis populations across
    trajectories, at each time step.

    Returns: pop_mean (n_times, N_site), pop_stderr (n_times, N_site)
    """
    N, n_times, n_traj = psi_traj.shape
    # Transform each trajectory to the site basis: psi_site = D @ psi_exc
    psi_site_traj = np.einsum('ia,atk->itk', eigenvectors, psi_traj)
    pop_traj = np.abs(psi_site_traj) ** 2   # (N_site, n_times, n_traj)

    pop_mean = np.mean(pop_traj, axis=2).T          # (n_times, N_site)
    pop_std = np.std(pop_traj, axis=2, ddof=1).T     # (n_times, N_site)
    pop_stderr = pop_std / np.sqrt(n_traj)

    return pop_mean, pop_stderr


# ====================
# Physical parameters
# ====================

# ---------------------------------------------------
# Physical constants and unit conversion (hbar = 1)
# ---------------------------------------------------
c_light_cm_fs = 2.99792458e-5          # speed of light, cm/fs
cm1_to_radfs = 2.0 * np.pi * c_light_cm_fs   # convert cm^-1 -> rad/fs
KB_cm1_per_K = 0.695034800            # Boltzmann constant, cm^-1/K

# =========================
# Environmental Parameters
# =========================
T_kelvin = 300.0
lam_cm1 = 35.0
Omega_cm1 = 106.1

lam = lam_cm1 * cm1_to_radfs
Omega = Omega_cm1 * cm1_to_radfs
beta = 1.0 / (KB_cm1_per_K * T_kelvin * cm1_to_radfs)  # inverse temperature, fs/rad

# -----------------
# Spectral density
# -----------------
C_func = lambda w: C_drude_lorentz(w, lam, Omega, beta)

# ---------------------------------------------------
# FMO exciton Hamiltonian (Adolphs & Renger), cm^-1
# ---------------------------------------------------
H_exc_cm1 = np.array([
    [200,  -96,    5,  -4.4,   4.7, -12.6,  -6.2],
    [-96,  320, 33.1,   6.8,   4.5,   7.4,  -0.3],
    [5,   33.1,    0, -51.1,   0.8,  -8.4,   7.6],
    [-4.4,  6.8,-51.1,   110, -76.6, -14.2,   -67],
    [4.7,   4.5,  0.8, -76.6,   270,  78.3,  -0.1],
    [-12.6, 7.4, -8.4, -14.2,  78.3,   420,  38.3],
    [-6.2, -0.3,  7.6,   -67,  -0.1,  38.3,   230]
], dtype=complex)

N_site = H_exc_cm1.shape[0]

# Convert to rad/fs
H_exc = H_exc_cm1 * cm1_to_radfs

# Eigenerengies and eigenvectors of the exciton Hamiltonian
eigenergies, eigenvectors = np.linalg.eigh(H_exc)

# ---------------------------------------------------
# Site projectors S_i = |i><i| (site basis)
# ---------------------------------------------------
S_site = np.zeros((N_site, N_site, N_site), dtype=complex)
for i in range(N_site):
    S_site[i, i, i] = 1.0

# S_i in the exciton basis
S_exc = np.zeros_like(S_site)
for i in range(N_site):
    S_exc[i] = eigenvectors.conj().T @ S_site[i] @ eigenvectors

# Weights and Geometric factors
s_weights = np.array([np.real(np.diag(S_exc[i])) for i in range(N_site)])
w_ab = compute_w_alphabeta(S_exc)


# ---------------------------------------------------
# MAIN: Sanity check & Comparison
# ---------------------------------------------------
if __name__ == "__main__":
    from scipy.integrate import quad  # Assicurati che quad sia importato
    
    dt = 1.0    # fs
    tf = 1000.0 # fs
    times = np.arange(0.0, tf, dt)
    n_times = len(times)
    n_steps = n_times - 1

    print("--- Inizializzazione ---")
    
    # 1. Stato Iniziale: eccitazione localizzata sul sito 1 (indice 0)
    rho0_site = np.zeros((N_site, N_site), dtype=complex)
    rho0_site[0, 0] = 1.0
    rho0_exc = eigenvectors.conj().T @ rho0_site @ eigenvectors

    psi0_site = np.zeros(N_site, dtype=complex)
    psi0_site[0] = 1.0
    psi0_exc = eigenvectors.conj().T @ psi0_site

    # 2. Setup Modello Collisionale
    channels = build_channel_unitaries(eigenergies, s_weights, w_ab, C_func, dt)
    U_H = build_free_evolution_unitary(eigenergies, dt)
    print(f"Number of collisional channels: {len(channels)} (expected {N_site + N_site*(N_site-1)})")

    # 3. Setup Master Equation di Redfield
    L_list, gamma_list = build_redfield_jump_operators(eigenergies, s_weights, w_ab, C_func)
    H_tot_exc = np.diag(eigenergies).astype(complex) # (Ignoriamo il Lamb Shift per questo test veloce)

    # 4. Setup Kraus operators (closed form)
    kraus_closed = build_kraus_operators(eigenergies, s_weights, w_ab, C_func, dt)

    # ==========================
    # EVOLUZIONE E CONFRONTO
    # ==========================
    
    print("\n--- Calcolo dinamica Collisionale (trace ancilla, deterministico)... ---")
    rho_traj_coll_exc = collisional_trace_evo(rho0_exc, channels, U_H, times)
    rho_traj_coll_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_traj_coll_exc, eigenvectors.conj())
    pop_site_coll = np.real(np.diagonal(rho_traj_coll_site, axis1=1, axis2=2))

    print("--- Calcolo dinamica Master Equation (Redfield)... ---")
    rho_traj_redfield_exc = Redfield_evo(rho0_exc, H_tot_exc, gamma_list, L_list, times)
    rho_traj_redfield_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_traj_redfield_exc, eigenvectors.conj())
    pop_site_redfield = np.real(np.diagonal(rho_traj_redfield_site, axis1=1, axis2=2))

    # ==========================
    # KRAUS: completeness + cross-check vs U extraction
    # ==========================
    print("\n--- Verifica Kraus operators... ---")
    kraus_extracted = extract_kraus_from_unitary(channels, N_site)
    max_dev_closed = check_kraus_completeness(kraus_closed, N_site)
    max_diff_K0 = max(np.max(np.abs(c['K0'] - e['K0'])) for c, e in zip(kraus_closed, kraus_extracted))
    max_diff_K1 = max(np.max(np.abs(c['K1'] - e['K1'])) for c, e in zip(kraus_closed, kraus_extracted))
    print(f"Kraus completeness deviation: {max_dev_closed:.3e}")
    print(f"Max |K0_closed - K0_extracted|: {max_diff_K0:.3e}")
    print(f"Max |K1_closed - K1_extracted|: {max_diff_K1:.3e}")

    # ==========================
    # MONTE CARLO (Numba)
    # ==========================
    print("\n--- Calcolo dinamica Monte Carlo (Numba, traiettorie complete)... ---")
    n_traj = 200

    # Cross-check: seeds independent of theta
    _, seeds_theta0 = compute_trajectories_numba(psi0_exc, U_H, kraus_closed, 0.0, n_traj, n_times, master_seed=42)
    _, seeds_theta90 = compute_trajectories_numba(psi0_exc, U_H, kraus_closed, np.pi/2, n_traj, n_times, master_seed=42)
    print("Seeds identical across theta:", np.array_equal(seeds_theta0, seeds_theta90))

    # Timing + cross-check vs pure-Python MC
    t0 = time.time()
    psi_traj_numba, _ = compute_trajectories_numba(psi0_exc, U_H, kraus_closed, 0.0, n_traj, n_times, master_seed=42)
    t_numba = time.time() - t0

    t0 = time.time()
    rho_avg_python_exc = average_trajectories(psi0_exc, U_H, kraus_closed, n_steps, n_traj, theta=0.0, seed=42)
    t_python = time.time() - t0

    rho_avg_numba_exc = psi_traj_to_rho_avg(psi_traj_numba)
    pop_numba = np.real(np.einsum('ia,tab,jb->tij', eigenvectors, rho_avg_numba_exc, eigenvectors.conj())
                         .diagonal(axis1=1, axis2=2))
    pop_python = np.real(np.einsum('ia,tab,jb->tij', eigenvectors, rho_avg_python_exc, eigenvectors.conj())
                          .diagonal(axis1=1, axis2=2))

    print(f"\nNumba time: {t_numba:.2f} s   |   Pure-Python time: {t_python:.2f} s   |   Speedup: {t_python/t_numba:.1f}x")
    print("Max abs diff (Numba vs pure-Python MC, same seeds):", np.max(np.abs(pop_numba[-1] - pop_python[-1])))

    # Population statistics directly from stored trajectories
    pop_mean, pop_stderr = site_population_stats(psi_traj_numba, eigenvectors)

    # ==========================
    # RISULTATI
    # ==========================
    print("\n--- Risultati finali ---")
    print(f"Trace at t=0 (Collisional): {np.real(np.trace(rho_traj_coll_site[0])):.5f}")
    print(f"Trace at t_f (Collisional): {np.real(np.trace(rho_traj_coll_site[-1])):.5f}")

    print("\nFinal-time site populations (Collisional, deterministic):")
    print(np.round(pop_site_coll[-1], 5))

    print("\nFinal-time site populations (Redfield):")
    print(np.round(pop_site_redfield[-1], 5))

    print("\nFinal-time site populations (Monte Carlo, Numba, theta=0):")
    print(np.round(pop_mean[-1], 5), "+/-", np.round(pop_stderr[-1], 5))

    max_diff = np.max(np.abs(pop_site_coll[-1] - pop_site_redfield[-1]))
    print(f"\n=> Max abs difference (Collisional vs Redfield) at t_f: {max_diff:.3e}")