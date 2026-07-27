import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import numba
from numba import njit, prange
import os
import time
import sys

sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
sp = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

# ===================================================
# Environment Spectral Function: Drude-Lorentz Bath
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
    ex = np.exp(x[neg])
    out[neg] = ex / (ex - 1.0)

    return out[0] if scalar_input else out


def C_drude_lorentz(omega, lam, Omega, beta, omega_tol=1e-10):
    """
    Drude-Lorentz spectral function C(omega) = 4*lam*omega*Omega/(omega^2+Omega^2) * 1/(1-exp(-omega*beta)).
    Handles the omega -> 0 limit analytically, numerically stable for large |omega|.
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
    Computes Lambda(omega) = (1/2pi) * P.V. Integral[ C(omega')/(omega-omega'), d omega' ].
    (Not used in the current dynamics -- Lamb shift neglected -- kept for future use.)
    """
    integrand = lambda wp: float(np.real(C_func(wp)))
    pv_result, _ = quad(integrand, -bound, bound, weight='cauchy', wvar=omega, limit=limit)
    return -pv_result / (2.0 * np.pi)


# =======================================
# Exciton Basis and Site Projectors
# =======================================

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

def build_redfield_jump_operators(eigenergies, s_weights, w_ab, C_func):
    """
    Builds Lindblad jump operators + rates for the full secular Redfield equation:
      - N Pure Dephasing operators:      L_i = S_i(0),          gamma_i = C(0)
      - N(N-1) Eigenstate Transition ops: L_{alpha,beta} = |alpha><beta|,
                                          gamma_{alpha,beta} = C(eps_b - eps_a) * w_{alpha,beta}
    Returns: L_list, gamma_list
    """
    N = len(eigenergies)
    L_list = []
    gamma_list = []

    C0 = np.real(C_func(0.0))
    for i in range(N):
        L_i = np.diag(s_weights[i]).astype(complex)
        L_list.append(L_i)
        gamma_list.append(C0)

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


def Liouvillian(H, gamma_k, L_k):
    """Builds the Liouvillian superoperator (row-major convention)."""
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


def Redfield_evo(rho0_exc, H_tot_exc, gamma_list, L_list, times):
    """
    Propagates the full secular Redfield equation in the exciton basis.
    Returns: rho_traj (n_times, N, N), exciton basis
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

def build_channel_unitaries(eigenenergies, s_weights, w_ab, C_func, dt):
    """
    Builds collisional unitary propagators U_channel = exp(-i H_channel dt):
      - Pure Dephasing:        H_i^PD    = sqrt(C(0)/dt) * (S_i(0) x sigma_x)
      - Eigenstate Transition: H_ab^Trans = sqrt(C(eps_b-eps_a) w_ab/dt) *
                                            (|a><b| x sigma_+ + |b><a| x sigma_-)
    Returns: list of dicts with keys 'U', 'type', 'label'
    """
    N = len(eigenenergies)
    C0 = np.real(C_func(0.0))
    channels = []

    for i in range(N):
        L_i = np.diag(s_weights[i]).astype(complex)
        c_i = np.sqrt(C0 / dt)
        H_i = c_i * np.kron(L_i, sx)
        U_i = expm(-1j * H_i * dt)
        channels.append({'U': U_i, 'type': 'PD', 'label': i})

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
    """Builds U_H(dt) = exp(-i H_exc dt), diagonal in the exciton basis."""
    return np.diag(np.exp(-1j * eigenenergies * dt)).astype(complex)


def apply_channel_and_trace(rho_sys, U_channel, rho_anc, dim_sys, dim_anc=2):
    """Applies a single collision (expand -> evolve -> partial trace over ancilla)."""
    rho_tot = np.kron(rho_sys, rho_anc)
    rho_tot = U_channel @ rho_tot @ U_channel.conj().T
    rho_tot_reshaped = rho_tot.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
    return np.einsum('ikjk->ij', rho_tot_reshaped)


def collisional_trace_evo(rho0_exc, channels, U_H, times):
    """
    Deterministic collisional evolution: U(dt) = U_exc-ph(dt) * U_H(dt) (Eq. V.11).
    Each channel uses a freshly reset ancilla in |0_a>, traced out after every collision.
    Corresponds to the exact infinite-trajectory-average limit.
    Returns: rho_traj (n_times, N, N), exciton basis
    """
    N = rho0_exc.shape[0]
    n_times = len(times)
    rho_anc = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    rho_traj = np.zeros((n_times, N, N), dtype=complex)
    rho_traj[0] = rho0_exc

    rho_sys = rho0_exc.copy()
    for t in range(1, n_times):
        rho_sys = U_H @ rho_sys @ U_H.conj().T
        for ch in channels:
            rho_sys = apply_channel_and_trace(rho_sys, ch['U'], rho_anc, N)
        rho_traj[t] = rho_sys

    return rho_traj


# ================
# Kraus operators
# ================

def build_kraus_operators(eigenenergies, s_weights, w_ab, C_func, dt):
    """
    Closed-form Kraus operators (K0, K1) for the N + N(N-1) collisional channels
    of the full secular Redfield model (exciton basis).
    """
    N = len(eigenenergies)
    I_sys = np.eye(N, dtype=complex)
    C0 = np.real(C_func(0.0))
    g_i_global = np.sqrt(C0 * dt)

    kraus_list = []

    for i in range(N):
        s_alpha = s_weights[i]
        K0_i = np.diag(np.cos(g_i_global * s_alpha)).astype(complex)
        K1_i = -1j * np.diag(np.sin(g_i_global * s_alpha)).astype(complex)
        kraus_list.append({'K0': K0_i, 'K1': K1_i, 'type': 'PD', 'label': i})

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


def check_kraus_completeness(kraus_list, N):
    """Sanity check: K0^dag K0 + K1^dag K1 = I for every channel. Returns max deviation."""
    I_sys = np.eye(N, dtype=complex)
    max_dev = 0.0
    for ch in kraus_list:
        completeness = ch['K0'].conj().T @ ch['K0'] + ch['K1'].conj().T @ ch['K1']
        max_dev = max(max_dev, np.max(np.abs(completeness - I_sys)))
    return max_dev

def build_channel_type_arrays(kraus_list):
    """
    For each of the n_channels collisional channels (in the same order as
    build_kraus_operators), builds:
      - channel_type  : 0 = Pure Dephasing, 1 = Eigenstate Transition
      - channel_alpha : for Trans channels, the TARGET exciton state alpha
                        that the system collapses onto if that channel jumps
                        (i.e. K1 ~ |alpha><beta|). -1 for PD channels (unused).
    """
    n_channels = len(kraus_list)
    channel_type = np.zeros(n_channels, dtype=np.int8)
    channel_alpha = -np.ones(n_channels, dtype=np.int32)

    for k, ch in enumerate(kraus_list):
        if ch['type'] == 'Trans':
            channel_type[k] = 1
            alpha, beta = ch['label']
            channel_alpha[k] = alpha

    return channel_type, channel_alpha

# ==========================================
# Monte Carlo trajectory algorithm (Numba)
# Stores psi (not full rho) for every single trajectory and time step.
# Per-trajectory seeds depend ONLY on (master_seed, n_traj), NOT on theta.
# Also tracks, per time step and trajectory, how many of the 49 channels
# resulted in a "jump" (M1 branch) -- meaningful as quantum-jump count at theta=0.
# ==========================================

def kraus_list_to_arrays(kraus_list, N):
    """Converts list-of-dicts Kraus representation into stacked arrays (n_channels, N, N)."""
    n_channels = len(kraus_list)
    K0_arr = np.zeros((n_channels, N, N), dtype=np.complex128)
    K1_arr = np.zeros((n_channels, N, N), dtype=np.complex128)
    for k, ch in enumerate(kraus_list):
        K0_arr[k] = ch['K0']
        K1_arr[k] = ch['K1']
    return K0_arr, K1_arr


def rotate_kraus_arrays(K0_arr, K1_arr, theta):
    """
    M0(theta) = cos(theta/2) K0 + sin(theta/2) K1
    M1(theta) = sin(theta/2) K0 - cos(theta/2) K1
    theta = 0 -> standard quantum jump; theta = pi/2 -> diffusive-type unravelling.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    M0_arr = (c * K0_arr + s * K1_arr).astype(np.complex128)
    M1_arr = (s * K0_arr - c * K1_arr).astype(np.complex128)
    return M0_arr, M1_arr


@njit(parallel=True, cache=True, fastmath=True)
def _mc_trajectories_core(psi0, U_H, M0_arr, M1_arr, n_traj, n_times, seeds):
    """
    Computes n_traj independent trajectories in parallel. Each trajectory:
    free evolution U_H, then sequential stochastic application of all channels.

    Returns:
    - psi_traj    : (N, n_times, n_traj), complex128
    - jump_counts : (n_times, n_traj), int32 -- number of channels (out of
                    n_channels) that resulted in the M1 ("jump") branch at
                    that time step, for that trajectory.
    """
    N = psi0.shape[0]
    n_channels = M0_arr.shape[0]
    psi_traj = np.zeros((N, n_times, n_traj), dtype=np.complex128)
    jump_counts = np.zeros((n_times, n_traj), dtype=np.int32)

    for traj in prange(n_traj):
        np.random.seed(seeds[traj])
        psi = psi0.copy()

        for i in range(N):
            psi_traj[i, 0, traj] = psi[i]

        for t in range(1, n_times):
            psi = np.dot(U_H, psi)
            n_jumps_step = 0

            for k in range(n_channels):
                psi_M0 = np.dot(M0_arr[k], psi)
                p0 = np.sum(np.abs(psi_M0)**2)
                if p0 > 1.0:
                    p0 = 1.0
                elif p0 < 0.0:
                    p0 = 0.0

                r = np.random.rand()
                if r < p0:
                    psi = psi_M0 / np.sqrt(p0)
                else:
                    psi_M1 = np.dot(M1_arr[k], psi)
                    p1 = np.sum(np.abs(psi_M1)**2)
                    psi = psi_M1 / np.sqrt(p1)
                    n_jumps_step += 1

            jump_counts[t, traj] = n_jumps_step

            for i in range(N):
                psi_traj[i, t, traj] = psi[i]

    return psi_traj, jump_counts


def compute_trajectories_numba(psi0_exc, U_H, kraus_list, theta, n_traj, n_times,
                                master_seed=42, batch_size=1000):
    """
    Computes n_traj Monte Carlo trajectories via the Numba JIT core, in batches
    (to limit peak memory). Per-trajectory seeds depend ONLY on (master_seed,
    n_traj), NOT on theta -- trajectory #k uses the identical seed across
    different theta values.

    Returns: psi_traj (N, n_times, n_traj), jump_counts (n_times, n_traj), seeds (n_traj,)
    """
    N = psi0_exc.shape[0]
    K0_arr, K1_arr = kraus_list_to_arrays(kraus_list, N)
    M0_arr, M1_arr = rotate_kraus_arrays(K0_arr, K1_arr, theta)

    rng_master = np.random.RandomState(master_seed)
    all_seeds = rng_master.randint(0, 2**30, size=n_traj)

    psi0 = psi0_exc.astype(np.complex128)
    U_H_np = U_H.astype(np.complex128)

    psi_traj_all = np.zeros((N, n_times, n_traj), dtype=np.complex128)
    jump_counts_all = np.zeros((n_times, n_traj), dtype=np.int32)

    n_done = 0
    n_batches = int(np.ceil(n_traj / batch_size))
    for b in range(n_batches):
        n_batch = min(batch_size, n_traj - n_done)
        seeds_b = all_seeds[n_done:n_done + n_batch]

        psi_batch, jumps_batch = _mc_trajectories_core(
            psi0, U_H_np, M0_arr, M1_arr, n_batch, n_times, seeds_b)

        psi_traj_all[:, :, n_done:n_done + n_batch] = psi_batch
        jump_counts_all[:, n_done:n_done + n_batch] = jumps_batch
        n_done += n_batch

    return psi_traj_all, jump_counts_all, all_seeds


def psi_traj_to_rho_avg(psi_traj):
    """Ensemble-averaged density matrix at each time step. Returns: (n_times, N, N)."""
    return np.einsum('itk,jtk->tij', psi_traj, np.conj(psi_traj)) / psi_traj.shape[2]


def site_population_stats(psi_traj, eigenvectors):
    """
    Mean and standard error of the SITE-basis populations across trajectories.
    Returns: pop_traj_mean (n_times, N_site), pop_traj_stderr (n_times, N_site)
    """
    N, n_times, n_traj = psi_traj.shape
    psi_site_traj = np.einsum('ia,atk->itk', eigenvectors, psi_traj)
    pop_traj = np.abs(psi_site_traj) ** 2

    pop_traj_mean = np.mean(pop_traj, axis=2).T
    pop_std = np.std(pop_traj, axis=2, ddof=1).T
    pop_traj_stderr = pop_std / np.sqrt(n_traj)

    return pop_traj_mean, pop_traj_stderr

# =======================================================================
# QJ dedicated evolution to analyse between which states the jumps occur
# =======================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _mc_trajectories_core_labeled(psi0, U_H, M0_arr, M1_arr, channel_type, channel_alpha,
                                   n_traj, n_times, seeds):
    """
    Same as _mc_trajectories_core, but additionally tracks, at every time step,
    the exciton-state label of the last Eigenstate-Transition jump ("which
    eigenstate the trajectory is currently collapsed onto"). Meaningful ONLY
    at theta=0 (quantum jump): -1 means "not yet collapsed" (still a genuine
    superposition, no Trans jump has occurred yet).

    Returns: psi_traj (N, n_times, n_traj), jump_counts (n_times, n_traj),
             label_traj (n_times, n_traj) int32
    """
    N = psi0.shape[0]
    n_channels = M0_arr.shape[0]
    psi_traj = np.zeros((N, n_times, n_traj), dtype=np.complex128)
    jump_counts = np.zeros((n_times, n_traj), dtype=np.int32)
    label_traj = np.full((n_times, n_traj), -1, dtype=np.int32)

    for traj in prange(n_traj):
        np.random.seed(seeds[traj])
        psi = psi0.copy()
        current_label = -1

        for i in range(N):
            psi_traj[i, 0, traj] = psi[i]
        label_traj[0, traj] = current_label

        for t in range(1, n_times):
            psi = np.dot(U_H, psi)
            n_jumps_step = 0

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
                    n_jumps_step += 1
                    if channel_type[k] == 1:      # Eigenstate Transition jump
                        current_label = channel_alpha[k]

            jump_counts[t, traj] = n_jumps_step
            label_traj[t, traj] = current_label

            for i in range(N):
                psi_traj[i, t, traj] = psi[i]

    return psi_traj, jump_counts, label_traj

def compute_trajectories_numba_labeled(psi0_exc, U_H, kraus_list, n_traj, n_times,
                                        master_seed=42, batch_size=1000):
    """
    Same as compute_trajectories_numba, fixed at theta=0 (standard quantum jump),
    additionally returning label_traj: the exciton-state index the trajectory
    has collapsed onto at each time step (see _mc_trajectories_core_labeled).
    """
    N = psi0_exc.shape[0]
    K0_arr, K1_arr = kraus_list_to_arrays(kraus_list, N)
    M0_arr, M1_arr = rotate_kraus_arrays(K0_arr, K1_arr, 0.0)   # theta=0: M0=K0, M1=K1
    channel_type, channel_alpha = build_channel_type_arrays(kraus_list)

    rng_master = np.random.RandomState(master_seed)
    all_seeds = rng_master.randint(0, 2**30, size=n_traj)

    psi0 = psi0_exc.astype(np.complex128)
    U_H_np = U_H.astype(np.complex128)

    psi_traj_all = np.zeros((N, n_times, n_traj), dtype=np.complex128)
    jump_counts_all = np.zeros((n_times, n_traj), dtype=np.int32)
    label_traj_all = np.full((n_times, n_traj), -1, dtype=np.int32)

    n_done = 0
    n_batches = int(np.ceil(n_traj / batch_size))
    for b in range(n_batches):
        n_batch = min(batch_size, n_traj - n_done)
        seeds_b = all_seeds[n_done:n_done + n_batch]

        psi_batch, jumps_batch, label_batch = _mc_trajectories_core_labeled(
            psi0, U_H_np, M0_arr, M1_arr, channel_type, channel_alpha,
            n_batch, n_times, seeds_b)

        psi_traj_all[:, :, n_done:n_done + n_batch] = psi_batch
        jump_counts_all[:, n_done:n_done + n_batch] = jumps_batch
        label_traj_all[:, n_done:n_done + n_batch] = label_batch
        n_done += n_batch

    return psi_traj_all, jump_counts_all, label_traj_all, all_seeds


# ====================
# Physical parameters
# ====================

c_light_cm_fs = 2.99792458e-5
cm1_to_radfs = 2.0 * np.pi * c_light_cm_fs
KB_cm1_per_K = 0.695034800

T_kelvin = 347.0
lam_cm1 = 35.0
# Omega_cm1 = 106.1

lam = lam_cm1 * cm1_to_radfs
# Omega = Omega_cm1 * cm1_to_radfs
Omega = 1.0   # already in fs^-1 units
beta = 1.0 / (KB_cm1_per_K * T_kelvin * cm1_to_radfs)

C_func = lambda w: C_drude_lorentz(w, lam, Omega, beta)

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
H_exc = H_exc_cm1 * cm1_to_radfs

eigenergies, eigenvectors = np.linalg.eigh(H_exc)

S_site = np.zeros((N_site, N_site, N_site), dtype=complex)
for i in range(N_site):
    S_site[i, i, i] = 1.0

S_exc = np.zeros_like(S_site)
for i in range(N_site):
    S_exc[i] = eigenvectors.conj().T @ S_site[i] @ eigenvectors

s_weights = np.array([np.real(np.diag(S_exc[i])) for i in range(N_site)])
w_ab = compute_w_alphabeta(S_exc)


# ============================================================
# MAIN: build components, run all three models, save to disk
# ============================================================
if __name__ == "__main__":

# ---------------------------------------------------
    # Run settings
    # ---------------------------------------------------
    dt = 1.0        # fs
    tf = 5000.0     # fs
    times = np.arange(0.0, tf, dt)
    n_times = len(times)

    n_traj = 10000
    master_seed = 42
    batch_size = 1000

    # Lettura degli argomenti passati dal file Bash:
    if len(sys.argv) > 1:
        theta_deg = float(sys.argv[1])
        bash_mode = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    else:
        theta_deg = 0.0
        bash_mode = "local_test"

    theta = np.radians(theta_deg)   # 0 = quantum jump; 90 = diffusive-type unravelling

    results_dir = "../Results/Data/"
    os.makedirs(results_dir, exist_ok=True)

    print(f"--- Inizializzazione (theta = {theta_deg} deg, n_traj = {n_traj}, mode = {bash_mode}) ---")

    # ---------------------------------------------------
    # Initial state: excitation localized at site 1
    # ---------------------------------------------------
    rho0_site = np.zeros((N_site, N_site), dtype=complex)
    rho0_site[0, 0] = 1.0
    rho0_exc = eigenvectors.conj().T @ rho0_site @ eigenvectors

    psi0_site = np.zeros(N_site, dtype=complex)
    psi0_site[0] = 1.0
    psi0_exc = eigenvectors.conj().T @ psi0_site

    # ---------------------------------------------------
    # Model setup
    # ---------------------------------------------------
    channels = build_channel_unitaries(eigenergies, s_weights, w_ab, C_func, dt)
    U_H = build_free_evolution_unitary(eigenergies, dt)
    print(f"Number of collisional channels: {len(channels)} (expected {N_site + N_site*(N_site-1)})")

    L_list, gamma_list = build_redfield_jump_operators(eigenergies, s_weights, w_ab, C_func)
    H_tot_exc = np.diag(eigenergies).astype(complex)   # Lamb shift neglected

    kraus_closed = build_kraus_operators(eigenergies, s_weights, w_ab, C_func, dt)

    # ---------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------
    print("\n--- Sanity checks ---")
    print(f"KMS relative error: {check_KMS(C_func, eigenergies[1]-eigenergies[0], beta):.3e}")
    print(f"Kraus completeness deviation: {check_kraus_completeness(kraus_closed, N_site):.3e}")

    # ---------------------------------------------------
    # 1. Redfield master equation
    # ---------------------------------------------------
    print("\n--- Calcolo dinamica Master Equation (Redfield)... ---")
    rho_redfield_exc = Redfield_evo(rho0_exc, H_tot_exc, gamma_list, L_list, times)
    rho_redfield_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_redfield_exc, eigenvectors.conj())
    pop_site_redfield = np.real(np.diagonal(rho_redfield_site, axis1=1, axis2=2))

    # ---------------------------------------------------
    # 2. Deterministic collisional (trace-ancilla, infinite-trajectory limit)
    # ---------------------------------------------------
    print("--- Calcolo dinamica Collisionale (trace ancilla, deterministico)... ---")
    rho_trace_coll_exc = collisional_trace_evo(rho0_exc, channels, U_H, times)
    rho_trace_coll_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_trace_coll_exc, eigenvectors.conj())
    pop_trace_coll_site = np.real(np.diagonal(rho_trace_coll_site, axis1=1, axis2=2))

    print(f"Trace at t=0 (Collisional): {np.real(np.trace(rho_trace_coll_site[0])):.5f}")
    print(f"Trace at t_f (Collisional): {np.real(np.trace(rho_trace_coll_site[-1])):.5f}")
    print(f"Max abs diff (Collisional vs Redfield) at t_f: "
          f"{np.max(np.abs(pop_trace_coll_site[-1] - pop_site_redfield[-1])):.3e}")

    # ---------------------------------------------------
    # 3. Monte Carlo trajectories (Numba)
    # ---------------------------------------------------
    print(f"\n--- Calcolo dinamica Monte Carlo (Numba, {n_traj} traiettorie)... ---")
    t0 = time.time()

    if theta_deg == 0.0:
        print("Theta=0 -> quantum jump regime: tracking eigenstate collapse labels")
        psi_traj, jump_counts, label_traj, seeds = compute_trajectories_numba_labeled(
            psi0_exc, U_H, kraus_closed, n_traj, n_times,
            master_seed=master_seed, batch_size=batch_size)
    else:
        psi_traj, jump_counts, seeds = compute_trajectories_numba(
            psi0_exc, U_H, kraus_closed, theta, n_traj, n_times,
            master_seed=master_seed, batch_size=batch_size)
        label_traj = None

    t_mc = time.time() - t0
    print(f"Elapsed time: {t_mc:.2f} s")

    pop_traj_mean, pop_traj_stderr = site_population_stats(psi_traj, eigenvectors)

    rho_traj_avg_exc = psi_traj_to_rho_avg(psi_traj)
    rho_traj_avg_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_traj_avg_exc, eigenvectors.conj())

    total_jumps = np.sum(jump_counts, axis=1)   # (n_times,), summed over trajectories

    print(f"Trace at t_f (MC, mean): {np.sum(pop_traj_mean[-1]):.5f}")
    print(f"Total jumps recorded (all trajectories, all channels, cumulative over run): "
          f"{np.sum(total_jumps)}")
    print(f"Final-time populations (MC): {np.round(pop_traj_mean[-1], 5)} +/- {np.round(pop_traj_stderr[-1], 5)}")

    # ==========================================================
    # SAVE RESULTS
    # ==========================================================
    def _make_fname_npz(results_dir, theta_deg, dt, n_traj):
        dt_str = f"{dt:.2f}".replace(".", "p")
        theta_str = f"{theta_deg:.3f}".replace(".", "p")
        return os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{n_traj}.npz")

    fname_npz = _make_fname_npz(results_dir, theta_deg, dt, n_traj)

    # 1. Creiamo il dizionario con tutti i dati base
    save_dict = {
        'rho_redfield_exc': rho_redfield_exc,
        'rho_redfield_site': rho_redfield_site,
        'rho_trace_coll_exc': rho_trace_coll_exc,
        'rho_trace_coll_site': rho_trace_coll_site,
        'psi_traj': psi_traj.astype(np.complex64),
        'jump_counts': jump_counts,
        'total_jumps': total_jumps,
        'seeds': seeds,
        'pop_traj_mean': pop_traj_mean,
        'pop_traj_stderr': pop_traj_stderr,
        'rho_traj_avg_exc': rho_traj_avg_exc,
        'rho_traj_avg_site': rho_traj_avg_site,
        'times': times, 'dt': dt, 'tf': tf, 'n_times': n_times,
        'theta': theta, 'theta_deg': theta_deg,
        'n_traj': n_traj, 'master_seed': master_seed,
        'N_site': N_site,
        'eigenergies': eigenergies, 'eigenvectors': eigenvectors,
        's_weights': s_weights, 'w_ab': w_ab,
        'lam_cm1': lam_cm1, 'Omega': Omega, 'T_kelvin': T_kelvin,
        'rho0_site': rho0_site, 'psi0_exc': psi0_exc
    }

    # 2. Aggiungiamo dinamicamente label_traj se esiste (regime quantum jump)
    if label_traj is not None:
        save_dict['label_traj'] = label_traj.astype(np.int16)   # -1 = not yet collapsed

    # 3. Salviamo scartando l'intero dizionario nel file .npz
    np.savez_compressed(fname_npz, **save_dict)

    print(f"\nSaved -> {os.path.basename(fname_npz)}")

    print("\n" + "=" * 40)
    print("COMPUTATION COMPLETED!")
    print(f"  - Angle (theta): {theta_deg} degrees ({theta:.4f} rad)")
    print(f"  - mode: {bash_mode}")
    print(f"  - n_traj = {n_traj}, dt = {dt} fs, tf = {tf} fs")
    print("=" * 40)