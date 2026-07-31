import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import numba
from numba import njit, prange
import os
import time
import sys
import tempfile

sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
sp = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

# ===================================================
# Environment Spectral Function: Drude-Lorentz Bath
# ===================================================

def check_KMS(C_func, omega_test, beta, rtol=1e-8):
    lhs = C_func(-omega_test)
    rhs = C_func(omega_test) * np.exp(-beta * omega_test)
    return np.abs(lhs - rhs) / (np.abs(rhs) + 1e-30)

def bose_factor_stable(x):
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
    integrand = lambda wp: float(np.real(C_func(wp)))
    pv_result, _ = quad(integrand, -bound, bound, weight='cauchy', wvar=omega, limit=limit)
    return -pv_result / (2.0 * np.pi)

# =======================================
# Exciton Basis and Site Projectors
# =======================================

def compute_w_alphabeta(S_exc):
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
# Deterministic trace-out-ancilla evolution
# ======================================================================

def build_channel_unitaries(eigenenergies, s_weights, w_ab, C_func, dt):
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
    return np.diag(np.exp(-1j * eigenenergies * dt)).astype(complex)

def apply_channel_and_trace(rho_sys, U_channel, rho_anc, dim_sys, dim_anc=2):
    rho_tot = np.kron(rho_sys, rho_anc)
    rho_tot = U_channel @ rho_tot @ U_channel.conj().T
    rho_tot_reshaped = rho_tot.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
    return np.einsum('ikjk->ij', rho_tot_reshaped)

def collisional_trace_evo(rho0_exc, channels, U_H, times):
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
    I_sys = np.eye(N, dtype=complex)
    max_dev = 0.0
    for ch in kraus_list:
        completeness = ch['K0'].conj().T @ ch['K0'] + ch['K1'].conj().T @ ch['K1']
        max_dev = max(max_dev, np.max(np.abs(completeness - I_sys)))
    return max_dev

def build_channel_type_arrays(kraus_list):
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
# ==========================================

def kraus_list_to_arrays(kraus_list, N):
    n_channels = len(kraus_list)
    K0_arr = np.zeros((n_channels, N, N), dtype=np.complex128)
    K1_arr = np.zeros((n_channels, N, N), dtype=np.complex128)
    for k, ch in enumerate(kraus_list):
        K0_arr[k] = ch['K0']
        K1_arr[k] = ch['K1']
    return K0_arr, K1_arr

def rotate_kraus_arrays(K0_arr, K1_arr, theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    M0_arr = (c * K0_arr + s * K1_arr).astype(np.complex128)
    M1_arr = (s * K0_arr - c * K1_arr).astype(np.complex128)
    return M0_arr, M1_arr

@njit(parallel=True, cache=True, fastmath=True)
def _mc_trajectories_core(psi0, U_H, M0_arr, M1_arr, n_traj, n_times, save_step, seeds):
    N = psi0.shape[0]
    n_channels = M0_arr.shape[0]
    
    n_saved = (n_times - 1) // save_step + 1
    
    psi_traj = np.zeros((N, n_saved, n_traj), dtype=np.complex128)
    jump_counts = np.zeros((n_saved, n_traj), dtype=np.int32)

    for traj in prange(n_traj):
        np.random.seed(seeds[traj])
        psi = psi0.copy()

        for i in range(N):
            psi_traj[i, 0, traj] = psi[i]
            
        save_idx = 0
        jumps_accum = 0

        for t in range(1, n_times):
            psi = np.dot(U_H, psi)
            n_jumps_step = 0

            for k in range(n_channels):
                psi_M0 = np.dot(M0_arr[k], psi)
                p0 = np.sum(np.abs(psi_M0)**2)
                if p0 > 1.0: p0 = 1.0
                elif p0 < 0.0: p0 = 0.0

                r = np.random.rand()
                if r < p0:
                    psi = psi_M0 / np.sqrt(p0)
                else:
                    psi_M1 = np.dot(M1_arr[k], psi)
                    p1 = np.sum(np.abs(psi_M1)**2)
                    psi = psi_M1 / np.sqrt(p1)
                    n_jumps_step += 1

            jumps_accum += n_jumps_step

            if t % save_step == 0:
                save_idx += 1
                for i in range(N):
                    psi_traj[i, save_idx, traj] = psi[i]
                jump_counts[save_idx, traj] = jumps_accum
                jumps_accum = 0

    return psi_traj, jump_counts


@njit(parallel=True, cache=True, fastmath=True)
def _mc_trajectories_core_labeled(psi0, U_H, M0_arr, M1_arr, channel_type, channel_alpha,
                                   n_traj, n_times, save_step, seeds):
    N = psi0.shape[0]
    n_channels = M0_arr.shape[0]
    
    n_saved = (n_times - 1) // save_step + 1
    
    psi_traj = np.zeros((N, n_saved, n_traj), dtype=np.complex128)
    jump_counts = np.zeros((n_saved, n_traj), dtype=np.int32)
    label_traj = np.full((n_saved, n_traj), -1, dtype=np.int32)

    for traj in prange(n_traj):
        np.random.seed(seeds[traj])
        psi = psi0.copy()
        current_label = -1

        for i in range(N):
            psi_traj[i, 0, traj] = psi[i]
        label_traj[0, traj] = current_label
        
        save_idx = 0
        jumps_accum = 0

        for t in range(1, n_times):
            psi = np.dot(U_H, psi)
            n_jumps_step = 0

            for k in range(n_channels):
                psi_M0 = np.dot(M0_arr[k], psi)
                p0 = np.real(np.vdot(psi_M0, psi_M0))
                if p0 > 1.0: p0 = 1.0
                elif p0 < 0.0: p0 = 0.0

                r = np.random.rand()
                if r < p0:
                    psi = psi_M0 / np.sqrt(p0)
                else:
                    psi_M1 = np.dot(M1_arr[k], psi)
                    p1 = np.real(np.vdot(psi_M1, psi_M1))
                    psi = psi_M1 / np.sqrt(p1)
                    n_jumps_step += 1
                    if channel_type[k] == 1:
                        current_label = channel_alpha[k]

            jumps_accum += n_jumps_step

            if t % save_step == 0:
                save_idx += 1
                for i in range(N):
                    psi_traj[i, save_idx, traj] = psi[i]
                jump_counts[save_idx, traj] = jumps_accum
                label_traj[save_idx, traj] = current_label
                jumps_accum = 0

    return psi_traj, jump_counts, label_traj


def compute_trajectories_numba(psi0_exc, U_H, kraus_list, theta, n_traj, n_times,
                                save_step=10, master_seed=42, batch_size=500):
    N = psi0_exc.shape[0]
    K0_arr, K1_arr = kraus_list_to_arrays(kraus_list, N)
    M0_arr, M1_arr = rotate_kraus_arrays(K0_arr, K1_arr, theta)

    rng_master = np.random.RandomState(master_seed)
    all_seeds = rng_master.randint(0, 2**30, size=n_traj)

    psi0 = psi0_exc.astype(np.complex128)
    U_H_np = U_H.astype(np.complex128)

    n_saved = (n_times - 1) // save_step + 1

    tmp_psi = tempfile.NamedTemporaryFile(delete=False)
    tmp_jumps = tempfile.NamedTemporaryFile(delete=False)

    psi_traj_all = np.memmap(tmp_psi.name, dtype=np.complex128, mode='w+', shape=(N, n_saved, n_traj))
    jump_counts_all = np.memmap(tmp_jumps.name, dtype=np.int32, mode='w+', shape=(n_saved, n_traj))

    n_done = 0
    n_batches = int(np.ceil(n_traj / batch_size))
    for b in range(n_batches):
        n_batch = min(batch_size, n_traj - n_done)
        seeds_b = all_seeds[n_done:n_done + n_batch]

        psi_batch, jumps_batch = _mc_trajectories_core(
            psi0, U_H_np, M0_arr, M1_arr, n_batch, n_times, save_step, seeds_b)

        psi_traj_all[:, :, n_done:n_done + n_batch] = psi_batch
        jump_counts_all[:, n_done:n_done + n_batch] = jumps_batch
        psi_traj_all.flush()
        jump_counts_all.flush()
        
        n_done += n_batch

    tmp_psi.close()
    tmp_jumps.close()

    return psi_traj_all, jump_counts_all, all_seeds, tmp_psi.name, tmp_jumps.name


def compute_trajectories_numba_labeled(psi0_exc, U_H, kraus_list, n_traj, n_times,
                                        save_step=10, master_seed=42, batch_size=500):
    N = psi0_exc.shape[0]
    K0_arr, K1_arr = kraus_list_to_arrays(kraus_list, N)
    M0_arr, M1_arr = rotate_kraus_arrays(K0_arr, K1_arr, 0.0)
    channel_type, channel_alpha = build_channel_type_arrays(kraus_list)

    rng_master = np.random.RandomState(master_seed)
    all_seeds = rng_master.randint(0, 2**30, size=n_traj)

    psi0 = psi0_exc.astype(np.complex128)
    U_H_np = U_H.astype(np.complex128)

    n_saved = (n_times - 1) // save_step + 1

    tmp_psi = tempfile.NamedTemporaryFile(delete=False)
    tmp_jumps = tempfile.NamedTemporaryFile(delete=False)
    tmp_labels = tempfile.NamedTemporaryFile(delete=False)

    psi_traj_all = np.memmap(tmp_psi.name, dtype=np.complex128, mode='w+', shape=(N, n_saved, n_traj))
    jump_counts_all = np.memmap(tmp_jumps.name, dtype=np.int32, mode='w+', shape=(n_saved, n_traj))
    label_traj_all = np.memmap(tmp_labels.name, dtype=np.int32, mode='w+', shape=(n_saved, n_traj))
    label_traj_all[:] = -1 

    n_done = 0
    n_batches = int(np.ceil(n_traj / batch_size))
    for b in range(n_batches):
        n_batch = min(batch_size, n_traj - n_done)
        seeds_b = all_seeds[n_done:n_done + n_batch]

        psi_batch, jumps_batch, label_batch = _mc_trajectories_core_labeled(
            psi0, U_H_np, M0_arr, M1_arr, channel_type, channel_alpha,
            n_batch, n_times, save_step, seeds_b)

        psi_traj_all[:, :, n_done:n_done + n_batch] = psi_batch
        jump_counts_all[:, n_done:n_done + n_batch] = jumps_batch
        label_traj_all[:, n_done:n_done + n_batch] = label_batch
        
        psi_traj_all.flush()
        jump_counts_all.flush()
        label_traj_all.flush()
        
        n_done += n_batch

    tmp_psi.close()
    tmp_jumps.close()
    tmp_labels.close()

    return psi_traj_all, jump_counts_all, label_traj_all, all_seeds, tmp_psi.name, tmp_jumps.name, tmp_labels.name


def psi_traj_to_rho_avg(psi_traj):
    return np.einsum('itk,jtk->tij', psi_traj, np.conj(psi_traj)) / psi_traj.shape[2]

def site_population_stats(psi_traj, eigenvectors):
    N, n_saved, n_traj = psi_traj.shape
    psi_site_traj = np.einsum('ia,atk->itk', eigenvectors, psi_traj)
    pop_traj = np.abs(psi_site_traj) ** 2

    pop_traj_mean = np.mean(pop_traj, axis=2).T
    pop_std = np.std(pop_traj, axis=2, ddof=1).T
    pop_traj_stderr = pop_std / np.sqrt(n_traj)

    return pop_traj_mean, pop_traj_stderr


# ====================
# Physical parameters
# ====================
c_light_cm_fs = 2.99792458e-5
cm1_to_radfs = 2.0 * np.pi * c_light_cm_fs
KB_cm1_per_K = 0.695034800

T_kelvin = 347.0
lam_cm1 = 35.0
lam = lam_cm1 * cm1_to_radfs  
Omega = 1.0
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

    dt = 1.0
    tf = 10000.0
    save_step = 10
    times = np.arange(0.0, tf, dt)
    n_times = len(times)

    n_traj = 10000
    master_seed = 42
    batch_size = 500

    if len(sys.argv) > 1:
        theta_deg = float(sys.argv[1])
        bash_mode = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    else:
        theta_deg = 0.0
        bash_mode = "local_test"

    theta = np.radians(theta_deg)

    results_dir = "../Results/Data/"
    os.makedirs(results_dir, exist_ok=True)

    print(f"--- Inizializzazione (theta = {theta_deg} deg, n_traj = {n_traj}, mode = {bash_mode}) ---")

    rho0_site = np.zeros((N_site, N_site), dtype=complex)
    rho0_site[0, 0] = 1.0
    rho0_exc = eigenvectors.conj().T @ rho0_site @ eigenvectors

    psi0_site = np.zeros(N_site, dtype=complex)
    psi0_site[0] = 1.0
    psi0_exc = eigenvectors.conj().T @ psi0_site

    channels = build_channel_unitaries(eigenergies, s_weights, w_ab, C_func, dt)
    U_H = build_free_evolution_unitary(eigenergies, dt)
    print(f"Number of collisional channels: {len(channels)} (expected {N_site + N_site*(N_site-1)})")

    L_list, gamma_list = build_redfield_jump_operators(eigenergies, s_weights, w_ab, C_func)
    H_tot_exc = np.diag(eigenergies).astype(complex)

    kraus_closed = build_kraus_operators(eigenergies, s_weights, w_ab, C_func, dt)

    print("\n--- Sanity checks ---")
    print(f"KMS relative error: {check_KMS(C_func, eigenergies[1]-eigenergies[0], beta):.3e}")
    print(f"Kraus completeness deviation: {check_kraus_completeness(kraus_closed, N_site):.3e}")

    # ==========================
    # Deterministic Evolutions
    # ==========================
    print("\n--- Calcolo dinamica Master Equation (Redfield)... ---")
    rho_redfield_exc = Redfield_evo(rho0_exc, H_tot_exc, gamma_list, L_list, times)
    rho_redfield_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_redfield_exc, eigenvectors.conj())
    
    print("--- Calcolo dinamica Collisionale (trace ancilla, deterministico)... ---")
    rho_trace_coll_exc = collisional_trace_evo(rho0_exc, channels, U_H, times)
    rho_trace_coll_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_trace_coll_exc, eigenvectors.conj())
    
    # Calculate populations before downsampling for the print statements
    pop_site_redfield = np.real(np.diagonal(rho_redfield_site, axis1=1, axis2=2))
    pop_trace_coll_site = np.real(np.diagonal(rho_trace_coll_site, axis1=1, axis2=2))

    print(f"Trace at t=0 (Collisional): {np.real(np.trace(rho_trace_coll_site[0])):.5f}")
    print(f"Trace at t_f (Collisional): {np.real(np.trace(rho_trace_coll_site[-1])):.5f}")
    print(f"Max abs diff (Collisional vs Redfield) at t_f: "
          f"{np.max(np.abs(pop_trace_coll_site[-1] - pop_site_redfield[-1])):.3e}")

    # DOWN-SAMPLE deterministic arrays to match Numba saved times
    times_saved = times[::save_step]
    n_saved = len(times_saved)
    
    rho_redfield_exc = rho_redfield_exc[::save_step]
    rho_redfield_site = rho_redfield_site[::save_step]
    rho_trace_coll_exc = rho_trace_coll_exc[::save_step]
    rho_trace_coll_site = rho_trace_coll_site[::save_step]

    # ==========================
    # Monte Carlo Evolution
    # ==========================
    print(f"\n--- Calcolo dinamica Monte Carlo (Numba, {n_traj} traiettorie)... ---")
    t0 = time.time()

    if theta_deg == 0.0:
        print("Theta=0 -> quantum jump regime: tracking eigenstate collapse labels")
        psi_traj, jump_counts, label_traj, seeds, tmp_psi_name, tmp_jumps_name, tmp_labels_name = compute_trajectories_numba_labeled(
            psi0_exc, U_H, kraus_closed, n_traj, n_times,
            save_step=save_step, master_seed=master_seed, batch_size=batch_size)
    else:
        psi_traj, jump_counts, seeds, tmp_psi_name, tmp_jumps_name = compute_trajectories_numba(
            psi0_exc, U_H, kraus_closed, theta, n_traj, n_times,
            save_step=save_step, master_seed=master_seed, batch_size=batch_size)
        label_traj = None
        tmp_labels_name = None

    t_mc = time.time() - t0
    print(f"Elapsed time: {t_mc:.2f} s")

    # ==========================
    # Metrics Computation
    # ==========================
    pop_traj_mean, pop_traj_stderr = site_population_stats(psi_traj, eigenvectors)
    rho_traj_avg_exc = psi_traj_to_rho_avg(psi_traj)
    rho_traj_avg_site = np.einsum('ia,tab,jb->tij', eigenvectors, rho_traj_avg_exc, eigenvectors.conj())
    total_jumps = np.sum(jump_counts, axis=1)

    print(f"Trace at t_f (MC, mean): {np.sum(pop_traj_mean[-1]):.5f}")
    print(f"Total jumps recorded (all trajectories, all channels, cumulative over run): {np.sum(total_jumps)}")
    print(f"Final-time populations (MC): {np.round(pop_traj_mean[-1], 5)} +/- {np.round(pop_traj_stderr[-1], 5)}")

    # ==========================
    # Data Saving
    # ==========================
    def _make_fname_npz(results_dir, theta_deg, dt, n_traj):
        dt_str = f"{dt:.2f}".replace(".", "p")
        theta_str = f"{theta_deg:.3f}".replace(".", "p")
        return os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{n_traj}.npz")

    fname_npz = _make_fname_npz(results_dir, theta_deg, dt, n_traj)

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
        'times': times_saved, 'dt': dt, 'tf': tf, 'n_times': n_saved,
        'theta': theta, 'theta_deg': theta_deg,
        'n_traj': n_traj, 'master_seed': master_seed,
        'N_site': N_site,
        'eigenergies': eigenergies, 'eigenvectors': eigenvectors,
        's_weights': s_weights, 'w_ab': w_ab,
        'lam_cm1': lam_cm1, 'Omega': Omega, 'T_kelvin': T_kelvin,
        'rho0_site': rho0_site, 'psi0_exc': psi0_exc
    }

    if label_traj is not None:
        save_dict['label_traj'] = label_traj.astype(np.int16)

    np.savez_compressed(fname_npz, **save_dict)
    print(f"\nSaved -> {os.path.basename(fname_npz)}")

    if tmp_psi_name is not None and os.path.exists(tmp_psi_name):
        os.remove(tmp_psi_name)
    if tmp_jumps_name is not None and os.path.exists(tmp_jumps_name):
        os.remove(tmp_jumps_name)
    if tmp_labels_name is not None and os.path.exists(tmp_labels_name):
        os.remove(tmp_labels_name)

    print("\n" + "=" * 40)
    print("COMPUTATION COMPLETED!")
    print(f"  - Angle (theta): {theta_deg} degrees ({theta:.4f} rad)")
    print(f"  - mode: {bash_mode}")
    print(f"  - n_traj = {n_traj}, dt = {dt} fs, tf = {tf} fs")
    print("=" * 40)