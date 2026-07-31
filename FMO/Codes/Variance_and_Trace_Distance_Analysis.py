#!/usr/bin/env python
# coding: utf-8

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to save files
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit
from scipy.optimize import curve_fit
from scipy.stats import poisson, norm, skew, kurtosis

# Suppress the specific scipy warning about catastrophic cancellation at t=0
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# =================================
# NUMBA OPTIMIZED METRIC FUNCTIONS
# =================================

@njit
def fidelity_generic_njit(rho, sigma):
    """Numba-compatible generalized fidelity."""
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho, 0.0) 
    
    diag_matrix = np.diag(np.sqrt(evals_rho)).astype(np.complex128)
    
    sqrt_rho = evecs_rho @ diag_matrix @ evecs_rho.conj().T
    inner_matrix = sqrt_rho @ sigma.astype(np.complex128) @ sqrt_rho
    inner_matrix = 0.5 * (inner_matrix + inner_matrix.conj().T)
    
    evals_inner = np.linalg.eigvalsh(inner_matrix)
    evals_inner = np.maximum(evals_inner, 0.0)
    
    fidelity = np.sum(np.sqrt(evals_inner))**2
    return min(1.0, fidelity)

@njit
def trace_distance_generic_njit(rho, sigma):
    """Numba-compatible generalized trace distance."""
    diff = rho - sigma
    diff = 0.5 * (diff + diff.conj().T)
    
    eigenvalues = np.linalg.eigvalsh(diff)
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    return min(1.0, t_dist)

# =================================
# GENERALIZED VARIANCE FUNCTIONS
# =================================

def compute_total_variance(observable_matrix, psi_traj):
    """
    Computes the Total Variance decomposition for a generic Hermitian observable.
    
    Parameters:
    -----------
    observable_matrix : ndarray, shape (N, N)
        The Hermitian matrix representing the quantum observable (O).
    psi_traj : ndarray, shape (N, n_times, n_traj)
        The pure state trajectories (must be in the same basis as observable_matrix).
        
    Returns:
    --------
    var_total : ndarray, shape (n_times,)
        The total variance of the observable over time.
    var_quant : ndarray, shape (n_times,)
        The intrinsic quantum variance (mean of internal variances).
    var_stat  : ndarray, shape (n_times,)
        The classical statistical variance (variance of the expected values).
    """
    
    # 1. Compute the square of the observable operator (O^2)
    observable_sq = observable_matrix @ observable_matrix
    
    # 2. Apply operators to the state vectors across all times and trajectories
    O_psi = np.tensordot(observable_matrix, psi_traj, axes=([1], [0]))
    O2_psi = np.tensordot(observable_sq, psi_traj, axes=([1], [0]))
    
    # 3. Compute conditional expectation values for each trajectory and time step
    E_k = np.real(np.sum(np.conj(psi_traj) * O_psi, axis=0))
    E2_k = np.real(np.sum(np.conj(psi_traj) * O2_psi, axis=0))
    
    # 4. Compute the Quantum Variance term inside each trajectory
    var_quant_k = E2_k - E_k**2
    
    # Ensure no negative variances due to numerical precision errors near zero
    var_quant_k = np.maximum(var_quant_k, 0.0)
    
    # 5. Law of Total Variance decomposition (averaging over the ensemble)
    var_quant = np.mean(var_quant_k, axis=1)  # Mean of intrinsic quantum variances
    var_stat = np.var(E_k, axis=1)            # Statistical variance of the trajectory means
    
    # The Total Variance is exactly the sum of the two terms
    var_total = var_quant + var_stat          
    
    return var_total, var_quant, var_stat


def get_exact_variance(observable_matrix, rho_t):
    """
    Computes the exact variance of a Hermitian observable using the full density matrix.
    Formula: Var(O) = Tr(O^2 * rho) - (Tr(O * rho))^2
    
    Parameters:
    -----------
    observable_matrix : ndarray, shape (N, N)
    rho_t : ndarray, shape (n_times, N, N)
    """
    observable_sq = observable_matrix @ observable_matrix
    
    # Using einsum to efficiently compute Trace(O * rho(t)) for each time step t
    # 'ik, tki -> t' means sum over i and k for each t
    E_O = np.real(np.einsum('ik,tki->t', observable_matrix, rho_t))
    E_O2 = np.real(np.einsum('ik,tki->t', observable_sq, rho_t))
    
    return np.maximum(E_O2 - E_O**2, 0.0)


# ==========================
# Input Parsing from Bash
# ==========================
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 90.0  # Default fallback if run manually

# --- Must match the values used in the main simulation script ---
dt = 1.0
N_traj = 10000

dt_str = f"{dt:.2f}".replace(".", "p")
theta_str = f"{theta_deg:.3f}".replace(".", "p")

results_dir = "../Results/Data/"
Output_dir = f"../Results/Plot/Variance_Analysis/{theta_str}"
os.makedirs(Output_dir, exist_ok=True)

fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Theta = {theta_deg} deg")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

# Data Extraction
times = data['times']
dt_val = float(data['dt'])
N_site = int(data['N_site'])
eigenergies = data['eigenergies']
eigenvectors = data['eigenvectors']
psi0_exc = data['psi0_exc']

psi_traj_exc = data['psi_traj']         # (N_site, n_times, n_traj), exciton basis, complex64
jump_counts = data['jump_counts']       # (n_times, n_traj) - M1 APPLICATIONS COUNT

# Load density matrices for Trace Distance convergence analysis
if 'rho_redfield_site' in data and 'rho_traj_avg_site' in data:
    rho_redfield_site = data['rho_redfield_site']
    rho_traj_avg_site = data['rho_traj_avg_site']
else:
    print("Warning: Density matrices (site basis) not found.")
    rho_redfield_site = None

if 'rho_redfield_exc' in data and 'rho_traj_avg_exc' in data:
    rho_redfield_exc = data['rho_redfield_exc']
    rho_traj_avg_exc = data['rho_traj_avg_exc']
else:
    print("Warning: Density matrices (exciton basis) not found.")
    rho_redfield_exc = None

n_times = len(times)
n_traj = psi_traj_exc.shape[2]

# ==========================
# Site-basis single-trajectory populations
# ==========================
psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)   # (N_site, n_times, n_traj)
pop_traj_site = np.abs(psi_traj_site) ** 2                             # (N_site, n_times, n_traj)

# ==========================
# STATISTICAL ANALYSIS: MEAN, VARIANCE, SKEWNESS, KURTOSIS
# ==========================
print("Computing Statistical Moments over time...")
mean_pop_time = np.mean(pop_traj_site, axis=2)
var_pop_time = np.var(pop_traj_site, axis=2)
skew_pop_time = skew(pop_traj_site, axis=2, nan_policy='omit')
kurt_pop_time = kurtosis(pop_traj_site, axis=2, fisher=True, nan_policy='omit')

# ===========================
# General plot setup
# ===========================
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': ':',
    'figure.autolayout': True
})

def save_fig(fig, filename):
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)

SITE_LABELS = [f"Site {i+1}" for i in range(N_site)]
colors = plt.cm.viridis(np.linspace(0, 1, N_site))

# ==========================================
# PLOT 1: All Statistical Moments over Time
# ==========================================
fig1, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
ax_mean, ax_var, ax_skew, ax_kurt = axes

for i in range(N_site):
    ax_mean.plot(times, mean_pop_time[i, :], color=colors[i], linewidth=2, label=SITE_LABELS[i])
    ax_var.plot(times, var_pop_time[i, :], color=colors[i], linewidth=2, label=SITE_LABELS[i])
    ax_skew.plot(times, skew_pop_time[i, :], color=colors[i], linewidth=2, label=SITE_LABELS[i])
    ax_kurt.plot(times, kurt_pop_time[i, :], color=colors[i], linewidth=2, label=SITE_LABELS[i])

ax_mean.set_title(f'Statistical Moments over Trajectories (Theta = {theta_deg}°)')
ax_mean.set_ylabel('Mean')
ax_mean.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

ax_var.set_ylabel('Variance')

ax_skew.set_ylabel('Skewness ($\\gamma_1$)')
ax_skew.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6) 
ax_skew.set_ylim(-5, 5)

ax_kurt.set_ylabel('Excess Kurtosis ($K - 3$)')
ax_kurt.set_xlabel('Time (fs)')
ax_kurt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6) 
ax_kurt.set_ylim(-5, 15)

save_fig(fig1, f'All_Statistical_Moments_Time_Theta_{theta_str}')


# ====================================================================
# NEW PLOTS: LAW OF TOTAL VARIANCE WITH GENERIC OBSERVABLES
# ====================================================================

# ---------------------------------------------------------
# A. Site Populations
# ---------------------------------------------------------
if rho_redfield_site is not None:
    print("Computing Total Variance Theorem: Site Populations...")
    
    fig_tv_site, axes_tv_site = plt.subplots(N_site, 1, figsize=(10, 2.5 * N_site), sharex=True)
    if N_site == 1: axes_tv_site = [axes_tv_site]
        
    for i in range(N_site):
        # Build Population Projector Observable |i><i|
        O_pop = np.zeros((N_site, N_site), dtype=np.complex128)
        O_pop[i, i] = 1.0
        
        # Calculate components using the generic functions
        var_tot_traj, var_quant, var_stat = compute_total_variance(O_pop, psi_traj_site)
        var_tot_exact = get_exact_variance(O_pop, rho_redfield_site)
        
        max_err = np.max(np.abs(var_tot_exact - var_tot_traj))
        print(f"  -> Site {i+1} Law of Total Variance max error: {max_err:.2e}")
        
        ax = axes_tv_site[i]
        ax.plot(times, var_tot_exact, color='black', linewidth=3, linestyle='--', label='Total Variance (Redfield)')
        ax.plot(times, var_stat, color='red', linewidth=2, alpha=0.8, label='Statistical Variance')
        ax.plot(times, var_quant, color='blue', linewidth=2, alpha=0.8, label='Quantum Variance')
        ax.plot(times, var_tot_traj, color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
        
        ax.set_ylabel(f'Site {i+1}')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f'Law of Total Variance - Site Populations (Theta = {theta_deg}°)')

    axes_tv_site[-1].set_xlabel('Time (fs)')
    fig_tv_site.tight_layout()
    save_fig(fig_tv_site, f'Law_Total_Variance_Sites_Theta_{theta_str}')


# ---------------------------------------------------------
# B. Exciton Populations
# ---------------------------------------------------------
if rho_redfield_exc is not None:
    print("Computing Total Variance Theorem: Exciton Populations...")
    
    fig_tv_exc, axes_tv_exc = plt.subplots(N_site, 1, figsize=(10, 2.5 * N_site), sharex=True)
    if N_site == 1: axes_tv_exc = [axes_tv_exc]
        
    for i in range(N_site):
        # Build Population Projector Observable |alpha><alpha|
        O_exc = np.zeros((N_site, N_site), dtype=np.complex128)
        O_exc[i, i] = 1.0
        
        var_tot_traj, var_quant, var_stat = compute_total_variance(O_exc, psi_traj_exc)
        var_tot_exact = get_exact_variance(O_exc, rho_redfield_exc)
        
        max_err = np.max(np.abs(var_tot_exact - var_tot_traj))
        print(f"  -> Exciton {i+1} Law of Total Variance max error: {max_err:.2e}")
        
        ax = axes_tv_exc[i]
        ax.plot(times, var_tot_exact, color='black', linewidth=3, linestyle='--', label='Total Variance (Redfield)')
        ax.plot(times, var_stat, color='red', linewidth=2, alpha=0.8, label='Statistical Variance')
        ax.plot(times, var_quant, color='blue', linewidth=2, alpha=0.8, label='Quantum Variance')
        ax.plot(times, var_tot_traj, color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
        
        ax.set_ylabel(f'Exciton {i+1}')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f'Law of Total Variance - Exciton Populations (Theta = {theta_deg}°)')

    axes_tv_exc[-1].set_xlabel('Time (fs)')
    fig_tv_exc.tight_layout()
    save_fig(fig_tv_exc, f'Law_Total_Variance_Excitons_Theta_{theta_str}')


# ---------------------------------------------------------
# C. Selected Coherences (Real and Imaginary Parts)
# ---------------------------------------------------------
if rho_redfield_site is not None:
    print("Computing Total Variance Theorem: Coherences (Site Basis)...")
    
    # Define a list of 10 relevant coherences (e.g., adjacent and some non-adjacent sites)
    # Python uses 0-based indexing (0 corresponds to Site 1)
    coherence_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # Nearest neighbors
        (0, 5), (1, 3), (2, 4), (0, 6)                   # Long-range coupling/correlations
    ]
    n_pairs = len(coherence_pairs)
    
    fig_creal, axes_creal = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    fig_cimag, axes_cimag = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    
    for idx, (m, n) in enumerate(coherence_pairs):
        
        # --- REAL PART OBSERVABLE: |m><n| + |n><m| ---
        O_real = np.zeros((N_site, N_site), dtype=np.complex128)
        O_real[m, n] = 1.0
        O_real[n, m] = 1.0
        
        vt_real_traj, vq_real, vs_real = compute_total_variance(O_real, psi_traj_site)
        vt_real_exact = get_exact_variance(O_real, rho_redfield_site)
        
        # --- IMAGINARY PART OBSERVABLE: -i|m><n| + i|n><m| ---
        O_imag = np.zeros((N_site, N_site), dtype=np.complex128)
        O_imag[m, n] = -1.0j
        O_imag[n, m] = 1.0j
        
        vt_imag_traj, vq_imag, vs_imag = compute_total_variance(O_imag, psi_traj_site)
        vt_imag_exact = get_exact_variance(O_imag, rho_redfield_site)
        
        # Logs
        err_real = np.max(np.abs(vt_real_exact - vt_real_traj))
        err_imag = np.max(np.abs(vt_imag_exact - vt_imag_traj))
        print(f"  -> Coherence ({m+1},{n+1}) | Max Error - Real: {err_real:.2e}, Imag: {err_imag:.2e}")
        
        # Plot Real Part
        ax_r = axes_creal[idx]
        ax_r.plot(times, vt_real_exact, color='black', linewidth=3, linestyle='--')
        ax_r.plot(times, vs_real, color='red', linewidth=2, alpha=0.8)
        ax_r.plot(times, vq_real, color='blue', linewidth=2, alpha=0.8)
        ax_r.plot(times, vt_real_traj, color='limegreen', linewidth=3, linestyle=':')
        ax_r.set_ylabel(f'Re( $\\rho_{{{m+1}{n+1}}}$ )')
        if idx == 0:
            ax_r.set_title(f'Law of Total Variance - Coherences REAL Part (Theta = {theta_deg}°)')
            # Adding custom legend to the first plot
            ax_r.plot([], [], color='black', linewidth=3, linestyle='--', label='Total Variance (Redfield)')
            ax_r.plot([], [], color='red', linewidth=2, label='Statistical Variance')
            ax_r.plot([], [], color='blue', linewidth=2, label='Quantum Variance')
            ax_r.plot([], [], color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
            ax_r.legend(loc='upper right', fontsize=8)

        # Plot Imaginary Part
        ax_i = axes_cimag[idx]
        ax_i.plot(times, vt_imag_exact, color='black', linewidth=3, linestyle='--')
        ax_i.plot(times, vs_imag, color='red', linewidth=2, alpha=0.8)
        ax_i.plot(times, vq_imag, color='blue', linewidth=2, alpha=0.8)
        ax_i.plot(times, vt_imag_traj, color='limegreen', linewidth=3, linestyle=':')
        ax_i.set_ylabel(f'Im( $\\rho_{{{m+1}{n+1}}}$ )')
        if idx == 0:
            ax_i.set_title(f'Law of Total Variance - Coherences IMAGINARY Part (Theta = {theta_deg}°)')
            ax_i.legend(loc='upper right', fontsize=8)
            ax_i.plot([], [], color='black', linewidth=3, linestyle='--', label='Total Variance (Redfield)')
            ax_i.plot([], [], color='red', linewidth=2, label='Statistical Variance')
            ax_i.plot([], [], color='blue', linewidth=2, label='Quantum Variance')
            ax_i.plot([], [], color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
            ax_i.legend(loc='upper right', fontsize=8)

    axes_creal[-1].set_xlabel('Time (fs)')
    axes_cimag[-1].set_xlabel('Time (fs)')
    
    fig_creal.tight_layout()
    fig_cimag.tight_layout()
    
    save_fig(fig_creal, f'Law_Total_Variance_Coherences_REAL_Theta_{theta_str}')
    save_fig(fig_cimag, f'Law_Total_Variance_Coherences_IMAG_Theta_{theta_str}')
    
    # # ==========================================
    # # PLOT B: BERRY-ESSEEN (Errore vs Skewness)
    # # ==========================================
    # # Il CLT dice che l'errore scala come SEM. 
    # # Berry-Esseen dice che l'approssimazione gaussiana fatica proporzionalmente alla Skewness.
    
    # # Calcolo del Standard Error of the Mean (SEM)
    # sem = np.sqrt(var_stat) / np.sqrt(n_traj)
    # # Modulo della skewness
    # abs_skew = np.abs(skew_pop_time[target_site, :])
    
    # fig_be, ax_be1 = plt.subplots(figsize=(8, 5))
    # ax_be2 = ax_be1.twinx()  # Secondo asse Y per la skewness
    
    # # Plottiamo l'errore standard e la skewness
    # line1, = ax_be1.plot(times, sem, color='darkblue', linewidth=2, label='Standard Error ($\\sigma_{stat} / \\sqrt{N}$)')
    # line2, = ax_be2.plot(times, abs_skew, color='crimson', linewidth=2, linestyle='--', label='Absolute Skewness $|\\gamma_1|$')
    
    # ax_be1.set_xlabel('Time (fs)')
    # ax_be1.set_ylabel('Monte Carlo Standard Error', color='darkblue')
    # ax_be2.set_ylabel('Berry-Esseen Penalty (Abs Skewness)', color='crimson')
    # ax_be1.set_title(f'Berry-Esseen & CLT Convergence - Site {target_site+1} (Theta = {theta_deg}°)')
    
    # # Colora i tick e le etichette degli assi per chiarezza
    # ax_be1.tick_params(axis='y', labelcolor='darkblue')
    # ax_be2.tick_params(axis='y', labelcolor='crimson')
    # ax_be1.grid(True, alpha=0.3)
    
    # # Uniamo le legende dei due assi gemelli
    # lines = [line1, line2]
    # labels = [l.get_label() for l in lines]
    # ax_be1.legend(lines, labels, loc='upper right')
    
    # save_fig(fig_be, f'Berry_Esseen_CLT_Site{target_site+1}_Theta_{theta_str}')


print("Statistical analysis and image saving successfully completed!")

# # ==========================================
# # PLOT 2: Trace distance (Redfield vs Avg Trajectories)
# # ==========================================
# if 'rho_redfield_site' in locals() and 'rho_traj_avg_site' in locals():
#     td_time = np.zeros(n_times)
#     for t in range(n_times):
#         td_time[t] = trace_distance_generic_njit(rho_redfield_site[t], rho_traj_avg_site[t])

#     fig2, ax2 = plt.subplots(figsize=(8, 5))
#     ax2.plot(times, td_time, color='red', linewidth=2, label='Trace Distance')
#     ax2.set_title(f'Trace Distance: Redfield vs Avg Trajectories (Theta = {theta_deg}°)')
#     ax2.set_xlabel('Time (fs)')
#     ax2.set_ylabel('Trace Distance')
#     ax2.set_yscale('log') 
#     ax2.legend(loc='best')
    
#     save_fig(fig2, f'Trace_Distance_Theta_{theta_str}')

# # ==========================================
# # PLOT 3: Convergenza Trace Distance vs N (Teorema Limite Centrale)
# # ==========================================
# N_list = np.array([100, 200, 500, 1000, 2000, 4000, 8000, 10000])
# mean_td_values = []
# max_td_values = []

# print("Computing Trace Distance convergence for various N...")
# for N_sub in N_list:
#     # WARNING: for consistency with rho_redfield_exc, we use psi_traj_exc!
#     psi_sub = psi_traj_exc[:, :, :N_sub]
#     rho_sub_avg = np.einsum('itk, jtk -> tij', psi_sub, np.conjugate(psi_sub)) / N_sub
    
#     # Calculate Trace Distance over time
#     td_time_sub = np.zeros(n_times)
#     for t in range(n_times):
#         td_time_sub[t] = trace_distance_generic_njit(rho_redfield_exc[t], rho_sub_avg[t])
        
#     # Define how many initial steps to ignore to avoid the t=0 transient
#     skip_steps = 100  # Ignore the first 100 fs (assuming dt=1 fs)
        
#     # Calculate Mean and Maximum ONLY on steady-state data and append to the list
#     mean_td_values.append(np.mean(td_time_sub[skip_steps:]))
#     max_td_values.append(np.max(td_time_sub[skip_steps:]))

# mean_td_values = np.array(mean_td_values)
# max_td_values = np.array(max_td_values)

# def clt_fit(N, a):
#     return a / np.sqrt(N)

# popt_mean, _ = curve_fit(clt_fit, N_list, mean_td_values)
# a_mean = popt_mean[0]

# popt_max, _ = curve_fit(clt_fit, N_list, max_td_values)
# a_max = popt_max[0]

# fig3, ax3 = plt.subplots(figsize=(8, 6))
# N_smooth = np.linspace(N_list[0], N_list[-1], 200)

# ax3.plot(N_list, mean_td_values, 'bo', markersize=8, label='Data: Time-Averaged TD')
# ax3.plot(N_smooth, clt_fit(N_smooth, a_mean), 'b--', linewidth=2, label='Fit 1/$\\sqrt{N}$')
# ax3.plot(N_list, max_td_values, 'ro', markersize=8, label='Data: Maximum TD')
# ax3.plot(N_smooth, clt_fit(N_smooth, a_max), 'r--', linewidth=2, label='Fit 1/$\\sqrt{N}$')

# ax3.set_xscale('log')
# ax3.set_yscale('log')
# ax3.set_xlabel('Number of Trajectories (N)')
# ax3.set_ylabel('Trace Distance Error')
# ax3.set_title(f'Monte Carlo Convergence Testing (Central Limit Theorem)\nTheta = {theta_deg}°')
# ax3.legend()
# ax3.grid(True, which="both", ls="--", alpha=0.5)

# save_fig(fig3, f'Convergence_CLT_Theta_{theta_str}')

# # ==========================================
# # PLOT 4: STATISTICAL DISTRIBUTION OF MEASUREMENT OUTCOMES (M1)
# # ==========================================
# print("Computing universal statistical distribution of M1 counts...")

# # Sum of all M1 applications over time per single trajectory
# n_jumps_total = jump_counts.sum(axis=0)  
# mean_jumps = np.mean(n_jumps_total)
# var_jumps = np.var(n_jumps_total)

# fig4, ax4 = plt.subplots(figsize=(8, 5))

# if theta_deg == 0.0:
#     # --- QUANTUM JUMP REGIME (Poisson Fit) ---
#     max_jumps = int(np.max(n_jumps_total))
#     bins = np.arange(-0.5, max_jumps + 1.5, 1) 
    
#     ax4.hist(n_jumps_total, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Simulated Jumps (M1)')
    
#     k_values = np.arange(0, max_jumps + 1)
#     poisson_pmf = poisson.pmf(k_values, mu=mean_jumps)
    
#     ax4.plot(k_values, poisson_pmf, 'ro--', markersize=6, linewidth=2, label=f'Poisson Fit ($\\lambda$ = {mean_jumps:.2f})')
#     ax4.set_title(f'Quantum Jump Regime ($\\Theta = 0^\\circ$)\nPoisson Distribution of Discrete Events')
    
#     if max_jumps < 20:
#         ax4.set_xticks(k_values) 

# else:
#     # --- DIFFUSIVE REGIME (Gaussian / De Moivre-Laplace Fit) ---
#     bins_c = np.linspace(np.min(n_jumps_total), np.max(n_jumps_total), 50)
    
#     ax4.hist(n_jumps_total, bins=bins_c, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Simulated Omodyne Clicks (M1)')
    
#     mu_gauss, std_gauss = norm.fit(n_jumps_total)
#     x_gauss = np.linspace(np.min(n_jumps_total)*0.9, np.max(n_jumps_total)*1.1, 200)
#     pdf_gauss = norm.pdf(x_gauss, mu_gauss, std_gauss)
    
#     ax4.plot(x_gauss, pdf_gauss, 'g--', linewidth=2.5, label=f'Gaussian Fit\n($\\mu$={mu_gauss:.1f}, $\\sigma$={std_gauss:.1f})')
#     ax4.set_title(f'Diffusive Limit ($\\Theta = {theta_deg}^\\circ$)\nGaussian Distribution of Measurement Outcomes')

# ax4.set_xlabel('Total Number of $M_1$ Applications')
# ax4.set_ylabel('Probability Density')
# ax4.legend()
# ax4.grid(True, alpha=0.3)

# save_fig(fig4, f'M1_Counts_Distribution_Theta_{theta_str}')

# # ==========================================
# # PLOT 5: FANO FACTOR EVOLUTION (Var/Mean) OVER TIME
# # ==========================================
# print("Computing Fano Factor evolution over time...")

# cumulative_jumps = np.cumsum(jump_counts, axis=0) 
# mean_t = np.mean(cumulative_jumps, axis=1)
# var_t = np.var(cumulative_jumps, axis=1)

# fano_t = np.zeros_like(mean_t)
# mask = mean_t > 0
# fano_t[mask] = var_t[mask] / mean_t[mask]

# # Set the first point to the expected theoretical limit to avoid ruining the plot
# fano_t[~mask] = 1.0 if theta_deg == 0.0 else 0.5  

# fig5, ax5 = plt.subplots(figsize=(8, 5))

# ax5.plot(times[mask], fano_t[mask], color='purple', linewidth=2, label='Simulated $\\text{Var}(N)/\\langle N \\rangle$')

# if theta_deg == 0.0:
#     ax5.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Poisson Theoretical Limit (1.0)')
# else:
#     # For a Binomial with p=0.5, Var/Mean = (Np(1-p)) / (Np) = 1-p = 0.5
#     ax5.axhline(0.5, color='green', linestyle='--', linewidth=2, label='Binomial Theoretical Limit (0.5)')

# ax5.set_xlabel('Time (fs)')
# ax5.set_ylabel('Variance / Mean')
# ax5.set_title(f'Statistical Index Evolution over Time (Theta = {theta_deg}°)')

# if len(fano_t[mask]) > 0:
#     y_max = max(1.2, np.max(fano_t[mask])*1.1)
#     y_min = min(0.3, np.min(fano_t[mask])*0.9)
#     ax5.set_ylim(y_min, y_max)

# ax5.legend()
# ax5.grid(True, alpha=0.3)

# save_fig(fig5, f'Statistical_Index_Evolution_Theta_{theta_str}')

# # ==========================================
# # PLOT 6: DENSITY HEATMAP (ONE PLOT PER SITE)
# # ==========================================
# print("Computing independent Density Heatmaps for each site...")

# # Define bins for the Y-axis (population goes strictly from 0 to 1)
# n_pop_bins = 100
# pop_bins = np.linspace(0.0, 1.0, n_pop_bins + 1)

# # For Matplotlib hist2d, we need the "edges" of the time bins
# dt_plot = times[1] - times[0]
# time_bins = np.append(times, times[-1] + dt_plot)

# # Create an X array repeating the time axis for each trajectory
# X_times = np.repeat(times, n_traj)

# # If the Redfield array was loaded successfully at the beginning, calculate its populations
# has_redfield = False
# if 'rho_redfield_site' in locals():
#     # Extract the diagonal elements of the density matrix at each time step
#     pop_redfield = np.real(np.diagonal(rho_redfield_site, axis1=1, axis2=2))
#     has_redfield = True

# # Create an independent plot for each of the 7 sites
# for i in range(N_site):
#     figD, axD = plt.subplots(figsize=(8, 5))
    
#     # Flatten the trajectories axis for site i
#     Y_pops = pop_traj_site[i, :, :].flatten()
    
#     # Draw the 2D histogram with 'Blues' colormap in logarithmic scale
#     h, xedges, yedges, im = axD.hist2d(X_times, Y_pops, bins=[time_bins, pop_bins], 
#                                       cmap='Blues', norm=LogNorm(), density=False)
    
#     # Overlay the exact deterministic mean Redfield dynamics in red
#     if has_redfield:
#         axD.plot(times, pop_redfield[:, i], color='red', linewidth=2.5, linestyle='--', 
#                  label='Redfield Exact (Mean Path)')
    
#     axD.set_xlabel('Time (fs)')
#     axD.set_ylabel(f'Population (Site {i+1})')
#     axD.set_ylim(0, 1)
#     axD.set_title(f'Trajectory Density: Site {i+1} ($\\Theta = {theta_deg}^\\circ$)')
    
#     # Add the colorbar
#     cbar = figD.colorbar(im, ax=axD, pad=0.02)
#     cbar.set_label('Number of Trajectories')
    
#     if has_redfield:
#         axD.legend(loc='upper right')
        
#     # Save the plot with the site index in the filename
#     save_fig(figD, f'Population_Heatmap_Site_{i+1}_Theta_{theta_str}')

print("Statistical analysis and image saving successfully completed!")