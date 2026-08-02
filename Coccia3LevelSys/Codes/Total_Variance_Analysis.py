#!/usr/bin/env python
# coding: utf-8
"""
Law-of-Total-Variance analysis for the 3-level collisional-model trajectories
(site basis only), matching the data saved by the trajectory-generation script
that calls compute_trajectory_wf(...) and np.savez_compressed(..., rho_tot_all=...).

Key difference vs. the FMO version: here each trajectory is stored as a FULL
density matrix rho_k(t) (shape N x N), not a state vector psi_k(t). Since the
Kraus-operator (M0/M1) evolution never mixes states, every trajectory stays
EXACTLY pure (rho_k(t) = |psi_k(t)><psi_k(t)|) - this script checks that
assumption once (purity diagnostic) and then exploits it: the same algebraic
identities used for the FMO case (O^2=O for population projectors,
O_real^2=O_imag^2=|m><m|+|n><n| for 2-level coherences) still hold, and are
now even more direct since we can read rho_mn,k(t) straight from the stored
array, with no need to reconstruct it from amplitudes.

"Exact" reference for the Law of Total Variance: the Lindblad master-equation
solution (rho_list_lindblad), since that is the actual open-system dynamics
that the stochastic trajectories are unraveling. rho_trace (the collision-
model dynamics obtained by tracing out the ancilla) is used only as an
independent cross-check that the two "exact" descriptions agree - it is not
itself the LTV reference.

NOTE on paths: results_dir / Output_dir below mirror the "../../Results/..."
relative depth used in the trajectory-generation script. Adjust if you place
this analysis script at a different depth in your folder structure.
"""

import sys
import os
import warnings
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# =================================
# NUMBA METRIC FUNCTIONS (for the convergence / cross-check plots)
# =================================

@njit
def fidelity_generic_njit(rho, sigma):
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho, 0.0)
    diag_matrix = np.diag(np.sqrt(evals_rho)).astype(np.complex128)
    sqrt_rho = evecs_rho @ diag_matrix @ evecs_rho.conj().T
    inner_matrix = sqrt_rho @ sigma.astype(np.complex128) @ sqrt_rho
    inner_matrix = 0.5 * (inner_matrix + inner_matrix.conj().T)
    evals_inner = np.linalg.eigvalsh(inner_matrix)
    evals_inner = np.maximum(evals_inner, 0.0)
    fidelity = np.sum(np.sqrt(evals_inner)) ** 2
    return min(1.0, fidelity)


@njit
def trace_distance_generic_njit(rho, sigma):
    diff = rho - sigma
    diff = 0.5 * (diff + diff.conj().T)
    eigenvalues = np.linalg.eigvalsh(diff)
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    return min(1.0, t_dist)


def compute_matrix_metric_series(rho_a, rho_b, metric_fn):
    """rho_a, rho_b: shape (n_times, N, N)."""
    n_times = rho_a.shape[0]
    result = np.empty(n_times)
    for t in range(n_times):
        result[t] = metric_fn(rho_a[t].astype(np.complex128), rho_b[t].astype(np.complex128))
    return result


# ==========================================
# EXACT VARIANCE FROM THE MASTER-EQUATION DENSITY MATRIX
# ==========================================

def get_exact_variance(observable_matrix, rho_t):
    """
    Var(O) = Tr(O^2 rho) - Tr(O rho)^2
    observable_matrix : (N, N)
    rho_t : (n_times, N, N)
    """
    observable_sq = observable_matrix @ observable_matrix
    E_O = np.real(np.einsum('ik,tki->t', observable_matrix, rho_t))
    E_O2 = np.real(np.einsum('ik,tki->t', observable_sq, rho_t))
    return np.maximum(E_O2 - E_O ** 2, 0.0)


# ==========================================
# FAST ANALYTIC LAW-OF-TOTAL-VARIANCE (populations & coherences)
# ==========================================

def total_variance_projector(pop_k):
    """
    pop_k : ndarray, shape (n_times, n_traj). O=|i><i| is a projector, O^2=O,
    so quantum variance per trajectory = p(1-p) exactly (see FMO script for
    the full derivation).
    """
    var_quant_k = pop_k * (1.0 - pop_k)
    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(pop_k, axis=1)
    return var_quant + var_stat, var_quant, var_stat


def total_variance_coherence_rho(rho_mn_k, pop_m_k, pop_n_k, part='real'):
    """
    Same identity as in the FMO script (O_real^2 = O_imag^2 = |m><m|+|n><n|),
    but here rho_mn,k(t) = <m|rho_k(t)|n> is read directly from the stored
    per-trajectory density matrix instead of being built from amplitudes:

        <O_real>  =  rho_nm,k + rho_mn,k  =  2 Re(rho_mn,k)
        <O_imag>  =  i(rho_mn,k - rho_nm,k)  =  -2 Im(rho_mn,k)

    rho_mn_k, pop_m_k, pop_n_k : ndarray, shape (n_times, n_traj)
    """
    E_k = 2.0 * np.real(rho_mn_k) if part == 'real' else -2.0 * np.imag(rho_mn_k)
    E2_k = pop_m_k + pop_n_k
    var_quant_k = np.maximum(E2_k - E_k ** 2, 0.0)
    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(E_k, axis=1)
    return var_quant + var_stat, var_quant, var_stat


# ==========================
# PLOTTING HELPERS
# ==========================

def save_fig(fig, filename, output_dir):
    path_png = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)


def plot_ltv_panel(ax, times, var_exact, var_stat, var_quant, var_sum, ylabel,
                    show_legend=False, title=None):
    ax.plot(times, var_exact, color='black', linewidth=3, linestyle='--', label='Total Variance (Lindblad)')
    ax.plot(times, var_stat, color='red', linewidth=2, alpha=0.8, label='Statistical Variance')
    ax.plot(times, var_quant, color='blue', linewidth=2, alpha=0.8, label='Quantum Variance')
    ax.plot(times, var_sum, color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc='upper right', fontsize=8)


def analyze_population_ltv(times, pop_traj, rho_exact, N_states, state_labels,
                            phi_deg, phi_str, output_dir):
    """pop_traj : ndarray, shape (N_states, n_times, n_traj)."""
    fig, axes = plt.subplots(N_states, 1, figsize=(10, 2.5 * N_states), sharex=True)
    if N_states == 1:
        axes = [axes]

    for i in range(N_states):
        var_tot_traj, var_quant, var_stat = total_variance_projector(pop_traj[i])

        O_pop = np.zeros((N_states, N_states), dtype=np.complex128)
        O_pop[i, i] = 1.0
        var_tot_exact = get_exact_variance(O_pop, rho_exact)

        max_err = np.max(np.abs(var_tot_exact - var_tot_traj))
        print(f"  -> {state_labels[i]} Law of Total Variance max error: {max_err:.2e}")

        plot_ltv_panel(
            axes[i], times, var_tot_exact, var_stat, var_quant, var_tot_traj,
            ylabel=state_labels[i],
            show_legend=(i == 0),
            title=f'Law of Total Variance - Populations (Phi = {phi_deg}°)' if i == 0 else None,
        )

    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    save_fig(fig, f'Law_Total_Variance_Populations_Phi_{phi_str}', output_dir)


def analyze_coherence_ltv(times, rho_tot_all, pop_traj, rho_exact, coherence_pairs,
                           N_states, phi_deg, phi_str, output_dir):
    """
    rho_tot_all : ndarray, shape (N_states, N_states, n_times, n_traj)
    pop_traj    : ndarray, shape (N_states, n_times, n_traj) - populations,
                  reused here for the E2_k = p_m + p_n identity.
    """
    n_pairs = len(coherence_pairs)
    fig_real, axes_real = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    fig_imag, axes_imag = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes_real, axes_imag = [axes_real], [axes_imag]

    for idx, (m, n) in enumerate(coherence_pairs):
        rho_mn_k = rho_tot_all[m, n, :, :]
        pop_m_k, pop_n_k = pop_traj[m], pop_traj[n]

        vt_real, vq_real, vs_real = total_variance_coherence_rho(rho_mn_k, pop_m_k, pop_n_k, part='real')
        vt_imag, vq_imag, vs_imag = total_variance_coherence_rho(rho_mn_k, pop_m_k, pop_n_k, part='imag')

        O_real = np.zeros((N_states, N_states), dtype=np.complex128)
        O_real[m, n], O_real[n, m] = 1.0, 1.0
        O_imag = np.zeros((N_states, N_states), dtype=np.complex128)
        O_imag[m, n], O_imag[n, m] = -1.0j, 1.0j

        vt_real_exact = get_exact_variance(O_real, rho_exact)
        vt_imag_exact = get_exact_variance(O_imag, rho_exact)

        err_real = np.max(np.abs(vt_real_exact - vt_real))
        err_imag = np.max(np.abs(vt_imag_exact - vt_imag))
        print(f"  -> Coherence ({m},{n}) | Max Error - Real: {err_real:.2e}, Imag: {err_imag:.2e}")

        plot_ltv_panel(
            axes_real[idx], times, vt_real_exact, vs_real, vq_real, vt_real,
            ylabel=f'Re( $\\rho_{{{m}{n}}}$ )', show_legend=(idx == 0),
            title=f'Law of Total Variance - Coherences REAL Part (Phi = {phi_deg}°)' if idx == 0 else None,
        )
        plot_ltv_panel(
            axes_imag[idx], times, vt_imag_exact, vs_imag, vq_imag, vt_imag,
            ylabel=f'Im( $\\rho_{{{m}{n}}}$ )', show_legend=(idx == 0),
            title=f'Law of Total Variance - Coherences IMAGINARY Part (Phi = {phi_deg}°)' if idx == 0 else None,
        )

    axes_real[-1].set_xlabel('Time')
    axes_imag[-1].set_xlabel('Time')
    fig_real.tight_layout()
    fig_imag.tight_layout()
    save_fig(fig_real, f'Law_Total_Variance_Coherences_REAL_Phi_{phi_str}', output_dir)
    save_fig(fig_imag, f'Law_Total_Variance_Coherences_IMAG_Phi_{phi_str}', output_dir)


def plot_convergence(times, rho_a, rho_b, label, phi_deg, phi_str, output_dir):
    """rho_a, rho_b : shape (n_times, N, N). Trace distance & fidelity between them."""
    trace_dist = compute_matrix_metric_series(rho_a, rho_b, trace_distance_generic_njit)
    fidelity = compute_matrix_metric_series(rho_a, rho_b, fidelity_generic_njit)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(times, trace_dist, color='darkorange', linewidth=2)
    ax1.set_ylabel('Trace Distance')
    ax1.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
    ax1.set_title(f'{label} (Phi = {phi_deg}°)')

    ax2.plot(times, fidelity, color='teal', linewidth=2)
    ax2.set_ylabel('Fidelity')
    ax2.set_xlabel('Time')
    ax2.axhline(1, color='black', linestyle=':', linewidth=1, alpha=0.6)

    fig.tight_layout()
    save_fig(fig, f'{label.replace(" ", "_")}_Phi_{phi_str}', output_dir)


# ==========================
# Input parsing & file location (mirrors the generation script exactly)
# ==========================
if len(sys.argv) > 1:
    phi_deg = float(sys.argv[1])
else:
    phi_deg = 90.0

phi_rad = np.deg2rad(phi_deg)

# --- Must match the values used in the trajectory-generation script ---
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
N_traj = 10000

results_dir = "../Results/Data/Complete_rho/"
phi_str = f"{phi_deg:.4f}".replace(".", "p")  # only this token gets the dot->p substitution
Output_dir = os.path.join("../Results/Plot/Variance_Analysis", phi_str)
os.makedirs(Output_dir, exist_ok=True)



def _make_fname_npz(results_dir, phi_deg, dt, N_traj):
    """Copied verbatim from the trajectory-generation script to guarantee a matching filename."""
    dt_str = f"{dt:.6f}".replace(".", "p")
    phi_str = f"{phi_deg:.4f}".replace(".", "p")
    return os.path.join(results_dir, f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}.npz")


phi_str = f"{phi_deg:.4f}".replace(".", "p")
fname = _make_fname_npz(results_dir, phi_rad, dt, N_traj)

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Phi = {phi_deg} deg")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

# ==========================
# Data extraction
# ==========================
times = data['times']
rho_tot_all = data['rho_tot_all']              # (N, N, n_times, N_traj), complex128
rho_list_lindblad = data['rho_list_lindblad']  # (n_times, N, N) - LTV "exact" reference
rho_trace = data['rho_trace']                  # (N, N, n_times) - independent collision-model cross-check

N_site = rho_tot_all.shape[0]
n_times = rho_tot_all.shape[2]
n_traj = rho_tot_all.shape[3]

# rho_trace is stored time-last; get_exact_variance / compute_matrix_metric_series expect time-first
rho_trace_tfirst = np.moveaxis(rho_trace, -1, 0)

# ==========================
# Purity sanity check
# ==========================
# The fast LTV shortcuts rely on every trajectory being exactly pure
# (Tr(rho_k(t)^2) = 1). This is guaranteed by construction for a Kraus-operator
# jump/no-jump unraveling, but it costs almost nothing to verify once.
purity = np.sum(np.abs(rho_tot_all) ** 2, axis=(0, 1))  # (n_times, n_traj), = Tr(rho_k(t)^2)
max_purity_dev = np.max(np.abs(purity - 1.0))
print(f"Purity check: max |Tr(rho_k(t)^2) - 1| over all trajectories/times = {max_purity_dev:.2e}")
if max_purity_dev > 1e-6:
    print("WARNING: trajectories are not numerically pure - the fast analytic LTV shortcuts "
          "(which assume O^2=O for projectors) may be inaccurate. Consider using the generic "
          "tensordot-based formulas from the FMO script instead.")

# ==========================
# Statistical moments over trajectories
# ==========================
print("Computing Statistical Moments over time...")
pop_traj_site = np.real(np.stack([rho_tot_all[i, i, :, :] for i in range(N_site)], axis=0))  # (N_site, n_times, n_traj)

mean_pop_time = np.mean(pop_traj_site, axis=2)
var_pop_time = np.var(pop_traj_site, axis=2)
skew_pop_time = skew(pop_traj_site, axis=2, nan_policy='omit')
kurt_pop_time = kurtosis(pop_traj_site, axis=2, fisher=True, nan_policy='omit')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': ':',
    'figure.autolayout': True
})

STATE_LABELS = [f'State {i}' for i in range(N_site)]
colors = plt.cm.viridis(np.linspace(0, 1, N_site))

fig1, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
ax_mean, ax_var, ax_skew, ax_kurt = axes

for i in range(N_site):
    ax_mean.plot(times, mean_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])
    ax_var.plot(times, var_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])
    ax_skew.plot(times, skew_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])
    ax_kurt.plot(times, kurt_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])

ax_mean.set_title(f'Statistical Moments over Trajectories (Phi = {phi_deg}°)')
ax_mean.set_ylabel('Mean')
ax_mean.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
ax_var.set_ylabel('Variance')
ax_skew.set_ylabel('Skewness ($\\gamma_1$)')
ax_skew.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax_skew.set_ylim(-5, 5)
ax_kurt.set_ylabel('Excess Kurtosis ($K - 3$)')
ax_kurt.set_xlabel('Time')
ax_kurt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax_kurt.set_ylim(-5, 15)

save_fig(fig1, f'All_Statistical_Moments_Time_Phi_{phi_str}', Output_dir)

# ==========================
# Convergence check: trajectory average vs Lindblad exact
# ==========================
print("Computing Convergence Check: Trajectory Average vs Lindblad...")
rho_traj_avg = np.moveaxis(np.mean(rho_tot_all, axis=3), -1, 0)  # (n_times, N, N)
plot_convergence(times, rho_list_lindblad, rho_traj_avg,
                  "Convergence_TrajAvg_vs_Lindblad", phi_deg, phi_str, Output_dir)

# Bonus cross-check: does the independent collision-model reduction (rho_trace,
# obtained by tracing out the ancilla from the full unitary) agree with the
# Lindblad master-equation solution? This validates the master-equation
# approximation itself, separately from the trajectory sampling error above.
print("Computing Cross-Check: Collision-Model Trace vs Lindblad...")
plot_convergence(times, rho_list_lindblad, rho_trace_tfirst,
                  "CrossCheck_CollisionTrace_vs_Lindblad", phi_deg, phi_str, Output_dir)

# ==========================
# A. Population LTV (site basis)
# ==========================
print("Computing Total Variance Theorem: Populations...")
analyze_population_ltv(times, pop_traj_site, rho_list_lindblad, N_site, STATE_LABELS,
                        phi_deg, phi_str, Output_dir)

# ==========================
# B. Coherence LTV (all pairs - only 3 for a 3-level system, no need to curate)
# ==========================
print("Computing Total Variance Theorem: Coherences...")
coherence_pairs = list(itertools.combinations(range(N_site), 2))
analyze_coherence_ltv(times, rho_tot_all, pop_traj_site, rho_list_lindblad, coherence_pairs,
                      N_site, phi_deg, phi_str, Output_dir)