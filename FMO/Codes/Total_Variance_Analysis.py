#!/usr/bin/env python
# coding: utf-8
"""
Law-of-Total-Variance analysis for FMO quantum trajectories.

Optimized version. Main changes vs. the original script (see chat for full
rationale):

1. FAST ANALYTIC PATHS for the observables actually used here (population
   projectors and 2-level coherence quadratures). For a projector O=|i><i|,
   O^2 = O, so <O^2> = <O> exactly -> quantum variance = p(1-p), no matrix
   algebra needed. For O_real=|m><n|+|n><m| and O_imag=-i|m><n|+i|n><m|, one
   can show O_real^2 = O_imag^2 = |m><m|+|n><n|, so <O^2> = p_m + p_n exactly.
   This removes the need to build (N, n_times, n_traj) intermediate arrays
   via tensordot for every site/exciton/coherence -> large memory and time
   savings, especially for big N_traj / n_times (the old approach allocated
   two full complex128 arrays the size of psi_traj on every single call).
   The generic tensordot-based functions are kept as documented fallbacks
   for observables that are NOT projectors or 2-level coherences.

2. DEAD CODE: fidelity_generic_njit / trace_distance_generic_njit and the
   loaded rho_traj_avg_site / rho_traj_avg_exc arrays were defined/loaded
   but never used. They look like an unfinished convergence check (comment
   said "Load density matrices for Trace Distance convergence analysis").
   I completed it: a new plot compares rho_redfield(t) vs the trajectory
   average rho_traj_avg(t) via trace distance and fidelity, in both bases,
   whenever the data is available. Remove this block if it wasn't intended.

3. BUG FIX: in the original, the `else` branches that handle missing
   density matrices only set rho_redfield_* = None, leaving
   rho_traj_avg_* undefined -> NameError if referenced later. Both are now
   set to None consistently.

4. Removed unused imports (poisson, norm, curve_fit, LogNorm).

5. Refactored the repeated 4-line plotting block (Redfield / stat / quant /
   sum) and the near-identical site/exciton population loops into small
   helper functions to cut duplication.

6. Added a bounds check on the hardcoded coherence_pairs list, since it
   assumes N_site >= 7; out-of-range pairs are now skipped with a warning
   instead of raising an IndexError.
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to save files
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import skew, kurtosis

# Suppress the specific scipy warning about catastrophic cancellation at t=0
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# =================================
# NUMBA OPTIMIZED METRIC FUNCTIONS (used by the convergence check below)
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

    fidelity = np.sum(np.sqrt(evals_inner)) ** 2
    return min(1.0, fidelity)


@njit
def trace_distance_generic_njit(rho, sigma):
    """Numba-compatible generalized trace distance."""
    diff = rho - sigma
    diff = 0.5 * (diff + diff.conj().T)

    eigenvalues = np.linalg.eigvalsh(diff)
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    return min(1.0, t_dist)


def compute_matrix_metric_series(rho_a, rho_b, metric_fn):
    """Apply a numba matrix-pair metric (fidelity, trace distance, ...) at every time step."""
    n_times = rho_a.shape[0]
    result = np.empty(n_times)
    for t in range(n_times):
        result[t] = metric_fn(rho_a[t].astype(np.complex128), rho_b[t].astype(np.complex128))
    return result


# ==========================================
# GENERIC LAW-OF-TOTAL-VARIANCE (fallback for arbitrary observables)
# ==========================================

def compute_total_variance(observable_matrix, psi_traj):
    """
    Generic Total Variance decomposition for an arbitrary Hermitian observable.
    Kept as a documented fallback / sanity-check reference; NOT used for the
    projector and 2-level coherence observables in this script (see the fast
    analytic versions below), since it allocates O(N * n_times * n_traj)
    intermediate arrays via tensordot which gets expensive fast.

    Parameters
    ----------
    observable_matrix : ndarray, shape (N, N)
    psi_traj : ndarray, shape (N, n_times, n_traj)
    """
    observable_sq = observable_matrix @ observable_matrix

    O_psi = np.tensordot(observable_matrix, psi_traj, axes=([1], [0]))
    O2_psi = np.tensordot(observable_sq, psi_traj, axes=([1], [0]))

    E_k = np.real(np.sum(np.conj(psi_traj) * O_psi, axis=0))
    E2_k = np.real(np.sum(np.conj(psi_traj) * O2_psi, axis=0))

    var_quant_k = np.maximum(E2_k - E_k ** 2, 0.0)

    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(E_k, axis=1)
    var_total = var_quant + var_stat

    return var_total, var_quant, var_stat


def get_exact_variance(observable_matrix, rho_t):
    """
    Exact variance of a Hermitian observable from the full density matrix.
    Var(O) = Tr(O^2 rho) - Tr(O rho)^2

    observable_matrix : ndarray, shape (N, N)
    rho_t : ndarray, shape (n_times, N, N)
    """
    observable_sq = observable_matrix @ observable_matrix
    E_O = np.real(np.einsum('ik,tki->t', observable_matrix, rho_t))
    E_O2 = np.real(np.einsum('ik,tki->t', observable_sq, rho_t))
    return np.maximum(E_O2 - E_O ** 2, 0.0)


# ==========================================
# FAST ANALYTIC LAW-OF-TOTAL-VARIANCE (used for populations & coherences)
# ==========================================

def total_variance_projector(pop_k):
    """
    Exact LTV decomposition for a rank-1 projector observable O=|i><i|.
    Since O^2 = O for a projector, <psi|O^2|psi> = <psi|O|psi> exactly, so
    the intrinsic (quantum) variance per trajectory is simply p*(1-p) -
    no matrix multiplication needed.

    pop_k : ndarray, shape (n_times, n_traj) - population of state i.
    """
    var_quant_k = pop_k * (1.0 - pop_k)
    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(pop_k, axis=1)
    return var_quant + var_stat, var_quant, var_stat


def total_variance_coherence(psi_m, psi_n, pop_m, pop_n, part='real'):
    """
    Exact LTV decomposition for the coherence quadrature observables
    O_real = |m><n| + |n><m|   and   O_imag = -i|m><n| + i|n><m|.

    Using |m><n|m><n| = 0, |m><n|n><m| = |m><m| (orthonormal basis), one gets
        O_real^2 = O_imag^2 = |m><m| + |n><n|
    so <O^2> = p_m + p_n exactly, reusing the population arrays already
    computed - again no tensordot needed.

    psi_m, psi_n : ndarray, shape (n_times, n_traj) - amplitudes on states m, n.
    pop_m, pop_n : ndarray, shape (n_times, n_traj) - populations of m, n.
    """
    z = np.conj(psi_m) * psi_n
    E_k = 2.0 * (np.real(z) if part == 'real' else np.imag(z))
    E2_k = pop_m + pop_n

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
    ax.plot(times, var_exact, color='black', linewidth=3, linestyle='--', label='Total Variance (Redfield)')
    ax.plot(times, var_stat, color='red', linewidth=2, alpha=0.8, label='Statistical Variance')
    ax.plot(times, var_quant, color='blue', linewidth=2, alpha=0.8, label='Quantum Variance')
    ax.plot(times, var_sum, color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc='upper right', fontsize=8)


def analyze_population_ltv(times, pop_traj, rho_redfield, N_states, basis_name,
                            state_label, theta_deg, theta_str, output_dir):
    """Population LTV plot for either the site or the exciton basis (fast path)."""
    fig, axes = plt.subplots(N_states, 1, figsize=(10, 2.5 * N_states), sharex=True)
    if N_states == 1:
        axes = [axes]

    for i in range(N_states):
        var_tot_traj, var_quant, var_stat = total_variance_projector(pop_traj[i])

        O_pop = np.zeros((N_states, N_states), dtype=np.complex128)
        O_pop[i, i] = 1.0
        var_tot_exact = get_exact_variance(O_pop, rho_redfield)

        max_err = np.max(np.abs(var_tot_exact - var_tot_traj))
        print(f"  -> {state_label} {i + 1} Law of Total Variance max error: {max_err:.2e}")

        plot_ltv_panel(
            axes[i], times, var_tot_exact, var_stat, var_quant, var_tot_traj,
            ylabel=f'{state_label} {i + 1}',
            show_legend=(i == 0),
            title=f'Law of Total Variance - {basis_name} (Theta = {theta_deg}°)' if i == 0 else None,
        )

    axes[-1].set_xlabel('Time (fs)')
    fig.tight_layout()
    save_fig(fig, f'Law_Total_Variance_{basis_name.replace(" ", "_")}_Theta_{theta_str}', output_dir)


def analyze_coherence_ltv(times, psi_traj, pop_traj, rho_redfield, coherence_pairs, N_states,
                           theta_deg, theta_str, output_dir, basis_tag="", basis_title="Coherences"):
    """
    Coherence (real & imaginary quadrature) LTV plots (fast path).
    Works for any basis (site or exciton) - just pass the corresponding
    psi_traj / pop_traj / rho_redfield and a basis_tag for the filename
    (e.g. "" for site, "_Exciton" for exciton) and basis_title for the plot title.
    """
    n_pairs = len(coherence_pairs)
    fig_real, axes_real = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    fig_imag, axes_imag = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes_real, axes_imag = [axes_real], [axes_imag]

    for idx, (m, n) in enumerate(coherence_pairs):
        psi_m, psi_n = psi_traj[m], psi_traj[n]
        pop_m, pop_n = pop_traj[m], pop_traj[n]

        vt_real, vq_real, vs_real = total_variance_coherence(psi_m, psi_n, pop_m, pop_n, part='real')
        vt_imag, vq_imag, vs_imag = total_variance_coherence(psi_m, psi_n, pop_m, pop_n, part='imag')

        O_real = np.zeros((N_states, N_states), dtype=np.complex128)
        O_real[m, n], O_real[n, m] = 1.0, 1.0
        O_imag = np.zeros((N_states, N_states), dtype=np.complex128)
        O_imag[m, n], O_imag[n, m] = -1.0j, 1.0j

        vt_real_exact = get_exact_variance(O_real, rho_redfield)
        vt_imag_exact = get_exact_variance(O_imag, rho_redfield)

        err_real = np.max(np.abs(vt_real_exact - vt_real))
        err_imag = np.max(np.abs(vt_imag_exact - vt_imag))
        print(f"  -> {basis_title} ({m + 1},{n + 1}) | Max Error - Real: {err_real:.2e}, Imag: {err_imag:.2e}")

        plot_ltv_panel(
            axes_real[idx], times, vt_real_exact, vs_real, vq_real, vt_real,
            ylabel=f'Re( $\\rho_{{{m + 1}{n + 1}}}$ )', show_legend=(idx == 0),
            title=f'Law of Total Variance - {basis_title} REAL Part (Theta = {theta_deg}°)' if idx == 0 else None,
        )
        plot_ltv_panel(
            axes_imag[idx], times, vt_imag_exact, vs_imag, vq_imag, vt_imag,
            ylabel=f'Im( $\\rho_{{{m + 1}{n + 1}}}$ )', show_legend=(idx == 0),
            title=f'Law of Total Variance - {basis_title} IMAGINARY Part (Theta = {theta_deg}°)' if idx == 0 else None,
        )

    axes_real[-1].set_xlabel('Time (fs)')
    axes_imag[-1].set_xlabel('Time (fs)')
    fig_real.tight_layout()
    fig_imag.tight_layout()
    save_fig(fig_real, f'Law_Total_Variance_Coherences{basis_tag}_REAL_Theta_{theta_str}', output_dir)
    save_fig(fig_imag, f'Law_Total_Variance_Coherences{basis_tag}_IMAG_Theta_{theta_str}', output_dir)


def plot_convergence(times, rho_redfield, rho_traj_avg, basis_name, theta_deg, theta_str, output_dir):
    """
    NEW: completes the trace-distance / fidelity convergence check that the
    original script set up (loaded the data, defined the metrics) but never
    plotted. Shows how well the trajectory-averaged density matrix matches
    the Redfield reference over time - i.e. whether N_traj is large enough.
    """
    trace_dist = compute_matrix_metric_series(rho_redfield, rho_traj_avg, trace_distance_generic_njit)
    fidelity = compute_matrix_metric_series(rho_redfield, rho_traj_avg, fidelity_generic_njit)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(times, trace_dist, color='darkorange', linewidth=2)
    ax1.set_ylabel('Trace Distance')
    ax1.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
    ax1.set_title(f'Trajectory-Average vs Redfield Convergence - {basis_name} (Theta = {theta_deg}°)')

    ax2.plot(times, fidelity, color='teal', linewidth=2)
    ax2.set_ylabel('Fidelity')
    ax2.set_xlabel('Time (fs)')
    ax2.axhline(1, color='black', linestyle=':', linewidth=1, alpha=0.6)

    fig.tight_layout()
    save_fig(fig, f'Convergence_TraceDistance_Fidelity_{basis_name}_Theta_{theta_str}', output_dir)


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

# Density matrices for the LTV "exact" reference AND for the convergence check.
# (Both rho_traj_avg_* are now set to None in the missing-data branch too, to
# avoid a NameError if referenced later - this was a latent bug in the original.)
if 'rho_redfield_site' in data and 'rho_traj_avg_site' in data:
    rho_redfield_site = data['rho_redfield_site']
    rho_traj_avg_site = data['rho_traj_avg_site']
else:
    print("Warning: Density matrices (site basis) not found.")
    rho_redfield_site = None
    rho_traj_avg_site = None

if 'rho_redfield_exc' in data and 'rho_traj_avg_exc' in data:
    rho_redfield_exc = data['rho_redfield_exc']
    rho_traj_avg_exc = data['rho_traj_avg_exc']
else:
    print("Warning: Density matrices (exciton basis) not found.")
    rho_redfield_exc = None
    rho_traj_avg_exc = None

n_times = len(times)
n_traj = psi_traj_exc.shape[2]

# ==========================
# Site-basis single-trajectory populations
# ==========================
psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)   # (N_site, n_times, n_traj)
pop_traj_site = np.abs(psi_traj_site) ** 2                             # (N_site, n_times, n_traj)
pop_traj_exc = np.abs(psi_traj_exc) ** 2                               # (N_site, n_times, n_traj)

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

SITE_LABELS = [f"Site {i + 1}" for i in range(N_site)]
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

save_fig(fig1, f'All_Statistical_Moments_Time_Theta_{theta_str}', Output_dir)


# ====================================================================
# CONVERGENCE CHECK: Trajectory average vs Redfield (trace distance & fidelity)
# ====================================================================
if rho_redfield_site is not None and rho_traj_avg_site is not None:
    print("Computing Convergence Check (Site basis): Trace Distance & Fidelity...")
    plot_convergence(times, rho_redfield_site, rho_traj_avg_site, "Site_Basis",
                      theta_deg, theta_str, Output_dir)

if rho_redfield_exc is not None and rho_traj_avg_exc is not None:
    print("Computing Convergence Check (Exciton basis): Trace Distance & Fidelity...")
    plot_convergence(times, rho_redfield_exc, rho_traj_avg_exc, "Exciton_Basis",
                      theta_deg, theta_str, Output_dir)


# ====================================================================
# LAW OF TOTAL VARIANCE WITH GENERIC OBSERVABLES (fast analytic paths)
# ====================================================================

# ---------------------------------------------------------
# A. Site Populations
# ---------------------------------------------------------
if rho_redfield_site is not None:
    print("Computing Total Variance Theorem: Site Populations...")
    analyze_population_ltv(times, pop_traj_site, rho_redfield_site, N_site,
                            "Site_Populations", "Site", theta_deg, theta_str, Output_dir)

# ---------------------------------------------------------
# B. Exciton Populations
# ---------------------------------------------------------
if rho_redfield_exc is not None:
    print("Computing Total Variance Theorem: Exciton Populations...")
    analyze_population_ltv(times, pop_traj_exc, rho_redfield_exc, N_site,
                            "Exciton_Populations", "Exciton", theta_deg, theta_str, Output_dir)

# ---------------------------------------------------------
# C. Selected Coherences (Real and Imaginary Parts) - Site Basis
# ---------------------------------------------------------
if rho_redfield_site is not None:
    print("Computing Total Variance Theorem: Coherences (Site Basis)...")

    coherence_pairs_site = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # Nearest neighbors
        (0, 5), (1, 3), (2, 4), (0, 6)                   # Long-range coupling/correlations
    ]
    valid_pairs_site = [(m, n) for (m, n) in coherence_pairs_site if max(m, n) < N_site]
    if len(valid_pairs_site) < len(coherence_pairs_site):
        skipped = [p for p in coherence_pairs_site if p not in valid_pairs_site]
        print(f"Warning: skipping site coherence pairs {skipped} - out of range for N_site={N_site}")

    if valid_pairs_site:
        analyze_coherence_ltv(times, psi_traj_site, pop_traj_site, rho_redfield_site,
                               valid_pairs_site, N_site, theta_deg, theta_str, Output_dir,
                               basis_tag="", basis_title="Coherences")

# ---------------------------------------------------------
# D. Selected Coherences (Real and Imaginary Parts) - Exciton Basis
#
# Physically motivated default: exciton-basis coherences oscillate at the
# exciton energy gap (H_S is diagonal in this basis), which is exactly the
# frequency probed by 2D electronic spectroscopy "quantum beat" maps, so
# adjacent-in-energy pairs are the most relevant ones for coherence-lifetime
# comparisons across unravelings. We use all energy-adjacent pairs (smallest
# gaps, typically longest-lived coherences) plus the (0, N-1) pair (largest
# gap, fastest expected dephasing) for contrast. eigenergies is assumed
# ascending, as returned by np.linalg.eigh - check this if your convention differs.
# ---------------------------------------------------------
if rho_redfield_exc is not None:
    print("Computing Total Variance Theorem: Coherences (Exciton Basis)...")

    coherence_pairs_exc = [(i, i + 1) for i in range(N_site - 1)]
    if N_site > 2:
        coherence_pairs_exc.append((0, N_site - 1))

    if coherence_pairs_exc:
        analyze_coherence_ltv(times, psi_traj_exc, pop_traj_exc, rho_redfield_exc,
                               coherence_pairs_exc, N_site, theta_deg, theta_str, Output_dir,
                               basis_tag="_Exciton", basis_title="Exciton Coherences")