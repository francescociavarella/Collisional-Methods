#!/usr/bin/env python
# coding: utf-8
"""
Law-of-Total-Variance analysis for FMO quantum trajectories.
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

# Import custom thesis style and saving function
from plot_style import set_thesis_style, save_fig

# Apply global thesis style settings
set_thesis_style()

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
    """Generic Total Variance decomposition for an arbitrary Hermitian observable."""
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
    """Exact variance of a Hermitian observable from the full density matrix."""
    observable_sq = observable_matrix @ observable_matrix
    E_O = np.real(np.einsum('ik,tki->t', observable_matrix, rho_t))
    E_O2 = np.real(np.einsum('ik,tki->t', observable_sq, rho_t))
    return np.maximum(E_O2 - E_O ** 2, 0.0)


# ==========================================
# FAST ANALYTIC LAW-OF-TOTAL-VARIANCE (used for populations & coherences)
# ==========================================

def total_variance_projector(pop_k):
    """Exact LTV decomposition for a rank-1 projector observable O=|i><i|."""
    var_quant_k = pop_k * (1.0 - pop_k)
    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(pop_k, axis=1)
    return var_quant + var_stat, var_quant, var_stat


def total_variance_coherence(psi_m, psi_n, pop_m, pop_n, part='real'):
    """Exact LTV decomposition for the coherence quadrature observables."""
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

def plot_ltv_panel(ax, times, var_exact, var_stat, var_quant, var_sum, ylabel,
                    show_legend=False, theta_deg=None):
    ax.plot(times, var_exact, color='black', linewidth=3, linestyle='--', label='Total Variance (Redfield)')
    ax.plot(times, var_stat, color='red', linewidth=2, alpha=0.8, label='Statistical Variance')
    ax.plot(times, var_quant, color='blue', linewidth=2, alpha=0.8, label='Quantum Variance')
    ax.plot(times, var_sum, color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
    ax.set_ylabel(ylabel)
    
    if show_legend:
        if theta_deg is not None:
            ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)
        else:
            ax.legend(loc='upper right')


def analyze_population_ltv(times, pop_traj, rho_redfield, N_states, basis_name,
                            state_label, theta_deg, theta_str, output_dir):
    """Population LTV plot for either the site or the exciton basis using a centered 4-top / 3-bottom grid layout."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 8)
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(gs[0, i*2:(i+1)*2]))
    for c_start, c_end in [(1, 3), (3, 5), (5, 7)]:
        axes.append(fig.add_subplot(gs[1, c_start:c_end]))
    axes = np.array(axes)

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
            theta_deg=theta_deg
        )
        axes[i].set_xlabel('Time (fs)')

    fig.tight_layout()
    save_fig(fig, f'Law_Total_Variance_{basis_name.replace(" ", "_")}_Theta_{theta_str}', output_dir)


def analyze_coherence_ltv(times, psi_traj, pop_traj, rho_redfield, coherence_pairs, N_states,
                           theta_deg, theta_str, output_dir, basis_tag="", basis_title="Coherences"):
    """Coherence (real & imaginary quadrature) LTV plots using a centered 4-top / 3-bottom grid layout when n_pairs == 7."""
    n_pairs = len(coherence_pairs)
    
    def create_grid_figure():
        if n_pairs == 7:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 8)
            axes = []
            for i in range(4):
                axes.append(fig.add_subplot(gs[0, i*2:(i+1)*2]))
            for c_start, c_end in [(1, 3), (3, 5), (5, 7)]:
                axes.append(fig.add_subplot(gs[1, c_start:c_end]))
            return fig, np.array(axes)
        else:
            fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
            if n_pairs == 1:
                axes = [axes]
            return fig, np.array(axes)

    fig_real, axes_real = create_grid_figure()
    fig_imag, axes_imag = create_grid_figure()

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
            ylabel=f'Re( $\\rho_{{{m + 1}{n + 1}}}$ )', show_legend=(idx == 0), theta_deg=theta_deg
        )
        plot_ltv_panel(
            axes_imag[idx], times, vt_imag_exact, vs_imag, vq_imag, vt_imag,
            ylabel=f'Im( $\\rho_{{{m + 1}{n + 1}}}$ )', show_legend=(idx == 0), theta_deg=theta_deg
        )

        axes_real[idx].set_xlabel('Time (fs)')
        axes_imag[idx].set_xlabel('Time (fs)')

    fig_real.tight_layout()
    fig_imag.tight_layout()
    save_fig(fig_real, f'Law_Total_Variance_Coherences{basis_tag}_REAL_Theta_{theta_str}', output_dir)
    save_fig(fig_imag, f'Law_Total_Variance_Coherences{basis_tag}_IMAG_Theta_{theta_str}', output_dir)


def plot_convergence(times, rho_redfield, rho_traj_avg, basis_name, theta_deg, theta_str, output_dir):
    """Trace-distance / fidelity convergence check plot."""
    trace_dist = compute_matrix_metric_series(rho_redfield, rho_traj_avg, trace_distance_generic_njit)
    fidelity = compute_matrix_metric_series(rho_redfield, rho_traj_avg, fidelity_generic_njit)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(times, trace_dist, color='darkorange', linewidth=2)
    ax1.set_ylabel('Trace Distance')
    ax1.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
    
    ax1.plot([], [], ' ', label=fr"$\theta = {theta_deg}^\circ$")
    ax1.legend(loc='best', frameon=False)

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
Output_dir = f"../Results/Plot/Total_Variance_Analysis/{theta_str}"
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

SITE_LABELS = [f"Site {i + 1}" for i in range(N_site)]
colors = plt.cm.viridis(np.linspace(0, 1, N_site))

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