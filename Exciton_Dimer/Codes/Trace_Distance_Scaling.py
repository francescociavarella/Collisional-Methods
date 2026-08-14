#!/usr/bin/env python
# coding: utf-8
"""
Trace Distance analysis for the Exciton Dimer model (extended version).

This script loads data for several target angles and, for EACH quantity,
produces BOTH:
  (a) an INDIVIDUAL plot per angle (saved under .../MODE/{angle}/), and
  (b) a single CUMULATIVE plot overlaying all analyzed angles together
      (saved under .../MODE/Comparison_All_Angles/).

Quantities:
1) Trace Distance SCALING vs N_traj (mean and max, log-log), with a linear
   fit. Individually: each angle gets its own fit + its own theoretical
   reference line (slope -0.5). Cumulative: all angles overlaid, with a
   SINGLE shared theoretical line (anchored on theta = 90) to keep the
   combined plot readable.

2) Time-resolved VARIANCE of the Site-1 population (|10> state), computed
   using a fixed sample of N_traj = 10000 trajectories.

3) Time-resolved TRACE DISTANCE between the N=10000-trajectory average
   density matrix and the exact Lindblad density matrix, as a function of
   time (no reduction to a single mean/max scalar).

4) Time-resolved VARIANCE of the trace distance, computed PER individual
   trajectory (not on the average), then the variance of these N=10000
   per-trajectory trace-distance values at each time step.

In every CUMULATIVE figure, theta = 90 is drawn on top of the other curves
(thicker line + explicit zorder).

OPTIMIZATION NOTE: the per-trajectory trace distance (used for quantity 4)
does not rely on a numba loop calling np.linalg.eigvalsh for every single
(time, trajectory) pair. For a 2x2 Hermitian matrix the trace distance has an
exact closed form (see compute_trace_distance_per_trajectory below), fully
vectorized with numpy -- mathematically identical to the eigenvalue-based
calculation, but much faster.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from numba import njit

# Import custom thesis style and saving function
from plot_style import set_thesis_style, save_fig

# Apply global thesis style settings
set_thesis_style()

# ==========================================
# FIXED COLOR MAP (per target theta angle)
# ==========================================
# Colorblind-safe palette (Wong, 2011), one color per target angle.
THETA_COLOR_MAP = {
    90: '#0072B2',
    0: '#D55E00',
    60: '#009E73',
    45: '#F0E442',
    30: '#CC79A7',
}


def get_theta_color(theta_target_deg, default='black'):
    """Returns the fixed color assigned to a given target theta angle (degrees)."""
    theta_key = int(round(theta_target_deg))
    return THETA_COLOR_MAP.get(theta_key, default)


def get_theta_style(theta_target_deg):
    """
    Returns (color, linewidth, zorder) for a given angle, so that theta = 90
    is always drawn thicker and on top of the other curves, and theta = 0 is
    also slightly thicker as the other reference limit.
    """
    color = get_theta_color(theta_target_deg)
    linewidth = 2.0 if theta_target_deg in (90.0, 0.0) else 1.8
    zorder = 10 if theta_target_deg == 90.0 else 5
    return color, linewidth, zorder


def get_angle_folder(theta_target_deg):
    """Formats the per-angle output subfolder name (integer angles without decimals)."""
    return str(int(theta_target_deg)) if float(theta_target_deg).is_integer() else str(theta_target_deg)


LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 16

# ==========================================
# FAST TRACE DISTANCE CALCULATION (averaged matrix)
# ==========================================
@njit
def compute_trace_distance_series(rho_a, rho_b):
    """
    Computes the trace distance between two series of density matrices over time.
    Trace distance T = 0.5 * sum(|eigenvalues(rho_a - rho_b)|)

    Parameters:
        rho_a, rho_b: 3D complex arrays of shape (n_times, 2, 2)
    Returns:
        1D float array of trace distances over time
    """
    n_times = rho_a.shape[0]
    t_dist = np.zeros(n_times)

    for t in range(n_times):
        diff = rho_a[t] - rho_b[t]
        # Ensure Hermiticity for numerical stability before diagonalization
        diff = 0.5 * (diff + diff.conj().T)
        eigenvalues = np.linalg.eigvalsh(diff)
        t_dist[t] = 0.5 * np.sum(np.abs(eigenvalues))

    return t_dist


# ==========================================
# TIME-RESOLVED VARIANCE OF THE ESTIMATOR
# (kept available, currently unused in the main flow)
# ==========================================
def compute_variance_series(raw_pop_10, raw_pop_01, raw_coh_10_01, raw_coh_01_10, idx):
    """
    Computes the time-resolved TOTAL variance of the density-matrix estimator
    (populations + coherences), using a fixed subsample of trajectories `idx`.

    Var_total(t) = Var[pop_10(t)] + Var[pop_01(t)]
                 + Var[Re(coh_10_01(t))] + Var[Im(coh_10_01(t))]
                 + Var[Re(coh_01_10(t))] + Var[Im(coh_01_10(t))]
    """
    var_pop10 = np.var(raw_pop_10[:, idx], axis=1)
    var_pop01 = np.var(raw_pop_01[:, idx], axis=1)

    var_coh1001 = np.var(raw_coh_10_01[:, idx].real, axis=1) + np.var(raw_coh_10_01[:, idx].imag, axis=1)
    var_coh0110 = np.var(raw_coh_01_10[:, idx].real, axis=1) + np.var(raw_coh_01_10[:, idx].imag, axis=1)

    var_total = var_pop10 + var_pop01 + var_coh1001 + var_coh0110
    return var_total


# ==========================================
# TRACE DISTANCE PER INDIVIDUAL TRAJECTORY
# (OPTIMIZED: closed-form, fully vectorized with numpy, no numba/eigvalsh loop)
# ==========================================
def compute_trace_distance_per_trajectory(pop_10, pop_01, coh_10_01, coh_01_10, rho_exact):
    """
    Computes the trace distance between EACH individual trajectory's own
    reconstructed 2x2 density matrix (no averaging over trajectories) and the
    exact Lindblad density matrix, at every time step -- using the closed-form
    expression for a 2x2 Hermitian matrix instead of an explicit eigendecomposition:

        a  = (diff_00 - diff_11) / 2          (real)
        b  = diff_01 (Hermitized)             (complex)
        mu = trace(diff) / 2                  (real, ~0 for normalized matrices)
        T  = max( sqrt(a^2 + |b|^2), |mu| )

    This is exact and vectorized over BOTH time and trajectories at once.

    Inputs:
        pop_10, pop_01: 2D real arrays, shape (n_times, N_traj)
        coh_10_01, coh_01_10: 2D complex arrays, shape (n_times, N_traj)
        rho_exact: 3D complex array, shape (n_times, 2, 2)
    Returns:
        td_matrix: 2D real array, shape (n_times, N_traj)
    """
    r00 = rho_exact[:, 0, 0].real[:, None]
    r11 = rho_exact[:, 1, 1].real[:, None]
    r01 = rho_exact[:, 0, 1][:, None]
    r10 = rho_exact[:, 1, 0][:, None]

    diff_00 = pop_10 - r00
    diff_11 = pop_01 - r11
    diff_01 = coh_01_10 - r01
    diff_10 = coh_10_01 - r10

    # Hermitize the off-diagonal element (diagonal is already real)
    b = 0.5 * (diff_01 + np.conj(diff_10))

    a = 0.5 * (diff_00 - diff_11)
    mu = 0.5 * (diff_00 + diff_11)

    r = np.sqrt(a**2 + np.abs(b)**2)
    td_matrix = np.maximum(r, np.abs(mu))

    return td_matrix


# ==========================================
# TRACE DISTANCE SCALING vs N_traj (mean & max, with fit)
# ==========================================
def compute_scaling_for_angle(raw_pop_10, raw_pop_01, raw_coh_10_01, raw_coh_01_10,
                               rho_exact, n_times, total_available_traj, n_bootstraps=1):
    """
    Computes the mean and max Trace Distance (in log10 space) as a function of
    the number of averaged trajectories N, and fits a straight line in
    log-log space (expected slope ~ -0.5 for Monte-Carlo averaging).
    """
    N_list = np.array([
        100, 200, 500, 1000, 2000, 3000, 4000, 5000,
        7500, 10000, 12500, 15000, 17500, 20000
    ])
    N_list = N_list[N_list <= total_available_traj]

    log_mean_td_list = []
    log_max_td_list = []

    for N in N_list:
        sample_log_means = []
        sample_log_maxs = []

        current_bootstraps = 1 if N == total_available_traj else n_bootstraps

        for b in range(current_bootstraps):
            if N == total_available_traj:
                idx = np.arange(total_available_traj)
            else:
                idx = np.random.choice(total_available_traj, N, replace=False)

            pop_10_avg = np.mean(raw_pop_10[:, idx], axis=1)
            pop_01_avg = np.mean(raw_pop_01[:, idx], axis=1)
            coh_10_01_avg = np.mean(raw_coh_10_01[:, idx], axis=1)
            coh_01_10_avg = np.mean(raw_coh_01_10[:, idx], axis=1)

            rho_avg_N = np.zeros((n_times, 2, 2), dtype=np.complex128)
            rho_avg_N[:, 0, 0] = pop_10_avg
            rho_avg_N[:, 1, 1] = pop_01_avg
            rho_avg_N[:, 1, 0] = coh_10_01_avg
            rho_avg_N[:, 0, 1] = coh_01_10_avg

            td_series = compute_trace_distance_series(rho_avg_N, rho_exact)

            # Skip the first 100 steps to avoid the artificial transient at t=0
            skip_idx = 100 if n_times > 200 else 0
            td_mean = np.mean(td_series[skip_idx:])
            td_max = np.max(td_series[skip_idx:])

            if td_mean > 0 and td_max > 0:
                sample_log_means.append(np.log10(td_mean))
                sample_log_maxs.append(np.log10(td_max))

        log_mean_td_list.append(np.mean(sample_log_means))
        log_max_td_list.append(np.mean(sample_log_maxs))

    log_mean_td = np.array(log_mean_td_list)
    log_max_td = np.array(log_max_td_list)
    log_N = np.log10(N_list)

    slope_mean, int_mean, r_mean, p_mean, err_mean_fit = linregress(log_N, log_mean_td)
    fit_mean_log = slope_mean * log_N + int_mean

    slope_max, int_max, r_max, p_max, err_max_fit = linregress(log_N, log_max_td)
    fit_max_log = slope_max * log_N + int_max

    print(f"  Mean Trace Distance Fit: y = {slope_mean:.4f}x + {int_mean:.4f} (R^2 = {r_mean**2:.4f})")
    print(f"  Max Trace Distance Fit:  y = {slope_max:.4f}x + {int_max:.4f} (R^2 = {r_max**2:.4f})")

    return {
        'log_N': log_N,
        'log_mean_td': log_mean_td,
        'log_max_td': log_max_td,
        'fit_mean_log': fit_mean_log,
        'fit_max_log': fit_max_log,
        'slope_mean': slope_mean,
        'int_mean': int_mean,
        'r_mean': r_mean,
        'slope_max': slope_max,
        'int_max': int_max,
        'r_max': r_max,
    }


# ==========================================
# DATA LOADING + METRIC COMPUTATION (per angle)
# ==========================================
def compute_metrics_for_angle(theta_target_deg, MODE='normal', dt=0.01, max_N=20000,
                               n_bootstraps=1, N_time_series=10000):
    """
    Loads Exciton Dimer data for a specific angle and computes (but does NOT
    plot) all quantities used later for both the individual and the combined
    figures:
      - scaling:       Trace Distance vs N_traj, mean & max, with fit
      - var_pop_site1: variance of the Site-1 population, N = N_time_series
      - td_series:     trace distance of the N_time_series-trajectory average
      - var_td:        variance of the per-trajectory trace distance

    Returns a dict with the results, or None if the input file is missing.
    """
    print(f"\n{'='*50}\nProcessing Target Theta = {theta_target_deg}° ({MODE})\n{'='*50}")

    # theta_plot = 90.0 - theta_target_deg
    theta_plot = theta_target_deg

    np.random.seed(42)

    theta_rad = np.radians(theta_target_deg)
    theta_str = f"{theta_rad:.6f}".replace(".", "p")
    dt_str = f"{dt:.6f}".replace(".", "p")

    if MODE == 'normal':
        Input_dir = "../Results/Data/Complete_rho/normal"
    elif MODE == 'close_to_90':
        Input_dir = "../Results/Data/Complete_rho/close_90_deg"
    else:
        raise ValueError(f"Unknown mode: {MODE}")

    filename = f"result_theta{theta_str}_dt{dt_str}_Ntraj{max_N}.npz"
    filepath = os.path.join(Input_dir, filename)

    try:
        data = np.load(filepath)
        print(f"Data loaded successfully from {filepath}")
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Skipping...")
        return None

    if 'rho_list_lindblad' not in data:
        print("Error: 'rho_list_lindblad' not found in data. Skipping...")
        return None
    rho_lindblad_complete = data['rho_list_lindblad']

    times = data['times']
    n_times = len(times)

    # Rebuild the 2x2 exact Lindblad reference matrix
    rho_exact = np.zeros((n_times, 2, 2), dtype=np.complex128)
    rho_exact[:, 0, 0] = rho_lindblad_complete[:, 2, 2]  # |10><10|
    rho_exact[:, 1, 1] = rho_lindblad_complete[:, 1, 1]  # |01><01|
    rho_exact[:, 0, 1] = rho_lindblad_complete[:, 2, 1]  # |10><01|
    rho_exact[:, 1, 0] = rho_lindblad_complete[:, 1, 2]  # |01><10|

    raw_pop_10 = data['pop_00']
    raw_pop_01 = data['pop_11']
    raw_coh_10_01 = data['coh_10_01']
    raw_coh_01_10 = data['coh_01_10']

    total_available_traj = raw_pop_10.shape[1]

    # --- Trace Distance scaling vs N_traj (mean & max), with fit ---
    print("Computing Trace Distance scaling vs N_traj...")
    scaling = compute_scaling_for_angle(
        raw_pop_10, raw_pop_01, raw_coh_10_01, raw_coh_01_10,
        rho_exact, n_times, total_available_traj, n_bootstraps=n_bootstraps
    )

    # Fix the trajectory subset so variance(t) and trace-distance(t) refer to
    # the SAME sample.
    N_fixed = min(N_time_series, total_available_traj)
    if N_fixed == total_available_traj:
        idx_fixed = np.arange(total_available_traj)
    else:
        idx_fixed = np.random.choice(total_available_traj, N_fixed, replace=False)

    # --- Variance of the Site-1 population (|10> state) ---
    print(f"Computing time-resolved variance of the Site-1 population for N = {N_fixed} trajectories...")
    var_pop_site1 = np.var(raw_pop_10[:, idx_fixed], axis=1)

    # --- Trace distance of the N_fixed-trajectory average ---
    print(f"Computing time-resolved trace distance for N = {N_fixed} trajectories...")
    pop_10_avg_fixed = np.mean(raw_pop_10[:, idx_fixed], axis=1)
    pop_01_avg_fixed = np.mean(raw_pop_01[:, idx_fixed], axis=1)
    coh_10_01_avg_fixed = np.mean(raw_coh_10_01[:, idx_fixed], axis=1)
    coh_01_10_avg_fixed = np.mean(raw_coh_01_10[:, idx_fixed], axis=1)

    rho_avg_fixed = np.zeros((n_times, 2, 2), dtype=np.complex128)
    rho_avg_fixed[:, 0, 0] = pop_10_avg_fixed
    rho_avg_fixed[:, 1, 1] = pop_01_avg_fixed
    rho_avg_fixed[:, 1, 0] = coh_10_01_avg_fixed
    rho_avg_fixed[:, 0, 1] = coh_01_10_avg_fixed

    td_series = compute_trace_distance_series(rho_avg_fixed, rho_exact)

    # --- Variance of the per-trajectory trace distance (optimized) ---
    print(f"Computing time-resolved variance of the per-trajectory trace distance for N = {N_fixed} trajectories...")
    td_matrix_traj = compute_trace_distance_per_trajectory(
        raw_pop_10[:, idx_fixed], raw_pop_01[:, idx_fixed],
        raw_coh_10_01[:, idx_fixed], raw_coh_01_10[:, idx_fixed],
        rho_exact
    )
    var_td = np.var(td_matrix_traj, axis=1)

    return {
        'theta_target_deg': theta_target_deg,
        'theta_plot': theta_plot,
        'theta_str': theta_str,
        'times': times,
        'N_fixed': N_fixed,
        'scaling': scaling,
        'var_pop_site1': var_pop_site1,
        'td_series': td_series,
        'var_td': var_td,
    }


# ==========================================
# INDIVIDUAL (SINGLE-ANGLE) PLOTS
# ==========================================
def plot_individual_scaling(res, MODE='normal'):
    """
    Builds the 2-panel scaling figure (mean & max Trace Distance vs N_traj)
    for a SINGLE angle, with its own fit and its own theoretical reference
    line (slope -0.5), anchored on its own first data point.
    """
    theta_target_deg = res['theta_target_deg']
    theta_plot = res['theta_plot']
    scaling = res['scaling']
    color, _, _ = get_theta_style(theta_target_deg)

    Output_dir = os.path.join("../Results/Plot/Trace_Distance_Scaling", MODE, get_angle_folder(theta_target_deg))
    os.makedirs(Output_dir, exist_ok=True)

    log_N = scaling['log_N']
    theory_mean_log = -0.5 * (log_N - log_N[0]) + scaling['log_mean_td'][0]
    theory_max_log = -0.5 * (log_N - log_N[0]) + scaling['log_max_td'][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(log_N, scaling['log_mean_td'], 'o', color=color, markeredgewidth=1.5, zorder=3, label='Raw Data')
    ax1.plot(log_N, scaling['fit_mean_log'], color='black', linestyle='-', linewidth=1.5, zorder=4,
              label=fr'Fit: $y = {scaling["slope_mean"]:.2f}x {scaling["int_mean"]:+.2f}$')
    ax1.plot(log_N, theory_mean_log, color='dimgray', linestyle='--', linewidth=1.5, zorder=2,
              label=r'Theory: slope = $-0.5$')
    ax1.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(r'$\log_{10} (\langle T \rangle_t)$', fontsize=LABEL_FONTSIZE)
    ax1.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right',
               title_fontsize=LEGEND_FONTSIZE, fontsize=LEGEND_FONTSIZE)

    ax2.plot(log_N, scaling['log_max_td'], 's', color=color, markeredgewidth=1.5, zorder=3, label='Raw Data')
    ax2.plot(log_N, scaling['fit_max_log'], color='black', linestyle='--', linewidth=1.5, zorder=4,
              label=fr'Fit: $y = {scaling["slope_max"]:.2f}x {scaling["int_max"]:+.2f}$')
    ax2.plot(log_N, theory_max_log, color='dimgray', linestyle='--', linewidth=1.5, zorder=2,
              label=r'Theory: slope = $-0.5$')
    ax2.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel(r'$\log_{10} (T_{\mathrm{max}})$', fontsize=LABEL_FONTSIZE)
    ax2.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right',
               title_fontsize=LEGEND_FONTSIZE, fontsize=LEGEND_FONTSIZE)

    scaling_filename = f"Trace_Distance_Scaling_Theta_{res['theta_str']}"
    save_fig(fig, scaling_filename, Output_dir)
    print(f"Individual scaling plot saved in {Output_dir}")


def plot_individual_time_series(res, MODE='normal'):
    """
    Builds the 3 individual time-resolved figures (Site-1 population variance,
    trace distance, trace distance variance) for a SINGLE angle.
    """
    theta_target_deg = res['theta_target_deg']
    theta_plot = res['theta_plot']
    times = res['times']
    N_fixed = res['N_fixed']
    theta_str = res['theta_str']
    color, linewidth, _ = get_theta_style(theta_target_deg)

    Output_dir = os.path.join("../Results/Plot/Trace_Distance_Scaling", MODE, get_angle_folder(theta_target_deg))
    os.makedirs(Output_dir, exist_ok=True)

    # --- Site-1 population variance ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(times, res['var_pop_site1'], color=color, linewidth=linewidth, label='Site 1 Population Variance')
    ax1.set_xlabel(r'$Time [1/V]$', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(r'$\mathrm{Var}\left(P_{1}(t)\right)$', fontsize=LABEL_FONTSIZE)
    ax1.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='lower right',
               title_fontsize=LEGEND_FONTSIZE, fontsize=LEGEND_FONTSIZE)
    save_fig(fig1, f"Variance_PopSite1_vs_Time_Theta_{theta_str}_N{N_fixed}", Output_dir)

    # --- Trace distance ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(times, res['td_series'], color=color, linewidth=linewidth, label='Trace Distance')
    ax2.set_xlabel(r'$Time [1/V]$', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel(r'$T(t)$', fontsize=LABEL_FONTSIZE)
    ax2.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right',
               title_fontsize=LEGEND_FONTSIZE, fontsize=LEGEND_FONTSIZE)
    save_fig(fig2, f"TraceDistance_vs_Time_Theta_{theta_str}_N{N_fixed}", Output_dir)

    # --- Trace distance variance ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(times, res['var_td'], color=color, linewidth=linewidth, label='Trace Distance Variance')
    ax3.set_xlabel(r'$Time [1/V]$', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel(r'$\mathrm{Var}\left(T(t)\right)$', fontsize=LABEL_FONTSIZE)
    ax3.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right',
               title_fontsize=LEGEND_FONTSIZE, fontsize=LEGEND_FONTSIZE)
    save_fig(fig3, f"Variance_TraceDistance_vs_Time_Theta_{theta_str}_N{N_fixed}", Output_dir)

    print(f"Individual time-resolved plots saved in {Output_dir}")


# ==========================================
# COMBINED SCALING PLOT (all angles overlaid)
# ==========================================
def plot_combined_scaling(results_by_angle, MODE='normal'):
    """
    Builds a 2-panel figure (mean and max Trace Distance vs N_traj), overlaying
    the raw data + fit of every angle. A SINGLE shared theoretical reference
    line (slope -0.5), anchored on theta = 90, is drawn in each panel instead
    of one theory line per angle, to keep the plot readable.
    """
    Output_dir = os.path.join("../Results/Plot/Trace_Distance_Scaling", MODE, "Comparison_All_Angles")
    os.makedirs(Output_dir, exist_ok=True)

    ref_angle = 90.0 if 90.0 in results_by_angle else next(iter(results_by_angle))
    ref_scaling = results_by_angle[ref_angle]['scaling']
    log_N_ref = ref_scaling['log_N']
    theory_mean_log = -0.5 * (log_N_ref - log_N_ref[0]) + ref_scaling['log_mean_td'][0]
    theory_max_log = -0.5 * (log_N_ref - log_N_ref[0]) + ref_scaling['log_max_td'][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    for theta_target_deg, res in results_by_angle.items():
        scaling = res['scaling']
        color, linewidth, zorder = get_theta_style(theta_target_deg)

        ax1.plot(scaling['log_N'], scaling['log_mean_td'], 'o', color=color, markeredgewidth=1.5,
                  zorder=zorder + 1, label=fr"$\theta = {res['theta_plot']}^\circ$")
        ax1.plot(scaling['log_N'], scaling['fit_mean_log'], color=color, linewidth=linewidth, zorder=zorder)

        ax2.plot(scaling['log_N'], scaling['log_max_td'], 's', color=color, markeredgewidth=1.5,
                  zorder=zorder + 1, label=fr"$\theta = {res['theta_plot']}^\circ$")
        ax2.plot(scaling['log_N'], scaling['fit_max_log'], color=color, linewidth=linewidth, zorder=zorder)

    ax1.plot(log_N_ref, theory_mean_log, color='black', linestyle='--', linewidth=1.5, zorder=1,
              label=r'Theory: slope = $-0.5$')
    ax2.plot(log_N_ref, theory_max_log, color='black', linestyle='--', linewidth=1.5, zorder=1,
              label=r'Theory: slope = $-0.5$')

    ax1.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(r'$\log_{10} (\langle T \rangle_t)$', fontsize=LABEL_FONTSIZE)
    ax1.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)

    ax2.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel(r'$\log_{10} (T_{\mathrm{max}})$', fontsize=LABEL_FONTSIZE)
    ax2.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)

    save_fig(fig, "Comparison_TraceDistance_Scaling_AllAngles", Output_dir)
    print(f"Combined trace distance scaling plot saved in {Output_dir}")


# ==========================================
# COMBINED TIME-RESOLVED PLOTS (all angles overlaid)
# ==========================================
def plot_combined_comparisons(results_by_angle, MODE='normal'):
    """
    Builds THREE figures, each overlaying the curves of every angle present
    in `results_by_angle`. theta = 90 is always drawn on top (thicker line +
    higher zorder).
    """
    Output_dir = os.path.join("../Results/Plot/Trace_Distance_Scaling", MODE, "Comparison_All_Angles")
    os.makedirs(Output_dir, exist_ok=True)

    # # ---------------------------------------------------------
    # # Figure: Variance of the Site-1 population, all angles
    # # ---------------------------------------------------------
    # fig1, ax1 = plt.subplots(figsize=(9, 6.5))
    # for theta_target_deg, res in results_by_angle.items():
    #     color, linewidth, zorder = get_theta_style(theta_target_deg)
    #     ax1.plot(res['times'], res['var_pop_site1'], color=color, linewidth=linewidth, zorder=zorder,
    #               label=fr"$\theta = {res['theta_plot']}^\circ$")
    # ax1.set_xlabel(r'$Time [1/V]$', fontsize=LABEL_FONTSIZE)
    # ax1.set_ylabel(r'$\mathrm{Var}\left(P_{1}(t)\right)$', fontsize=LABEL_FONTSIZE)
    # ax1.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)
    # save_fig(fig1, "Comparison_Variance_PopSite1_AllAngles", Output_dir)
    # print(f"Combined Site-1 population variance plot saved in {Output_dir}")

    # # ---------------------------------------------------------
    # # Figure: Trace distance (average matrix), all angles
    # # ---------------------------------------------------------
    # fig2, ax2 = plt.subplots(figsize=(9, 6.5))
    # for theta_target_deg, res in results_by_angle.items():
    #     color, linewidth, zorder = get_theta_style(theta_target_deg)
    #     ax2.plot(res['times'], res['td_series'], color=color, linewidth=linewidth, zorder=zorder,
    #               label=fr"$\theta = {res['theta_plot']}^\circ$")
    # ax2.set_xlabel(r'$Time [1/V]$', fontsize=LABEL_FONTSIZE)
    # ax2.set_ylabel(r'$T(t)$', fontsize=LABEL_FONTSIZE)
    # ax2.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
    # save_fig(fig2, "Comparison_TraceDistance_AllAngles", Output_dir)
    # print(f"Combined trace distance plot saved in {Output_dir}")

    # # ---------------------------------------------------------
    # # Figure: Variance of the trace distance, all angles
    # # ---------------------------------------------------------
    # fig3, ax3 = plt.subplots(figsize=(9, 6.5))
    # for theta_target_deg, res in results_by_angle.items():
    #     color, linewidth, zorder = get_theta_style(theta_target_deg)
    #     ax3.plot(res['times'], res['var_td'], color=color, linewidth=linewidth, zorder=zorder,
    #               label=fr"$\theta = {res['theta_plot']}^\circ$")
    # ax3.set_xlabel(r'$Time [1/V]$', fontsize=LABEL_FONTSIZE)
    # ax3.set_ylabel(r'$\mathrm{Var}\left(T(t)\right)$', fontsize=LABEL_FONTSIZE)
    # ax3.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
    # save_fig(fig3, "Comparison_Variance_TraceDistance_AllAngles", Output_dir)
    # print(f"Combined trace distance variance plot saved in {Output_dir}")


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":

    # target_angles = [0.0, 90.0, 30.0, 45.0, 60.0]
    target_angles = [0.0, 90.0]
    mode = 'normal'

    results_by_angle = {}
    for angle in target_angles:
        result = compute_metrics_for_angle(angle, MODE=mode)
        if result is not None:
            results_by_angle[angle] = result

            # (a) Individual plots for THIS single angle
            plot_individual_scaling(result, MODE=mode)
            plot_individual_time_series(result, MODE=mode)

    # (b) Cumulative plots overlaying ALL analyzed angles together
    plot_combined_scaling(results_by_angle, MODE=mode)
    plot_combined_comparisons(results_by_angle, MODE=mode)

    print("\nAll analysis completed: individual per-angle plots + combined comparison plots generated.")