#!/usr/bin/env python
# coding: utf-8
"""
Trace Distance analysis for the Exciton Dimer model (extended version).

This script reconstructs the 2x2 average density matrix on the fly from saved
populations and coherences, and produces THREE separate figures:

1) Trace Distance SCALING vs N_traj (mean and max), WITHOUT error bars.
   (n_bootstraps=1: a single random draw per N, no repeated resampling.)

2) Time-resolved VARIANCE of the density-matrix estimator, computed using a
   single fixed sample of N_traj = 10000 trajectories.
   Var_total(t) = Var[rho_00(t)] + Var[rho_11(t)]
                + Var[Re(rho_01(t))] + Var[Im(rho_01(t))]
   (summed over the two independently-stored coherence arrays coh_10_01 and
   coh_01_10, consistent with how the density matrix is reconstructed below).

3) Time-resolved TRACE DISTANCE between the N=10000-trajectory average
   density matrix and the exact Lindblad density matrix, plotted as a
   function of time (no reduction to a single mean/max scalar).
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
    0: '#0072B2',
    90: '#D55E00',
    60: '#009E73',
    45: '#F0E442',
    30: '#CC79A7',
}


def get_theta_color(theta_target_deg, default='black'):
    """Returns the fixed color assigned to a given target theta angle (degrees)."""
    theta_key = int(round(theta_target_deg))
    return THETA_COLOR_MAP.get(theta_key, default)


# ==========================================
# FAST TRACE DISTANCE CALCULATION
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
# ==========================================
def compute_variance_series(raw_pop_10, raw_pop_01, raw_coh_10_01, raw_coh_01_10, idx):
    """
    Computes the time-resolved TOTAL variance of the density-matrix estimator
    (populations + coherences), using a fixed subsample of trajectories `idx`.

    Var_total(t) = Var[pop_10(t)] + Var[pop_01(t)]
                 + Var[Re(coh_10_01(t))] + Var[Im(coh_10_01(t))]
                 + Var[Re(coh_01_10(t))] + Var[Im(coh_01_10(t))]

    Parameters:
        raw_pop_10, raw_pop_01: 2D real arrays, shape (n_times, n_traj)
        raw_coh_10_01, raw_coh_01_10: 2D complex arrays, shape (n_times, n_traj)
        idx: 1D int array of trajectory indices to use
    Returns:
        1D float array, the total variance at each time step
    """
    var_pop10 = np.var(raw_pop_10[:, idx], axis=1)
    var_pop01 = np.var(raw_pop_01[:, idx], axis=1)

    var_coh1001 = np.var(raw_coh_10_01[:, idx].real, axis=1) + np.var(raw_coh_10_01[:, idx].imag, axis=1)
    var_coh0110 = np.var(raw_coh_01_10[:, idx].real, axis=1) + np.var(raw_coh_01_10[:, idx].imag, axis=1)

    var_total = var_pop10 + var_pop01 + var_coh1001 + var_coh0110
    return var_total


# ==========================================
# MAIN ANALYSIS FUNCTION
# ==========================================
def analyze_scaling(theta_target_deg, MODE='normal', dt=0.01, max_N=20000,
                     n_bootstraps=1, N_time_series=10000):
    """
    Loads Exciton Dimer data for a specific angle, reconstructs average density
    matrices, and produces:
      1) Trace Distance scaling vs N_traj (mean & max), NO error bars.
      2) Time-resolved total variance of the estimator at N = N_time_series.
      3) Time-resolved trace distance at N = N_time_series.
    """
    print(f"\n{'='*50}\nProcessing Target Theta = {theta_target_deg}° ({MODE})\n{'='*50}")

    # Mathematically invert the angle for plotting/labels as requested
    theta_plot = 90.0 - theta_target_deg

    # Fixed color assigned to this angle (used across all figures below)
    color_theta = get_theta_color(theta_target_deg)

    # Set seed for reproducible sampling
    np.random.seed(42)

    # Convert angle to match the saved filename convention
    theta_rad = np.radians(theta_target_deg)
    theta_str = f"{theta_rad:.6f}".replace(".", "p")
    dt_str = f"{dt:.6f}".replace(".", "p")

    if MODE == 'normal':
        Input_dir = "../Results/Data/Complete_rho/normal"
    elif MODE == 'close_to_90':
        Input_dir = "../Results/Data/Complete_rho/close_90_deg"
    else:
        raise ValueError(f"Unknown mode: {MODE}")

    # Format the angle folder properly
    angle_folder = str(int(theta_target_deg)) if theta_target_deg.is_integer() else str(theta_target_deg)
    Output_dir = os.path.join("../Results/Plot/Trace_Distance_Scaling", MODE, angle_folder)
    os.makedirs(Output_dir, exist_ok=True)

    filename = f"result_theta{theta_str}_dt{dt_str}_Ntraj{max_N}.npz"
    filepath = os.path.join(Input_dir, filename)

    try:
        data = np.load(filepath)
        print(f"Data loaded successfully from {filepath}")
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Skipping...")
        return

    # Extract baseline Lindblad dynamics
    if 'rho_list_lindblad' not in data:
        print("Error: 'rho_list_lindblad' not found in data. Skipping...")
        return
    rho_lindblad_complete = data['rho_list_lindblad']

    times = data['times']
    n_times = len(times)

    # Rebuild the 2x2 exact Lindblad reference matrix
    # Based on the dimer structure: index 2,2 -> |10>, index 1,1 -> |01>
    rho_exact = np.zeros((n_times, 2, 2), dtype=np.complex128)
    rho_exact[:, 0, 0] = rho_lindblad_complete[:, 2, 2]  # |10><10|
    rho_exact[:, 1, 1] = rho_lindblad_complete[:, 1, 1]  # |01><01|
    rho_exact[:, 0, 1] = rho_lindblad_complete[:, 2, 1]  # |10><01|
    rho_exact[:, 1, 0] = rho_lindblad_complete[:, 1, 2]  # |01><10|

    # Extract raw trajectory elements
    raw_pop_10 = data['pop_00']
    raw_pop_01 = data['pop_11']
    raw_coh_10_01 = data['coh_10_01']
    raw_coh_01_10 = data['coh_01_10']

    total_available_traj = raw_pop_10.shape[1]

    # # ==========================================================
    # # PART 1: TRACE DISTANCE SCALING vs N_traj -- NO ERROR BARS
    # # ==========================================================
    # N_list = np.array([
    #     100, 200, 500, 1000, 2000, 3000, 4000, 5000,
    #     7500, 10000, 12500, 15000, 17500, 20000
    # ])
    # N_list = N_list[N_list <= total_available_traj]

    # log_mean_td_list = []
    # log_max_td_list = []

    # print("Computing Trace Distance scaling (no error bars)...")
    # for N in N_list:
    #     sample_log_means = []
    #     sample_log_maxs = []

    #     current_bootstraps = 1 if N == total_available_traj else n_bootstraps

    #     for b in range(current_bootstraps):
    #         if N == total_available_traj:
    #             idx = np.arange(total_available_traj)
    #         else:
    #             idx = np.random.choice(total_available_traj, N, replace=False)

    #         pop_10_avg = np.mean(raw_pop_10[:, idx], axis=1)
    #         pop_01_avg = np.mean(raw_pop_01[:, idx], axis=1)
    #         coh_10_01_avg = np.mean(raw_coh_10_01[:, idx], axis=1)
    #         coh_01_10_avg = np.mean(raw_coh_01_10[:, idx], axis=1)

    #         rho_avg_N = np.zeros((n_times, 2, 2), dtype=np.complex128)
    #         rho_avg_N[:, 0, 0] = pop_10_avg
    #         rho_avg_N[:, 1, 1] = pop_01_avg
    #         rho_avg_N[:, 1, 0] = coh_10_01_avg
    #         rho_avg_N[:, 0, 1] = coh_01_10_avg

    #         td_series = compute_trace_distance_series(rho_avg_N, rho_exact)

    #         # Skip the first 100 steps to avoid the artificial transient at t=0
    #         skip_idx = 100 if n_times > 200 else 0
    #         td_mean = np.mean(td_series[skip_idx:])
    #         td_max = np.max(td_series[skip_idx:])

    #         if td_mean > 0 and td_max > 0:
    #             sample_log_means.append(np.log10(td_mean))
    #             sample_log_maxs.append(np.log10(td_max))

    #     # Average across bootstraps (kept only to stabilize the estimate,
    #     # NOT plotted as an error bar)
    #     log_mean_td_list.append(np.mean(sample_log_means))
    #     log_max_td_list.append(np.mean(sample_log_maxs))

    # log_mean_td = np.array(log_mean_td_list)
    # log_max_td = np.array(log_max_td_list)

    # # --- Log10 fitting ---
    # log_N = np.log10(N_list)

    # slope_mean, int_mean, r_mean, p_mean, err_mean_fit = linregress(log_N, log_mean_td)
    # fit_mean_log = slope_mean * log_N + int_mean

    # slope_max, int_max, r_max, p_max, err_max_fit = linregress(log_N, log_max_td)
    # fit_max_log = slope_max * log_N + int_max

    # theory_mean_log = -0.5 * (log_N - log_N[0]) + log_mean_td[0]
    # theory_max_log = -0.5 * (log_N - log_N[0]) + log_max_td[0]

    # print(f"Mean Trace Distance Fit: y = {slope_mean:.4f}x + {int_mean:.4f} (R^2 = {r_mean**2:.4f})")
    # print(f"Max Trace Distance Fit: y = {slope_max:.4f}x + {int_max:.4f} (R^2 = {r_max**2:.4f})")

    # # --- Plotting (no error bars) ---
    # fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ax1.plot(log_N, log_mean_td, 'o', color=color_theta, markeredgewidth=1.5,
    #           label='Raw Data', zorder=3)
    # ax1.plot(log_N, fit_mean_log, color='black', linestyle='-', linewidth=1.5,
    #           label=fr'Fit: $y = {slope_mean:.2f}x {int_mean:+.2f}$', zorder=4)
    # ax1.plot(log_N, theory_mean_log, color='dimgray', linestyle='--', linewidth=1.5,
    #           label=r'Theory: slope = $-0.5$', zorder=2)

    # ax1.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$')
    # ax1.set_ylabel(r'$\log_{10} (\langle T \rangle_t)$')
    # ax1.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right', title_fontsize=16)

    # ax2.plot(log_N, log_max_td, 's', color=color_theta, markeredgewidth=1.5,
    #           label='Raw Data', zorder=3)
    # ax2.plot(log_N, fit_max_log, color='black', linestyle='--', linewidth=1.5,
    #           label=fr'Fit: $y = {slope_max:.2f}x {int_max:+.2f}$', zorder=4)
    # ax2.plot(log_N, theory_max_log, color='dimgray', linestyle='--', linewidth=1.5,
    #           label=r'Theory: slope = $-0.5$', zorder=2)

    # ax2.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$')
    # ax2.set_ylabel(r'$\log_{10} (T_{\mathrm{max}})$')
    # ax2.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right', title_fontsize=16)

    # scaling_filename = f"Trace_Distance_Scaling_NoErrorBars_Theta_{theta_str}"
    # save_fig(fig1, scaling_filename, Output_dir)
    # print(f"Scaling plot (no error bars) saved in {Output_dir}")

    # ==========================================================
    # Fix the trajectory subset used for PARTS 2 and 3 below,
    # so that variance(t) and trace-distance(t) refer to the SAME sample.
    # ==========================================================
    N_fixed = min(N_time_series, total_available_traj)
    if N_fixed == total_available_traj:
        idx_fixed = np.arange(total_available_traj)
    else:
        idx_fixed = np.random.choice(total_available_traj, N_fixed, replace=False)

    # # ==========================================================
    # # PART 2: TIME-RESOLVED VARIANCE at N = N_time_series
    # # ==========================================================
    # print(f"Computing time-resolved variance for N = {N_fixed} trajectories...")
    # var_series = compute_variance_series(raw_pop_10, raw_pop_01, raw_coh_10_01, raw_coh_01_10, idx_fixed)

    # fig2, ax3 = plt.subplots(figsize=(8, 6))
    # ax3.plot(times, var_series, color=color_theta, linewidth=1.8, label='Trajectories Variance')
    # ax3.set_xlabel(r'$Time [1/V]$')
    # ax3.set_ylabel(r'$\mathrm{Var}(t)$')
    # # ax3.set_title(fr"Estimator Variance vs Time — $\theta = {theta_plot}^\circ$, $N_{{\mathrm{{traj}}}} = {N_fixed}$")
    # ax3.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right', title_fontsize=16)

    # variance_filename = f"Variance_vs_Time_Theta_{theta_str}_N{N_fixed}"
    # save_fig(fig2, variance_filename, Output_dir)
    # print(f"Variance-vs-time plot saved in {Output_dir}")

    # ==========================================================
    # PART 3: TIME-RESOLVED TRACE DISTANCE at N = N_time_series
    # ==========================================================
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

    td_series_fixed = compute_trace_distance_series(rho_avg_fixed, rho_exact)

    fig3, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(times, td_series_fixed, color=color_theta, linewidth=1.8, label='Trace Distance')
    ax4.set_xlabel(r'$Time [1/V]$')
    ax4.set_ylabel(r'$T(t)$')
    # ax4.set_title(fr"Trace Distance vs Time — $\theta = {theta_plot}^\circ$, $N_{{\mathrm{{traj}}}} = {N_fixed}$")
    ax4.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right', title_fontsize=16)

    td_time_filename = f"TraceDistance_vs_Time_Theta_{theta_str}_N{N_fixed}"
    save_fig(fig3, td_time_filename, Output_dir)
    print(f"Trace-distance-vs-time plot saved in {Output_dir}")


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":

    # Target angles: 0 (Diffusive Limit mapping to 90 plot) and 90 (Jump Limit mapping to 0 plot)
    # target_angles = [0.0, 30.0, 45.0, 60.0, 90.0]
    target_angles = [0.0, 90.0]
    mode = 'normal'

    for angle in target_angles:
        analyze_scaling(angle, MODE=mode)

    print("\nAll analysis (scaling + variance/time + trace distance/time) completed.")