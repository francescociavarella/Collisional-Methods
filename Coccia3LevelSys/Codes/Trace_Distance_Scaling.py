#!/usr/bin/env python
# coding: utf-8

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
# FAST TRACE DISTANCE CALCULATION
# ==========================================
@njit
def compute_trace_distance_series(rho_a, rho_b):
    """
    Computes the trace distance between two series of density matrices over time.
    Trace distance T = 0.5 * sum(|eigenvalues(rho_a - rho_b)|)
    
    Parameters:
        rho_a, rho_b: 3D complex arrays of shape (n_times, N_dim, N_dim)
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
# MAIN ANALYSIS FUNCTION
# ==========================================
def analyze_scaling(theta_deg, dt=0.01, max_N=10000, n_bootstraps=10):
    """
    Loads data for a specific angle, computes mean and max trace distance 
    for varying N_traj using random sampling to estimate error bars, 
    fits the scaling in log10 space, generates the scaling plots, 
    and plots the trace distance over time for the maximum number of trajectories.
    """
    print(f"\n{'='*50}\nProcessing Theta = {theta_deg}°\n{'='*50}")
    
    # Set seed for reproducible error bars
    np.random.seed(42)
    
    # Convert angle to match the saved filename convention (radians)
    theta_rad = np.radians(theta_deg)
    dt_str = f"{dt:.6f}".replace(".", "p")
    phi_str = f"{theta_rad:.4f}".replace(".", "p")
    
    results_dir = "../Results/Data/Complete_rho/"
    Output_dir = f"../Results/Plot/Trace_Distance_Scaling/{phi_str}"
    os.makedirs(Output_dir, exist_ok=True)
    
    fname = os.path.join(results_dir, f"result_phi{phi_str}_dt{dt_str}_Ntraj{max_N}.npz")
    
    try:
        data = np.load(fname)
        print(f"Data loaded successfully from {fname}")
    except FileNotFoundError:
        print(f"Error: File {fname} not found. Skipping...")
        return
        
    # Extract time array and baseline Lindblad dynamics
    times = data['times']
    rho_lindblad = data['rho_list_lindblad']
    
    # Extract the full raw trajectories (N_dim, N_dim, n_times, N_traj)
    rho_tot_all = data['rho_tot_all']
    
    # Move axes to get (N_traj, n_times, N_dim, N_dim) for easier slicing
    rho_tot_all = np.moveaxis(rho_tot_all, [0, 1, 2, 3], [2, 3, 1, 0])
    total_available_traj = rho_tot_all.shape[0]
    
    # ==========================================
    # 1. TRACE DISTANCE OVER TIME (FOR MAX TRAJECTORIES)
    # ==========================================
    print("Computing Trace Distance over time for the full trajectory set...")
    rho_avg_max_traj = np.mean(rho_tot_all, axis=0)
    td_time_series = compute_trace_distance_series(rho_avg_max_traj, rho_lindblad)
    
    # Plot Trace Distance vs Time
    fig_time, ax_time = plt.subplots(figsize=(10, 5))
    ax_time.plot(times, td_time_series, color='crimson', linewidth=1.5, label=fr'Trace Distance ($N_{{\mathrm{{traj}}}} = {total_available_traj}$')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel(r'Trace Distance $\mathcal{T}(t)$')
    ax_time.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
    
    filename_time = f"Trace_Distance_vs_Time_Theta_{phi_str}"
    save_fig(fig_time, filename_time, Output_dir)
    print(f"Trace distance vs time plot saved successfully in {Output_dir}")

    # ==========================================
    # 2. SCALING ANALYSIS FOR VARYING N_traj
    # ==========================================
    N_list = np.array([
        100, 200, 300, 400, 500, 600, 700, 800, 900, 
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
    ])
    
    log_mean_td_list = []
    log_mean_err_list = []
    
    log_max_td_list = []
    log_max_err_list = []
    
    print("Computing scaling metrics and error bars via random sampling...")
    for N in N_list:
        sample_log_means = []
        sample_log_maxs = []
        
        current_bootstraps = 1 if N == total_available_traj else n_bootstraps
        
        for b in range(current_bootstraps):
            if N == total_available_traj:
                idx = np.arange(total_available_traj)
            else:
                idx = np.random.choice(total_available_traj, N, replace=False)
                
            rho_avg_N = np.mean(rho_tot_all[idx, :, :, :], axis=0)
            td_series = compute_trace_distance_series(rho_avg_N, rho_lindblad)
            
            sample_log_means.append(np.log10(np.mean(td_series)))
            sample_log_maxs.append(np.log10(np.max(td_series)))
            
        log_mean_td_list.append(np.mean(sample_log_means))
        log_mean_err_list.append(np.std(sample_log_means))
        
        log_max_td_list.append(np.mean(sample_log_maxs))
        log_max_err_list.append(np.std(sample_log_maxs))
        
    log_mean_td = np.array(log_mean_td_list)
    log_mean_err = np.array(log_mean_err_list)
    
    log_max_td = np.array(log_max_td_list)
    log_max_err = np.array(log_max_err_list)
    
    # ==========================================
    # LOG10 FITTING
    # ==========================================
    log_N = np.log10(N_list)
    
    slope_mean, int_mean, r_mean, p_mean, err_mean = linregress(log_N, log_mean_td)
    fit_mean_log = slope_mean * log_N + int_mean
    
    slope_max, int_max, r_max, p_max, err_max = linregress(log_N, log_max_td)
    fit_max_log = slope_max * log_N + int_max
    
    theory_mean_log = -0.5 * (log_N - log_N[0]) + log_mean_td[0]
    theory_max_log = -0.5 * (log_N - log_N[0]) + log_max_td[0]

    print(f"Mean TD Fit: y = {slope_mean:.4f}x + {int_mean:.4f} (R^2 = {r_mean**2:.4f})")
    print(f"Max TD Fit: y = {slope_max:.4f}x + {int_max:.4f} (R^2 = {r_max**2:.4f})")

    # ==========================================
    # PLOTTING SCALING RESULTS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel 1: Mean Trace Distance ---
    ax1.errorbar(log_N, log_mean_td, yerr=log_mean_err, fmt='o', color='royalblue', 
                 ecolor='royalblue', capsize=4, elinewidth=1.5, markeredgewidth=1.5, 
                 label='Raw Data', zorder=3)
    ax1.plot(log_N, fit_mean_log, color='red', linestyle='-', linewidth=2, 
             label=fr'Fit: $y = {slope_mean:.2f}x {int_mean:+.2f}$', zorder=4)
    ax1.plot(log_N, theory_mean_log, color='dimgray', linestyle='--', linewidth=1.5, 
             label=r'Theory: slope = $-0.5$', zorder=2)
    
    ax1.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$')
    ax1.set_ylabel(r'$\log_{10} (\langle \mathcal{T} \rangle_t)$')
    ax1.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right')

    # --- Panel 2: Max Trace Distance ---
    ax2.errorbar(log_N, log_max_td, yerr=log_max_err, fmt='s', color='mediumseagreen', 
                 ecolor='mediumseagreen', capsize=4, elinewidth=1.5, markeredgewidth=1.5, 
                 label='Raw Data', zorder=3)
    ax2.plot(log_N, fit_max_log, color='red', linestyle='--', linewidth=2, 
             label=fr'Fit: $y = {slope_max:.2f}x {int_max:+.2f}$', zorder=4)
    ax2.plot(log_N, theory_max_log, color='dimgray', linestyle='--', linewidth=1.5, 
             label=r'Theory: slope = $-0.5$', zorder=2)
    
    ax2.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$')
    ax2.set_ylabel(r'$\log_{10} (\mathcal{T}_{\mathrm{max}})$')
    ax2.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right')

    # Save scaling figure
    filename = f"Trace_Distance_Scaling_Theta_{phi_str}"
    save_fig(fig, filename, Output_dir)
    print(f"Scaling plot saved successfully in {Output_dir}")


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # Target angles: 0 (Quantum Jump limit) and 90 (Diffusive limit)
    target_angles = [0.0, 90.0]
    
    for angle in target_angles:
        analyze_scaling(angle)
        
    print("\nAll analysis completed successfully.")