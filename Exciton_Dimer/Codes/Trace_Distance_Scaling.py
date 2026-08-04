#!/usr/bin/env python
# coding: utf-8
"""
Trace Distance scaling analysis with bootstrapping error bars for the Exciton Dimer model.
Reconstructs the 2x2 average density matrix on the fly from saved populations and coherences.
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
# MAIN ANALYSIS FUNCTION
# ==========================================
def analyze_scaling(theta_target_deg, MODE='normal', dt=0.01, max_N=20000, n_bootstraps=10):
    """
    Loads Exciton Dimer data for a specific angle, reconstructs average density matrices,
    computes mean and max Trace Distance for varying N_traj using random sampling, 
    fits the scaling in log10 space, and generates the plots.
    """
    print(f"\n{'='*50}\nProcessing Target Theta = {theta_target_deg}° ({MODE})\n{'='*50}")
    
    # Mathematically invert the angle for plotting/labels as requested
    theta_plot = 90.0 - theta_target_deg
    
    # Set seed for reproducible error bars
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
    
    n_times = len(data['times'])
    
    # Rebuild the 2x2 exact Lindblad reference matrix
    # Based on the dimer structure: index 2,2 -> |10>, index 1,1 -> |01>
    rho_exact = np.zeros((n_times, 2, 2), dtype=np.complex128)
    rho_exact[:, 0, 0] = rho_lindblad_complete[:, 2, 2] # |10><10|
    rho_exact[:, 1, 1] = rho_lindblad_complete[:, 1, 1] # |01><01|
    rho_exact[:, 0, 1] = rho_lindblad_complete[:, 2, 1] # |10><01|
    rho_exact[:, 1, 0] = rho_lindblad_complete[:, 1, 2] # |01><10|
    
    # Extract raw trajectory elements
    raw_pop_10 = data['pop_00']
    raw_pop_01 = data['pop_11']
    raw_coh_10_01 = data['coh_10_01']
    raw_coh_01_10 = data['coh_01_10']
    
    total_available_traj = raw_pop_10.shape[1]
    
    # Define the subset of trajectories to evaluate
    N_list = np.array([
        100, 200, 500, 1000, 2000, 3000, 4000, 5000, 
        7500, 10000, 12500, 15000, 17500, 20000
    ])
    
    # Ensure we don't request more trajectories than available
    N_list = N_list[N_list <= total_available_traj]
    
    log_mean_td_list = []
    log_mean_err_list = []
    
    log_max_td_list = []
    log_max_err_list = []
    
    print("Computing Trace Distance scaling and error bars via random sampling...")
    for N in N_list:
        sample_log_means = []
        sample_log_maxs = []
        
        # If N equals the total available trajectories, we can only extract 1 unique combination
        current_bootstraps = 1 if N == total_available_traj else n_bootstraps
        
        for b in range(current_bootstraps):
            if N == total_available_traj:
                idx = np.arange(total_available_traj)
            else:
                # Randomly pick N trajectories without replacement
                idx = np.random.choice(total_available_traj, N, replace=False)
                
            # Compute averages of density matrix elements over the selected subset
            pop_10_avg = np.mean(raw_pop_10[:, idx], axis=1)
            pop_01_avg = np.mean(raw_pop_01[:, idx], axis=1)
            coh_10_01_avg = np.mean(raw_coh_10_01[:, idx], axis=1)
            coh_01_10_avg = np.mean(raw_coh_01_10[:, idx], axis=1)
            
            # Reconstruct the 2x2 average density matrix for this sampling
            # CRITICAL FIX: The coherences are mapped correctly matching the Lindblad array
            rho_avg_N = np.zeros((n_times, 2, 2), dtype=np.complex128)
            rho_avg_N[:, 0, 0] = pop_10_avg
            rho_avg_N[:, 1, 1] = pop_01_avg
            rho_avg_N[:, 1, 0] = coh_10_01_avg  # mapped to c10 previously
            rho_avg_N[:, 0, 1] = coh_01_10_avg  # mapped to c01 previously
            
            # Calculate the Trace Distance over time using the robust generic function
            td_series = compute_trace_distance_series(rho_avg_N, rho_exact)
            
            # Skip the first 100 steps to avoid the artificial transient at t=0
            skip_idx = 100 if n_times > 200 else 0
            td_mean = np.mean(td_series[skip_idx:])
            td_max = np.max(td_series[skip_idx:])
            
            if td_mean > 0 and td_max > 0:
                sample_log_means.append(np.log10(td_mean))
                sample_log_maxs.append(np.log10(td_max))
            
        # Compute the final mean and standard deviation (error bar) in log10 space
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
    
    # Linear regression for Mean Trace Distance
    slope_mean, int_mean, r_mean, p_mean, err_mean_fit = linregress(log_N, log_mean_td)
    fit_mean_log = slope_mean * log_N + int_mean
    
    # Linear regression for Max Trace Distance
    slope_max, int_max, r_max, p_max, err_max_fit = linregress(log_N, log_max_td)
    fit_max_log = slope_max * log_N + int_max
    
    # Theoretical 1/sqrt(N) line in log10 space (slope = -0.5)
    theory_mean_log = -0.5 * (log_N - log_N[0]) + log_mean_td[0]
    theory_max_log = -0.5 * (log_N - log_N[0]) + log_max_td[0]

    print(f"Mean Trace Distance Fit: y = {slope_mean:.4f}x + {int_mean:.4f} (R^2 = {r_mean**2:.4f})")
    print(f"Max Trace Distance Fit: y = {slope_max:.4f}x + {int_max:.4f} (R^2 = {r_max**2:.4f})")

    # ==========================================
    # PLOTTING
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
    ax1.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right', title_fontsize=11)

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
    ax2.legend(title=fr"$\theta = {theta_plot}^\circ$", loc='upper right', title_fontsize=11)

    # Save figure
    filename = f"Trace_Distance_Scaling_Theta_{theta_str}"
    save_fig(fig, filename, Output_dir)
    print(f"Plot saved successfully in {Output_dir}")


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # Target angles: 0 (Diffusive Limit mapping to 90 plot) and 90 (Jump Limit mapping to 0 plot)
    target_angles = [0.0, 90.0]
    mode = 'normal'
    
    for angle in target_angles:
        analyze_scaling(angle, MODE=mode)
        
    print("\nAll scaling analysis completed.")