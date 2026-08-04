#!/usr/bin/env python
# coding: utf-8
"""
Trace Distance scaling analysis with bootstrapping error bars for the FMO model.
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


def get_average_density_matrix(psi_sub):
    """
    Constructs the average density matrix from a subset of pure state trajectories.
    
    Parameters:
        psi_sub: 3D complex array of shape (N_site, n_times, N_subset)
    Returns:
        rho_avg: 3D complex array of shape (n_times, N_site, N_site)
    """
    N_subset = psi_sub.shape[2]
    # Build |psi><psi| and average over the subset axis (k)
    rho_avg = np.einsum('itk, jtk -> tij', psi_sub, np.conjugate(psi_sub)) / N_subset
    return rho_avg


# ==========================================
# MAIN ANALYSIS FUNCTION
# ==========================================
def analyze_scaling(theta_deg, dt=1.0, max_N=10000, n_bootstraps=10):
    """
    Loads FMO data for a specific angle, computes mean and max trace distance 
    for varying N_traj using random sampling to estimate error bars, 
    fits the scaling in log10 space, and generates the plot.
    """
    print(f"\n{'='*50}\nProcessing Theta = {theta_deg}°\n{'='*50}")
    
    # Set seed for reproducible error bars
    np.random.seed(42)
    
    dt_str = f"{dt:.2f}".replace(".", "p")
    theta_str = f"{theta_deg:.3f}".replace(".", "p")
    
    results_dir = "../Results/Data/"
    Output_dir = f"../Results/Plot/Scaling/{theta_str}"
    os.makedirs(Output_dir, exist_ok=True)
    
    fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{max_N}.npz")
    
    try:
        data = np.load(fname)
        print(f"Data loaded successfully from {fname}")
    except FileNotFoundError:
        print(f"Error: File {fname} not found. Skipping...")
        return
        
    # Extract baseline Redfield dynamics (n_times, N_site, N_site)
    if 'rho_redfield_site' not in data:
        print("Error: 'rho_redfield_site' not found in data. Skipping...")
        return
    rho_redfield = data['rho_redfield_site']
    
    # Extract eigenvectors to transform basis
    eigenvectors = data['eigenvectors']
    
    # Extract the pure state trajectories (N_site, n_times, n_traj) in exciton basis
    psi_traj_exc = data['psi_traj']
    
    # Transform pure states to the site basis to match rho_redfield_site
    print("Transforming trajectories to site basis...")
    psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)
    
    total_available_traj = psi_traj_site.shape[2]
    
    # Define the subset of trajectories to evaluate
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
        
        # If N equals the total available trajectories, we can only extract 1 unique combination
        current_bootstraps = 1 if N == total_available_traj else n_bootstraps
        
        for b in range(current_bootstraps):
            if N == total_available_traj:
                idx = np.arange(total_available_traj)
            else:
                # Randomly pick N trajectories without replacement
                idx = np.random.choice(total_available_traj, N, replace=False)
                
            # Slice the selected trajectories
            psi_sub = psi_traj_site[:, :, idx]
            
            # Construct the average density matrix for this subset
            rho_avg_N = get_average_density_matrix(psi_sub)
            
            # Calculate trace distance against Redfield over time
            td_series = compute_trace_distance_series(rho_avg_N, rho_redfield)
            
            # Since the very first steps might have TD near 0 (transient), we avoid log10(0)
            td_mean = np.mean(td_series)
            td_max = np.max(td_series)
            
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
    slope_mean, int_mean, r_mean, p_mean, err_mean = linregress(log_N, log_mean_td)
    fit_mean_log = slope_mean * log_N + int_mean
    
    # Linear regression for Max Trace Distance
    slope_max, int_max, r_max, p_max, err_max = linregress(log_N, log_max_td)
    fit_max_log = slope_max * log_N + int_max
    
    # Theoretical 1/sqrt(N) line in log10 space (slope = -0.5)
    theory_mean_log = -0.5 * (log_N - log_N[0]) + log_mean_td[0]
    theory_max_log = -0.5 * (log_N - log_N[0]) + log_max_td[0]

    print(f"Mean TD Fit: y = {slope_mean:.4f}x + {int_mean:.4f} (R^2 = {r_mean**2:.4f})")
    print(f"Max TD Fit: y = {slope_max:.4f}x + {int_max:.4f} (R^2 = {r_max**2:.4f})")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel 1: Mean Trace Distance ---
    # Blue dots with error bars for raw data
    ax1.errorbar(log_N, log_mean_td, yerr=log_mean_err, fmt='o', color='royalblue', 
                 ecolor='royalblue', capsize=4, elinewidth=1.5, markeredgewidth=1.5, 
                 label='Raw Data', zorder=3)
                 
    # Solid red line for fit
    ax1.plot(log_N, fit_mean_log, color='red', linestyle='-', linewidth=2, 
             label=fr'Fit: $y = {slope_mean:.2f}x {int_mean:+.2f}$', zorder=4)
             
    # Dark gray dashed line for theory
    ax1.plot(log_N, theory_mean_log, color='dimgray', linestyle='--', linewidth=1.5, 
             label=r'Theory: slope = $-0.5$', zorder=2)
    
    ax1.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$')
    ax1.set_ylabel(r'$\log_{10} (\langle \mathcal{T} \rangle_t)$')
    ax1.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)

    # --- Panel 2: Max Trace Distance ---
    # Green squares with error bars for raw data
    ax2.errorbar(log_N, log_max_td, yerr=log_max_err, fmt='s', color='mediumseagreen', 
                 ecolor='mediumseagreen', capsize=4, elinewidth=1.5, markeredgewidth=1.5, 
                 label='Raw Data', zorder=3)
                 
    # Dashed red line for fit
    ax2.plot(log_N, fit_max_log, color='red', linestyle='--', linewidth=2, 
             label=fr'Fit: $y = {slope_max:.2f}x {int_max:+.2f}$', zorder=4)
             
    # Dark gray dashed line for theory
    ax2.plot(log_N, theory_max_log, color='dimgray', linestyle='--', linewidth=1.5, 
             label=r'Theory: slope = $-0.5$', zorder=2)
    
    ax2.set_xlabel(r'$\log_{10}(N_{\mathrm{traj}})$')
    ax2.set_ylabel(r'$\log_{10} (\mathcal{T}_{\mathrm{max}})$')
    ax2.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)

    # Save figure
    filename = f"Trace_Distance_Scaling_Theta_{theta_str}"
    save_fig(fig, filename, Output_dir)
    print(f"Plot saved successfully in {Output_dir}")


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # Target angles: only 0 (Quantum Jump limit) and 90 (Diffusive limit)
    target_angles = [0.0, 90.0]
    
    for angle in target_angles:
        analyze_scaling(angle)
        
    print("\nAll scaling analysis completed.")