#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, wasserstein_distance

# ==========================
# Input Parsing and Setup
# ==========================
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 90.0  

dt = 1.0
N_traj = 10000

dt_str = f"{dt:.2f}".replace(".", "p")
theta_str = f"{theta_deg:.3f}".replace(".", "p")

results_dir = "../Results/Data/"
Output_dir = f"../Results/Plot/TimeSlices/{theta_str}"
os.makedirs(Output_dir, exist_ok=True)

fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Loading data for Theta = {theta_deg} deg...")
except FileNotFoundError:
    print(f"Error: File {fname} not found.")
    sys.exit(1)

times = data['times']
eigenvectors = data['eigenvectors']
psi_traj_exc = data['psi_traj']         # Exciton basis

# ==========================
# Compute Populations
# ==========================
# Exciton Basis
pop_traj_exc = np.abs(psi_traj_exc) ** 2

# Site Basis
psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)
pop_traj_site = np.abs(psi_traj_site) ** 2

# ==========================
# Identify Time Slices
# ==========================
# The requested times in fs
target_times = [10, 71, 100, 500, 1000, 2000, times[-1]]
time_indices = []

for t_target in target_times:
    # Find the closest available time index in the array
    idx = (np.abs(times - t_target)).argmin()
    time_indices.append(idx)

# ===========================
# Plotting Configuration
# ===========================
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'figure.autolayout': True
})

def save_fig(fig, filename):
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)

# ==========================
# Core Plotting Function
# ==========================
def plot_time_slices(pop_data, title_prefix, file_prefix, color_hist):
    """
    Generates a row of histograms for the specified time slices.
    """
    num_slices = len(time_indices)
    fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4), sharey=False)
    
    # Bins from 0.0 to 1.0 since populations are probabilities
    bins = np.linspace(0.0, 1.0, 50)
    bin_width = bins[1] - bins[0]
    
    for k, idx in enumerate(time_indices):
        ax = axes[k]
        t_actual = times[idx]
        
        # Extract the population values of all trajectories at this specific time step
        data_slice = pop_data[idx, :]
        
        # Array of weights so that the sum of all bin heights equals 1.0
        weights = np.ones_like(data_slice) / len(data_slice)
        
        # Plot Histogram (Relative Frequency)
        ax.hist(data_slice, bins=bins, weights=weights, alpha=0.7, color=color_hist, edgecolor='black')
        
        # ==========================
        # Gaussian Fit & Metrics
        # ==========================
        mu, std = norm.fit(data_slice)
        x_fit = np.linspace(0.0, 1.0, 100)
        pdf_fit = norm.pdf(x_fit, mu, std) * bin_width
        
        # 1. Kolmogorov-Smirnov Test
        # D statistic: 0 = perfect match, 1 = maximum divergence
        ks_stat, _ = kstest(data_slice, 'norm', args=(mu, std))
        
        # 2. Wasserstein Distance (Earth Mover's Distance)
        # We generate a theoretical normal sample (with fixed seed for reproducibility) 
        # to compare against the empirical data
        theoretical_sample = norm.rvs(loc=mu, scale=std, size=len(data_slice), random_state=42)
        w_dist = wasserstein_distance(data_slice, theoretical_sample)
        
        # Custom label for the legend including all metrics
        fit_label = (f'Gauss Fit\n'
                     f'$\\mu$={mu:.2f}, $\\sigma$={std:.2f}\n'
                     f'K-S: {ks_stat:.2f}\n'
                     f'$W_D$: {w_dist:.3f}')
        
        ax.plot(x_fit, pdf_fit, 'r--', linewidth=2, label=fit_label)
        
        ax.set_title(f"Time = {t_actual:.0f} fs")
        ax.set_xlabel("Population")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        
        if k == 0:
            ax.set_ylabel("Fraction of Trajectories")

        # Adjust legend fontsize to fit the new text
        ax.legend(loc='upper right', fontsize=7)

    fig.suptitle(f"{title_prefix} | Theta = {theta_deg} deg", fontsize=16, y=1.05)
    save_fig(fig, f"{file_prefix}_Theta_{theta_str}")


# ==========================
# Execute Plotting
# ==========================
print("Generating histograms for Site 1...")
# We select index 0 for Site 1
plot_time_slices(pop_data=pop_traj_site[0, :, :], 
                 title_prefix="Population Distribution: SITE 1", 
                 file_prefix="Hist_Slice_Site1", 
                 color_hist='skyblue')

print("Generating histograms for Exciton 1...")
# We select index 0 for Exciton 1
plot_time_slices(pop_data=pop_traj_exc[0, :, :], 
                 title_prefix="Population Distribution: EXCITON 1", 
                 file_prefix="Hist_Slice_Exciton1", 
                 color_hist='lightgreen')

print("Time-slice analysis completed successfully!")