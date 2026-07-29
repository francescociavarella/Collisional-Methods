#!/usr/bin/env python
# coding: utf-8

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress warnings for deterministic initial states (t=0)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# Extract Data
times = data['times']
eigenvectors = data['eigenvectors']
psi_traj_exc = data['psi_traj']         # Exciton basis

# Extract number of sites dynamically from the shape of the array
N_site = psi_traj_exc.shape[0]

# ==========================
# Compute Populations
# ==========================
# Exciton Basis Population
pop_traj_exc = np.abs(psi_traj_exc) ** 2

# Site Basis Population
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
    'figure.autolayout': False # Disabled autolayout to manually adjust large grids
})

def save_fig(fig, filename):
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)

# Generate a colormap to differentiate rows (sites/excitons)
colors = plt.cm.viridis(np.linspace(0, 1, N_site))

# ==========================
# Core Plotting Function
# ==========================
def plot_grid(pop_data, basis_name, file_prefix, use_density):
    """
    Generates a full grid of histograms without fits or statistical metrics.
    Rows = Sites / Excitons
    Columns = Time Slices
    use_density = True plots Probability Density (Area=1).
    use_density = False plots Trajectory Fraction (Sum of heights=1).
    """
    num_cols = len(time_indices)
    num_rows = N_site
    
    # Create a large figure to hold the N_site x N_times grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3 * num_rows), sharex=True, sharey=False)
    
    # Handle the edge case where N_site is 1 (axes would be 1D)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    bins = np.linspace(0.0, 1.0, 50)
    
    for i in range(num_rows):
        row_color = colors[i]
        
        for k, idx in enumerate(time_indices):
            ax = axes[i, k]
            t_actual = times[idx]
            
            # Extract data for the specific site/exciton 'i' at time 'idx'
            data_slice = pop_data[i, idx, :]
            
            if use_density:
                # Plot Probability Density (Area = 1)
                ax.hist(data_slice, bins=bins, density=True, alpha=0.7, color=row_color, edgecolor='black')
                # Let the Y-axis adjust automatically because density peaks can exceed 1.0
                ax.set_ylim(bottom=0)
            else:
                # Plot Fraction of Trajectories (Sum of heights = 1)
                weights = np.ones_like(data_slice) / len(data_slice)
                ax.hist(data_slice, bins=bins, weights=weights, alpha=0.7, color=row_color, edgecolor='black')
                # Lock Y-axis to 1.0 since probabilities cannot exceed 100%
                ax.set_ylim(0, 1.0)
            
            # Formatting Labels and Titles
            if i == 0:
                ax.set_title(f"Time = {t_actual:.0f} fs")
            if i == num_rows - 1:
                ax.set_xlabel("Population")
            if k == 0:
                ylabel_str = f"{basis_name} {i+1}\n" + ("Density" if use_density else "Fraction")
                ax.set_ylabel(ylabel_str, fontweight='bold')
                
            ax.set_xlim(0, 1.0)

    # Main Title
    norm_type = "Probability Density" if use_density else "Fraction of Trajectories"
    fig.suptitle(f"{basis_name} Basis Distribution ({norm_type}) | Theta = {theta_deg} deg", fontsize=20, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Add explicit tags to filenames to distinguish the normalization method
    suffix = "Density" if use_density else "Fraction"
    save_fig(fig, f"{file_prefix}_{suffix}_Theta_{theta_str}")


# ==========================
# Execute Plotting
# ==========================
print("Generating Grid Plots for Site Basis...")
plot_grid(pop_data=pop_traj_site, 
          basis_name="Site", 
          file_prefix="Grid_Hist_Site", 
          use_density=False) # Fraction

plot_grid(pop_data=pop_traj_site, 
          basis_name="Site", 
          file_prefix="Grid_Hist_Site", 
          use_density=True)  # PDF

print("Generating Grid Plots for Exciton Basis...")
plot_grid(pop_data=pop_traj_exc, 
          basis_name="Exciton", 
          file_prefix="Grid_Hist_Exciton", 
          use_density=False) # Fraction

plot_grid(pop_data=pop_traj_exc, 
          basis_name="Exciton", 
          file_prefix="Grid_Hist_Exciton", 
          use_density=True)  # PDF

print("Time-slice grid analysis completed successfully!")